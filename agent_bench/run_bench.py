"""用固定题组无人值守地测试各 agent（本地模型），输出 results.json 与 report.md。

用法：
    python agent_bench/run_bench.py                         # 所有已安装的 agent × 所有题目
    python agent_bench/run_bench.py --agents qwen,claude --tasks svg,bugfix

安全：每个工具用各自原生的权限机制，只放行题目需要的操作（均已实测生效）：
    Claude Code   acceptEdits（工作目录内改文件自动批准）+ --allowedTools 命令白名单
    OpenCode      permission 规则：按命令放行，其余 deny
    Codex         自带沙箱 workspace-write：只能写工作目录、不能联网
    Aider         不执行命令，只做 svg / bugfix（测试命令由我们固定传入）
    Qwen Code / Gemini CLI  命令白名单无人值守时不生效（实测 whoami 照样执行），
                  只做无需命令的 svg；其余题目请交互式测试（逐条确认命令）
测试用独立配置目录 agents/bench-<工具>/，不影响日常使用的配置。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
BASE = BENCH.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BENCH))
import agents  # noqa: E402
from tasks import TASKS  # noqa: E402

GATEWAY = "http://127.0.0.1:11434"
SERVER = "http://127.0.0.1:8080"
ALIAS, KEY = "llama-deploy-local", "local-no-key-needed"
# 哪些 agent 只能无人值守地做部分题目（其余 agent 做全部题）：
#   aider        不执行命令，只做不需要它自己跑命令的题（bugfix 的测试命令由我们固定传入）
#   qwen/gemini  实测（Qwen Code 0.24）无人值守时 --allowed-tools 的命令白名单不生效：
#                只放行 run_shell_command(ffprobe)，whoami 照样 auto_accept 执行。
#                需要跑命令的题等于完全放开权限，不做；这几题请交互式测试（逐条确认命令）
#   qwen         另外实测 Qwen Code 0.24 无人值守时主会话里没有写文件工具（select:write_file -> Not found），
#                连 svg 也只能打印或转交子 agent、15 分钟超时，因此整体交给交互式测试
ONLY = {"aider": {"svg", "bugfix"}, "qwen": set(), "gemini": {"svg"}}
SKIP_REASON = {"aider": "不执行命令",
               "qwen": "无人值守模式受限（无写文件工具、命令白名单不生效），需交互式测试",
               "gemini": "命令白名单无人值守时不生效，需交互式测试"}


def ensure_server():
    """确保模型服务在 agent + 看图模式、网关在运行。"""
    mode_file = BASE / ".llama-server.mode.json"
    mode = {}
    try:
        mode = json.loads(mode_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        urllib.request.urlopen(SERVER + "/health", timeout=3).read()
        up = True
    except Exception:
        up = False
    if not (up and mode.get("agent") and mode.get("vision")):
        subprocess.run([sys.executable, str(BASE / "run.py"), "stop"], capture_output=True)
        subprocess.run([sys.executable, str(BASE / "run.py"), "server", "--agent", "--vision", "--background"],
                       capture_output=True)
        for _ in range(180):
            try:
                urllib.request.urlopen(SERVER + "/health", timeout=3).read()
                break
            except Exception:
                time.sleep(2)
    subprocess.run([sys.executable, str(BASE / "compat.py"), "start"], capture_output=True)
    props = json.loads(urllib.request.urlopen(SERVER + "/props", timeout=10).read())
    return int(props["default_generation_settings"]["n_ctx"])


class Thermal:
    """每 10 秒记录一次 GPU 温度与功耗。"""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                t, p = [float(x) for x in out.stdout.strip().split(",")[:2]]
                self.samples.append((t, p))
            except Exception:
                pass
            self._stop.wait(10)

    def stop(self):
        self._stop.set()
        return {"max_temp_c": max((t for t, _ in self.samples), default=None),
                "max_power_w": max((p for _, p in self.samples), default=None)}


def one_line(prompt: str) -> str:
    """把多行提示词压成一行。

    Windows 上 claude / codex / qwen / gemini / opencode 都是 npm 生成的 .CMD 批处理包装，
    批处理会在参数的第一个换行处截断：提示词只剩第一行，排在它后面的参数
    （--permission-mode、--allowedTools、--skip-trust……）也全部丢失。实测据此得出的
    一整轮结果都是错的。% 和双引号在批处理里有特殊含义，同样不能出现。
    """
    line = " ".join(part.strip() for part in prompt.splitlines() if part.strip())
    if "%" in line or '"' in line:
        raise ValueError("提示词里不能含 % 或双引号：它们会被 Windows 批处理包装脚本解释")
    return line


def build_command(agent: str, task: dict, prompt: str, work: Path, home: Path) -> list:
    exe = str(agents.command_path(agent))
    prompt = one_line(prompt)
    cmds, dirs = task["commands"], task["read_dirs"]()
    if agent in ("qwen", "gemini"):
        # 只做无需命令的题（见 ONLY）：auto-edit 自动批准工作区内改文件，不传 --allowed-tools，
        # 实测此时 shell 命令一律被拦（ffprobe、whoami 都 blocked）
        assert not cmds, "Qwen Code / Gemini 的命令白名单无人值守时不生效，不能跑需要命令的题"
        cmd = [exe] + (["-p", prompt, "--skip-trust"] if agent == "gemini" else [prompt])
        # 注意两者拼写不同：Qwen Code 是 auto-edit，Gemini CLI 是 auto_edit（写错会直接打印帮助退出）
        return cmd + ["--approval-mode", "auto_edit" if agent == "gemini" else "auto-edit"]
    if agent == "claude":
        # acceptEdits：工作目录内改文件自动批准（实测 Write(./**) 这类路径规则匹配不上，写文件被当成待审批而拒绝）；
        # 命令仍只放行白名单
        allowed = " ".join(["Read", "Glob", "Grep"] + [f"Bash({c}:*)" for c in cmds])
        cmd = [exe, "-p", prompt, "--permission-mode", "acceptEdits", "--allowedTools", allowed, "--max-turns", "80"]
        for d in dirs:
            cmd += ["--add-dir", d]
        return cmd
    if agent == "codex":
        return [exe, "exec", "--skip-git-repo-check", "-s", "workspace-write", prompt]
    if agent == "opencode":
        return [exe, "run", prompt]
    if agent == "aider":
        cmd = [exe] + agents.command_args("aider", home, ALIAS) + ["--no-check-update", "--analytics-disable",
                                                                    "--message", prompt]
        if task["title"].startswith("修 bug"):
            cmd += ["--test-cmd", "python -m pytest -q", "--auto-test"]
            cmd += [str(p.relative_to(work)) for p in sorted((work / "invoice").glob("*.py"))]
        else:
            cmd += ["pelican.svg"]
        return cmd
    raise ValueError(agent)


def prepare(agent: str, task: dict, work: Path, n_ctx: int) -> tuple:
    home = BASE / "agents" / f"bench-{agent}"
    agents.prepare_home(agent, home, GATEWAY, ALIAS, n_ctx, vision=task["vision"])
    if agent == "opencode":
        cfg_path = home / "opencode.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        bash = {"*": "deny"}
        for c in task["commands"]:
            bash[c] = "allow"
            bash[c + " *"] = "allow"
        cfg["permission"] = {"edit": "allow", "bash": bash, "webfetch": "deny", "doom_loop": "deny",
                             "external_directory": "allow" if task["read_dirs"]() else "deny"}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    for k, v in agents.build_env(agent, home, GATEWAY, ALIAS, KEY, n_ctx).items():
        if v == "":
            env.pop(k, None)
        else:
            env[k] = v
    env["PWD"] = str(work)
    extra = [str(p) for p in agents.find_tool_dirs()]
    exe = agents.command_path(agent)
    if exe:
        extra.append(str(exe.parent))
    env["PATH"] = os.pathsep.join([env.get("PATH", "")] + extra)
    if agent == "aider":
        subprocess.run(["git", "init", "-q"], cwd=work)
        subprocess.run(["git", "add", "-A"], cwd=work)
        subprocess.run(["git", "-c", "user.name=bench", "-c", "user.email=bench@local", "commit", "-qm", "start",
                        "--allow-empty"], cwd=work)
    return home, env


def run_one(agent: str, task_id: str, root: Path, n_ctx: int) -> dict:
    task = TASKS[task_id]
    work = root / f"{agent}-{task_id}"
    work.mkdir(parents=True, exist_ok=True)
    task["setup"](work)
    home, env = prepare(agent, task, work, n_ctx)
    cmd = build_command(agent, task, task["prompt"], work, home)
    log_path = root / "logs" / f"{agent}-{task_id}.log"
    log_path.parent.mkdir(exist_ok=True)
    thermal = Thermal()
    started = time.time()
    timed_out = False
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=task["timeout"])
        except subprocess.TimeoutExpired:
            timed_out, rc = True, None
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    duration = time.time() - started
    result = task["check"](work)
    passed = sum(result["checks"].values())
    return {"agent": agent, "task": task_id, "title": task["title"], "rc": rc, "timed_out": timed_out,
            "minutes": round(duration / 60, 1), "passed": passed, "total": len(result["checks"]),
            "checks": result["checks"], "artifacts": result["artifacts"], "work": str(work), "log": str(log_path),
            **thermal.stop()}


def write_report(root: Path, results: list, n_ctx: int):
    (root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    agents_order = list(dict.fromkeys(r["agent"] for r in results))
    tasks_order = list(dict.fromkeys(r["task"] for r in results))
    lines = [f"# agent 测试报告 {root.name}", "",
             f"模型服务：agent 模式 + 看图，上下文 {n_ctx}。分数为自动检查通过项数。", "",
             "| agent | " + " | ".join(TASKS[t]["title"] for t in tasks_order) + " |",
             "|---|" + "---|" * len(tasks_order)]
    for a in agents_order:
        cells = []
        for t in tasks_order:
            r = next((x for x in results if x["agent"] == a and x["task"] == t), None)
            if r:
                cells.append(f"{r['passed']}/{r['total']} · {r['minutes']} 分钟" + (" · 超时" if r["timed_out"] else ""))
            else:
                cells.append("— " + SKIP_REASON.get(a, "未运行"))
        lines.append(f"| {agents.AGENTS[a]['name']} | " + " | ".join(cells) + " |")
    lines += ["", "## 明细", ""]
    for r in results:
        lines.append(f"### {agents.AGENTS[r['agent']]['name']} · {r['title']}")
        lines.append(f"- 用时 {r['minutes']} 分钟，退出码 {r['rc']}{'，超时' if r['timed_out'] else ''}；"
                     f"GPU 最高 {r['max_temp_c']}°C / {r['max_power_w']}W")
        for name, ok in r["checks"].items():
            lines.append(f"- {'✅' if ok else '❌'} {name}")
        lines.append(f"- 产物：`{json.dumps(r['artifacts'], ensure_ascii=False)}`")
        lines.append(f"- 工作目录：`{r['work']}`；日志：`{r['log']}`")
        lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default=",".join(a for a in agents.AGENTS if agents.installed(a)))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", default=str(agents.default_workspace() / "bench"))
    args = ap.parse_args()
    root = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    n_ctx = ensure_server()
    print(f"结果目录: {root}  上下文: {n_ctx}", flush=True)
    results = []
    for agent in [a.strip() for a in args.agents.split(",") if a.strip()]:
        for task_id in [t.strip() for t in args.tasks.split(",") if t.strip()]:
            if task_id not in ONLY.get(agent, TASKS):
                continue
            print(f"▶ {agent} · {task_id} ...", flush=True)
            r = run_one(agent, task_id, root, n_ctx)
            results.append(r)
            print(f"  {r['passed']}/{r['total']}  {r['minutes']} 分钟{'  超时' if r['timed_out'] else ''}", flush=True)
            write_report(root, results, n_ctx)
    print(f"报告: {root / 'report.md'}")


if __name__ == "__main__":
    main()
