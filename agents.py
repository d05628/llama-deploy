"""一键用本地模型启动 agent 软件（Claude Code / Codex / Gemini CLI）。

隔离原则：只影响本工具打开的那个终端窗口。
- 环境变量只写进生成的启动器（.cmd / .sh），不改系统或用户级环境变量；
- 每个 agent 使用 llama-deploy/agents/<id>/ 下独立的配置目录
  （CLAUDE_CONFIG_DIR / CODEX_HOME / GEMINI_CLI_HOME），官方的 ~/.claude、
  ~/.codex、~/.gemini 一个字节都不碰，登录状态和会话历史互不混用；
- 平时直接运行 claude / codex / gemini，仍然走官方通道。
"""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

IS_WIN = platform.system() == "Windows"

AGENTS = {
    # Qwen Code 是 Qwen 官方的 CLI agent，工具调用格式针对 Qwen 系列调过，本地跑 Qwen 模型首选
    "qwen": {"name": "Qwen Code", "command": "qwen", "tag": "推荐",
             "desc": "Qwen 官方出品，工具调用针对 Qwen 模型调优；能写代码、读写文件、执行命令，开「看图」后可操作桌面软件",
             "install": "npm install -g @qwen-code/qwen-code@latest"},
    "claude": {"name": "Claude Code", "command": "claude",
               "desc": "Anthropic 出品，功能最全的编程 agent（子任务、计划模式、MCP）；提示词较长，本地模型下偶尔会在同一条命令上反复重试",
               "install": "npm install -g @anthropic-ai/claude-code"},
    "codex": {"name": "Codex", "command": "codex",
              "desc": "OpenAI 出品，偏重在沙箱里改代码、跑命令，操作前会征求确认",
              "install": "npm install -g @openai/codex"},
    # 提示词针对 Gemini 模型调过：实测本地 Qwen 工具调用 3 次成功 2 次，偶尔把文件写到它的临时目录
    "gemini": {"name": "Gemini CLI", "command": "gemini", "tag": "不推荐",
               "desc": "Google 出品；提示词针对 Gemini 调优，本地 Qwen 下偶尔把文件写到它自己的临时目录（Qwen Code 是它的 Qwen 调优分支）。若弹出登录选项，选「2. Use Gemini API Key」即可，无需登录 Google",
               "install": "npm install -g @google/gemini-cli"},
    "opencode": {"name": "OpenCode", "command": "opencode",
                 "desc": "开源、界面美观，内置 plan/build 两种模式，支持任意 OpenAI 兼容模型",
                 "install": "npm install -g opencode-ai@latest"},
    # Aider 以 git 为中心、提示词精简，适合上下文有限的本地模型
    "aider": {"name": "Aider", "command": "aider",
              "desc": "以 git 为中心的结对编程工具：改文件无需逐次确认，每次改动自动 git 提交（/undo 可撤销）；不会执行命令（如渲染需自己运行），提示词精简",
              "install": "python -m pip install aider-install && aider-install"},
}

# 启动器里的值会被 cmd 解析，这些字符无法安全地写进 set 语句
_UNSAFE_CHARS = set('"%\r\n')
# 窗口标题前缀：「关闭 agent」按它找到本工具打开的窗口
WINDOW_TITLE_PREFIX = "llama-deploy agent - "
# cmd 按系统代码页（中文系统为 GBK）解析 .cmd 文件；mbcs 编解码器只在 Windows 上存在
LAUNCHER_ENCODING = "mbcs" if os.name == "nt" else "utf-8"


def find_tool_dirs() -> list:
    """找到 agent 常用、但通常不在 PATH 里的工具目录（目前是 Blender），取最新版本。"""
    if not IS_WIN:
        return []
    found = []
    for drive in "CDEFGHIJ":
        for root in (f"{drive}:\\Program Files\\Blender Foundation", f"{drive}:\\Blender Foundation"):
            base = Path(root)
            if base.is_dir():
                found += [p.parent for p in base.glob("*/blender.exe")]
    if not found:
        return []

    def version(p: Path):
        nums = [int(x) for x in p.name.replace("Blender", "").strip().split(".") if x.isdigit()]
        return nums or [0]
    return [max(found, key=version)]


def command_path(agent: str):
    """找到 agent 可执行文件。PATH 里没有时再看常见安装位置：

    aider-install 装到 ~/.local/bin，虽然会改用户 PATH，但已在运行的管理器进程看不到新 PATH，
    实测装完后界面仍显示"未安装"。
    """
    command = AGENTS[agent]["command"]
    found = shutil.which(command)
    if found:
        return Path(found)
    for candidate in (Path.home() / ".local" / "bin" / (command + (".exe" if IS_WIN else "")),):
        if candidate.is_file():
            return candidate
    return None


def installed(agent: str) -> bool:
    return command_path(agent) is not None


def agent_home(base_dir: Path, agent: str) -> Path:
    return base_dir / "agents" / agent


def build_env(agent: str, home: Path, gateway: str, alias: str, api_key: str, n_ctx: int) -> dict:
    """返回注入到该 agent 进程的环境变量。值为空字符串表示"在该窗口里清除继承来的同名变量"。"""
    if agent == "claude":
        return {
            "CLAUDE_CONFIG_DIR": str(home),
            "ANTHROPIC_BASE_URL": gateway,
            "ANTHROPIC_AUTH_TOKEN": api_key,
            # 清掉可能继承自用户环境的官方 Key，否则它会覆盖网关、把请求发往官方
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": alias,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": alias,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": alias,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": alias,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": alias,
            # 告诉 Claude Code 本地模型的真实窗口，自动压缩才会在撑爆前触发
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(n_ctx),
            # 长提示词在本地可能要处理几分钟
            "API_TIMEOUT_MS": "3000000",
            # Glob 背后是 ripgrep，默认 20 秒超时；实测在 20 万目录的数据盘上全盘遍历要 6-8 分钟，
            # 默认值下"在整个硬盘里找文件"必然失败（未公开的变量，已核对 Claude Code 2.1.183）
            "CLAUDE_CODE_GLOB_TIMEOUT_SECONDS": "600",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
        }
    if agent == "codex":
        return {"CODEX_HOME": str(home), "LLAMA_DEPLOY_API_KEY": api_key}
    if agent == "gemini":
        return {
            "GEMINI_CLI_HOME": str(home),
            "GOOGLE_GEMINI_BASE_URL": gateway,
            "GEMINI_API_KEY": api_key,
        }
    if agent == "qwen":
        # QWEN_HOME 直接替代 ~/.qwen（已核对 qwen-code 0.24 源码）
        return {"QWEN_HOME": str(home), "LLAMA_DEPLOY_API_KEY": api_key}
    if agent == "opencode":
        # OPENCODE_CONFIG 指向我们目录里的配置；XDG_* 把它的全局配置、会话库、缓存都
        # 重定向到本目录（它按 XDG 规范存放，默认在 ~/.local/share/opencode 等处）。
        # 这些变量只在这个只运行 opencode 的窗口里生效。
        return {
            "OPENCODE_CONFIG": str(home / "opencode.json"),
            "XDG_CONFIG_HOME": str(home / "xdg" / "config"),
            "XDG_DATA_HOME": str(home / "xdg" / "data"),
            "XDG_STATE_HOME": str(home / "xdg" / "state"),
            "XDG_CACHE_HOME": str(home / "xdg" / "cache"),
            "LLAMA_DEPLOY_API_KEY": api_key,
        }
    if agent == "aider":
        return {"OPENAI_API_BASE": f"{gateway}/v1", "OPENAI_API_KEY": api_key}
    raise ValueError(f"未知 agent: {agent}")


def command_args(agent: str, home: Path, alias: str) -> list:
    """agent 命令后面追加的参数。"""
    if agent == "aider":
        return ["--model", f"openai/{alias}",
                "--model-metadata-file", str(home / "model-metadata.json"),
                "--no-show-model-warnings",
                # 文件改动自动确认（每次改动都会 git 提交，可 /undo 撤销）；
                # 不自动执行 shell 命令：Aider 的 --yes-always 会连"运行命令"也一并同意，
                # 所以干脆不让它提议执行命令
                "--yes-always", "--no-suggest-shell-commands"]
    if agent == "codex" and IS_WIN:
        # Codex 0.157 起默认启动共享后台服务，在管理员窗口里拒绝运行：
        # "start the Windows daemon from a non-elevated terminal"
        return ["--no-daemon"]
    return []


def codex_config(gateway: str, alias: str, n_ctx: int) -> str:
    # Codex 0.15x 起只支持 Responses 协议（wire_api = "chat" 已移除），网关的 /v1/responses 对应它
    return (
        "# 由 llama-deploy 生成，每次一键启动时覆盖；只作用于 CODEX_HOME 指向本目录的会话\n"
        f'model = "{alias}"\n'
        'model_provider = "llama_deploy"\n'
        f"model_context_window = {n_ctx}\n"
        f"model_auto_compact_token_limit = {int(n_ctx * 0.8)}\n"
        "\n"
        "[model_providers.llama_deploy]\n"
        'name = "llama-deploy (local)"\n'
        f'base_url = "{gateway}/v1"\n'
        'wire_api = "responses"\n'
        'env_key = "LLAMA_DEPLOY_API_KEY"\n'
    )


def prepare_home(agent: str, home: Path, gateway: str, alias: str, n_ctx: int, vision: bool = False):
    """写入该 agent 独立配置目录里需要的文件（只写我们自己的目录）。

    vision：模型服务开了看图时，要在各工具的模型配置里显式声明支持图片。
    Qwen Code / OpenCode 对未知模型名一律按纯文本处理，不声明就不会把图片发给模型，
    实测它们都回答"当前模型不支持图片"。
    """
    home.mkdir(parents=True, exist_ok=True)
    if agent == "codex":
        (home / "config.toml").write_text(codex_config(gateway, alias, n_ctx), encoding="utf-8")
    elif agent == "qwen":
        path = home / "settings.json"
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        settings["modelProviders"] = {"openai": [{
            "id": alias,
            "name": "llama-deploy (local)",
            "baseUrl": f"{gateway}/v1",
            "envKey": "LLAMA_DEPLOY_API_KEY",
            "generationConfig": {"contextWindowSize": n_ctx, "modalities": {"image": bool(vision)}},
        }]}
        settings.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "openai"
        settings.setdefault("model", {})["name"] = alias
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    elif agent == "opencode":
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"llama-deploy": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llama-deploy (local)",
                "options": {"baseURL": f"{gateway}/v1", "apiKey": "{env:LLAMA_DEPLOY_API_KEY}"},
                "models": {alias: {"name": "本地模型", "attachment": bool(vision),
                                   "modalities": {"input": ["text", "image"] if vision else ["text"],
                                                  "output": ["text"]},
                                   "limit": {
                    "context": n_ctx, "output": max(4096, min(32768, n_ctx // 4))}}},
            }},
            "model": f"llama-deploy/{alias}",
            # 标题生成等后台任务也走本地模型，不去连云端
            "small_model": f"llama-deploy/{alias}",
            # 快照会对整个工作目录做 git 备份：实测工作目录是用户主目录时一直卡在快照、毫无反应
            "snapshot": False,
        }
        (home / "opencode.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    elif agent == "aider":
        metadata = {f"openai/{alias}": {
            "max_input_tokens": n_ctx, "max_tokens": max(4096, min(32768, n_ctx // 4)),
            "input_cost_per_token": 0, "output_cost_per_token": 0,
            "litellm_provider": "openai", "mode": "chat", "supports_vision": bool(vision),
        }}
        (home / "model-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elif agent == "gemini":
        settings_dir = home / ".gemini"
        settings_dir.mkdir(exist_ok=True)
        path = settings_dir / "settings.json"
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        # 预选 API Key 认证，否则首次启动会弹出 Google 登录
        settings.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "gemini-api-key"
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def default_workspace() -> Path:
    """agent 默认工作目录。不用用户主目录：agent 会扫描/快照整个工作目录，主目录太大。"""
    return Path.home() / "llama-agent-workspace"


def workspace_warning(path: Path) -> str:
    """工作目录过大（用户主目录、盘符根目录）时返回提示；正常返回空字符串。"""
    resolved = path.resolve()
    if resolved == Path.home().resolve():
        return "工作目录是整个用户目录：agent 会扫描其中所有文件，OpenCode 等会因此长时间无响应，建议换成具体的项目文件夹"
    if resolved.parent == resolved:
        return "工作目录是整个磁盘根目录：agent 会扫描整盘文件，建议换成具体的项目文件夹"
    return ""


def validate_cwd(cwd: str) -> Path:
    if not (cwd or "").strip():
        path = default_workspace()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path = Path(os.path.expanduser(cwd)).resolve()
    if not path.is_dir():
        raise ValueError(f"工作目录不存在: {path}")
    if _UNSAFE_CHARS & set(str(path)):
        raise ValueError("工作目录路径含有 \" 或 % 等无法安全传给终端的字符")
    return path


def write_launcher(base_dir: Path, agent: str, env: dict, cwd: Path, args: list = (), extra_path: list = ()) -> Path:
    """生成可重复双击使用的启动器。环境变量只在这个窗口的进程里生效。"""
    for key, value in list(env.items()) + [("args", a) for a in args]:
        if _UNSAFE_CHARS & set(value):
            raise ValueError(f"{key} 的值含有无法安全写入启动器的字符")
    name = AGENTS[agent]["name"]
    command = AGENTS[agent]["command"]
    base_command = command
    out_dir = base_dir / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    if IS_WIN:
        # 不切换到 UTF-8 代码页（chcp 65001）：Windows 10 旧版控制台在该模式下，Qwen Code 这类
        # 全屏终端界面写屏会报 "write UNKNOWN" 直接闪退。启动器改用系统代码页（中文系统为 GBK）保存。
        lines = ["@echo off", f"title {WINDOW_TITLE_PREFIX}{name}"]
        lines += [f'set "{k}={v}"' for k, v in env.items()]
        # cmd 的 cd 不会更新 PWD；从 Git Bash 等环境继承来的旧 PWD 会让部分工具
        # （实测 OpenCode）把文件写到启动管理器的目录，而不是用户选的工作目录
        lines.append(f'set "PWD={cwd}"')
        for tool_dir in extra_path:
            if not (_UNSAFE_CHARS & set(str(tool_dir))):
                lines.append(f'set "PATH=%PATH%;{tool_dir}"')
        command = " ".join([command] + [f'"{a}"' if " " in a else a for a in args])
        lines += [f'cd /d "{cwd}"',
                  f"echo {name} 正在使用本地模型（仅此窗口生效；平时直接运行 {base_command} 仍走官方）",
                  command]
        path = out_dir / f"launch-{agent}.cmd"
        # newline="" 原样写入：否则 Windows 文本模式会把 \n 再转一次，变成 \r\r\n
        with open(path, "w", encoding=LAUNCHER_ENCODING, errors="replace", newline="") as f:
            f.write("\r\n".join(lines) + "\r\n")
    else:
        import shlex
        lines = ["#!/bin/sh"]
        lines += [f"unset {k}" if v == "" else f"export {k}={shlex.quote(v)}" for k, v in env.items()]
        lines += [f"cd {shlex.quote(str(cwd))}", "exec " + " ".join([command] + [shlex.quote(a) for a in args])]
        path = out_dir / f"launch-{agent}.sh"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
    return path


def open_terminal(launcher: Path):
    """在新终端窗口里运行启动器。"""
    if IS_WIN:
        subprocess.Popen(["cmd.exe", "/c", "start", "", "cmd.exe", "/k", str(launcher)],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", "-a", "Terminal", str(launcher)])
        return
    for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
        if shutil.which(term):
            subprocess.Popen([term, "-e", str(launcher)])
            return
    raise RuntimeError(f"未找到图形终端，请手动运行: {launcher}")


def agent_window_pids(base_dir: Path) -> list:
    """本工具打开的 agent 窗口（cmd.exe /k <base>\\agents\\launch-*.cmd）的 PID。

    按命令行识别而不是窗口标题：Qwen Code 等启动后会改写窗口标题（实测变成 "npm view ..."）。
    """
    if not IS_WIN:
        return []
    marker = str(base_dir / "agents" / "launch-").lower()
    script = ("Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | "
              "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, errors="replace", timeout=20).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        pid, _, cmdline = line.partition("|")
        if pid.strip().isdigit() and marker in cmdline.lower():
            pids.append(int(pid))
    return pids


def close_agent_windows(base_dir: Path) -> int:
    """关闭本工具打开的 agent 窗口，连同窗口里的 agent 进程树。返回关闭的窗口数。"""
    pids = agent_window_pids(base_dir)
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    return len(pids)
