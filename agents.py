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
             "install": "npm install -g @qwen-code/qwen-code@latest"},
    "claude": {"name": "Claude Code", "command": "claude",
               "install": "npm install -g @anthropic-ai/claude-code"},
    "codex": {"name": "Codex", "command": "codex",
              "install": "npm install -g @openai/codex"},
    # 提示词针对 Gemini 模型调过：实测本地 Qwen 工具调用 3 次成功 2 次，偶尔把文件写到它的临时目录
    "gemini": {"name": "Gemini CLI", "command": "gemini", "tag": "不推荐",
               "install": "npm install -g @google/gemini-cli"},
    "opencode": {"name": "OpenCode", "command": "opencode",
                 "install": "npm install -g opencode-ai@latest"},
    # Aider 以 git 为中心、提示词精简，适合上下文有限的本地模型
    "aider": {"name": "Aider", "command": "aider",
              "install": "python -m pip install aider-install && aider-install"},
}

# 启动器里的值会被 cmd 解析，这些字符无法安全地写进 set 语句
_UNSAFE_CHARS = set('"%\r\n')


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


def installed(agent: str) -> bool:
    return shutil.which(AGENTS[agent]["command"]) is not None


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
                "--no-show-model-warnings"]
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


def prepare_home(agent: str, home: Path, gateway: str, alias: str, n_ctx: int):
    """写入该 agent 独立配置目录里需要的文件（只写我们自己的目录）。"""
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
            "generationConfig": {"contextWindowSize": n_ctx},
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
                "models": {alias: {"name": "本地模型", "limit": {
                    "context": n_ctx, "output": max(4096, min(32768, n_ctx // 4))}}},
            }},
            "model": f"llama-deploy/{alias}",
            # 标题生成等后台任务也走本地模型，不去连云端
            "small_model": f"llama-deploy/{alias}",
        }
        (home / "opencode.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    elif agent == "aider":
        metadata = {f"openai/{alias}": {
            "max_input_tokens": n_ctx, "max_tokens": max(4096, min(32768, n_ctx // 4)),
            "input_cost_per_token": 0, "output_cost_per_token": 0,
            "litellm_provider": "openai", "mode": "chat",
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


def validate_cwd(cwd: str) -> Path:
    path = Path(os.path.expanduser(cwd or "~")).resolve()
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
        lines = ["@echo off", "chcp 65001 >nul", f"title {name} · 本地模型 (llama-deploy)"]
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
        path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
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
