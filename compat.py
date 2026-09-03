#!/usr/bin/env python3
"""
Compatibility gateway for llama-deploy.

It exposes Ollama, Anthropic Messages, and OpenAI-compatible endpoints and
forwards generation to the local llama.cpp OpenAI server.
"""
import io
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.jsonc"
MODELS_DIR = BASE_DIR / "models"
PID_FILE = BASE_DIR / ".compat-server.pid"
LOG_FILE = BASE_DIR / ".compat-server.log"
IS_WIN = platform.system() == "Windows"
VERSION = "1.0.0"
CLIENT_GONE_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


def parse_jsonc(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<!:)//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def load_config() -> dict:
    return parse_jsonc(CONFIG_FILE)


def compat_config(cfg: dict) -> dict:
    cc = cfg.get("compat", {}) or {}
    sc = cfg.get("server", {}) or {}
    upstream = cc.get("upstream_url") or f"http://127.0.0.1:{int(sc.get('port', 8080) or 8080)}"
    return {
        "host": cc.get("host", "0.0.0.0"),
        "port": int(cc.get("port", 11434) or 11434),
        "upstream_url": str(upstream).rstrip("/"),
        "model_alias": cc.get("model_alias", "llama-deploy-local"),
        "api_key": cc.get("api_key", "local-no-key-needed"),
        "request_timeout": int(cc.get("request_timeout", 600) or 600),
        "claude_tool_mode": cc.get("claude_tool_mode", "repair"),
    }


def active_model_name(cfg: dict) -> str:
    mc = cfg.get("model", {}) or {}
    return Path(mc.get("model_file") or "local-model").name


def model_size(cfg: dict) -> int:
    mc = cfg.get("model", {}) or {}
    model_file = mc.get("model_file", "")
    if model_file:
        path = MODELS_DIR / model_file
        if path.exists():
            return path.stat().st_size
    return int(mc.get("model_size", 0) or 0)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def estimate_tokens(text: str) -> int:
    text = text or ""
    # Good enough for preflight/count-token APIs. llama.cpp still enforces real limits.
    return max(1, int(len(text) / 3.8))


def pid_running(pid: int) -> bool:
    try:
        if IS_WIN:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            )
            return r.returncode == 0 and str(pid) in (r.stdout or "")
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return isinstance(sys.exc_info()[1], PermissionError)
    except Exception:
        return False


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    out = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
            continue
        if not isinstance(block, dict):
            out.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            out.append(block.get("text", ""))
        elif btype == "image":
            out.append("[image omitted by compatibility gateway]")
        elif btype == "tool_result":
            result = content_to_text(block.get("content", ""))
            out.append(f"[tool_result {block.get('tool_use_id', '')}]\n{result}")
        elif btype == "tool_use":
            out.append(f"[tool_use {block.get('name', '')}]\n{json.dumps(block.get('input', {}), ensure_ascii=False)}")
        else:
            out.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(x for x in out if x)


def anthropic_system_to_text(system) -> str:
    return content_to_text(system)


def openai_message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = item.get("type")
                if typ in ("input_text", "output_text", "text"):
                    parts.append(item.get("text", ""))
                elif typ in ("input_image", "image_url"):
                    parts.append("[image omitted by compatibility gateway]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(content)


def normalize_openai_messages(messages: list) -> list:
    system_parts = []
    normalized = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            normalized.append({"role": "user", "content": str(msg)})
            continue

        role = msg.get("role", "user")
        if role == "developer":
            role = "system"
        if role == "system":
            text = openai_message_text(msg.get("content", ""))
            if text:
                system_parts.append(text)
            continue

        clean = dict(msg)
        clean["role"] = role if role in ("user", "assistant", "tool") else "user"
        clean["content"] = openai_message_text(clean.get("content", ""))
        normalized.append(clean)

    if system_parts:
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + normalized
    return normalized or [{"role": "user", "content": ""}]


def prepare_openai_chat_payload(payload: dict) -> dict:
    prepared = dict(payload or {})
    prepared["messages"] = normalize_openai_messages(prepared.get("messages") or [])
    return prepared


def request_thinking_enabled(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "enabled", "medium", "high")
    if isinstance(value, dict):
        typ = str(value.get("type", "")).strip().lower()
        if typ in ("disabled", "off", "none"):
            return False
        if typ in ("enabled", "on"):
            return True
        effort = str(value.get("effort", "")).strip().lower()
        if effort:
            return effort not in ("minimal", "none", "off", "disabled")
        for key in ("budget_tokens", "thinkingBudget", "thinking_budget"):
            if key in value:
                try:
                    return int(value.get(key) or 0) > 0
                except (TypeError, ValueError):
                    return default
        if "includeThoughts" in value:
            return bool(value.get("includeThoughts"))
    return default


def apply_template_thinking(payload: dict, enabled: bool) -> dict:
    out = dict(payload or {})
    kwargs = dict(out.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = bool(enabled)
    out["chat_template_kwargs"] = kwargs
    return out


def responses_input_to_messages(payload: dict) -> list:
    messages = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": openai_message_text(instructions)})
    if payload.get("messages"):
        for msg in payload.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role not in ("system", "user", "assistant", "developer"):
                role = "user"
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": openai_message_text(msg.get("content", ""))})
        return normalize_openai_messages(messages)
    inp = payload.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                typ = item.get("type")
                if typ in ("message", None):
                    role = item.get("role", "user")
                    if role not in ("system", "user", "assistant", "developer"):
                        role = "user"
                    if role == "developer":
                        role = "system"
                    messages.append({"role": role, "content": openai_message_text(item.get("content", ""))})
                elif typ == "function_call_output":
                    messages.append({"role": "tool", "content": openai_message_text(item.get("output", ""))})
                else:
                    messages.append({"role": "user", "content": json.dumps(item, ensure_ascii=False)})
            else:
                messages.append({"role": "user", "content": str(item)})
    return normalize_openai_messages(messages)


def openai_tools_from_responses(payload: dict) -> list:
    out = []
    for tool in payload.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and tool.get("name"):
            out.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            })
    return out


def tool_schemas_from_payload(payload: dict) -> dict:
    schemas = {}
    for tool in payload.get("tools", []) or []:
        if isinstance(tool, dict) and tool.get("name"):
            schemas[tool["name"]] = tool.get("input_schema") or {"type": "object", "properties": {}}
    return schemas


def tool_hint_text(tools: list) -> str:
    names = [t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name")]
    if not names:
        return ""
    return (
        "Tool-use compatibility rules for Claude Code: when you use a tool, emit exactly one structured tool call "
        "with JSON arguments matching the provided schema. Do not write textual pseudo-tags like [tool_use], "
        "<parameter=...>, or XML blocks. For Bash, arguments must include a non-empty command string and a short "
        "non-empty description string. Prefer simple, valid tool calls over parallel calls. Available tools: "
        + ", ".join(names)
    )


def anthropic_to_openai(payload: dict, cfg: dict) -> dict:
    cc = compat_config(cfg)
    messages = []
    system = anthropic_system_to_text(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    if payload.get("tools") and cc.get("claude_tool_mode") != "text_only":
        messages.append({"role": "system", "content": tool_hint_text(payload.get("tools") or [])})
    for msg in payload.get("messages", []) or []:
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = content_to_text(msg.get("content", ""))
        messages.append({"role": role, "content": text})

    out = {
        "model": payload.get("model") or cc["model_alias"],
        "messages": normalize_openai_messages(messages),
        "stream": False,
        "max_tokens": int(payload.get("max_tokens") or 1024),
    }
    for key in ("temperature", "top_p", "top_k"):
        if key in payload:
            out[key] = payload[key]
    if payload.get("stop_sequences"):
        out["stop"] = payload.get("stop_sequences")
    if payload.get("tools") and cc.get("claude_tool_mode") != "text_only":
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in payload.get("tools", [])
            if isinstance(t, dict) and t.get("name")
        ]
        out["parallel_tool_calls"] = False
        out["tool_choice"] = "auto"
    if payload.get("tool_choice") and cc.get("claude_tool_mode") != "text_only":
        choice = payload["tool_choice"]
        if isinstance(choice, dict) and choice.get("type") == "tool" and choice.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return apply_template_thinking(out, request_thinking_enabled(payload.get("thinking"), False))


def _default_schema_value(prop: str, schema: dict, tool_name: str):
    default = schema.get("default")
    if default is not None:
        return default
    typ = schema.get("type", "string")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), typ[0] if typ else "string")
    prop_l = prop.lower()
    tool_l = tool_name.lower()
    if typ == "string":
        if prop_l == "description" and tool_l == "bash":
            return "Run a shell command"
        if prop_l in ("path", "folder", "dir", "directory") and tool_l in ("ls", "list"):
            return "."
        return None
    if typ == "array":
        return []
    if typ == "object":
        return {}
    if typ == "boolean":
        return False
    if typ in ("number", "integer"):
        return 0
    return None


def _coerce_schema_value(prop: str, value, schema: dict, tool_name: str):
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
    if typ == "object":
        return value if isinstance(value, dict) else {}
    if typ == "array":
        return value if isinstance(value, list) else []
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if typ == "integer":
        try:
            return int(value)
        except Exception:
            return _default_schema_value(prop, schema, tool_name)
    if typ == "number":
        try:
            return float(value)
        except Exception:
            return _default_schema_value(prop, schema, tool_name)
    if value is None:
        return _default_schema_value(prop, schema, tool_name)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_tool_args(tool_name: str, raw_args):
    if isinstance(raw_args, dict):
        args = dict(raw_args)
    elif isinstance(raw_args, str):
        text = raw_args.strip()
        try:
            args = json.loads(text) if text else {}
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {"command": text} if tool_name.lower() == "bash" else {"raw": text}
    else:
        args = {}

    if tool_name.lower() == "bash":
        if "command" not in args:
            for alias in ("cmd", "shell", "script"):
                if args.get(alias):
                    args["command"] = args[alias]
                    break
        if args.get("command") and not args.get("description"):
            args["description"] = "Run a shell command"
    return args


def sanitize_tool_call(tool_name: str, raw_args, schema: dict):
    args = _normalize_tool_args(tool_name, raw_args)
    schema = schema or {"type": "object", "properties": {}}
    props = schema.get("properties") or {}
    required = schema.get("required") or []

    if props:
        clean = {}
        for key, val in args.items():
            if key in props:
                clean[key] = _coerce_schema_value(key, val, props[key], tool_name)
        for key in required:
            missing = key not in clean or clean.get(key) in (None, "")
            if missing:
                default = _default_schema_value(key, props.get(key, {}), tool_name)
                if default is None:
                    return None, f"missing required tool parameter: {key}"
                clean[key] = default
        args = clean
    elif not isinstance(args, dict):
        args = {}

    if tool_name.lower() == "bash" and (not args.get("command")):
        return None, "missing Bash command"
    if tool_name.lower() == "bash" and not args.get("description"):
        args["description"] = "Run a shell command"
    return args, ""


def openai_to_anthropic(resp: dict, cfg: dict, model_name: str = "", tool_schemas: dict = None) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_blocks = []
    tool_calls = message.get("tool_calls") or []
    tool_schemas = tool_schemas or {}
    accepted_tool_calls = 0
    for call in tool_calls:
        fn = call.get("function") or {}
        tool_name = fn.get("name", "")
        try:
            raw_args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            raw_args = fn.get("arguments", "")
        args, err = sanitize_tool_call(tool_name, raw_args, tool_schemas.get(tool_name, {}))
        if args is None:
            content_blocks.append({
                "type": "text",
                "text": f"[兼容网关已拦截一个无效工具调用: {tool_name or 'unknown'} ({err})]",
            })
            continue
        accepted_tool_calls += 1
        content_blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": tool_name,
            "input": args,
        })
    text = message.get("content") or ""
    if text:
        content_blocks.append({"type": "text", "text": text})
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    finish = choice.get("finish_reason") or "stop"
    stop_reason = "tool_use" if accepted_tool_calls else ("max_tokens" if finish == "length" else "end_turn")
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model_name or active_model_name(cfg),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def http_json(method: str, url: str, payload=None, timeout: int = 600) -> tuple:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"error": body or str(e)}
    except Exception as e:
        return 502, {"error": {"message": str(e), "type": "upstream_error"}}


def iter_openai_chat_stream(url: str, payload: dict, timeout: int = 600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            item = line[5:].strip()
            if item == "[DONE]":
                break
            try:
                chunk = json.loads(item)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            if text:
                yield text


def ollama_model_entries(cfg: dict, alias: str) -> list:
    active = active_model_name(cfg)
    size = model_size(cfg)
    names = []
    for name in (alias, active):
        if name and name not in names:
            names.append(name)
    return [
        {
            "name": name,
            "model": name,
            "modified_at": now_iso(),
            "size": size,
            "digest": "local",
            "details": {
                "format": "gguf",
                "family": "local",
                "families": ["local"],
                "parameter_size": "local",
                "quantization_level": "local",
            },
        }
        for name in names
    ]


def gemini_model_entries(cfg: dict, alias: str) -> list:
    names = []
    for name in (alias, "gemini-2.5-pro", "gemini-2.5-flash", active_model_name(cfg)):
        if name and name not in names:
            names.append(name)
    return [
        {
            "name": f"models/{name}",
            "version": "001",
            "displayName": name,
            "description": "Local llama-deploy model exposed through Gemini-compatible API",
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
        }
        for name in names
    ]


def gemini_part_text(part) -> str:
    if not isinstance(part, dict):
        return str(part)
    if "text" in part:
        return str(part.get("text", ""))
    if "functionCall" in part:
        fc = part.get("functionCall") or {}
        return "[function_call {}]\n{}".format(fc.get("name", ""), json.dumps(fc.get("args", {}), ensure_ascii=False))
    if "functionResponse" in part:
        fr = part.get("functionResponse") or {}
        return "[function_response {}]\n{}".format(fr.get("name", ""), json.dumps(fr.get("response", {}), ensure_ascii=False))
    if "inlineData" in part or "fileData" in part:
        return "[media omitted by compatibility gateway]"
    return json.dumps(part, ensure_ascii=False)


def gemini_to_openai(payload: dict, model: str) -> dict:
    messages = []
    sys_inst = payload.get("systemInstruction")
    if sys_inst:
        parts = sys_inst.get("parts", []) if isinstance(sys_inst, dict) else []
        text = "\n".join(gemini_part_text(p) for p in parts)
        if text:
            messages.append({"role": "system", "content": text})

    for content in payload.get("contents", []) or []:
        if not isinstance(content, dict):
            continue
        role = content.get("role", "user")
        role = "assistant" if role == "model" else "user"
        parts = content.get("parts", []) or []
        text = "\n".join(gemini_part_text(p) for p in parts)
        messages.append({"role": role, "content": text})

    gen = payload.get("generationConfig") or {}
    out = {
        "model": model,
        "messages": normalize_openai_messages(messages),
        "stream": False,
    }
    if "temperature" in gen:
        out["temperature"] = gen["temperature"]
    if "topP" in gen:
        out["top_p"] = gen["topP"]
    if "topK" in gen:
        out["top_k"] = gen["topK"]
    if "maxOutputTokens" in gen:
        out["max_tokens"] = gen["maxOutputTokens"]
    out = apply_template_thinking(out, request_thinking_enabled(gen.get("thinkingConfig"), False))

    tools = []
    for tool in payload.get("tools", []) or []:
        for fd in tool.get("functionDeclarations", []) or []:
            if isinstance(fd, dict) and fd.get("name"):
                tools.append({
                    "type": "function",
                    "function": {
                        "name": fd.get("name"),
                        "description": fd.get("description", ""),
                        "parameters": fd.get("parameters") or {"type": "object", "properties": {}},
                    },
                })
    if tools:
        out["tools"] = tools
        out["parallel_tool_calls"] = False
        out["tool_choice"] = "auto"
    return out


def openai_to_gemini(resp: dict) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    parts = []
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"raw": fn.get("arguments", "")}
        parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
    text = message.get("content") or ""
    if text:
        parts.append({"text": text})
    if not parts:
        parts.append({"text": ""})
    finish = choice.get("finish_reason") or "stop"
    usage = resp.get("usage") or {}
    return {
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": "MAX_TOKENS" if finish == "length" else "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": int(usage.get("prompt_tokens", 0) or 0),
            "candidatesTokenCount": int(usage.get("completion_tokens", 0) or 0),
            "totalTokenCount": int(usage.get("total_tokens", 0) or 0),
        },
        "modelVersion": resp.get("model", ""),
    }


class CompatHandler(BaseHTTPRequestHandler):
    server_version = f"llama-deploy-compat/{VERSION}"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key, Anthropic-Version, Anthropic-Beta")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n).decode("utf-8", errors="replace") if n else "{}"
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}

    def _model_id(self, cfg: dict) -> str:
        return (compat_config(cfg).get("model_alias") or active_model_name(cfg))

    def _call_openai_chat(self, payload: dict, cfg: dict) -> tuple:
        cc = compat_config(cfg)
        payload = prepare_openai_chat_payload(payload)
        payload["stream"] = False
        return http_json("POST", cc["upstream_url"] + "/v1/chat/completions", payload, cc["request_timeout"])

    def _stream_ollama_chat(self, openai_payload: dict, cfg: dict, model: str):
        cc = compat_config(cfg)
        payload = prepare_openai_chat_payload(openai_payload)
        payload["stream"] = True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        try:
            for text in iter_openai_chat_stream(cc["upstream_url"] + "/v1/chat/completions", payload, cc["request_timeout"]):
                item = {
                    "model": model,
                    "created_at": now_iso(),
                    "message": {"role": "assistant", "content": text},
                    "done": False,
                }
                self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            final = {"model": model, "created_at": now_iso(), "message": {"role": "assistant", "content": ""}, "done": True}
            self.wfile.write((json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        except CLIENT_GONE_ERRORS:
            return
        except Exception as e:
            err = {"model": model, "created_at": now_iso(), "message": {"role": "assistant", "content": ""}, "done": True, "error": str(e)}
            try:
                self.wfile.write((json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except CLIENT_GONE_ERRORS:
                return

    def _stream_ollama_generate(self, openai_payload: dict, cfg: dict, model: str):
        cc = compat_config(cfg)
        payload = prepare_openai_chat_payload(openai_payload)
        payload["stream"] = True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        try:
            for text in iter_openai_chat_stream(cc["upstream_url"] + "/v1/chat/completions", payload, cc["request_timeout"]):
                item = {"model": model, "created_at": now_iso(), "response": text, "done": False}
                self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            final = {"model": model, "created_at": now_iso(), "response": "", "done": True}
            self.wfile.write((json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        except CLIENT_GONE_ERRORS:
            return
        except Exception as e:
            err = {"model": model, "created_at": now_iso(), "response": "", "done": True, "error": str(e)}
            try:
                self.wfile.write((json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except CLIENT_GONE_ERRORS:
                return

    def _stream_gemini(self, openai_payload: dict, cfg: dict):
        cc = compat_config(cfg)
        payload = prepare_openai_chat_payload(openai_payload)
        payload["stream"] = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        try:
            for text in iter_openai_chat_stream(cc["upstream_url"] + "/v1/chat/completions", payload, cc["request_timeout"]):
                chunk = {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "index": 0,
                    }]
                }
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            final = {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP", "index": 0}]}
            self.wfile.write(("data: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except CLIENT_GONE_ERRORS:
            return
        except Exception as e:
            err = {"error": {"message": str(e), "status": "UNAVAILABLE"}}
            try:
                self.wfile.write(("data: " + json.dumps(err, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except CLIENT_GONE_ERRORS:
                return

    def _stream_openai_response(self, response: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

        def event(name: str, data: dict):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        event("response.created", {"type": "response.created", "response": {k: v for k, v in response.items() if k != "output"}})
        text = response.get("output_text", "")
        if text:
            event("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_0", "output_index": 0, "content_index": 0, "delta": text})
        event("response.completed", {"type": "response.completed", "response": response})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_anthropic_stream(self, message: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

        def event(name: str, data: dict):
            raw = f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(raw)
            self.wfile.flush()

        msg = dict(message)
        content = msg.pop("content", [])
        msg["content"] = []
        event("message_start", {"type": "message_start", "message": msg})
        for idx, block in enumerate(content):
            if block.get("type") == "text":
                event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                text = block.get("text", "")
                if text:
                    event("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text}})
                event("content_block_stop", {"type": "content_block_stop", "index": idx})
            elif block.get("type") == "tool_use":
                start_block = {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}
                event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": start_block})
                partial = json.dumps(block.get("input", {}), ensure_ascii=False)
                if partial:
                    event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": partial},
                    })
                event("content_block_stop", {"type": "content_block_stop", "index": idx})
        event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": message.get("stop_reason", "end_turn"), "stop_sequence": None},
            "usage": {"output_tokens": message.get("usage", {}).get("output_tokens", 0)},
        })
        event("message_stop", {"type": "message_stop"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_HEAD(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/health", "/api/version", "/api/tags", "/api/ps", "/v1/models", "/v1beta/models", "/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            self.send_response(404)
        self._cors()
        self.end_headers()

    def do_GET(self):
        cfg = load_config()
        cc = compat_config(cfg)
        path = urllib.parse.urlparse(self.path).path
        model = self._model_id(cfg)
        if path in ("/", "/health"):
            self._send_json({
                "status": "ok",
                "service": "llama-deploy compatibility gateway",
                "version": VERSION,
                "upstream": cc["upstream_url"],
                "model": active_model_name(cfg),
                "alias": model,
                "endpoints": ["openai", "ollama", "anthropic"],
            })
        elif path == "/api/version":
            self._send_json({"version": f"llama-deploy-compat-{VERSION}"})
        elif path == "/api/tags":
            self._send_json({"models": ollama_model_entries(cfg, model)})
        elif path == "/api/ps":
            self._send_json({"models": ollama_model_entries(cfg, model)})
        elif path == "/v1/models":
            if self.headers.get("anthropic-version"):
                self._send_json({
                    "data": [{"id": model, "type": "model", "display_name": model, "created_at": now_iso()}],
                    "has_more": False,
                    "first_id": model,
                    "last_id": model,
                })
            else:
                self._send_json({"object": "list", "data": [{"id": model, "object": "model", "created": int(time.time()), "owned_by": "llama-deploy"}]})
        elif path in ("/v1beta/models", "/v1/models"):
            self._send_json({"models": gemini_model_entries(cfg, model)})
        elif re.match(r"^/(v1beta|v1)/models/([^:/]+)$", path):
            name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
            entries = gemini_model_entries(cfg, model)
            found = next((m for m in entries if m["name"] == f"models/{name}" or m["displayName"] == name), entries[0])
            self._send_json(found)
        else:
            self._send_json({"error": f"not found: {path}"}, 404)

    def do_POST(self):
        cfg = load_config()
        cc = compat_config(cfg)
        path = urllib.parse.urlparse(self.path).path
        data = self._read_json()
        model = self._model_id(cfg)

        if path == "/v1/messages/count_tokens":
            text = anthropic_system_to_text(data.get("system"))
            for msg in data.get("messages", []) or []:
                text += "\n" + content_to_text(msg.get("content", ""))
            self._send_json({"input_tokens": estimate_tokens(text)})
            return

        if path == "/v1/messages":
            openai_payload = anthropic_to_openai(data, cfg)
            tool_schemas = tool_schemas_from_payload(data)
            status, upstream = self._call_openai_chat(openai_payload, cfg)
            if status >= 400:
                self._send_json(upstream, status)
                return
            message = openai_to_anthropic(upstream, cfg, data.get("model") or model, tool_schemas)
            if data.get("stream"):
                self._send_anthropic_stream(message)
            else:
                self._send_json(message)
            return

        if path == "/api/chat":
            messages = data.get("messages") or []
            openai_payload = {
                "model": data.get("model") or model,
                "messages": [{"role": m.get("role", "user"), "content": content_to_text(m.get("content", ""))} for m in messages],
                "stream": False,
            }
            options = data.get("options") or {}
            if "temperature" in options:
                openai_payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                openai_payload["max_tokens"] = options["num_predict"]
            openai_payload = apply_template_thinking(
                openai_payload,
                request_thinking_enabled(data.get("think", options.get("think", options.get("enable_thinking"))), False),
            )
            if data.get("stream", True):
                self._stream_ollama_chat(openai_payload, cfg, data.get("model") or model)
                return
            status, upstream = self._call_openai_chat(openai_payload, cfg)
            if status >= 400:
                self._send_json({"error": upstream}, status)
                return
            choice = (upstream.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "")
            resp = {
                "model": data.get("model") or model,
                "created_at": now_iso(),
                "message": {"role": "assistant", "content": text},
                "done": True,
            }
            self._send_json(resp)
            return

        if path == "/api/generate":
            messages = []
            if data.get("system"):
                messages.append({"role": "system", "content": data.get("system", "")})
            messages.append({"role": "user", "content": data.get("prompt", "")})
            openai_payload = {"model": data.get("model") or model, "messages": messages, "stream": False}
            options = data.get("options") or {}
            if "temperature" in options:
                openai_payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                openai_payload["max_tokens"] = options["num_predict"]
            openai_payload = apply_template_thinking(
                openai_payload,
                request_thinking_enabled(data.get("think", options.get("think", options.get("enable_thinking"))), False),
            )
            if data.get("stream", True):
                self._stream_ollama_generate(openai_payload, cfg, data.get("model") or model)
                return
            status, upstream = self._call_openai_chat(openai_payload, cfg)
            if status >= 400:
                self._send_json({"error": upstream}, status)
                return
            choice = (upstream.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "")
            resp = {"model": data.get("model") or model, "created_at": now_iso(), "response": text, "done": True}
            self._send_json(resp)
            return

        if path == "/api/show":
            self._send_json({
                "modelfile": f"FROM {active_model_name(cfg)}",
                "parameters": "",
                "template": "",
                "details": {"format": "gguf", "family": "local", "parameter_size": "local", "quantization_level": "local"},
                "model_info": {"general.name": active_model_name(cfg), "llama-deploy.alias": model},
            })
            return

        if path in ("/api/pull", "/api/create"):
            resp = {"status": "success", "done": True}
            if data.get("stream", True):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            else:
                self._send_json(resp)
            return

        m = re.match(r"^/(v1beta|v1)/models/([^:/]+)(?::(generateContent|streamGenerateContent|countTokens))$", path)
        if m:
            gemini_model = urllib.parse.unquote(m.group(2))
            method = m.group(3)
            if method == "countTokens":
                text = ""
                for content in data.get("contents", []) or []:
                    for part in content.get("parts", []) or []:
                        text += "\n" + gemini_part_text(part)
                self._send_json({"totalTokens": estimate_tokens(text)})
                return
            openai_payload = gemini_to_openai(data, gemini_model or model)
            if method == "streamGenerateContent":
                self._stream_gemini(openai_payload, cfg)
                return
            status, upstream = self._call_openai_chat(openai_payload, cfg)
            if status >= 400:
                self._send_json(upstream, status)
                return
            self._send_json(openai_to_gemini(upstream))
            return

        if path == "/v1/responses":
            messages = responses_input_to_messages(data)
            openai_payload = {
                "model": data.get("model") or model,
                "messages": messages,
                "stream": False,
                "max_tokens": int(data.get("max_output_tokens") or data.get("max_tokens") or 1024),
            }
            if data.get("temperature") is not None:
                openai_payload["temperature"] = data.get("temperature")
            openai_payload = apply_template_thinking(
                openai_payload,
                request_thinking_enabled(data.get("reasoning"), False),
            )
            tools = openai_tools_from_responses(data)
            if tools:
                openai_payload["tools"] = tools
                openai_payload["parallel_tool_calls"] = False
                openai_payload["tool_choice"] = "auto"
            status, upstream = self._call_openai_chat(openai_payload, cfg)
            if status >= 400:
                self._send_json(upstream, status)
                return
            choice = (upstream.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            resp_id = f"resp_{uuid.uuid4().hex}"
            output_items = []
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                output_items.append({
                    "id": call.get("id") or f"fc_{uuid.uuid4().hex[:16]}",
                    "type": "function_call",
                    "status": "completed",
                    "name": fn.get("name", ""),
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                    "arguments": fn.get("arguments") or "{}",
                })
            if text or not output_items:
                output_items.append({
                    "id": f"msg_{uuid.uuid4().hex[:16]}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                })
            usage = upstream.get("usage") or {}
            response = {
                "id": resp_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": data.get("model") or model,
                "output": output_items,
                "output_text": text,
                "parallel_tool_calls": False,
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
            }
            if data.get("stream"):
                self._stream_openai_response(response)
            else:
                self._send_json(response)
            return

        if path in ("/v1/chat/completions", "/v1/completions"):
            upstream_payload = prepare_openai_chat_payload(data) if path == "/v1/chat/completions" else data
            if path == "/v1/chat/completions" and not upstream_payload.get("chat_template_kwargs"):
                thinking_value = upstream_payload.get("reasoning", upstream_payload.get("enable_thinking"))
                upstream_payload = apply_template_thinking(
                    upstream_payload,
                    request_thinking_enabled(thinking_value, False),
                )
            status, upstream = http_json("POST", cc["upstream_url"] + path, upstream_payload, cc["request_timeout"])
            self._send_json(upstream, status)
            return

        self._send_json({"error": f"not found: {path}"}, 404)


def serve() -> int:
    cfg = load_config()
    cc = compat_config(cfg)
    server = ThreadingHTTPServer((cc["host"], cc["port"]), CompatHandler)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    print(f"llama-deploy compatibility gateway {VERSION}")
    print(f"listening: http://{cc['host']}:{cc['port']}")
    print(f"upstream:  {cc['upstream_url']}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        PID_FILE.unlink(missing_ok=True)
    return 0


def cmd_start() -> int:
    if PID_FILE.exists():
        text = PID_FILE.read_text(encoding="utf-8", errors="replace").strip()
        if text.isdigit() and pid_running(int(text)):
            print(f"compat gateway already running (PID: {text})")
            return 0
        PID_FILE.unlink(missing_ok=True)

    try:
        log_f = open(LOG_FILE, "w", encoding="utf-8")
    except OSError as e:
        print(f"cannot open log file: {e}")
        return 1
    try:
        kwargs = {"stdout": log_f, "stderr": subprocess.STDOUT, "cwd": str(BASE_DIR)}
        if IS_WIN:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve"], **kwargs)
        time.sleep(1.5)
        if proc.poll() is not None:
            log_f.close()
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-1000:] if LOG_FILE.exists() else ""
            print(f"compat gateway failed (exit {proc.returncode})\n{tail}")
            return 1
        print(f"compat gateway started (PID: {proc.pid})")
        return 0
    finally:
        try:
            log_f.close()
        except Exception:
            pass


def cmd_stop() -> int:
    if not PID_FILE.exists():
        print("compat gateway is not running")
        return 0
    text = PID_FILE.read_text(encoding="utf-8", errors="replace").strip()
    if not text.isdigit():
        PID_FILE.unlink(missing_ok=True)
        print("invalid PID file removed")
        return 0
    pid = int(text)
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
        else:
            os.kill(pid, signal.SIGTERM)
        print(f"compat gateway stopped (PID: {pid})")
        return 0
    except ProcessLookupError:
        print("compat gateway process is gone")
        return 0
    except Exception as e:
        print(f"stop failed: {e}")
        return 1
    finally:
        PID_FILE.unlink(missing_ok=True)


def cmd_status() -> int:
    cfg = load_config()
    cc = compat_config(cfg)
    running = False
    pid = ""
    if PID_FILE.exists():
        pid = PID_FILE.read_text(encoding="utf-8", errors="replace").strip()
        running = pid.isdigit() and pid_running(int(pid))
    print("compat gateway status")
    print(f"  running:  {running}")
    print(f"  pid:      {pid or '-'}")
    print(f"  listen:   http://{cc['host']}:{cc['port']}")
    print(f"  upstream: {cc['upstream_url']}")
    return 0


def print_help():
    print("""Usage:
  python compat.py start
  python compat.py stop
  python compat.py status
  python compat.py serve
""")


def main() -> int:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "help").lower()
    if cmd == "start":
        return cmd_start()
    if cmd == "stop":
        return cmd_stop()
    if cmd == "status":
        return cmd_status()
    if cmd == "serve":
        return serve()
    print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
