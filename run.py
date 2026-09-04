#!/usr/bin/env python3
import csv
import io
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
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
LLAMA_DIR = BASE_DIR / "llama.cpp"
MODELS_DIR = BASE_DIR / "models"
CONFIG_FILE = BASE_DIR / "config.jsonc"
PID_FILE = BASE_DIR / ".llama-server.pid"
LOG_FILE = BASE_DIR / ".llama-server.log"
IS_WIN = platform.system() == "Windows"
HELP_CACHE = {}
META_CACHE = {}
TENSOR_CACHE = {}
PHYSICAL_CPU_CACHE = None
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


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


def rc(cmd: list, timeout: int = 5) -> subprocess.CompletedProcess:
    """执行子命令，强制 UTF-8 输出，跨平台安全。"""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        # 命令不存在（如 nvidia-smi 未安装），返回虚拟失败对象
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")


def help_text(binary: Path) -> str:
    """获取二进制的帮助文本，用于 supports() 特性检测。结果缓存。"""
    key = str(binary.resolve()) if binary.exists() else str(binary)
    if key not in HELP_CACHE:
        text = ""
        for probe in ([str(binary), "--help"], [str(binary), "-h"]):
            try:
                res = subprocess.run(
                    probe,
                    capture_output=True,
                    timeout=10,
                )
                # 帮助文本可能在 stdout 或 stderr，合并处理
                stdout = res.stdout.decode("utf-8", errors="replace") if res.stdout else ""
                stderr = res.stderr.decode("utf-8", errors="replace") if res.stderr else ""
                text = (stdout + "\n" + stderr).strip()
                if text:
                    break
            except FileNotFoundError:
                break  # 二进制不存在，不必再试
            except subprocess.TimeoutExpired:
                break  # 超时，放弃
            except Exception:
                pass
        HELP_CACHE[key] = text
    return HELP_CACHE[key]


def supports(binary: Path, *flags: str) -> bool:
    return binary.exists() and all(flag in help_text(binary) for flag in flags)


def supports_value(binary: Path, option: str, value: str) -> bool:
    """检查某个参数的帮助行是否明确列出了指定值，避免全文误匹配。"""
    if not binary.exists():
        return False
    lines = help_text(binary).splitlines()
    for index, line in enumerate(lines):
        if option in line:
            block_lines = [line]
            for continuation in lines[index + 1:index + 4]:
                if continuation.lstrip().startswith("-"):
                    break
                block_lines.append(continuation)
            block = " ".join(block_lines).lower()
            if re.search(rf"(?:^|[^a-z0-9_-]){re.escape(value.lower())}(?:$|[^a-z0-9_-])", block):
                return True
    return False


def find_binary(name: str) -> Path:
    aliases = {"llama-cli": ["llama-cli", "main"], "llama-server": ["llama-server", "server"]}.get(name, [name])
    envs = [os.environ.get(f"{name.upper().replace('-', '_')}_BIN", ""), os.environ.get("LLAMA_CPP_BIN", "")]
    for item in envs:
        if item and Path(item).exists():
            return Path(item)
    roots = [Path(v) for k in ("LLAMA_CPP_DIR", "LLAMA_BIN_DIR") if (v := os.environ.get(k, "").strip())]
    roots.append(LLAMA_DIR)
    for alias in aliases:
        exe = f"{alias}.exe" if IS_WIN else alias
        for root in roots:
            for cand in [root / exe, root / "build" / "bin" / exe, root / "build" / "bin" / "Release" / exe, root / "bin" / exe]:
                if cand.exists():
                    return cand
        if which := shutil.which(alias):
            return Path(which)
        for root in roots:
            if root.exists():
                for cand in root.rglob(exe):
                    if cand.is_file():
                        return cand
    return LLAMA_DIR / (f"{aliases[0]}.exe" if IS_WIN else aliases[0])


def resolve_model(model_path: Path, mc: dict) -> Path:
    model_file, shard_files = mc.get("model_file", ""), mc.get("shard_files", [])
    if model_path.exists() and model_path.stat().st_size > 1000:
        m = re.match(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", model_path.name)
        if m and m.group(2) != "00001":
            first = model_path.parent / f"{m.group(1)}-00001-of-{m.group(3)}.gguf"
            if first.exists():
                print(f"   自动修正分片入口: {first.name}")
                return first
        return model_path
    search = sorted(shard_files)[0] if shard_files else model_file
    m = re.match(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", search)
    if m and m.group(2) != "00001":
        search = f"{m.group(1)}-00001-of-{m.group(3)}.gguf"
    for cand in MODELS_DIR.rglob(search):
        if cand.is_file() and cand.stat().st_size > 1000:
            return cand
    if m:
        for cand in MODELS_DIR.rglob(f"{m.group(1)}-00001-of-*.gguf"):
            if cand.is_file() and cand.stat().st_size > 1000:
                return cand
    for cand in MODELS_DIR.rglob(model_file):
        if cand.is_file() and cand.stat().st_size > 1000:
            return cand
    return None


def model_files(model_path: Path) -> list:
    m = re.match(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", model_path.name)
    if not m:
        return [model_path] if model_path.exists() else []
    seen, out = set(), []
    for it in (model_path.parent.glob(f"{m.group(1)}-*-of-*.gguf"), MODELS_DIR.glob(f"{m.group(1)}-*-of-*.gguf"), MODELS_DIR.rglob(f"{m.group(1)}-*-of-*.gguf")):
        for f in it:
            key = str(f.resolve())
            if f.is_file() and key not in seen:
                seen.add(key)
                out.append(f)
    return sorted(out)


def model_size_mb(model_path: Path) -> float:
    files = model_files(model_path)
    if len(files) > 1:
        size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"   检测到 {len(files)} 个分片，总计 {size / 1024:.1f}GB")
        return size
    return files[0].stat().st_size / (1024 * 1024) if files else 0.0


def _rx(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("bad gguf")
    return data


def _u32(f): return struct.unpack("<I", _rx(f, 4))[0]
def _u64(f): return struct.unpack("<Q", _rx(f, 8))[0]
def _s(f): return _rx(f, _u64(f)).decode("utf-8", errors="replace")


def _scalar(f, t):
    return {
        0: lambda: struct.unpack("<B", _rx(f, 1))[0],
        1: lambda: struct.unpack("<b", _rx(f, 1))[0],
        2: lambda: struct.unpack("<H", _rx(f, 2))[0],
        3: lambda: struct.unpack("<h", _rx(f, 2))[0],
        4: lambda: struct.unpack("<I", _rx(f, 4))[0],
        5: lambda: struct.unpack("<i", _rx(f, 4))[0],
        6: lambda: struct.unpack("<f", _rx(f, 4))[0],
        7: lambda: struct.unpack("<?", _rx(f, 1))[0],
        8: lambda: _s(f),
        10: lambda: struct.unpack("<Q", _rx(f, 8))[0],
        11: lambda: struct.unpack("<q", _rx(f, 8))[0],
        12: lambda: struct.unpack("<d", _rx(f, 8))[0],
    }[t]()


def _skip(f, t):
    if t in FIXED:
        f.seek(FIXED[t], 1)
    elif t == 8:
        f.seek(_u64(f), 1)
    elif t == 9:
        item_t, length = _u32(f), _u64(f)
        if item_t in FIXED:
            f.seek(FIXED[item_t] * length, 1)
        else:
            for _ in range(length):
                _skip(f, item_t)
    else:
        raise ValueError(f"bad gguf type {t}")


# gguf_meta 按后缀提取的架构字段（去掉 "<arch>." 前缀后存一份短名）。
# 注意力相关字段用于估算 KV cache 占用，full_attention_interval 是混合注意力
# 模型（如 qwen35：每 4 层才有 1 层全注意力）的关键，缺了它会把 KV 高估数倍。
META_SUFFIXES = (
    ".block_count",
    ".context_length",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".full_attention_interval",
    ".nextn_predict_layers",
)


def gguf_meta(model_path: Path) -> dict:
    key = str(model_path.resolve()) if model_path and model_path.exists() else str(model_path)
    if key in META_CACHE:
        return META_CACHE[key]
    meta = {}
    try:
        with open(model_path, "rb") as f:
            if _rx(f, 4) != b"GGUF":
                META_CACHE[key] = meta
                return meta
            if _u32(f) < 2:
                META_CACHE[key] = meta
                return meta
            _u64(f)
            wanted = {
                "general.architecture",
                "general.name",
                "general.type",
                "general.basename",
                "general.finetune",
                "clip.vision.projector_type",
                # chat_template 用于判断模型是否原生强制思考（如 MiniMax）
                "tokenizer.chat_template",
            }
            n_kv = _u64(f)
            for _ in range(n_kv):
                k, t = _s(f), _u32(f)
                if k in wanted or k.endswith(META_SUFFIXES):
                    meta[k] = _scalar(f, t)
                else:
                    _skip(f, t)
                # mmproj 文件：读到关键字段就可以停
                arch = meta.get("general.architecture")
                if meta.get("general.type") == "mmproj" and meta.get("clip.vision.projector_type"):
                    break
                # 普通模型：需要读到 chat_template（可能在文件靠后的 KV），
                # 但如果已经读到所有目标字段就提前停
                if arch and f"{arch}.block_count" in meta and "tokenizer.chat_template" in meta:
                    break
    except Exception:
        meta = {}
    arch = meta.get("general.architecture")
    if arch:
        meta["block_count"] = meta.get(f"{arch}.block_count", 0)
        meta["context_length"] = meta.get(f"{arch}.context_length", 0)
        for suffix in META_SUFFIXES:
            meta.setdefault(suffix.lstrip("."), meta.get(f"{arch}{suffix}", 0))
    META_CACHE[key] = meta
    return meta


def gguf_tensor_names(model_path: Path) -> list:
    key = str(model_path.resolve()) if model_path and model_path.exists() else str(model_path)
    if key in TENSOR_CACHE:
        return TENSOR_CACHE[key]
    names = []
    try:
        with open(model_path, "rb") as f:
            if _rx(f, 4) != b"GGUF":
                TENSOR_CACHE[key] = names
                return names
            if _u32(f) < 2:
                TENSOR_CACHE[key] = names
                return names
            n_tensors = _u64(f)
            n_kv = _u64(f)
            for _ in range(n_kv):
                _s(f)
                _skip(f, _u32(f))
            for _ in range(n_tensors):
                names.append(_s(f))
                n_dims = _u32(f)
                f.seek(8 * n_dims, 1)
                f.seek(4 + 8, 1)  # type + offset
    except Exception:
        names = []
    TENSOR_CACHE[key] = names
    return names


def has_mtp_head(model_path: Path) -> bool:
    name = model_path.name.lower() if model_path else ""
    if "mtp" in name or "nextn" in name:
        return True
    for t in gguf_tensor_names(model_path):
        low = t.lower()
        if ".nextn." in low or ".mtp." in low or "mtp_" in low:
            return True
    return False


def _config_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "auto")
    return bool(value)


def _model_words(path: Path, meta: dict) -> set:
    text = " ".join(str(x) for x in [
        path.name if path else "",
        meta.get("general.name", ""),
        meta.get("general.basename", ""),
        meta.get("general.architecture", ""),
        meta.get("clip.vision.projector_type", ""),
    ])
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    ignore = {
        "gguf", "mmproj", "f16", "q4", "q5", "q6", "q8", "k", "m", "it",
        "instruct", "model", "models", "vision", "clip", "unsloth",
    }
    return {w for w in words if len(w) > 1 and w not in ignore}


def is_reasoning_model(model_path: Path, meta: dict) -> bool:
    """
    判断模型是否具备思考/推理能力（即能响应 --reasoning on/off）。
    优先从 chat_template 中检测思考标记，其次按模型名关键词兜底。
    """
    # 从 chat_template 检测：含 <think> 标记说明模型原生支持思考
    template = meta.get("tokenizer.chat_template", "")
    if template and "<think>" in template:
        return True

    # 关键词兜底：覆盖未嵌入 chat_template 的模型
    words = _model_words(model_path, meta)
    name = " ".join(words)
    return any(x in words or x in name for x in (
        "qwen3", "qwen35moe", "deepseek", "r1", "thinking", "reasoning",
    ))


def detect_thinking_mode(model_path: Path, meta: dict) -> str:
    """
    检测模型的思考开关行为，返回以下之一：
      "controllable"  — 支持 --reasoning on/off，开关有效
      "always_on"     — 模型原生强制思考，--reasoning off 无效（如 MiniMax）
      "none"          — 不是思考型模型
    判断依据：chat_template 中是否含思考标记，以及是否有条件分支控制思考。
    """
    template = meta.get("tokenizer.chat_template", "")

    if not template:
        # 无 chat_template，按模型名关键词判断，保守视为 controllable
        return "controllable" if is_reasoning_model(model_path, meta) else "none"

    has_think_tag = "<think>" in template or "</think>" in template

    if not has_think_tag:
        return "none"

    # 如果 template 里有条件判断思考开关（enable_thinking / thinking_mode / add_thinking 等），
    # 说明模型支持通过参数控制，属于 controllable
    controllable_signals = (
        "enable_thinking" in template
        or "thinking_mode" in template
        or "add_thinking" in template
        or "if thinking" in template.lower()
        # llama.cpp --reasoning 在 jinja 模板里注入的变量名
        or "reasoning" in template
    )
    if controllable_signals:
        return "controllable"

    # 有 <think> 但没有条件分支：模型每次都输出思考，无法关闭
    return "always_on"


def compatible_mmproj(model_path: Path, mmproj_path: Path, model_meta: dict = None) -> bool:
    if not model_path or not mmproj_path or not mmproj_path.exists():
        return False
    model_meta = model_meta or gguf_meta(model_path)
    mm_meta = gguf_meta(mmproj_path)
    if mm_meta.get("general.type") and mm_meta.get("general.type") != "mmproj":
        return False
    model_words = _model_words(model_path, model_meta)
    mm_words = _model_words(mmproj_path, mm_meta)
    arch = str(model_meta.get("general.architecture", "")).lower()
    projector = str(mm_meta.get("clip.vision.projector_type", "")).lower()
    if arch.startswith("gemma4"):
        return "gemma4" in projector or "gemma" in mm_words
    if "qwen" in arch or any(w.startswith("qwen") for w in model_words):
        return any(w.startswith("qwen") for w in mm_words)
    return bool(model_words & mm_words)


def find_compatible_mmproj(model_path: Path, configured: Path, model_meta: dict) -> tuple:
    if configured and configured.exists():
        # 管理器保存的是用户选择或同仓库自动绑定的精确文件名。许多官方仓库
        # 使用通用名 mmproj-model-f16.gguf，单靠文件名没有家族信息，不能误判为不兼容。
        configured_meta = gguf_meta(configured)
        configured_type = str(configured_meta.get("general.type", "")).lower()
        if "mmproj" in configured.name.lower() and configured_type in ("", "mmproj"):
            return configured, ""
    candidates = []
    vision_dir = MODELS_DIR / "vision"
    if vision_dir.exists():
        for cand in vision_dir.rglob("*.gguf"):
            if "mmproj" in cand.name.lower() and compatible_mmproj(model_path, cand, model_meta):
                candidates.append(cand)
    if not candidates:
        return None, ""
    model_words = _model_words(model_path, model_meta)
    candidates.sort(key=lambda p: (len(model_words & _model_words(p, gguf_meta(p))), -p.stat().st_size), reverse=True)
    return candidates[0], candidates[0].name


def meminfo() -> dict:
    total = avail = 0.0
    try:
        if platform.system() == "Windows":
            import ctypes
            class M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            s = M(); s.dwLength = ctypes.sizeof(s); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            total, avail = s.ullTotalPhys / (1024 ** 3), s.ullAvailPhys / (1024 ** 3)
        elif platform.system() == "Darwin":
            if (r := rc(["sysctl", "-n", "hw.memsize"], 5)).returncode == 0 and r.stdout.strip().isdigit():
                total = int(r.stdout.strip()) / (1024 ** 3)
            if (r := rc(["vm_stat"], 5)).returncode == 0:
                ps = int(re.search(r"page size of (\d+) bytes", r.stdout).group(1)) if re.search(r"page size of (\d+) bytes", r.stdout) else 4096
                pages = {}
                for line in r.stdout.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1); v = v.strip().rstrip(".").replace(".", "")
                        if v.isdigit():
                            pages[k.strip()] = int(v)
                avail = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0)) * ps / (1024 ** 3)
        else:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"): total = int(line.split()[1]) / (1024 ** 2)
                    elif line.startswith("MemAvailable"): avail = int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return {"total_gb": round(total, 1), "avail_gb": round(avail, 1)}


def logical_cpu_count() -> int:
    return os.cpu_count() or 4


def physicalish_cpu_count() -> int:
    """
    推理线程数估算（-t 参数）。
    SMT/HT 对 llama.cpp 顺序推理帮助不大，用全逻辑核反而引发 cache 竞争。
    AMD 5600: 12 逻辑核 -> 6 物理核 -> 返回 6。
    batch 线程（-tb）应单独设为逻辑核数以充分利用并行带宽，见 logical_cpu_count()。
    """
    global PHYSICAL_CPU_CACHE
    if PHYSICAL_CPU_CACHE:
        return PHYSICAL_CPU_CACHE
    logical = logical_cpu_count()
    physical = 0
    try:
        system = platform.system()
        if system == "Windows":
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if ps:
                result = rc([
                    ps, "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Sum NumberOfCores).Sum",
                ], 8)
                value = result.stdout.strip()
                physical = int(value) if result.returncode == 0 and value.isdigit() else 0
        elif system == "Darwin":
            result = rc(["sysctl", "-n", "hw.physicalcpu"], 5)
            value = result.stdout.strip()
            physical = int(value) if result.returncode == 0 and value.isdigit() else 0
        else:
            pairs = set()
            physical_id = core_id = None
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for raw in list(cpuinfo) + ["\n"]:
                    line = raw.strip()
                    if not line:
                        if physical_id is not None and core_id is not None:
                            pairs.add((physical_id, core_id))
                        physical_id = core_id = None
                    elif line.startswith("physical id"):
                        physical_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
            physical = len(pairs)
    except Exception:
        physical = 0
    if not (0 < physical <= logical):
        physical = max(1, logical // 2) if logical >= 8 else logical
    PHYSICAL_CPU_CACHE = physical
    return physical


def has_vulkan() -> bool:
    sdk = os.environ.get("VULKAN_SDK", "").strip()
    if sdk and (Path(sdk) / "Bin" / "vulkaninfo.exe").exists():
        return True
    if IS_WIN and any(Path(p).exists() for p in (
        "C:/Windows/System32/vulkan-1.dll",
        "C:/Windows/System32/vulkaninfo.exe",
    )):
        return True
    if not (v := shutil.which("vulkaninfo")):
        return False
    try:
        return rc([v, "--summary"], 8).returncode == 0
    except Exception:
        return False


def accel(preferred="auto") -> dict:
    info = {"name": "", "gpu_count": 0, "vram_mb": 0, "vram_free_mb": 0, "compute_capability": "", "available_backends": ["cpu"], "selected_backend": "cpu", "reason": ""}
    try:
        r = rc(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], 5)
        if r.returncode == 0 and r.stdout.strip():
            names, totals, free_values = [], [], []
            for line in r.stdout.strip().splitlines():
                # GPU 名字可能含逗号，从右侧取两个数字字段。
                parts = [x.strip() for x in line.rsplit(",", 2)]
                if len(parts) != 3:
                    continue
                name_part, total_part, free_part = parts
                names.append(name_part)
                totals.append(int(total_part) if total_part.isdigit() else 0)
                free_values.append(int(free_part) if free_part.isdigit() else 0)
            if names:
                info.update({
                    "name": " + ".join(names),
                    "gpu_count": len(names),
                    "vram_mb": sum(totals),
                    "vram_free_mb": sum(free_values),
                })
            info["available_backends"].append("cuda")
            cap_result = rc(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"], 5)
            if cap_result.returncode == 0:
                capabilities = [
                    value.strip() for value in cap_result.stdout.splitlines()
                    if re.fullmatch(r"\d+(?:\.\d+)?", value.strip())
                ]
                if capabilities:
                    info["compute_capability"] = min(
                        capabilities, key=lambda value: tuple(int(x) for x in value.split("."))
                    )
    except Exception:
        pass
    if platform.system() == "Darwin":
        info["available_backends"].append("metal")
        info["name"] = info["name"] or f"Apple Metal ({platform.machine()})"
    if has_vulkan():
        info["available_backends"].append("vulkan")
        info["name"] = info["name"] or "Vulkan GPU"
    avail = set(info["available_backends"])
    if preferred == "auto":
        for b in ("cuda", "metal", "vulkan", "cpu"):
            if b in avail:
                info["selected_backend"] = b; break
    elif preferred in avail:
        info["selected_backend"] = preferred
    else:
        info["reason"] = f"requested backend '{preferred}' is unavailable"
    return info


def _version_tuple(value) -> tuple:
    match = re.search(r"(\d+)(?:\.(\d+))?", str(value or ""))
    return (int(match.group(1)), int(match.group(2) or 0)) if match else ()


def is_blackwell(gpu: dict) -> bool:
    return _version_tuple(gpu.get("compute_capability", "")) >= (12, 0)


def model_identity(model_path: Path, meta: dict) -> str:
    return " ".join(str(value).lower() for value in (
        model_path.name if model_path else "",
        meta.get("general.name", ""),
        meta.get("general.basename", ""),
        meta.get("general.architecture", ""),
    ))


def is_qwen38_flash_next(model_path: Path, meta: dict) -> bool:
    identity = model_identity(model_path, meta)
    arch = str(meta.get("general.architecture", "")).lower()
    return arch.startswith("qwen4exp") or (
        "qwen" in identity and "flash" in identity and "next" in identity
    )


def is_qwen38_dense_27b(model_path: Path, meta: dict) -> bool:
    identity = model_identity(model_path, meta)
    if is_qwen38_flash_next(model_path, meta):
        return False
    return (
        "qwen3.8" in identity or "qwen3_8" in identity or "qwen38" in identity
        or str(meta.get("general.architecture", "")).lower().startswith("qwen35")
    ) and ("27b" in identity or int(meta.get("block_count") or 0) == 64)


# 每个元素的字节数：q8_0 是 34 字节 / 32 个权重，q4_0 是 18 / 32，以此类推。
_CACHE_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}


def kv_cache_mb(meta: dict, ctx_size: int, cache_type_k: str, cache_type_v: str) -> float:
    """估算 KV cache 显存占用（MiB）。元数据不全时返回 0，调用方据此跳过判断。"""
    n_layer   = _safe_int(meta.get("block_count"), 0) - _safe_int(meta.get("nextn_predict_layers"), 0)
    n_kv_head = _safe_int(meta.get("attention.head_count_kv"), 0)
    k_len     = _safe_int(meta.get("attention.key_length"), 0)
    v_len     = _safe_int(meta.get("attention.value_length"), 0)
    if min(n_layer, n_kv_head, k_len, v_len, ctx_size) <= 0:
        return 0.0
    # 混合注意力模型每 interval 层才有一层带 KV，其余是常数大小的循环状态。
    interval = max(1, _safe_int(meta.get("full_attention_interval"), 1))
    attn_layers = max(1, n_layer // interval)
    per_token = attn_layers * n_kv_head * (
        k_len * _CACHE_TYPE_BYTES.get(cache_type_k, 2.0)
        + v_len * _CACHE_TYPE_BYTES.get(cache_type_v, 2.0)
    )
    return per_token * ctx_size / (1024 * 1024)


def max_ctx_for_vram(meta: dict, size_mb: float, free_mb: float,
                     cache_type_k: str, cache_type_v: str, overhead_mb: float) -> int:
    """返回权重仍能全部驻留显存的最大上下文长度；权重本身就装不下时返回 0。"""
    per_token_mb = kv_cache_mb(meta, 1024, cache_type_k, cache_type_v) / 1024
    budget = free_mb - size_mb - overhead_mb
    if per_token_mb <= 0 or budget <= 0:
        return 0
    return int(budget / per_token_mb) // 1024 * 1024


def performance_tuning(model_path: Path, meta: dict, gpu: dict, pc: dict,
                       ctx_size: int, vision=False, mmproj_mb=0.0) -> dict:
    """按模型和实际空闲显存生成可移植的单路推理参数。显式配置始终优先。"""
    profile = str(pc.get("profile", "auto") or "auto").strip().lower()
    if profile not in {"auto", "maximum", "compatible"}:
        profile = "auto"
    size_mb = model_size_mb(model_path)
    total_mb = int(gpu.get("vram_mb", 0) or 0)
    free_mb = int(gpu.get("vram_free_mb", 0) or total_mb)
    cuda = gpu.get("selected_backend") == "cuda"
    denominator = max(1, min(value for value in (total_mb, free_mb) if value > 0)) if (total_mb > 0 or free_mb > 0) else 1
    pressure = size_mb / denominator if cuda else 0.0
    dense_27b = is_qwen38_dense_27b(model_path, meta)
    flash_next = is_qwen38_flash_next(model_path, meta)
    notes = []

    def cache_value(key):
        configured = str(pc.get(key, "auto") or "auto").lower()
        if configured != "auto":
            return configured
        if profile == "compatible" or not cuda:
            return "f16"
        # 高显存压力和长上下文使用 Q8 KV，把节省的空间让给模型层。
        if profile == "maximum" or pressure >= 0.68 or ctx_size >= 8192:
            return "q8_0"
        return "f16"

    configured_target = _safe_int(pc.get("fit_target_mb", 0), 0)
    # 先算出"普通模式下会取多少"，视觉模式再覆盖。
    # 保留这个值，是为了在视觉模式下能算清楚"关掉视觉能省多少显存"。
    if configured_target:
        fit_target = configured_target
    elif not cuda:
        fit_target = 0
    elif profile == "compatible":
        fit_target = 1024
    elif dense_27b and total_mb >= 15000:
        # 权重比显存还大，留给驱动的余量每多 100MiB 就少一点权重驻留 GPU。
        # 实测 16GB 卡上 64 比 256 快约 15%，代价是显存余量更紧。
        fit_target = 64 if profile == "maximum" else 256
        notes.append("Qwen3.8 27B / 16GB 显存高压模式：优先保留更多模型层在 GPU")
    elif pressure >= 0.80:
        fit_target = 384 if profile == "maximum" else 512
    elif is_blackwell(gpu):
        fit_target = 512 if profile == "maximum" else 768
    else:
        fit_target = 1024

    text_mode_fit_target = fit_target
    if cuda and vision and not configured_target:
        # 视觉编码器的中间张量峰值很高，要给它留出 mmproj 量级的余量
        fit_target = max(1024, min(4096, int(mmproj_mb * 1.1) + 512))

    available_memory_mb = float(meminfo().get("avail_gb", 0) or 0) * 1024 + free_mb
    if flash_next and size_mb > available_memory_mb * 0.92:
        notes.append(
            f"Flash-Next GGUF 约 {size_mb / 1024:.1f}GB，已接近或超过当前 RAM+VRAM 可用容量；"
            "即使能映射加载也可能频繁换页，无法获得理想吐字速度"
        )
    # ── 显存预算体检 ────────────────────────────────────────────────────────
    # 权重装不下时 llama.cpp 会把层甩到 CPU，吐字速度会掉到原来的几分之一，
    # 但过程里没有任何提示。这里按元数据把账算清楚，直接告诉用户瓶颈在哪。
    kv_mb = kv_cache_mb(meta, ctx_size, cache_value("cache_type_k"), cache_value("cache_type_v"))
    if cuda and free_mb > 0 and kv_mb > 0:
        # 计算缓冲 + 循环状态 + CUDA context 的经验值。视觉模式还要算上常驻显存的
        # mmproj 权重，且 fit_target 会被抬到 mmproj 的量级，两项一起吃掉的显存
        # 足以把本来装得下的模型重新挤回 CPU —— 这正是开视觉后吐字变慢的原因。
        vision_mb = mmproj_mb if (vision and mmproj_mb > 0) else 0.0
        overhead_mb = 600.0 + vision_mb
        # llama.cpp 按 fit_target 保留一块显存不用，这部分不能算进权重预算
        usable_mb = max(0.0, free_mb - fit_target)
        vision_note = (
            f"（视觉模块 {vision_mb:.0f}MiB + 预留 {fit_target}MiB；"
            f"不需要图片理解时改用普通模式启动可显著提速）" if vision_mb else ""
        )
        if size_mb + overhead_mb >= usable_mb:
            budget_gb = max(0.0, usable_mb - overhead_mb - kv_mb) / 1024
            head = (f"权重 {size_mb / 1024:.1f}GB 超出可用显存 {usable_mb / 1024:.1f}GB，"
                    f"必然有一部分层留在 CPU 上，这是吐字慢的主因；")
            if vision_mb:
                # 视觉模式下先看看关掉视觉是否就能装下——这是最省事的解法
                text_usable = free_mb - text_mode_fit_target
                fits_without_vision = size_mb + 600.0 + kv_mb <= text_usable
                notes.append(
                    f"视觉模式额外占用约 {vision_mb + fit_target:.0f}MiB 显存"
                    + vision_note
                    + ("；实测该模型在普通模式下可完整驻留显存，开视觉会把约 "
                       f"{size_mb + overhead_mb - usable_mb:.0f}MiB 权重挤回 CPU"
                       if fits_without_vision else
                       f"；当前权重 {size_mb / 1024:.1f}GB 已超出视觉模式下的 "
                       f"{usable_mb / 1024:.1f}GB 预算")
                )
            elif budget_gb < 1.0:
                # 空闲显存少到放不下任何模型，多半是别的进程占着，而不是模型选大了
                notes.append(
                    head + f"当前几乎没有空闲显存可用，请先确认是否有其它进程"
                           f"（残留的 llama-server、浏览器硬件加速等）占用了显存"
                )
            else:
                # 已经在用 4bit KV 时就别再建议换 KV 类型了
                lever = ("调小 ctx_size"
                         if _CACHE_TYPE_BYTES.get(cache_value("cache_type_k"), 2.0) <= 0.65
                         else "调小 ctx_size 或改用 q4_0 KV")
                notes.append(
                    head + f"当前 ctx_size={ctx_size} 还要额外占用 {kv_mb:.0f}MiB KV cache，"
                           f"{lever} 能把这部分显存让给权重。"
                           f"若要让权重完整驻留显存，需换用体积 ≤{budget_gb:.1f}GB 的量化档位"
                )
        else:
            max_ctx = max_ctx_for_vram(meta, size_mb, usable_mb,
                                       cache_value("cache_type_k"), cache_value("cache_type_v"),
                                       overhead_mb)
            if 0 < max_ctx < ctx_size:
                notes.append(
                    f"ctx_size={ctx_size} 需要 {kv_mb:.0f}MiB KV cache，加上权重会超出可用显存，"
                    f"部分层会被挤到 CPU 上；当前显存下建议 ctx_size ≤{max_ctx}"
                    + vision_note
                )

    if is_blackwell(gpu):
        notes.append(f"已启用 RTX 50 / Blackwell 自适应策略（compute capability {gpu.get('compute_capability')}）")

    return {
        "profile": profile,
        "model_size_mb": size_mb,
        "pressure": pressure,
        "cache_type_k": cache_value("cache_type_k"),
        "cache_type_v": cache_value("cache_type_v"),
        "fit_target_mb": fit_target,
        "notes": notes,
    }


def auto_layers(model_path: Path, gpu: dict, vision: bool, mmproj: Path, meta: dict) -> int:
    size_mb = model_size_mb(model_path)
    if size_mb <= 0:
        print("   无法获取模型大小，自动 GPU 卸载已禁用")
        return 0
    free_mb = gpu.get("vram_free_mb", 0)
    if free_mb <= 0:
        print("   GPU 可用显存未知，自动卸载层数已禁用；如需强制使用 GPU，请手动设置 gpu_layers")
        return 0

    mmproj_mb = 0.0
    if vision and mmproj and mmproj.exists():
        try:
            mmproj_mb = mmproj.stat().st_size / (1024 * 1024)
        except OSError:
            pass
    reserve = 500 + mmproj_mb * 1.2
    avail_gb = (free_mb - reserve) / 1024
    if avail_gb <= 0.2:
        print(f"   模型 {size_mb / 1024:.1f}GB / 可用显存不足 -> 使用 CPU")
        return 0

    ram_gb = meminfo().get("avail_gb", 0) or 16
    if size_mb / 1024 > ram_gb + avail_gb:
        print(f"   警告: 模型 {size_mb / 1024:.1f}GB 超过可用内存 {ram_gb:.1f}GB + 显存 {avail_gb:.1f}GB")

    layers = int(meta.get("block_count") or 0)
    if layers <= 0:
        # 按文件大小粗估层数
        layers = (
            24 if size_mb < 820 else
            28 if size_mb < 2048 else
            32 if size_mb < 4096 else
            36 if size_mb < 8192 else
            40 if size_mb < 15360 else
            64 if size_mb < 30720 else
            80 if size_mb < 61440 else
            100
        )

    # MoE 只减少每 token 的计算量，不减少已卸载专家权重的显存驻留量。
    # 因此始终按完整 GGUF 大小估算，避免在小显存设备上严重过量卸载。
    if layers <= 0:
        print("   层数估算为 0，无法计算卸载层数")
        return 0
    per_gb = (size_mb / 1024 / layers) * 1.15
    if per_gb <= 0:
        print("   每层显存估算异常，跳过自动卸载")
        return 0

    fit = int(avail_gb / per_gb)
    if fit >= layers:
        print(f"   模型 {size_mb / 1024:.1f}GB / 可用显存 {avail_gb:.1f}GB -> 全部卸载")
        return layers
    if fit > 0:
        print(f"   模型 {size_mb / 1024:.1f}GB / 可用显存 {avail_gb:.1f}GB -> 卸载 {fit}/{layers} 层")
        return fit
    print(f"   模型 {size_mb / 1024:.1f}GB / 可用显存 {avail_gb:.1f}GB -> 显存不足，使用 CPU")
    return 0


def diagnose(model_path: Path) -> str:
    if not LOG_FILE.exists():
        return ""
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace").lower()
    arch = gguf_meta(model_path).get("general.architecture", "unknown")
    if "unknown model architecture" in text:
        return f"当前 llama.cpp 不支持模型架构 '{arch}'。这不是固定机器调参问题，而是后端版本兼容性不足；需要升级整套 llama.cpp 二进制/DLL，或更换已支持的模型。"
    if "cuda" in text and "error" in text:
        return "日志包含 CUDA 错误。迁移到不同平台时建议保留 backend=auto，让运行时自动回退。"
    if "not enough memory" in text:
        return "日志提示内存不足。需要减小模型、降低上下文，或在更强平台上运行。"
    return ""


def process_name(pid: int) -> str:
    """返回该 PID 的进程名，取不到时返回空字符串。"""
    try:
        if IS_WIN:
            r = rc(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], timeout=5)
            if r.returncode != 0 or not (r.stdout or "").strip():
                return ""
            row = next(csv.reader(io.StringIO(r.stdout.strip())), [])
            return (row[0] if row else "").strip('"')
        r = rc(["ps", "-p", str(pid), "-o", "comm="], timeout=5)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def pid_running(pid: int, expect: str = "") -> bool:
    """跨平台检测进程是否存活；给了 expect 就同时核对进程名。

    PID 会被系统回收复用，只判断"存在这个 PID"就去 kill，
    在 PID 文件过期时会误杀无关进程。
    """
    if pid <= 0:
        return False
    name = process_name(pid)
    if not name:
        # 取不到进程名（权限不足等），退回到仅判断存活
        try:
            if IS_WIN:
                return False
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except Exception:
            return False
    return expect.lower() in name.lower() if expect else True


def _safe_int(val, default: int) -> int:
    """安全转 int，失败返回 default。"""
    try:
        v = int(val)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _safe_int_min(val, default: int, minimum: int) -> int:
    """安全转 int，并允许指定最小值。"""
    try:
        v = int(val)
        return v if v >= minimum else default
    except (TypeError, ValueError):
        return default


def runtime(cfg: dict, mode: str, vision=False) -> dict:
    sc  = cfg.get("server", {})
    sp  = cfg.get("sampling", {})
    gc  = cfg.get("gpu", {})
    mc  = cfg.get("model", {})
    pc  = cfg.get("performance", {})

    binary = find_binary("llama-cli" if mode == "chat" else "llama-server")
    if not binary.exists():
        raise RuntimeError(f"找不到可执行文件: {binary}\n请先运行 python deploy.py 完成部署")

    model = resolve_model(MODELS_DIR / mc.get("model_file", ""), mc)
    if not model:
        raise RuntimeError(
            f"模型文件不存在: {MODELS_DIR / mc.get('model_file', '')}\n"
            "请检查 config.jsonc 中 model.model_file 是否与 models/ 目录下的文件名一致"
        )
    configured_mmproj = (
        MODELS_DIR / "vision" / mc.get("mmproj_file", "")
        if mc.get("mmproj_file", "")
        else None
    )
    meta = gguf_meta(model)
    gpu  = accel(gc.get("backend", "auto"))
    warn = []
    mmproj = configured_mmproj

    # ── 视觉模块匹配 ────────────────────────────────────────────────────────
    if vision:
        found_mmproj, auto_name = find_compatible_mmproj(model, configured_mmproj, meta)
        if configured_mmproj and configured_mmproj.exists() and not compatible_mmproj(model, configured_mmproj, meta):
            warn.append(f"已忽略不匹配的视觉模块: {configured_mmproj.name}")
        if found_mmproj:
            mmproj = found_mmproj
            if auto_name and (not configured_mmproj or configured_mmproj.name != auto_name):
                warn.append(f"已自动匹配视觉模块: {auto_name}")
        else:
            mmproj = None

    mmproj_mb = 0.0
    if vision and mmproj and mmproj.exists():
        try:
            mmproj_mb = mmproj.stat().st_size / (1024 * 1024)
        except OSError:
            pass
    ctx_size = _safe_int(sc.get("ctx_size"), 2048)
    tuning = performance_tuning(model, meta, gpu, pc, ctx_size, vision, mmproj_mb)
    warn.extend(tuning["notes"])

    # ── 基础参数 ────────────────────────────────────────────────────────────
    args = [str(binary), "-m", str(model)]
    if supports(binary, "--jinja"):
        args.append("--jinja")

    wants_thinking = bool(sc.get("enable_thinking", False))
    reasoning_budget = _safe_int_min(sc.get("reasoning_budget", 512), 512, -1)

    # ── 思考模式检测 ─────────────────────────────────────────────────────────
    # detect_thinking_mode 从 chat_template 精确判断：
    #   controllable — 支持 --reasoning on/off
    #   always_on    — 模型强制思考，无法关闭（如部分 MiniMax 版本）
    #   none         — 非思考型模型
    thinking_mode = detect_thinking_mode(model, meta)

    if supports(binary, "--reasoning"):
        if thinking_mode == "controllable":
            # 模型支持开关，尊重用户配置
            args += ["--reasoning", "on" if wants_thinking else "off"]
            if wants_thinking and supports(binary, "--reasoning-budget"):
                args += ["--reasoning-budget", str(reasoning_budget)]
        elif thinking_mode == "always_on":
            # 模型强制思考，传 off 也没用，不传此参数
            # 警告会在后面的 warn 段统一输出
            pass
        elif thinking_mode == "none":
            # 非思考型模型，明确关闭（防止某些版本 llama.cpp 默认开启）
            args += ["--reasoning", "off"]

    # 推理线程：物理核；批处理线程：逻辑核（充分利用 SMT 带宽）
    threads       = _safe_int(sc.get("threads") or pc.get("threads") or 0, 0) or physicalish_cpu_count()
    threads_batch = _safe_int(pc.get("threads_batch") or sc.get("threads_batch") or 0, 0) or logical_cpu_count()

    common = ["-t", str(threads)]
    if supports(binary, "-tb"):
        common += ["-tb", str(threads_batch)]

    spec_setting = str(pc.get("spec_type", pc.get("speculative", "auto"))).strip().lower()
    spec_requested = spec_setting not in ("", "0", "false", "off", "none", "no")
    mtp_available = has_mtp_head(model)
    spec_mtp_supported = supports(binary, "--spec-type") and "draft-mtp" in help_text(binary)
    spec_mtp_enabled = False
    if mode == "server" and spec_requested:
        if vision:
            warn.append("视觉模式下暂不启用 MTP speculative decoding，避免 multimodal slot/OOM 兼容问题")
        elif is_qwen38_flash_next(model, meta) and not _config_bool(pc.get("allow_experimental_mtp", False)):
            warn.append("Flash-Next 的 llama.cpp MTP 路径仍属实验实现；默认不启用，如需测试请显式设置 allow_experimental_mtp=true")
        elif spec_setting in ("auto", "true", "on", "1", "mtp", "draft-mtp"):
            if mtp_available and spec_mtp_supported:
                spec_mtp_enabled = True
            elif spec_setting not in ("auto", "true", "on", "1"):
                warn.append("已请求 MTP，但当前模型或 llama.cpp 不支持 draft-mtp，已跳过")
        else:
            warn.append(f"未知 speculative 类型: {spec_setting}，已跳过")

    common += [
        "--ctx-size",        str(ctx_size),
        "--temp",            str(sp.get("temperature", 0.7)),
        "--top-k",           str(_safe_int(sp.get("top_k"), 20)),
        "--top-p",           str(sp.get("top_p", 0.8)),
        "--presence-penalty",str(sp.get("presence_penalty", 1.5)),
    ]

    # 0 表示交给当前 llama.cpp 的 --fit 自动选择，避免固定值在不同显存机器上 OOM。
    if pc.get("batch_size", 0) and supports(binary, "-b"):
        common += ["-b", str(_safe_int(pc.get("batch_size"), 2048))]
    if pc.get("ubatch_size", 0) and supports(binary, "-ub"):
        common += ["-ub", str(_safe_int(pc.get("ubatch_size"), 512))]
    if pc.get("priority", 0) and supports(binary, "--prio"):
        common += ["--prio", str(_safe_int(pc.get("priority"), 0))]
    if pc.get("priority_batch", pc.get("priority", 0)) and supports(binary, "--prio-batch"):
        common += ["--prio-batch", str(_safe_int(pc.get("priority_batch", pc.get("priority", 0)), 0))]

    # ── 内存锁定（mlock）────────────────────────────────────────────────────
    # 将模型锁入物理内存，防止被 swap 出去，消除推理延迟抖动。
    # 代价是提前占用一块内存；内存充裕时强烈建议开启。
    # Windows 无效（llama.cpp 忽略），Linux/macOS 需要足够的 ulimit -l。
    if pc.get("mlock", False) and supports(binary, "--mlock"):
        common.append("--mlock")

    # ── 内存映射（mmap）─────────────────────────────────────────────────────
    # mmap=True（默认）：按需加载模型页，首次推理慢但启动快。
    # mmap=False：启动时全量载入内存，推理更稳定，适合 SSD 充足内存的场景。
    # 配置里 mmap: false 则显式关闭。
    if pc.get("mmap", True) is False and supports(binary, "--no-mmap"):
        common.append("--no-mmap")

    # ── KV cache 卸载控制 ────────────────────────────────────────────────────
    # 默认 llama.cpp 会把 KV cache 也卸载到 GPU（随 -ngl 走）。
    # 若显存刚好够模型但不够 KV cache，可以关掉（--no-kv-offload）。
    if pc.get("no_kv_offload", False) and supports(binary, "--no-kv-offload"):
        common.append("--no-kv-offload")

    # ── KV cache 类型 ───────────────────────────────────────────────────────
    # auto 会在模型挤压显存或长上下文时采用 q8_0，让更多权重驻留 GPU；
    # explicit f16/q4 等设置仍完全尊重用户选择。
    cache_type_k = tuning["cache_type_k"]
    cache_type_v = tuning["cache_type_v"]
    allowed_cache_types = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
    if cache_type_k != "f16" and cache_type_k in allowed_cache_types and supports(binary, "--cache-type-k"):
        common += ["--cache-type-k", cache_type_k]
    if cache_type_v != "f16" and cache_type_v in allowed_cache_types and supports(binary, "--cache-type-v"):
        common += ["--cache-type-v", cache_type_v]

    # ── RoPE 长上下文扩展 ────────────────────────────────────────────────────
    # ctx_size 超过模型原生上下文时，需要 RoPE scaling 防止位置编码崩溃。
    # yarn 是目前最主流的扩展方式（Qwen/LLaMA3 均支持）。
    native_ctx = _safe_int(meta.get("context_length", 0), 0)
    if native_ctx > 0 and ctx_size > native_ctx:
        rope_type = pc.get("rope_scaling", "yarn")   # 可选：yarn / linear
        if rope_type == "yarn" and supports(binary, "--rope-scaling"):
            common += ["--rope-scaling", "yarn"]
            # yarn-ext-factor：-1 让 llama.cpp 自动推算
            if supports(binary, "--yarn-ext-factor"):
                common += ["--yarn-ext-factor", "-1"]

    if mode == "chat":
        # --color auto 是新语法，老版本只支持 --color（flag 形式）
        if supports(binary, "--color auto"):
            color_args = ["--color", "auto"]
        elif supports(binary, "--color"):
            color_args = ["--color"]
        else:
            color_args = []
        args += color_args + common + ["-n", str(_safe_int(sp.get("max_tokens"), 512))]
    else:
        args += [
            "--host", sc.get("host", "0.0.0.0"),
            "--port", str(_safe_int(sc.get("port"), 8080)),
        ] + common

        # ── 连续批处理（cont-batching）──────────────────────────────────────
        # 允许多个请求共享同一个 batch，极大提升多并发吞吐量。
        # 单用户单并发时基本无害，多用户时效果显著。
        if pc.get("cont_batching", True) and supports(binary, "--cont-batching"):
            args.append("--cont-batching")

        # ── KV cache 碎片整理 ────────────────────────────────────────────────
        # 长对话后 KV cache 出现空洞，defrag 会定期整理，提升长对话速度。
        # 阈值 0.1 = 碎片率超过 10% 时触发。
        defrag = pc.get("defrag_thold", 0.1)
        if defrag is not False and supports(binary, "--defrag-thold"):
            args += ["--defrag-thold", str(defrag)]

        # ── Prometheus 指标端点 ─────────────────────────────────────────────
        # 开启后可在 http://localhost:{port}/metrics 查看实时 token/s 等指标。
        if pc.get("metrics", False) and supports(binary, "--metrics"):
            args.append("--metrics")

        # ── 连接槽数量 ───────────────────────────────────────────────────────
        # parallel 和 --slots 共同决定最大并发数，两者取较大值。
        # --slots 在新版 llama.cpp 中控制 server 的连接槽数。
        parallel = _safe_int(pc.get("parallel"), 1)
        if spec_mtp_enabled and parallel != 1:
            warn.append("MTP speculative decoding 当前仅适合 parallel=1，已自动降为 1")
            parallel = 1
        if (pc.get("parallel", 0) or spec_mtp_enabled) and supports(binary, "-np"):
            args += ["-np", str(parallel)]
        if pc.get("cache_reuse", 0) and supports(binary, "--cache-reuse"):
            args += ["--cache-reuse", str(_safe_int(pc.get("cache_reuse"), 0))]
        if _config_bool(pc.get("kv_unified", True), True) and supports(binary, "--kv-unified"):
            args.append("--kv-unified")
        if spec_mtp_enabled:
            args += ["--spec-type", "draft-mtp"]
            if supports(binary, "--spec-draft-n-max"):
                args += ["--spec-draft-n-max", str(_safe_int(pc.get("spec_draft_n_max", 3), 3) or 3)]
            if supports(binary, "--spec-draft-ngl"):
                args += ["--spec-draft-ngl", str(pc.get("spec_draft_ngl", "auto") or "auto")]
            if pc.get("ctx_checkpoints", None) is not None and supports(binary, "--ctx-checkpoints"):
                args += ["--ctx-checkpoints", str(_safe_int(pc.get("ctx_checkpoints"), 32))]
        if not vision and supports(binary, "--no-mmproj"):
            args.append("--no-mmproj")
        if vision and mmproj and mmproj.exists():
            args += ["--mmproj", str(mmproj)]

    # ── GPU 卸载层数 ─────────────────────────────────────────────────────────
    # gpu_layers: -1 = 自动（按显存估算或 -ngl auto）；0 = 禁用 GPU；>0 = 手动指定层数
    layers = 0
    try:
        gpu_layers_cfg = int(gc.get("gpu_layers", -1))
    except (TypeError, ValueError):
        gpu_layers_cfg = -1

    if gpu.get("selected_backend") != "cpu":
        # 优先使用 llama.cpp 内置的 -ngl auto（若支持）
        if (gpu_layers_cfg == -1
                and pc.get("auto_gpu_layers", True)
                and supports(binary, "-ngl")
                and supports_value(binary, "-ngl", "auto")):
            layers = -1
            args += ["-ngl", "auto"]
            if supports(binary, "--fit"):
                args += ["--fit", "on"]
            if supports(binary, "--fit-target"):
                args += ["--fit-target", str(tuning["fit_target_mb"])]
        else:
            layers = (
                auto_layers(model, gpu, vision, mmproj, meta)
                if gpu_layers_cfg == -1
                else max(0, gpu_layers_cfg)
            )

    if pc.get("cpu_moe", False) and supports(binary, "--cpu-moe"):
        args.append("--cpu-moe")
    elif _safe_int(pc.get("n_cpu_moe", 0), 0) > 0 and supports(binary, "--n-cpu-moe"):
        args += ["--n-cpu-moe", str(_safe_int(pc.get("n_cpu_moe"), 0))]

    fa_flags = ["-fa"] if supports(binary, "-fa") else (["--flash-attn"] if supports(binary, "--flash-attn") else [])
    want_fa  = bool(gc.get("flash_attention", True)) and bool(fa_flags)

    if layers > 0:
        args += ["-ngl", str(layers)]
        if want_fa:
            args += fa_flags + (["on"] if fa_flags == ["-fa"] else [])
    elif layers == -1:
        if want_fa:
            args += fa_flags + (["on"] if fa_flags == ["-fa"] else [])
    else:
        # 强制纯 CPU
        if supports(binary, "-ngl"):
            args += ["-ngl", "0"]
        if supports(binary, "--device"):
            args += ["--device", "none"]
        if supports(binary, "--no-op-offload"):
            args.append("--no-op-offload")

    # ── 警告收集 ─────────────────────────────────────────────────────────────
    if thinking_mode == "always_on":
        warn.append(
            f"当前模型（{meta.get('general.name', model.name)}）的 chat_template 中未检测到思考开关条件分支，"
            "模型可能会无论 enable_thinking 设置如何都先输出思考过程。"
            "这是模型本身的设计，非 llama.cpp 或配置问题。"
        )
    elif thinking_mode == "none" and wants_thinking:
        warn.append("当前模型不含思考能力，enable_thinking 已自动忽略")
    elif thinking_mode == "controllable" and not supports(binary, "--reasoning"):
        warn.append("当前 llama.cpp 不支持 --reasoning 参数，enable_thinking 设置将被忽略；建议升级 llama.cpp")
    if want_fa and layers > 0 and not fa_flags:
        warn.append("当前 llama.cpp 不支持 flash attention 参数，已自动跳过")
    if vision and (not mmproj or not mmproj.exists()):
        warn.append("视觉模式已请求，但 mmproj 不存在，已按纯文本模式启动")
    if gc.get("backend", "auto") != "cpu" and gpu.get("selected_backend") == "cpu":
        warn.append(f"将以 CPU 模式运行: {gpu.get('reason') or '未检测到可用 GPU 后端'}")

    return {
        "args":   args,
        "binary": binary,
        "cwd":    str(binary.parent if binary.parent.exists() else LLAMA_DIR),
        "model":  model,
        "mmproj": mmproj,
        "meta":   meta,
        "gpu":    gpu,
        "tuning": tuning,
        "layers": layers,
        "warn":   warn,
        "host":   sc.get("host", "0.0.0.0"),
        "port":   _safe_int(sc.get("port"), 8080),
    }


def summary(rt: dict, mode: str, vision: bool):
    meta, gpu = rt["meta"], rt["gpu"]
    print(f"   引擎: {rt['binary']}")
    print(f"   模型: {rt['model']}")
    if meta:
        print(f"   元数据: name={meta.get('general.name', rt['model'].name)}, arch={meta.get('general.architecture', 'unknown')}, blocks={meta.get('block_count', 'unknown')}")
    if gpu.get("selected_backend") == "cpu":
        print("   后端: CPU")
    else:
        msg = f"   后端: {gpu['selected_backend']} | {gpu.get('name') or gpu['selected_backend'].upper()}"
        if gpu.get("vram_free_mb", 0) > 0: msg += f" | 可用显存: {gpu['vram_free_mb']}MB"
        if gpu.get("compute_capability"): msg += f" | CC {gpu['compute_capability']}"
        print(msg)
    tuning = rt.get("tuning", {})
    if tuning:
        print(
            f"   性能策略: {tuning.get('profile', 'auto')} | KV {tuning.get('cache_type_k')}/{tuning.get('cache_type_v')}"
            f" | 显存余量 {tuning.get('fit_target_mb', 0)}MB"
        )
    if rt["layers"] == -1: print("   GPU 卸载层数: auto")
    elif rt["layers"] > 0: print(f"   GPU 卸载层数: {rt['layers']}")
    if mode == "server": print(f"   服务模式: {'vision+text' if vision else 'text'} | http://{rt['host']}:{rt['port']}")
    if mode == "server" and vision and rt.get("mmproj"): print(f"   视觉模块: {rt['mmproj']}")
    for w in rt["warn"]: print(f"   注意: {w}")


def cmd_chat(cfg: dict) -> int:
    rt = runtime(cfg, "chat")
    print("启动聊天模式..."); summary(rt, "chat", False); print("-" * 50)
    try: return subprocess.run(rt["args"], cwd=rt["cwd"]).returncode
    except KeyboardInterrupt: print("\n已退出"); return 0
    except FileNotFoundError as e: print(f"启动失败: {e}"); return 1


def cmd_server(cfg: dict, vision=False) -> int:
    rt = runtime(cfg, "server", vision)
    print("启动 API 服务...")
    summary(rt, "server", vision)
    print(f"   日志: {LOG_FILE}")
    print("-" * 50)
    proc = None
    log_f = None
    try:
        log_f = open(LOG_FILE, "w", encoding="utf-8")
        proc = subprocess.Popen(
            rt["args"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=rt["cwd"],
        )
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        print(f"   PID: {proc.pid}")

        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            log_f.write(line + "\n")
            log_f.flush()
            print(f"   {line}")
            low = line.lower()
            if "listening" in low or "server is listening" in low:
                print(f"\n✅ 服务已启动: http://localhost:{rt['port']}")
            if "error" in low and "cuda" in low:
                print("\n⚠️  检测到 CUDA 报错，建议保留 backend=auto 让运行时自动回退")

        proc.wait()
        if proc.returncode not in (0, -15):   # -15 = SIGTERM（正常停止）
            print(f"\n服务退出，代码: {proc.returncode}")
            if note := diagnose(rt["model"]):
                print(f"   诊断: {note}")
            print(f"   查看日志: {LOG_FILE}")
        return proc.returncode if proc.returncode >= 0 else 0

    except KeyboardInterrupt:
        print("\n正在停止服务...")
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("服务已停止")
        return 0
    except FileNotFoundError as e:
        print(f"启动失败（找不到可执行文件）: {e}")
        return 1
    except Exception as e:
        print(f"异常: {e}")
        return 1
    finally:
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        PID_FILE.unlink(missing_ok=True)


def cmd_server_background(cfg: dict, vision=False) -> int:
    rt = runtime(cfg, "server", vision)
    summary(rt, "server", vision)
    try:
        log_f = open(LOG_FILE, "w", encoding="utf-8")
    except OSError as e:
        print(f"无法打开日志文件: {e}")
        return 1

    try:
        popen_kwargs = dict(
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=rt["cwd"],
        )
        if IS_WIN:
            # Windows：脱离当前控制台，避免随父进程退出
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            )
        else:
            # Unix：新会话，脱离终端控制
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(rt["args"], **popen_kwargs)
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")

        # 等待 3 秒确认进程未立即退出
        time.sleep(3)
        if proc.poll() is not None:
            # 进程已退出，关闭日志再读取
            log_f.close()
            log_f = None
            PID_FILE.unlink(missing_ok=True)
            print(f"服务启动失败 (退出码: {proc.returncode})")
            if note := diagnose(rt["model"]):
                print(f"诊断: {note}")
            if LOG_FILE.exists():
                tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-800:]
                print(f"日志:\n{tail}")
            return 1

        print(f"服务已在后台启动 (PID: {proc.pid})")
        print(f"地址: http://localhost:{rt['port']}")
        print(f"日志: {LOG_FILE}")
        return 0

    except Exception as e:
        PID_FILE.unlink(missing_ok=True)
        print(f"启动异常: {e}")
        return 1
    finally:
        # log_f 交给后台子进程继续写入（Windows 下文件描述符继承），
        # 父进程这里关闭自己持有的句柄；子进程仍可写。
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass


def cmd_stop() -> int:
    if not PID_FILE.exists():
        print("没有运行中的服务（PID 文件不存在）")
        return 0
    try:
        pid_text = PID_FILE.read_text(encoding="utf-8").strip()
        if not pid_text.isdigit():
            print(f"PID 文件内容无效: {pid_text!r}，清除残留文件")
            PID_FILE.unlink(missing_ok=True)
            return 1
        pid = int(pid_text)
    except Exception as e:
        print(f"读取 PID 文件失败: {e}")
        PID_FILE.unlink(missing_ok=True)
        return 1

    if not pid_running(pid, "llama-server"):
        print(f"PID {pid} 已不是 llama-server 进程（可能已退出且 PID 被复用），仅清除 PID 文件")
        PID_FILE.unlink(missing_ok=True)
        return 0

    try:
        if IS_WIN:
            r = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                print(f"已停止服务 (PID: {pid})")
            else:
                # taskkill 返回 128 表示进程不存在
                print(f"进程 {pid} 可能已不存在: {r.stderr.strip()}")
        else:
            os.kill(pid, signal.SIGTERM)
            print(f"已发送 SIGTERM 至进程 {pid}")
        return 0
    except ProcessLookupError:
        print(f"进程 {pid} 不存在（已自行退出）")
        return 0
    except PermissionError:
        print(f"无权限终止进程 {pid}（可能需要 sudo）")
        return 1
    except Exception as e:
        print(f"停止失败: {e}")
        return 1
    finally:
        PID_FILE.unlink(missing_ok=True)


def cmd_status(cfg: dict) -> int:
    mc, gc, sc = cfg.get("model", {}), cfg.get("gpu", {}), cfg.get("server", {})
    model = resolve_model(MODELS_DIR / mc.get("model_file", ""), mc) if mc.get("model_file") else None
    meta, gpu, mem = (gguf_meta(model) if model else {}), accel(gc.get("backend", "auto")), meminfo()
    cli, srv = find_binary("llama-cli"), find_binary("llama-server")
    mmproj = MODELS_DIR / "vision" / mc.get("mmproj_file", "") if mc.get("mmproj_file", "") else None
    print("llama-deploy 状态"); print("-" * 50)
    print(f"   llama-cli:    {'OK' if cli.exists() else 'NO'} {cli}")
    print(f"   llama-server: {'OK' if srv.exists() else 'NO'} {srv}")
    print(f"   模型文件:      {'OK' if model and model.exists() else 'NO'} {model or mc.get('model_file', '')}")
    print(f"   视觉模块:      {'OK' if mmproj and mmproj.exists() else 'NO'} {mc.get('mmproj_file', '') or '(none)'}")
    print(f"   内存:          total={mem.get('total_gb', 0)}GB avail={mem.get('avail_gb', 0)}GB")
    if meta: print(f"   模型元数据:    name={meta.get('general.name', 'unknown')}, arch={meta.get('general.architecture', 'unknown')}, blocks={meta.get('block_count', 'unknown')}, ctx={meta.get('context_length', 'unknown')}")
    if gpu.get("selected_backend") == "cpu": print(f"   GPU 后端:      CPU ({gc.get('backend', 'auto')})")
    else:
        capability = f" | CC={gpu['compute_capability']}" if gpu.get("compute_capability") else ""
        print(f"   GPU 后端:      {gpu['selected_backend']} | {gpu.get('name') or gpu['selected_backend'].upper()} | free={gpu.get('vram_free_mb', 0)}MB{capability}")
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        if pid_running(pid, "llama-server"): print(f"   服务状态:      运行中 (PID: {pid})"); print(f"   地址:          http://localhost:{sc.get('port', 8080)}")
        else: print("   服务状态:      未运行 (PID 文件残留)"); PID_FILE.unlink(missing_ok=True)
    else:
        print("   服务状态:      未运行")
    if LOG_FILE.exists():
        print(f"   最近日志:      {LOG_FILE}")
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in lines[-5:]: print(f"     | {line}")
        if model and (note := diagnose(model)): print(f"   诊断:          {note}")
    return 0


def cmd_benchmark(cfg: dict, sweep=False) -> int:
    """用 llama-bench 测量真实 prompt/decode 吞吐；sweep 比较不同显存余量。"""
    mc, gc, sc, pc = (
        cfg.get("model", {}), cfg.get("gpu", {}), cfg.get("server", {}),
        cfg.get("performance", {}),
    )
    model = resolve_model(MODELS_DIR / mc.get("model_file", ""), mc)
    if not model:
        raise RuntimeError("未找到当前模型，请先在管理器中选择或下载模型")
    binary = find_binary("llama-bench")
    if not binary.exists():
        raise RuntimeError("当前 llama.cpp 包缺少 llama-bench，请先运行 python deploy.py --upgrade-llama")

    meta = gguf_meta(model)
    gpu = accel(gc.get("backend", "auto"))
    tuning = performance_tuning(
        model, meta, gpu, pc, _safe_int(sc.get("ctx_size"), 8192), False, 0.0
    )
    fit_targets = (
        "256,384,512,768,1024" if sweep and gpu.get("selected_backend") != "cpu"
        else str(tuning["fit_target_mb"])
    )
    args = [
        str(binary), "-m", str(model), "-p", "512", "-n", "128", "-r", "3",
        "-t", str(physicalish_cpu_count()), "-b", "2048", "-ub", "512",
        "-ctk", tuning["cache_type_k"], "-ctv", tuning["cache_type_v"],
        "--prio", str(_safe_int(pc.get("priority", 2), 2)), "-o", "md", "--progress",
    ]
    if gpu.get("selected_backend") != "cpu":
        args += ["-ngl", "-1", "-fa", "on", "--fit-target", fit_targets]

    print("llama.cpp 本机性能基准")
    print("-" * 50)
    print(f"   模型: {model.name} ({tuning['model_size_mb'] / 1024:.1f}GB)")
    print(f"   GPU: {gpu.get('name') or 'CPU'} | 可用显存 {gpu.get('vram_free_mb', 0)}MB")
    print(f"   KV: {tuning['cache_type_k']}/{tuning['cache_type_v']} | fit-target: {fit_targets}MB")
    print("   结果中 pp512 是提示词处理速度，tg128 是单路吐字速度（token/s）。")
    if sweep:
        print("   sweep 会比较 5 档显存余量；优先采用 tg128 最高且能稳定完成的档位。")
    print("-" * 50)
    try:
        return subprocess.run(args, cwd=str(binary.parent)).returncode
    except KeyboardInterrupt:
        print("\n基准测试已中止")
        return 130


def print_help():
    print("""
llama-deploy 管理工具

用法:
    python run.py chat
    python run.py server
    python run.py server --vision
    python run.py server --background
    python run.py stop
    python run.py status
    python run.py benchmark
    python run.py benchmark --sweep
    python run.py help
""")


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "help":
        print_help(); return 0
    cfg = parse_jsonc(CONFIG_FILE)
    if not cfg:
        print("找不到或无法解析 config.jsonc"); return 1
    try:
        if args[0].lower() == "chat": return cmd_chat(cfg)
        if args[0].lower() == "server": return cmd_server_background(cfg, "--vision" in args or "-v" in args) if ("--background" in args or "-bg" in args) else cmd_server(cfg, "--vision" in args or "-v" in args)
        if args[0].lower() == "stop": return cmd_stop()
        if args[0].lower() == "status": return cmd_status(cfg)
        if args[0].lower() == "benchmark": return cmd_benchmark(cfg, "--sweep" in args)
        print(f"未知命令: {args[0]}"); print_help(); return 1
    except RuntimeError as e:
        print(str(e)); return 1


if __name__ == "__main__":
    sys.exit(main())
