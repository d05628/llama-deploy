#!/usr/bin/env python3
"""
llama-deploy 智能管理器
启动 Web UI，支持搜索模型、编辑配置、一键部署

用法：python manager.py [--port 9090]
"""

# ============================================================
#  Windows UTF-8 修复（必须在最前面）
# ============================================================
import sys
import io
import os

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

import http.server
import ipaddress
import json
import csv
import platform
import re
import signal
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# ============================================================
#  常量
# ============================================================

VERSION = "1.1.1"
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.jsonc"
PID_FILE = BASE_DIR / ".llama-server.pid"
COMPAT_PID_FILE = BASE_DIR / ".compat-server.pid"
COMPAT_LOG_FILE = BASE_DIR / ".compat-server.log"
LLAMA_DIR = BASE_DIR / "llama.cpp"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_PORT = 9090
LLAMA_VERSION_CACHE = {"key": None, "value": None, "ts": 0.0}

# ============================================================
#  JSONC 解析器
# ============================================================

def parse_jsonc(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    text = filepath.read_text(encoding="utf-8")
    text = re.sub(r'(?<!:)//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def save_config(config: dict):
    comments = {
        "model": "模型配置",
        "download": "下载配置（镜像加速）",
        "server": "服务器配置",
        "compat": "兼容网关（Ollama / Claude Code / OpenAI）",
        "gpu": "GPU 配置",
        "sampling": "采样参数",
        "performance": "性能优化",
        "build": "编译配置",
        "ui": "界面配置",
    }
    lines = ["{"]
    sections = list(config.items())
    for i, (key, value) in enumerate(sections):
        if key in comments:
            lines.append(f'  // --- {comments[key]} ---')
        val_str = json.dumps(value, ensure_ascii=False, indent=4)
        val_str = val_str.replace("\n", "\n  ")
        comma = "," if i < len(sections) - 1 else ""
        lines.append(f'  "{key}": {val_str}{comma}')
    lines.append("}")
    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")


def default_config():
    """返回默认配置字典。字段需与 run.py runtime() 的读取逻辑保持一致。"""
    return {
        "model": {
            "repo_id": "",
            "model_file": "",
            "mmproj_file": "",
            "mmproj_use_xet": True,
            "mmproj_map": {},
            "mmproj_bindings": {},
        },
        "download": {
            "hf_mirror": "https://hf-mirror.com",
            "github_mirror": "",
            "timeout": 300,
            "retries": 3,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "threads": 0,         # 0 = run.py 自动计算（物理核数）
            "ctx_size": 8192,
            "enable_thinking": False,
            "reasoning_budget": 512,
        },
        "compat": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 11434,
            "upstream_url": "http://127.0.0.1:8080",
            "model_alias": "llama-deploy-local",
            "api_key": "local-no-key-needed",
            "claude_tool_mode": "repair",
            "request_timeout": 600,
        },
        "gpu": {
            "backend": "auto",
            "gpu_layers": -1,
            "flash_attention": True,
        },
        "sampling": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "max_tokens": 2048,
        },
        "performance": {
            "profile": "auto",
            "parallel": 1,
            "threads_batch": 0,   # 0 = run.py 自动计算（逻辑核数）
            "batch_size": 0,
            "ubatch_size": 0,
            "fit_target_mb": 0,
            "priority": 2,
            "priority_batch": 2,
            "cache_reuse": 512,
            "auto_gpu_layers": True,
            "cache_type_k": "auto",
            "cache_type_v": "auto",
            "spec_type": "off",
            "spec_draft_n_max": 3,
            "spec_draft_ngl": "auto",
            "allow_experimental_mtp": False,
            "kv_unified": True,
            "ctx_checkpoints": 32,
            "cpu_moe": False,
            "n_cpu_moe": 0,
        },
        "build": {"use_openblas": True, "jobs": 0},
        "ui": {"language": "zh", "verbose": True},
    }


def pid_running(pid: int) -> bool:
    try:
        if platform.system() == "Windows":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            return r.returncode == 0 and str(pid) in (r.stdout or "")
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return isinstance(sys.exc_info()[1], PermissionError)
    except Exception:
        return False


def running_llama_processes() -> list:
    names = []
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            for row in csv.reader((result.stdout or "").splitlines()):
                name = row[0].strip() if row else ""
                if re.fullmatch(r"(?:llama(?:-[\w-]+)?|ggml-rpc-server|rpc-server|server|main)\.exe", name, re.I):
                    names.append(name)
        else:
            result = subprocess.run(
                ["ps", "-eo", "comm="],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            for line in (result.stdout or "").splitlines():
                name = Path(line.strip()).name
                if re.fullmatch(r"llama(?:-[\w-]+)?|ggml-rpc-server|rpc-server|server|main", name, re.I):
                    names.append(name)
    except Exception:
        return []
    return sorted(set(names), key=str.lower)


def get_lan_ips() -> list:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                ips.append(ip)
        finally:
            s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    def score(ip: str):
        try:
            addr = ipaddress.ip_address(ip)
            parts = [int(x) for x in ip.split(".")]
            if parts[0] == 192 and parts[1] == 168:
                return (0, ip)
            if parts[0] == 10:
                return (1, ip)
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return (2, ip)
            if parts[0] == 198 and parts[1] == 18:
                return (8, ip)
            if addr.is_private:
                return (3, ip)
        except Exception:
            pass
        return (9, ip)

    return sorted(ips, key=score) if ips else ["127.0.0.1"]


def ensure_compat_config(cfg: dict) -> dict:
    sc = cfg.setdefault("server", {})
    cc = cfg.setdefault("compat", {})
    cc.setdefault("enabled", True)
    cc.setdefault("host", "0.0.0.0")
    cc.setdefault("port", 11434)
    cc.setdefault("upstream_url", f"http://127.0.0.1:{sc.get('port', 8080)}")
    cc.setdefault("model_alias", "llama-deploy-local")
    cc.setdefault("api_key", "local-no-key-needed")
    cc.setdefault("claude_tool_mode", "repair")
    cc.setdefault("request_timeout", 600)
    return cc


def model_tokens(name: str) -> set:
    ignore = {
        "gguf", "mmproj", "model", "vision", "projector", "f16", "bf16", "f32",
        "q2", "q3", "q4", "q5", "q6", "q8", "iq2", "iq3", "iq4", "k", "m",
        "s", "l", "xl", "it", "instruct", "chat", "ud",
    }
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 1 and w not in ignore}


def local_model_name(remote_name: str) -> str:
    """将仓库内相对路径规范化为本地文件名，同时保留原路径供下载使用。"""
    return str(remote_name or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def paired_mmproj_name(model_file: str, mmproj_file: str) -> str:
    """生成一眼可识别、复制到其他设备后仍可自动匹配的视觉模型名。"""
    model_name = local_model_name(model_file)
    mmproj_name = local_model_name(mmproj_file)
    if not model_name or not mmproj_name:
        return mmproj_name
    model_stem = re.sub(
        r"-\d{5}-of-\d{5}$", "", Path(model_name).stem, flags=re.I
    )
    if mmproj_name.lower().startswith((model_stem + ".mmproj").lower()):
        return mmproj_name
    variants = re.findall(
        r"(?:^|[-_.])(bf16|f16|f32|q[2-8](?:_[0-9a-z]+)?|iq[1-4](?:_[0-9a-z]+)?)(?=$|[-_.])",
        Path(mmproj_name).stem,
        flags=re.I,
    )
    suffix = "-" + variants[-1].lower() if variants else ""
    return f"{model_stem}.mmproj{suffix}.gguf"


def ensure_paired_mmproj_alias(model_file: str, mmproj_file: str) -> str:
    """为已存在的视觉模型创建配对别名；优先硬链接，不额外占用模型空间。"""
    source_name = local_model_name(mmproj_file)
    paired_name = paired_mmproj_name(model_file, source_name)
    if not source_name or paired_name == source_name:
        return source_name
    vision_dir = MODELS_DIR / "vision"
    source = next(
        (p for p in vision_dir.rglob("*.gguf") if p.name.casefold() == source_name.casefold()),
        None,
    ) if vision_dir.exists() else None
    target = vision_dir / paired_name
    if source and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(source), str(target))
        except OSError:
            shutil.copy2(source, target)
    return paired_name if target.exists() else source_name


def model_family(name: str) -> str:
    text = str(name or "").lower().replace("_", "-")
    patterns = (
        (r"qwen-?(\d+(?:[.-]\d+)?)", "qwen"),
        (r"gemma-?(\d+(?:[.-]\d+)?)", "gemma"),
        (r"llama-?(\d+(?:[.-]\d+)?)", "llama"),
        (r"minicpm-?v?-?(\d+(?:[.-]\d+)?)?", "minicpm"),
        (r"internvl-?(\d+(?:[.-]\d+)?)?", "internvl"),
        (r"glm-?(\d+(?:[.-]\d+)?)?", "glm"),
        (r"pixtral", "pixtral"),
        (r"mistral", "mistral"),
        (r"phi-?(\d+(?:[.-]\d+)?)?", "phi"),
    )
    for pattern, prefix in patterns:
        match = re.search(pattern, text)
        if match:
            version = (match.group(1) if match.lastindex else "") or ""
            return prefix + version.replace("-", ".")
    return ""


def mmproj_match_score(model_file: str, mmproj_file: str) -> int:
    model_family_name = model_family(model_file)
    mmproj_family_name = model_family(mmproj_file)
    if model_family_name and mmproj_family_name and model_family_name != mmproj_family_name:
        return -1
    model = model_tokens(model_file)
    mmproj = model_tokens(mmproj_file)
    overlap = model & mmproj
    score = sum(max(2, len(token)) for token in overlap)
    if model_family_name and model_family_name == mmproj_family_name:
        score += 100
    return score


def mmproj_matches_model(model_file: str, mmproj_file: str) -> bool:
    return mmproj_match_score(model_file, mmproj_file) > 0


def find_matching_mmproj(model_file: str, preferred: str = "") -> str:
    vision_dir = MODELS_DIR / "vision"
    if not model_file or not vision_dir.exists():
        return ""
    preferred_name = local_model_name(preferred)
    if preferred_name:
        for candidate in vision_dir.rglob(preferred_name):
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate.name
    matches = []
    for cand in vision_dir.rglob("*.gguf"):
        if "mmproj" not in cand.name.lower():
            continue
        score = mmproj_match_score(model_file, cand.name)
        if score > 0:
            quality = 2 if re.search(r"(?:^|[-_.])(bf16|f16)(?:[-_.]|$)", cand.name, re.I) else 1
            matches.append((score, quality, cand.stat().st_size, cand.name))
    if not matches:
        return ""
    matches.sort(reverse=True)
    return matches[0][3]


# ============================================================
#  系统信息
# ============================================================

def get_system_info() -> dict:
    ram_gb = 0.0
    avail_gb = 0.0
    try:
        if platform.system() == "Windows":
            import ctypes
            class MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMSTAT()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
        elif platform.system() == "Darwin":
            total = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if total.returncode == 0 and total.stdout.strip().isdigit():
                ram_gb = int(total.stdout.strip()) / (1024 ** 3)

            vm_stat = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True, timeout=5
            )
            if vm_stat.returncode == 0:
                page_size = 4096
                m = re.search(r"page size of (\d+) bytes", vm_stat.stdout)
                if m:
                    page_size = int(m.group(1))

                pages = {}
                for line in vm_stat.stdout.splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    value = value.strip().rstrip(".").replace(".", "")
                    if value.isdigit():
                        pages[key.strip()] = int(value)

                avail_pages = (
                    pages.get("Pages free", 0) +
                    pages.get("Pages inactive", 0) +
                    pages.get("Pages speculative", 0)
                )
                avail_gb = avail_pages * page_size / (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) / (1024 ** 2)
                    if line.startswith("MemAvailable"):
                        avail_gb = int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass

    try:
        st = os.statvfs(str(BASE_DIR))
        disk_free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        disk_total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
    except Exception:
        try:
            total, used, free = shutil.disk_usage(str(BASE_DIR))
            disk_free_gb = free / (1024 ** 3)
            disk_total_gb = total / (1024 ** 3)
        except Exception:
            disk_free_gb = 0
            disk_total_gb = 0

    # GPU 检测
    gpu_name = ""
    gpu_vram_mb = 0
    gpu_vram_free_mb = 0
    gpu_backend = "cpu"
    # ── NVIDIA GPU 检测 ───────────────────────────────────────────────────────
    # nvidia-smi 格式：name, memory.total, memory.free
    # GPU 名字可能含逗号，从右侧取两个数字字段更安全
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            parts = [x.strip() for x in first_line.rsplit(",", 2)]
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                gpu_name       = parts[0]
                gpu_vram_mb    = int(parts[1])
                gpu_vram_free_mb = int(parts[2])
                gpu_backend    = "cuda"
            elif len(parts) >= 2 and parts[-1].isdigit():
                gpu_name    = parts[0]
                gpu_vram_mb = int(parts[-1])
                gpu_backend = "cuda"
    except FileNotFoundError:
        pass   # nvidia-smi 未安装
    except Exception:
        pass

    if not gpu_name and platform.system() == "Darwin":
        gpu_name = f"Apple Metal ({platform.machine()})"
        gpu_backend = "metal"
    elif not gpu_name:
        try:
            vulkaninfo = shutil.which("vulkaninfo")
            if vulkaninfo:
                result = subprocess.run(
                    [vulkaninfo, "--summary"],
                    capture_output=True, text=True, timeout=8
                )
                if result.returncode == 0:
                    gpu_name = "Vulkan GPU"
                    gpu_backend = "vulkan"
        except Exception:
            pass

    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 4,
        "ram_gb": round(ram_gb, 1),
        "ram_avail_gb": round(avail_gb, 1),
        "disk_free_gb": round(disk_free_gb, 1),
        "disk_total_gb": round(disk_total_gb, 1),
        "python": platform.python_version(),
        "is_arm": platform.machine().lower() in ("aarch64", "armv7l", "arm64"),
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "gpu_vram_free_mb": gpu_vram_free_mb,
        "gpu_backend": gpu_backend,
    }


def find_llama_binary(name: str) -> Path:
    exe = f"{name}.exe" if platform.system() == "Windows" else name
    candidates = [
        LLAMA_DIR / exe,
        LLAMA_DIR / "build" / "bin" / exe,
        LLAMA_DIR / "build" / "bin" / "Release" / exe,
        LLAMA_DIR / "bin" / exe,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return LLAMA_DIR / exe


def get_llama_version() -> dict:
    binary = find_llama_binary("llama-server")
    if not binary.exists():
        return {"binary": str(binary), "version": "", "build": "", "available": False}

    try:
        cache_key = (str(binary), binary.stat().st_mtime)
    except Exception:
        cache_key = (str(binary), None)
    now = time.time()
    if (
        LLAMA_VERSION_CACHE["key"] == cache_key and
        LLAMA_VERSION_CACHE["value"] is not None and
        now - LLAMA_VERSION_CACHE["ts"] < 30
    ):
        return dict(LLAMA_VERSION_CACHE["value"])

    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=10, cwd=str(binary.parent)
        )
        output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        version = ""
        build = ""
        for line in output.splitlines():
            line = line.strip()
            if line.lower().startswith("version:"):
                version = line.split(":", 1)[1].strip()
            elif line.lower().startswith("built with"):
                build = line
        value = {
            "binary": str(binary),
            "version": version,
            "build": build,
            "available": True,
        }
        LLAMA_VERSION_CACHE.update({"key": cache_key, "value": value, "ts": now})
        return dict(value)
    except Exception as e:
        value = {
            "binary": str(binary),
            "version": "",
            "build": str(e),
            "available": True,
        }
        LLAMA_VERSION_CACHE.update({"key": cache_key, "value": value, "ts": now})
        return dict(value)


# ============================================================
#  模型源适配器
# ============================================================

class ModelSource:
    USER_AGENT = "llama-deploy/1.0"
    TIMEOUT = 15

    def _fetch_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def search(self, query: str, limit: int = 20) -> list:
        raise NotImplementedError

    def list_files(self, repo_id: str) -> list:
        raise NotImplementedError


class HuggingFaceSource(ModelSource):
    def __init__(self, mirror: str = ""):
        self.base = (mirror or "https://huggingface.co").rstrip("/")
        self.api_base = self.base

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(f"{query} gguf")
        url = f"{self.api_base}/api/models?search={q}&sort=downloads&direction=-1&limit={limit}"
        data = self._fetch_json(url)
        if isinstance(data, dict) and "error" in data:
            return [data]
        results = []
        for item in (data if isinstance(data, list) else []):
            model_id = item.get("modelId", "") or item.get("id", "")
            tags = item.get("tags", [])
            if not any(k in model_id.lower() for k in ["gguf"]) and \
               "gguf" not in [t.lower() for t in tags]:
                continue
            results.append({
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "tags": tags[:10],
                "updated": item.get("lastModified", ""),
                "source": "huggingface",
            })
        return results

    def list_files(self, repo_id: str) -> list:
        # 获取文件列表
        url = f"{self.api_base}/api/models/{repo_id}"
        data = self._fetch_json(url)
        if isinstance(data, dict) and "error" in data:
            return [data]

        siblings = data.get("siblings", [])

        # 尝试 tree API 获取文件大小
        size_map = {}
        try:
            tree_url = f"{self.api_base}/api/models/{repo_id}/tree/main"
            tree_data = self._fetch_json(tree_url)
            if isinstance(tree_data, list):
                for item in tree_data:
                    if isinstance(item, dict):
                        path = item.get("path", "")
                        size = item.get("size", 0) or item.get("lfs", {}).get("size", 0) or 0
                        if path and size:
                            size_map[path] = size
        except Exception:
            pass

        # 解析所有 GGUF 文件
        raw_files = []
        for sib in siblings:
            if not isinstance(sib, dict):
                continue
            fname = sib.get("rfilename", "")
            if not fname.lower().endswith(".gguf"):
                continue
            size = size_map.get(fname, 0) or sib.get("size", 0) or 0
            quant = "unknown"
            for q in ["Q2_K","Q3_K_S","Q3_K_M","Q3_K_L","Q4_0","Q4_K_S",
                       "Q4_K_M","Q5_0","Q5_K_S","Q5_K_M","Q6_K","Q8_0",
                       "F16","BF16","F32","IQ1_S","IQ2_XXS","IQ2_XS",
                       "IQ3_XXS","IQ3_XS","IQ4_XS","IQ4_NL"]:
                if q.lower() in fname.lower() or q in fname:
                    quant = q
                    break
            is_mmproj = "mmproj" in fname.lower()
            raw_files.append({
                "filename": fname, "size": size, "quant": quant,
                "is_mmproj": is_mmproj,
            })

        # 检测并合并分片文件
        import re
        shard_pattern = re.compile(r'^(.+)-(\d{5})-of-(\d{5})\.gguf$')
        shard_groups = {}  # base_name -> [files]
        single_files = []

        for f in raw_files:
            m = shard_pattern.match(f["filename"])
            if m:
                base = m.group(1)
                total = int(m.group(3))
                key = f"{base}|{total}"
                if key not in shard_groups:
                    shard_groups[key] = {
                        "base": base, "total": total,
                        "quant": f["quant"], "is_mmproj": f["is_mmproj"],
                        "shards": [],
                    }
                shard_groups[key]["shards"].append(f)
            else:
                single_files.append(f)

        # 构建最终文件列表
        files = []

        # 单文件
        for f in single_files:
            size = f["size"]
            ram_est = round(size / (1024**3) * 1.2 + 0.8, 1) if size else 0
            files.append({
                "filename": f["filename"],
                "size": size,
                "size_mb": round(size / (1024**2)) if size else 0,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "quant": f["quant"],
                "ram_estimate_gb": ram_est,
                "is_mmproj": f["is_mmproj"],
                "is_shard": False,
                "shard_count": 1,
                "shard_files": [f["filename"]],
                "download_url": f"{self.base}/{repo_id}/resolve/main/{f['filename']}",
            })

        # 分片文件（合并为一个条目）
        for key, group in shard_groups.items():
            total_size = sum(s["size"] for s in group["shards"])
            shard_filenames = sorted([s["filename"] for s in group["shards"]])
            shard_file_sizes = {s["filename"]: s.get("size", 0) for s in group["shards"]}
            shard_download_urls = {
                name: f"{self.base}/{repo_id}/resolve/main/{name}"
                for name in shard_filenames
            }
            found = len(group["shards"])
            total_expected = group["total"]
            display_name = f"{group['base']}-*.gguf ({found}/{total_expected} 分片)"
            ram_est = round(total_size / (1024**3) * 1.2 + 0.8, 1) if total_size else 0
            # 第一个分片文件名（llama.cpp 自动检测后续分片）
            first_shard = shard_filenames[0] if shard_filenames else ""

            files.append({
                "filename": display_name,
                "size": total_size,
                "size_mb": round(total_size / (1024**2)) if total_size else 0,
                "size_gb": round(total_size / (1024**3), 2) if total_size else 0,
                "quant": group["quant"],
                "ram_estimate_gb": ram_est,
                "is_mmproj": group["is_mmproj"],
                "is_shard": True,
                "shard_count": total_expected,
                "shard_files": shard_filenames,
                "shard_file_sizes": shard_file_sizes,
                "shard_download_urls": shard_download_urls,
                "first_shard": first_shard,
                "download_url": f"{self.base}/{repo_id}/resolve/main/{first_shard}",
            })

        files.sort(key=lambda x: (x["is_mmproj"], x["size"]))
        return files

class ModelScopeSource(ModelSource):
    """ModelScope 模型源"""

    def __init__(self):
        self.bases = [
            "https://www.modelscope.cn",
            "https://modelscope.cn",
        ]

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(query)
        # 基于网页 URL 参数构造 API 请求
        api_paths = [
            f"/api/v1/models?libraries=GGUF&Query={q}&PageSize={limit}&sort=latest",
            f"/api/v1/models?Query={q}&PageSize={limit}&SortBy=GmtModified",
            f"/api/v1/models?query={q}&pageSize={limit}&libraries=GGUF",
            f"/api/v1/dolphin/models?Query={q}&PageSize={limit}",
            f"/api/v1/hub/models?query={q}&pageSize={limit}",
        ]
        for base in self.bases:
            for path in api_paths:
                try:
                    data = self._fetch_json(base + path)
                    if isinstance(data, dict) and "error" not in data:
                        results = self._parse(data)
                        if results:
                            return results
                except Exception:
                    continue

        # 全部失败，回退到 HuggingFace 搜索
        hf = HuggingFaceSource("")
        hf_results = hf.search(query, limit)
        for r in hf_results:
            r["source"] = "modelscope(via HF)"
        return hf_results

    def _parse(self, data) -> list:
        models = []
        if isinstance(data, dict):
            # 递归搜索列表
            for k1 in ["Data", "data", "Result", "result"]:
                d = data.get(k1)
                if isinstance(d, list) and len(d) > 0:
                    models = d
                    break
                if isinstance(d, dict):
                    for k2 in ["Models","models","Model","Items","items","List","Records"]:
                        arr = d.get(k2)
                        if isinstance(arr, list) and len(arr) > 0:
                            models = arr
                            break
                    if models:
                        break
        elif isinstance(data, list):
            models = data

        results = []
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = (item.get("Path") or item.get("ModelId") or
                        item.get("Name") or item.get("id") or "")
            if not model_id:
                continue
            name = (item.get("ChineseName") or item.get("Name") or
                    item.get("name") or model_id)
            downloads = 0
            for k in ["Downloads","downloads","DownloadCount"]:
                v = item.get(k)
                if v:
                    try: downloads = int(v); break
                    except: pass
            likes = 0
            for k in ["Stars","stars","Likes","likes"]:
                v = item.get(k)
                if v:
                    try: likes = int(v); break
                    except: pass
            tags = item.get("Tags") or item.get("tags") or []
            if not isinstance(tags, list): tags = []
            clean_tags = []
            for t in tags[:8]:
                if isinstance(t, dict): clean_tags.append(str(t.get("Name","")))
                elif isinstance(t, str): clean_tags.append(t)
            results.append({
                "id": model_id, "name": name,
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "downloads": downloads, "likes": likes, "tags": clean_tags,
                "updated": item.get("GmtModified") or item.get("LastUpdatedDate") or "",
                "source": "modelscope",
            })
        return results

    def list_files(self, repo_id: str) -> list:
        for base in self.bases:
            urls = [
                f"{base}/api/v1/models/{repo_id}/repo/files?Recursive=true&PageSize=200",
                f"{base}/api/v1/models/{repo_id}/repo?Recursive=true&PageSize=200",
            ]
            for url in urls:
                try:
                    data = self._fetch_json(url)
                    if isinstance(data, dict) and "error" not in data:
                        files = self._parse_files(data, repo_id)
                        if files:
                            return files
                except Exception:
                    continue

        # 回退 HuggingFace
        hf = HuggingFaceSource("")
        return hf.list_files(repo_id)

    def _parse_files(self, data, repo_id: str) -> list:
        raw = []
        if isinstance(data, list): raw = data
        elif isinstance(data, dict):
            for path in [["Data","Files"],["Data","files"],["Data"],["Files"],["files"]]:
                obj = data
                for k in path:
                    obj = obj.get(k) if isinstance(obj, dict) else None
                if isinstance(obj, list):
                    raw = obj; break
        files = []
        for item in (raw if isinstance(raw, list) else []):
            if not isinstance(item, dict): continue
            fname = ""
            for k in ["Path","path","Name","name","FilePath"]:
                v = item.get(k)
                if v: fname = str(v); break
            if not fname.lower().endswith(".gguf"): continue
            size = 0
            for k in ["Size","size","Bytes"]:
                v = item.get(k)
                if v:
                    try: size = int(v); break
                    except: pass
            ram_est = round(size/(1024**3)*1.2+0.8, 1) if size else 0
            quant = "unknown"
            for q in ["Q2_K","Q3_K_S","Q3_K_M","Q3_K_L","Q4_0","Q4_K_S",
                       "Q4_K_M","Q5_0","Q5_K_S","Q5_K_M","Q6_K","Q8_0","F16","BF16","F32"]:
                if q.lower() in fname.lower(): quant = q; break
            is_mmproj = "mmproj" in fname.lower()
            base = self.bases[0]
            files.append({
                "filename": fname, "size": size,
                "size_mb": round(size/(1024**2)) if size else 0,
                "size_gb": round(size/(1024**3),2) if size else 0,
                "quant": quant, "ram_estimate_gb": ram_est, "is_mmproj": is_mmproj,
                "download_url": f"{base}/models/{repo_id}/resolve/master/{fname}",
            })
        files.sort(key=lambda x: (x["is_mmproj"], x["size"]))
        return files
class OllamaLibrarySource(ModelSource):
    """搜索主流 GGUF 发布者的模型（bartowski, unsloth, TheBloke 等）"""

    POPULAR_AUTHORS = ["bartowski", "unsloth", "TheBloke", "QuantFactory", "mradermacher"]

    def __init__(self):
        pass

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(f"{query} gguf")
        # 同时搜索多个知名 GGUF 发布者
        all_results = []
        # 先搜全局
        url = f"https://huggingface.co/api/models?search={q}&sort=downloads&direction=-1&limit={limit}"
        data = self._fetch_json(url)
        if isinstance(data, list):
            for item in data:
                model_id = item.get("modelId", "") or item.get("id", "")
                tags = item.get("tags", [])
                # 必须是 GGUF 相关
                if "gguf" not in model_id.lower() and "gguf" not in [t.lower() for t in tags]:
                    continue
                # 优先显示知名发布者
                author = model_id.split("/")[0] if "/" in model_id else ""
                is_popular = author.lower() in [a.lower() for a in self.POPULAR_AUTHORS]
                all_results.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                    "author": author,
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": tags[:10],
                    "updated": item.get("lastModified", ""),
                    "source": "gguf",
                    "_popular": is_popular,
                })

        # 知名发布者排前面
        all_results.sort(key=lambda x: (not x.get("_popular", False), -(x.get("downloads", 0))))
        # 清理内部字段
        for r in all_results:
            r.pop("_popular", None)
        return all_results[:limit]

    def list_files(self, repo_id: str) -> list:
        hf = HuggingFaceSource("")
        return hf.list_files(repo_id)


class UnslothSource(ModelSource):
    """
    Unsloth 精选 GGUF 源。
    Unsloth 以高质量量化著称，支持动态量化（Dynamic GGUF），
    提供比 TheBloke 更新、更准确的量化版本。
    直接搜索 unsloth 账号下的全部模型。
    """

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(f"{query} gguf")
        # 只搜 unsloth 作者名下的模型，保证质量
        url = (
            f"https://huggingface.co/api/models"
            f"?search={q}&author=unsloth&sort=downloads&direction=-1&limit={limit}"
        )
        data = self._fetch_json(url)
        if isinstance(data, dict) and "error" in data:
            return [data]
        results = []
        for item in (data if isinstance(data, list) else []):
            model_id = item.get("modelId", "") or item.get("id", "")
            tags = item.get("tags", [])
            results.append({
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "author": "unsloth",
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "tags": [t for t in tags if isinstance(t, str)][:8],
                "updated": item.get("lastModified", ""),
                "source": "unsloth",
            })
        return results

    def list_files(self, repo_id: str) -> list:
        hf = HuggingFaceSource("")
        return hf.list_files(repo_id)


class HFMirrorCNSource(ModelSource):
    """
    HuggingFace 国内加速镜像源（hf-mirror.com）。
    API 与 HuggingFace 完全兼容，国内无需代理直接访问。
    搜索走 hf-mirror API，下载链接也指向镜像。
    """

    MIRROR = "https://hf-mirror.com"

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(f"{query} gguf")
        url = f"{self.MIRROR}/api/models?search={q}&sort=downloads&direction=-1&limit={limit}"
        data = self._fetch_json(url)
        if isinstance(data, dict) and "error" in data:
            return [data]
        results = []
        for item in (data if isinstance(data, list) else []):
            model_id = item.get("modelId", "") or item.get("id", "")
            tags = item.get("tags", [])
            if not any(k in model_id.lower() for k in ["gguf"]) and \
               "gguf" not in [t.lower() for t in tags if isinstance(t, str)]:
                continue
            results.append({
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "tags": [t for t in tags if isinstance(t, str)][:8],
                "updated": item.get("lastModified", ""),
                "source": "hf-mirror",
            })
        return results

    def list_files(self, repo_id: str) -> list:
        # 文件列表 API 和 HuggingFace 相同，但 base 换成镜像地址
        hf = HuggingFaceSource(self.MIRROR)
        return hf.list_files(repo_id)


class GiteeAISource(ModelSource):
    """
    Gitee AI 模型库（ai.gitee.com）。
    国内自研平台，收录了大量主流中文模型的 GGUF 版本，
    访问速度快，对中文模型（Qwen、DeepSeek、GLM 等）收录完整。
    """

    BASE = "https://ai.gitee.com"

    def search(self, query: str, limit: int = 20) -> list:
        q = urllib.parse.quote(query)
        # Gitee AI 公开 API
        urls = [
            f"{self.BASE}/api/v1/models?q={q}&format=gguf&limit={limit}&sort=downloads",
            f"{self.BASE}/api/v1/models?search={q}&limit={limit}",
        ]
        for url in urls:
            try:
                data = self._fetch_json(url)
                results = self._parse(data)
                if results:
                    return results
            except Exception:
                continue
        # Gitee AI API 失败时回退到 HF 搜索（搜索结果标记来源）
        hf = HuggingFaceSource("")
        hf_results = hf.search(query, limit)
        for r in hf_results:
            r["source"] = "gitee-ai(via HF)"
        return hf_results

    def _parse(self, data) -> list:
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("data", "models", "items", "results"):
                val = data.get(key)
                if isinstance(val, list):
                    items = val
                    break
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("path") or item.get("id") or item.get("name") or ""
            if not model_id:
                continue
            results.append({
                "id": model_id,
                "name": item.get("name") or model_id.split("/")[-1],
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "downloads": item.get("downloads_count") or item.get("downloads", 0),
                "likes": item.get("stars_count") or item.get("likes", 0),
                "tags": item.get("tags", [])[:8] if isinstance(item.get("tags"), list) else [],
                "updated": item.get("updated_at") or item.get("lastModified", ""),
                "source": "gitee-ai",
            })
        return results

    def list_files(self, repo_id: str) -> list:
        # 优先尝试 Gitee AI 自有 API，失败回退 HF
        urls = [
            f"{self.BASE}/api/v1/repos/{repo_id}/git/trees/main?recursive=true",
        ]
        for url in urls:
            try:
                data = self._fetch_json(url)
                files = self._parse_files(data, repo_id)
                if files:
                    return files
            except Exception:
                pass
        hf = HuggingFaceSource("")
        return hf.list_files(repo_id)

    def _parse_files(self, data, repo_id: str) -> list:
        items = []
        if isinstance(data, dict):
            items = data.get("tree", [])
        elif isinstance(data, list):
            items = data
        files = []
        for item in items:
            if not isinstance(item, dict):
                continue
            fname = item.get("path", "") or item.get("name", "")
            if not str(fname).lower().endswith(".gguf"):
                continue
            size = int(item.get("size", 0) or 0)
            quant = "unknown"
            for q in ["Q2_K","Q3_K_M","Q4_0","Q4_K_S","Q4_K_M","Q5_K_M","Q6_K","Q8_0","F16","BF16"]:
                if q.lower() in str(fname).lower():
                    quant = q
                    break
            is_mmproj = "mmproj" in str(fname).lower()
            files.append({
                "filename": fname,
                "size": size,
                "size_mb": round(size / (1024**2)) if size else 0,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "quant": quant,
                "ram_estimate_gb": round(size / (1024**3) * 1.2 + 0.8, 1) if size else 0,
                "is_mmproj": is_mmproj,
                "download_url": f"{self.BASE}/{repo_id}/resolve/main/{fname}",
            })
        files.sort(key=lambda x: (x["is_mmproj"], x["size"]))
        return files


def get_source(name: str, config: dict = None) -> ModelSource:
    """
    根据名称返回对应的模型源实例。
    config 用于读取用户配置的镜像地址。
    """
    if config is None:
        config = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else {}
    mirror = config.get("download", {}).get("hf_mirror", "")

    source_map = {
        "modelscope":       ModelScopeSource,
        "ollama":           OllamaLibrarySource,
        "gguf":             OllamaLibrarySource,    # 兼容旧名称
        "unsloth":          UnslothSource,
        "hf_mirror":        lambda: HFMirrorCNSource(),
        "huggingface_mirror": lambda: HFMirrorCNSource(),
        "giteeai":          GiteeAISource,
    }
    cls = source_map.get(name)
    if cls:
        return cls() if callable(cls) else cls()
    # 默认：HuggingFace（可带镜像）
    return HuggingFaceSource(mirror)

# ============================================================
#  RAM 推荐引擎
# ============================================================

def get_recommendations(ram_gb: float, vram_mb: int = 0) -> dict:
    usable = ram_gb * 0.65
    max_model_gb = usable - 0.3
    recs = {
        "max_model_size_gb": round(max_model_gb, 1),
        "recommended_ctx": 2048 if ram_gb >= 8 else (1024 if ram_gb >= 4 else 512),
        "recommended_quant": "Q4_K_M",
        "tips": [],
    }
    if vram_mb > 0:
        vram_gb = vram_mb / 1024
        recs["gpu_max_model_gb"] = round(vram_gb * 0.85, 1)
        recs["tips"].append(f"检测到 GPU 显存 {vram_gb:.1f}GB，可加载 ≤{recs['gpu_max_model_gb']}GB 的模型到 GPU")
        recs["tips"].append("建议设置 gpu_layers=-1 将全部层卸载到 GPU")
        if vram_gb >= 6:
            recs["tips"].append("显存充足，可使用 Q5_K_M 或 Q8_0 量化获得更好质量")
    if ram_gb <= 4:
        recs["tips"].extend(["建议 0.5-1B 参数模型", "Q4_K_M 量化最佳", "关闭思考模式节省内存"])
    elif ram_gb <= 8:
        recs["tips"].append("可运行 1-3B 参数模型")
    else:
        recs["tips"].append("可运行 3-7B+ 参数模型")
    return recs


# ============================================================
#  部署管理器
# ============================================================

class DeployManager:
    def __init__(self):
        self.deploy_process = None
        self.deploy_log = []
        self.is_deploying = False

    def _start_task(self, command: list, start_message: str):
        if self.is_deploying:
            return {"status": "error", "message": "已有任务在运行，请等待当前任务完成"}
        self.deploy_log = []
        self.is_deploying = True

        def _run():
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                if sys.platform == "win32":
                    env["PYTHONUTF8"] = "1"
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=str(BASE_DIR), env=env,
                )
                self.deploy_process = proc
                for raw_line in iter(proc.stdout.readline, b''):
                    try:
                        line = raw_line.decode('utf-8', errors='replace').rstrip()
                    except Exception:
                        line = raw_line.decode('gbk', errors='replace').rstrip()
                    self.deploy_log.append(line)
                proc.wait()
                status = "✅ 完成" if proc.returncode == 0 else "❌ 失败"
                self.deploy_log.append(f"\n{status}")
            except Exception as e:
                self.deploy_log.append(f"❌ 错误: {e}")
            finally:
                self.is_deploying = False

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "ok", "message": start_message}

    def start_deploy(self):
        """启动一键部署任务（后台线程执行 deploy.py）"""
        return self._start_task(
            [sys.executable, str(BASE_DIR / "deploy.py")],
            "部署已启动"
        )

    def upgrade_llama(self):
        running = running_llama_processes()
        if running:
            return {
                "status": "error",
                "message": "请先停止所有 llama.cpp 进程再升级: " + ", ".join(running),
            }
        return self._start_task(
            [sys.executable, str(BASE_DIR / "deploy.py"), "--upgrade-llama"],
            "llama.cpp 升级已启动"
        )

    def get_deploy_log(self, since: int = 0) -> dict:
        return {
            "lines": self.deploy_log[since:],
            "total": len(self.deploy_log),
            "running": self.is_deploying,
        }

    def _compat_status(self, config: dict = None) -> dict:
        cfg = config or (parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config())
        cc = ensure_compat_config(cfg)
        ips = get_lan_ips()
        port = int(cc.get("port", 11434) or 11434)
        pid = None
        running = False
        if COMPAT_PID_FILE.exists():
            try:
                pid_text = COMPAT_PID_FILE.read_text(encoding="utf-8").strip()
                if pid_text.isdigit():
                    pid = int(pid_text)
                    running = pid_running(pid)
                if not running:
                    COMPAT_PID_FILE.unlink(missing_ok=True)
            except Exception:
                running = False
                COMPAT_PID_FILE.unlink(missing_ok=True)
        base_urls = [f"http://{ip}:{port}" for ip in ips]
        return {
            "gateway_running": running,
            "gateway_pid": pid if running else None,
            "gateway_port": port,
            "gateway_host": cc.get("host", "0.0.0.0"),
            "gateway_upstream": cc.get("upstream_url", ""),
            "gateway_model_alias": cc.get("model_alias", "llama-deploy-local"),
            "gateway_api_key": cc.get("api_key", "local-no-key-needed"),
            "gateway_urls": base_urls,
            "gateway_openai_urls": [u + "/v1" for u in base_urls],
            "gateway_ollama_urls": base_urls,
            "gateway_anthropic_urls": base_urls,
        }

    def start_gateway(self) -> dict:
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config()
        ensure_compat_config(cfg)
        save_config(cfg)
        st = self._compat_status(cfg)
        if st["gateway_running"]:
            return {"status": "ok", "message": f"兼容网关已在运行 (PID: {st['gateway_pid']})", **st}
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "compat.py"), "start"],
            capture_output=True,
            timeout=20,
            cwd=str(BASE_DIR),
            env=env,
        )
        out = (result.stdout or b"").decode("utf-8", errors="replace")
        err = (result.stderr or b"").decode("utf-8", errors="replace")
        st = self._compat_status(cfg)
        if result.returncode == 0 and st["gateway_running"]:
            return {"status": "ok", "message": "兼容网关已启动", **st}
        log_tail = ""
        if COMPAT_LOG_FILE.exists():
            log_tail = COMPAT_LOG_FILE.read_text(encoding="utf-8", errors="replace")[-600:]
        return {"status": "error", "message": f"兼容网关启动失败:\n{out}\n{err}\n{log_tail}".strip(), **st}

    def stop_gateway(self) -> dict:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "compat.py"), "stop"],
            capture_output=True,
            timeout=15,
            cwd=str(BASE_DIR),
            env=env,
        )
        out = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        st = self._compat_status()
        return {"status": "ok" if result.returncode == 0 else "error", "message": out or "操作完成", **st}

    def publish_lan(self) -> dict:
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config()
        cfg.setdefault("server", {})["host"] = "0.0.0.0"
        server_port = int(cfg.get("server", {}).get("port", 8080) or 8080)
        cc = ensure_compat_config(cfg)
        cc["enabled"] = True
        cc["host"] = "0.0.0.0"
        cc["upstream_url"] = f"http://127.0.0.1:{server_port}"
        save_config(cfg)

        server_msg = ""
        if not self.get_server_status().get("server_running"):
            server_start = self.start_server(False)
            server_msg = server_start.get("message", "")
            if server_start.get("status") == "error":
                return {"status": "error", "message": "服务器启动失败，无法发布到局域网:\n" + server_msg}

        gateway_start = self.start_gateway()
        st = self.get_server_status()
        msg = "已发布到局域网"
        if server_msg:
            msg += "\n" + server_msg
        if gateway_start.get("status") == "error":
            msg += "\n兼容网关启动失败: " + gateway_start.get("message", "")
            return {"status": "error", "message": msg, **st}
        return {"status": "ok", "message": msg, **st}

    def open_firewall(self) -> dict:
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config()
        cc = ensure_compat_config(cfg)
        server_port = int(cfg.get("server", {}).get("port", 8080) or 8080)
        gateway_port = int(cc.get("port", 11434) or 11434)
        ports = [server_port, gateway_port]
        if platform.system() != "Windows":
            return {"status": "ok", "message": f"非 Windows 系统，请自行放行端口: {', '.join(map(str, ports))}"}
        outputs = []
        ok = True
        for port in ports:
            name = f"llama-deploy {port}"
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={name}", "dir=in", "action=allow", "protocol=TCP", f"localport={port}",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            outputs.append((r.stdout or r.stderr or "").strip())
            ok = ok and r.returncode == 0
        if ok:
            return {"status": "ok", "message": f"已尝试放行端口: {', '.join(map(str, ports))}"}
        commands = "\n".join(
            f'netsh advfirewall firewall add rule name="llama-deploy {p}" dir=in action=allow protocol=TCP localport={p}'
            for p in ports
        )
        return {
            "status": "error",
            "message": "防火墙规则添加失败，可能需要以管理员身份运行。\n\n请在管理员 PowerShell 执行:\n" + commands + "\n\n" + "\n".join(outputs),
        }

    def get_server_status(self) -> dict:
        pid = None
        running = False
        if PID_FILE.exists():
            try:
                pid_text = PID_FILE.read_text(encoding="utf-8").strip()
                if pid_text.isdigit():
                    pid = int(pid_text)
                    running = pid_running(pid)
                if not running:
                    PID_FILE.unlink(missing_ok=True)
            except Exception:
                running = False
                PID_FILE.unlink(missing_ok=True)
        config = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else {}
        ensure_compat_config(config)
        model_cfg = config.get("model", {})
        model_file = model_cfg.get("model_file", "")
        model_path = MODELS_DIR / model_file
        model_exists = bool(model_file and model_path.exists())
        model_actual_size = model_path.stat().st_size if model_exists else 0
        model_expected_size = int(model_cfg.get("model_size", 0) or 0)
        model_complete = bool(model_exists and model_actual_size > 1000)
        model_size_mismatch = bool(
            model_complete and model_expected_size and model_actual_size != model_expected_size
        )
        mmproj_file = model_cfg.get("mmproj_file", "")
        mmproj_path = MODELS_DIR / "vision" / mmproj_file if mmproj_file else None
        llama_info = get_llama_version()
        server_port = config.get("server", {}).get("port", 8080)
        lan_ips = get_lan_ips()
        status = {
            "server_running": running,
            "server_pid": pid if running else None,
            "model_deployed": model_complete,
            "model_exists": model_exists,
            "model_complete": model_complete,
            "model_size_mismatch": model_size_mismatch,
            "model_file": model_file,
            "model_size_mb": round(model_actual_size / (1024**2)) if model_actual_size else 0,
            "model_expected_size_mb": round(model_expected_size / (1024**2)) if model_expected_size else 0,
            "mmproj_deployed": mmproj_path is not None and mmproj_path.exists() and mmproj_path.stat().st_size > 1000,
            "mmproj_file": mmproj_file,
            "host": config.get("server", {}).get("host", "0.0.0.0"),
            "port": server_port,
            "lan_ips": lan_ips,
            "lan_urls": [f"http://{ip}:{server_port}" for ip in lan_ips],
            "openai_urls": [f"http://{ip}:{server_port}/v1" for ip in lan_ips],
            "llama_version": llama_info.get("version", ""),
            "llama_build": llama_info.get("build", ""),
            "llama_binary": llama_info.get("binary", ""),
        }
        status.update(self._compat_status(config))
        return status

    def start_server(self, vision: bool = False):
        st = self.get_server_status()
        if st["server_running"]:
            return {"status": "error", "message": "服务器已在运行"}

        cmd = [sys.executable, str(BASE_DIR / "run.py"), "server", "--background"]
        if vision:
            cmd.append("--vision")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=15,
                cwd=str(BASE_DIR), env=env,
            )
            output = result.stdout.decode('utf-8', errors='replace')
            err = result.stderr.decode('utf-8', errors='replace')

            if result.returncode == 0 and PID_FILE.exists():
                pid = PID_FILE.read_text().strip()
                port = parse_jsonc(CONFIG_FILE).get("server", {}).get("port", 8080)
                return {
                    "status": "ok",
                    "message": f"服务器已启动 (PID: {pid})\n地址: http://localhost:{port}",
                }
            else:
                # 读取日志获取详细错误
                log_msg = ""
                log_file = BASE_DIR / ".llama-server.log"
                if log_file.exists():
                    try:
                        log_msg = log_file.read_text(encoding='utf-8')[-300:]
                    except Exception:
                        pass
                return {
                    "status": "error",
                    "message": f"启动失败:\n{output}\n{err}\n{log_msg}".strip(),
                }
        except subprocess.TimeoutExpired:
            # 15 秒超时，检查是否实际已启动
            if PID_FILE.exists():
                pid = PID_FILE.read_text().strip()
                return {"status": "ok", "message": f"服务器启动中 (PID: {pid})"}
            return {"status": "error", "message": "启动超时"}
        except Exception as e:
            return {"status": "error", "message": f"启动异常: {e}"}
    def stop_server(self):
        killed = False

        # 方法1：按 PID 文件杀
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True)
                else:
                    os.kill(pid, signal.SIGTERM)
                killed = True
            except Exception:
                pass
            PID_FILE.unlink(missing_ok=True)

        # 方法2：按进程名杀（确保彻底）
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    capture_output=True, text=True
                )
                if "SUCCESS" in (result.stdout or ""):
                    killed = True
            else:
                subprocess.run(["pkill", "-f", "llama-server"],
                               capture_output=True)
                killed = True
        except Exception:
            pass

        PID_FILE.unlink(missing_ok=True)

        if killed:
            return {"status": "ok", "message": "✅ 服务器已停止（已清理所有 llama-server 进程）"}
        else:
            return {"status": "ok", "message": "ℹ️ 没有找到运行中的服务器"}
# ============================================================
#  模型库管理
# ============================================================

class ModelLibrary:
    """管理本地已下载的模型"""

    @staticmethod
    def scan() -> list:
        """扫描 models 目录下所有 GGUF 文件"""
        models = []
        if not MODELS_DIR.exists():
            return models

        for f in MODELS_DIR.rglob("*.gguf"):
            # 跳过 vision 子目录中的 mmproj 文件
            is_mmproj = "mmproj" in f.name.lower()
            rel_path = f.relative_to(MODELS_DIR)

            # 解析量化类型
            quant = "unknown"
            for q in ["Q2_K","Q3_K_S","Q3_K_M","Q3_K_L","Q4_0","Q4_K_S",
                       "Q4_K_M","Q5_0","Q5_K_S","Q5_K_M","Q6_K","Q8_0",
                       "F16","BF16","F32","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ4_NL"]:
                if q.lower() in f.name.lower() or q in f.name:
                    quant = q
                    break

            size = f.stat().st_size
            models.append({
                "filename": f.name,
                "path": str(rel_path),
                "full_path": str(f),
                "size_mb": round(size / (1024**2)),
                "size_gb": round(size / (1024**3), 2),
                "quant": quant,
                "is_mmproj": is_mmproj,
                "is_active": False,  # 下面会标记
            })

        # 标记当前激活的模型
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else {}
        active_model = cfg.get("model", {}).get("model_file", "")
        active_mmproj = cfg.get("model", {}).get("mmproj_file", "")

        for m in models:
            if not m["is_mmproj"] and m["filename"] == active_model:
                m["is_active"] = True
            if m["is_mmproj"] and m["filename"] == active_mmproj:
                m["is_active"] = True

        # 排序：激活的在前，然后按大小
        models.sort(key=lambda x: (not x["is_active"], x["is_mmproj"], -x["size_mb"]))
        return models

    @staticmethod
    def activate(filename: str, is_mmproj: bool = False) -> dict:
        """切换激活的模型，自动匹配 mmproj"""
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config()
        model_cfg = cfg.setdefault("model", {})
        mmproj_map = model_cfg.setdefault("mmproj_map", {})
        mmproj_bindings = model_cfg.setdefault("mmproj_bindings", {})
        filename = local_model_name(filename)

        if is_mmproj:
            active_model = local_model_name(model_cfg.get("model_file", ""))
            model_cfg["mmproj_file"] = ensure_paired_mmproj_alias(active_model, filename)
            if active_model:
                mmproj_bindings[active_model] = model_cfg["mmproj_file"]
            # 绑定到当前 repo
            repo = model_cfg.get("repo_id", "")
            if repo:
                mmproj_map[repo] = model_cfg["mmproj_file"]
            save_config(cfg)
            return {"status": "ok", "message": f"✅ 视觉模块已切换为: {model_cfg['mmproj_file']}"}
        else:
            # 保存旧 repo 的 mmproj 映射
            old_repo = model_cfg.get("repo_id", "")
            old_mmproj = model_cfg.get("mmproj_file", "")
            if old_repo and old_mmproj:
                mmproj_map[old_repo] = old_mmproj

            model_cfg["model_file"] = filename
            # 清除分片信息（激活单文件）
            model_cfg.pop("shard_files", None)
            model_cfg.pop("shard_count", None)

            # 尝试自动匹配 mmproj（优先按模型文件名匹配，避免跨架构复用）
            preferred = mmproj_bindings.get(filename, "")
            matched = find_matching_mmproj(filename, preferred)
            if matched:
                model_cfg["mmproj_file"] = matched
                mmproj_bindings[filename] = matched

            # 方法1：从 mmproj_map 中按文件名关键词匹配
            name_lower = filename.lower()
            for repo_key, mmproj_name in mmproj_map.items() if not matched else []:
                # 检查 repo 名和模型文件名是否有共同关键词
                repo_parts = repo_key.lower().replace("/", " ").replace("-", " ").split()
                for part in repo_parts:
                    if len(part) > 3 and part in name_lower and mmproj_matches_model(filename, mmproj_name):
                        mmproj_path = MODELS_DIR / "vision" / mmproj_name
                        if mmproj_path.exists() and mmproj_path.stat().st_size > 1000:
                            matched = mmproj_name
                            model_cfg["mmproj_file"] = matched
                            mmproj_bindings[filename] = matched
                            break
                if matched:
                    break

            if not matched:
                model_cfg["mmproj_file"] = ""

            save_config(cfg)
            msg = f"✅ 已切换为: {filename}"
            if matched:
                msg += f"\n👁️ 自动匹配视觉模块: {matched}"
            else:
                msg += "\n💡 如需视觉功能，请在模型市场下载对应的 mmproj"
            return {"status": "ok", "message": msg}

    @staticmethod
    def delete(filename: str) -> dict:
        """删除模型文件"""
        # 安全检查
        if ".." in filename or "/" in filename or "\\" in filename:
            return {"status": "error", "message": "非法文件名"}

        # 搜索文件
        found = None
        for f in MODELS_DIR.rglob(filename):
            found = f
            break

        if not found or not found.exists():
            return {"status": "error", "message": f"文件不存在: {filename}"}

        # 不允许删除当前激活的模型
        cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else {}
        active = cfg.get("model", {}).get("model_file", "")
        if filename == active:
            return {"status": "error", "message": "不能删除当前激活的模型，请先切换到其他模型"}

        size_mb = found.stat().st_size // (1024 * 1024)
        found.unlink()
        return {"status": "ok", "message": f"✅ 已删除: {filename} ({size_mb}MB)"}


model_lib = ModelLibrary()
deploy_mgr = DeployManager()


# ============================================================
#  HTTP Handler
# ============================================================

class APIHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        """安全的日志方法，不会因为类型错误崩溃"""
        try:
            msg = str(format) % args if args else str(format)
            if "/api/" in msg:
                sys.stderr.write(f"{self.address_string()} - [{self.log_date_time_string()}] {msg}\n")
        except Exception:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/system":
            self._json_resp(get_system_info())
        elif path == "/api/config":
            cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else {}
            self._json_resp(cfg)
        elif path == "/api/search":
            src = params.get("source", "huggingface")
            q = params.get("q", "")
            lim = int(params.get("limit", "20"))
            if not q:
                self._json_resp({"error": "请输入搜索关键词"})
                return
            source = get_source(src)
            results = source.search(q, lim)
            self._json_resp({"results": results, "source": src})
        elif path.startswith("/api/files/"):
            repo_id = urllib.parse.unquote(path[len("/api/files/"):])
            src = params.get("source", "huggingface")
            try:
                source = get_source(src)
                files = source.list_files(repo_id)
                self._json_resp({"files": files, "repo_id": repo_id})
            except Exception as e:
                self._json_resp({"files": [], "repo_id": repo_id, "error": str(e)})
        elif path == "/api/recommend":
            ram = float(params.get("ram", "4"))
            vram = int(params.get("vram", "0"))
            self._json_resp(get_recommendations(ram, vram))

        elif path == "/api/models":
            self._json_resp({"models": model_lib.scan()})

        elif path == "/api/status":
            self._json_resp(deploy_mgr.get_server_status())
        elif path == "/api/deploy/log":
            since = int(params.get("since", "0"))
            self._json_resp(deploy_mgr.get_deploy_log(since))
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
        try:
            data = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/config":
            save_config(data)
            self._json_resp({"status": "ok", "message": "配置已保存"})
        elif path == "/api/config/model":
            cfg = parse_jsonc(CONFIG_FILE) if CONFIG_FILE.exists() else default_config()
            model_cfg = cfg.setdefault("model", {})
            mmproj_map = model_cfg.setdefault("mmproj_map", {})
            mmproj_bindings = model_cfg.setdefault("mmproj_bindings", {})

            old_repo = model_cfg.get("repo_id", "")
            old_mmproj = model_cfg.get("mmproj_file", "")
            new_repo = data.get("repo_id", old_repo)

            resp_extra = {}

            # 保存当前 repo 的 mmproj 映射（切换前先记住）
            if old_repo and old_mmproj:
                mmproj_map[old_repo] = old_mmproj

            # 更新基本字段
            if "repo_id" in data:
                model_cfg["repo_id"] = data["repo_id"]
            if "source" in data:
                model_cfg["source"] = data["source"]
            if "model_file" in data:
                remote_model_file = data.get("model_repo_file", data["model_file"])
                model_cfg["model_file"] = local_model_name(data["model_file"])
                model_cfg["model_repo_file"] = remote_model_file
                if "model_size" in data:
                    model_cfg["model_size"] = data.get("model_size", 0)
                if "model_download_url" in data:
                    model_cfg["model_download_url"] = data.get("model_download_url", "")
                if "shard_file_sizes" in data:
                    model_cfg["shard_file_sizes"] = data.get("shard_file_sizes", {})

            # 视觉模块处理
            if data.get("bind_mmproj"):
                # 用户手动选了 mmproj，绑定到当前 repo
                mmproj_remote = data.get("mmproj_repo_file", data.get("mmproj_file", ""))
                active_model = local_model_name(model_cfg.get("model_file", ""))
                model_cfg["mmproj_file"] = paired_mmproj_name(
                    active_model, data.get("mmproj_file", "") or mmproj_remote
                )
                model_cfg["mmproj_repo_file"] = mmproj_remote
                if "mmproj_size" in data:
                    model_cfg["mmproj_size"] = data.get("mmproj_size", 0)
                if "mmproj_download_url" in data:
                    model_cfg["mmproj_download_url"] = data.get("mmproj_download_url", "")
                bound_mmproj = model_cfg["mmproj_file"]
                mmproj_map[new_repo] = bound_mmproj
                if active_model and bound_mmproj:
                    mmproj_bindings[active_model] = bound_mmproj
                if bound_mmproj:
                    resp_extra["mmproj_matched"] = bound_mmproj
            elif "mmproj_file" in data:
                mmproj_remote = data.get("mmproj_repo_file", data["mmproj_file"])
                active_model = local_model_name(model_cfg.get("model_file", ""))
                model_cfg["mmproj_file"] = paired_mmproj_name(
                    active_model, data["mmproj_file"] or mmproj_remote
                )
                model_cfg["mmproj_repo_file"] = mmproj_remote
                if "mmproj_size" in data:
                    model_cfg["mmproj_size"] = data.get("mmproj_size", 0)
                if "mmproj_download_url" in data:
                    model_cfg["mmproj_download_url"] = data.get("mmproj_download_url", "")
                if model_cfg["mmproj_file"]:
                    mmproj_map[new_repo] = model_cfg["mmproj_file"]
                    if active_model:
                        mmproj_bindings[active_model] = model_cfg["mmproj_file"]
                    resp_extra["mmproj_matched"] = model_cfg["mmproj_file"]

            # 切换主模型时自动匹配 mmproj
            if data.get("auto_match_mmproj") and "mmproj_file" not in data and ("model_file" in data or new_repo != old_repo):
                active_model = model_cfg.get("model_file", "")
                preferred = mmproj_bindings.get(local_model_name(active_model), "")
                matched = find_matching_mmproj(active_model, preferred) or mmproj_map.get(new_repo, "")
                if matched:
                    # 检查文件是否真的存在
                    mmproj_path = MODELS_DIR / "vision" / matched
                    if mmproj_path.exists() and mmproj_path.stat().st_size > 1000 and mmproj_matches_model(active_model, matched):
                        model_cfg["mmproj_file"] = matched
                        mmproj_bindings[local_model_name(active_model)] = matched
                        resp_extra["mmproj_matched"] = matched
                    else:
                        model_cfg["mmproj_file"] = ""
                        model_cfg.pop("mmproj_size", None)
                        resp_extra["mmproj_cleared"] = True
                else:
                    model_cfg["mmproj_file"] = ""
                    model_cfg.pop("mmproj_size", None)
                    resp_extra["mmproj_cleared"] = True

            if "mmproj_use_xet" in data:
                model_cfg["mmproj_use_xet"] = data["mmproj_use_xet"]

            # 分片信息
            if "shard_files" in data:
                model_cfg["shard_files"] = data["shard_files"]
                model_cfg["shard_count"] = data.get("shard_count", len(data["shard_files"]))
                if "shard_download_urls" in data:
                    model_cfg["shard_download_urls"] = data.get("shard_download_urls", {})
            elif "model_file" in data and "shard_files" not in data:
                model_cfg.pop("shard_files", None)
                model_cfg.pop("shard_count", None)
                model_cfg.pop("shard_download_urls", None)

            save_config(cfg)
            resp = {"status": "ok", "message": "模型配置已更新"}
            resp.update(resp_extra)
            self._json_resp(resp)
        elif path == "/api/deploy":
            self._json_resp(deploy_mgr.start_deploy())
        elif path == "/api/llama/update":
            self._json_resp(deploy_mgr.upgrade_llama())
        elif path == "/api/server/start":
            vision = data.get("vision", False)
            self._json_resp(deploy_mgr.start_server(vision))
        elif path == "/api/server/stop":
            self._json_resp(deploy_mgr.stop_server())
        elif path == "/api/gateway/start":
            self._json_resp(deploy_mgr.start_gateway())
        elif path == "/api/gateway/stop":
            self._json_resp(deploy_mgr.stop_gateway())
        elif path == "/api/lan/publish":
            self._json_resp(deploy_mgr.publish_lan())
        elif path == "/api/lan/firewall":
            self._json_resp(deploy_mgr.open_firewall())

        elif path == "/api/models/activate":
            filename = data.get("filename", "")
            is_mmproj = data.get("is_mmproj", False)
            self._json_resp(model_lib.activate(filename, is_mmproj))
        elif path == "/api/models/delete":
            filename = data.get("filename", "")
            self._json_resp(model_lib.delete(filename))

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_resp(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)


# ============================================================
#  完整 HTML 页面
# ============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llama-deploy 管理器</title>
<style>
:root {
  --bg:#0f1117;--bg2:#1a1d27;--bg3:#252832;--text:#e4e4e7;--text2:#a1a1aa;--text3:#71717a;
  --primary:#3b82f6;--primary-hover:#2563eb;--primary-bg:#1e3a5f;
  --green:#22c55e;--green-bg:#14532d;--red:#ef4444;--red-bg:#450a0a;
  --yellow:#eab308;--yellow-bg:#422006;--border:#2e3140;--radius:10px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Noto Sans SC","Microsoft YaHei",sans-serif;
     background:var(--bg);color:var(--text);line-height:1.6}
a{color:var(--primary);text-decoration:none}
.app{display:flex;min-height:100vh}
.sidebar{width:220px;background:var(--bg2);border-right:1px solid var(--border);
         padding:20px 0;flex-shrink:0;position:fixed;height:100vh;overflow-y:auto}
.main{margin-left:220px;flex:1;padding:24px 32px;max-width:1200px}
.logo{padding:0 20px 20px;border-bottom:1px solid var(--border);margin-bottom:12px}
.logo h1{font-size:20px;display:flex;align-items:center;gap:8px}
.logo span{font-size:12px;color:var(--text3)}
.nav-item{display:flex;align-items:center;gap:10px;padding:10px 20px;cursor:pointer;
          color:var(--text2);transition:all 0.2s;font-size:14px}
.nav-item:hover{background:var(--bg3);color:var(--text)}
.nav-item.active{background:var(--primary-bg);color:var(--primary);border-right:3px solid var(--primary)}
.nav-item .icon{font-size:18px;width:24px;text-align:center}
.sys-info{padding:16px 20px;border-top:1px solid var(--border);font-size:12px;
          color:var(--text3);position:absolute;bottom:0;width:100%}
.sys-info div{margin:3px 0}
.page{display:none}.page.active{display:block}
.page-title{font-size:22px;font-weight:600;margin-bottom:20px;display:flex;align-items:center;gap:10px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
      padding:20px;margin-bottom:16px}
.card-title{font-size:16px;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.search-bar{display:flex;gap:10px;margin-bottom:20px}
.search-bar input{flex:1;padding:10px 16px;background:var(--bg3);border:1px solid var(--border);
                  border-radius:var(--radius);color:var(--text);font-size:14px;outline:none}
.search-bar input:focus{border-color:var(--primary)}
.search-bar select{padding:10px 12px;background:var(--bg3);border:1px solid var(--border);
                   border-radius:var(--radius);color:var(--text);font-size:14px}
.btn{padding:8px 18px;border-radius:var(--radius);border:none;cursor:pointer;font-size:14px;
     font-weight:500;transition:all 0.2s;display:inline-flex;align-items:center;gap:6px}
.btn-primary{background:var(--primary);color:white}
.btn-primary:hover{background:var(--primary-hover)}
.btn-success{background:var(--green);color:white}
.btn-danger{background:var(--red);color:white}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text2)}
.btn-ghost:hover{background:var(--bg3);color:var(--text)}
.btn-sm{padding:5px 12px;font-size:12px}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500}
.tag-blue{background:var(--primary-bg);color:var(--primary)}
.tag-green{background:var(--green-bg);color:var(--green)}
.tag-red{background:var(--red-bg);color:var(--red)}
.tag-yellow{background:var(--yellow-bg);color:var(--yellow)}
.model-list{display:grid;gap:12px}
.model-card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);
            padding:16px;cursor:pointer;transition:all 0.2s}
.model-card:hover{border-color:var(--primary);transform:translateY(-1px)}
.model-card .name{font-weight:600;font-size:15px;margin-bottom:6px}
.model-card .meta{display:flex;gap:12px;color:var(--text3);font-size:12px;margin-bottom:8px}
.model-card .tags{display:flex;gap:4px;flex-wrap:wrap}
.file-table{width:100%;border-collapse:collapse}
.file-table th{text-align:left;padding:10px 12px;color:var(--text3);font-size:12px;
               border-bottom:1px solid var(--border);font-weight:500}
.file-table td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:13px}
.file-table tr:hover{background:var(--bg3)}
.ram-bar{height:6px;border-radius:3px;background:var(--bg);overflow:hidden;width:100px}
.ram-bar-fill{height:100%;border-radius:3px;transition:width 0.3s}
.form-group{margin-bottom:16px}
.form-label{display:block;font-size:13px;color:var(--text2);margin-bottom:4px}
.form-hint{font-size:11px;color:var(--text3);margin-top:2px}
.form-input{width:100%;padding:8px 12px;background:var(--bg3);border:1px solid var(--border);
            border-radius:6px;color:var(--text);font-size:14px;outline:none}
.form-input:focus{border-color:var(--primary)}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.form-toggle{display:flex;align-items:center;gap:8px;cursor:pointer}
.toggle-switch{width:40px;height:22px;background:var(--bg);border-radius:11px;
               position:relative;transition:0.3s;border:1px solid var(--border)}
.toggle-switch.on{background:var(--primary);border-color:var(--primary)}
.toggle-switch::after{content:'';position:absolute;width:16px;height:16px;border-radius:50%;
                      background:white;top:2px;left:2px;transition:0.3s}
.toggle-switch.on::after{left:20px}
.log-box{background:#0a0c10;border:1px solid var(--border);border-radius:var(--radius);
         padding:16px;font-family:"JetBrains Mono","Fira Code",monospace;font-size:13px;
         max-height:400px;overflow-y:auto;white-space:pre-wrap;color:#8b949e;line-height:1.8}
.log-box .log-ok{color:var(--green)}.log-box .log-err{color:var(--red)}
.log-box .log-warn{color:var(--yellow)}.log-box .log-info{color:var(--primary)}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.status-dot.green{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.red{background:var(--red)}
.status-dot.yellow{background:var(--yellow);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.loading{display:inline-block;width:16px;height:16px;border:2px solid var(--border);
         border-top-color:var(--primary);border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:40px;color:var(--text3)}
.empty .icon{font-size:48px;margin-bottom:12px}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:100;
               align-items:center;justify-content:center;padding:20px}
.modal-overlay.show{display:flex}
.modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
           max-width:900px;width:100%;max-height:80vh;overflow:auto;padding:24px}
@media(max-width:768px){
  .sidebar{width:60px}.sidebar .nav-text,.sidebar .logo span{display:none}
  .main{margin-left:60px;padding:16px}.form-row{grid-template-columns:1fr}.sys-info{display:none}
}
</style>
</head>
<body>
<div class="app">
  <nav class="sidebar">
    <div class="logo"><h1>🦙 <span>llama-deploy</span></h1><span>智能管理器</span></div>
    <div class="nav-item active" onclick="switchPage('market',this)">
      <span class="icon">🔍</span><span class="nav-text">模型市场</span></div>
    <div class="nav-item" onclick="switchPage('library',this)">
      <span class="icon">📚</span><span class="nav-text">模型库</span></div>
    <div class="nav-item" onclick="switchPage('config',this)">
      <span class="icon">⚙️</span><span class="nav-text">配置编辑</span></div>
    <div class="nav-item" onclick="switchPage('deploy',this)">
      <span class="icon">🚀</span><span class="nav-text">部署管理</span></div>
    <div class="nav-item" onclick="switchPage('system',this)">
      <span class="icon">📊</span><span class="nav-text">系统信息</span></div>
    <div class="sys-info" id="sidebarInfo"></div>
  </nav>

  <div class="main">
    <!-- 模型市场 -->
    <div id="page-market" class="page active">
      <div class="page-title">🔍 模型市场</div>
      <div class="card">
        <div class="search-bar">
          <select id="searchSource" onchange="onSourceChange()">
            <optgroup label="🌍 国际">
              <option value="huggingface">🤗 HuggingFace</option>
              <option value="gguf">🔥 GGUF精选发布者</option>
              <option value="unsloth">⚡ Unsloth精选</option>
            </optgroup>
            <optgroup label="🇨🇳 国内加速">
              <option value="hf_mirror" selected>🪞 HF镜像(推荐·国内)</option>
              <option value="modelscope">🏠 魔搭社区(ModelScope)</option>
              <option value="giteeai">🔴 Gitee AI</option>
            </optgroup>
          </select>
          <input type="text" id="searchInput" placeholder="搜索模型，例如：qwen3、llama3、gemma、deepseek..."
                 onkeydown="if(event.key==='Enter')doSearch()">
          <button class="btn btn-primary" onclick="doSearch()">🔍 搜索</button>
        </div>
        <div id="sourceHint" style="font-size:12px;color:var(--text3);padding:4px 0 8px">
          🪞 HF镜像：HuggingFace 国内加速，速度快，资源与官方同步
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
          <span style="font-size:12px;color:var(--text3);align-self:center">快捷搜索：</span>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('qwen3')">Qwen3</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('deepseek')">DeepSeek</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('llama3')">LLaMA3</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('gemma3')">Gemma3</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('glm4')">GLM4</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('mistral')">Mistral</button>
          <button class="btn btn-ghost btn-sm" onclick="quickSearch('phi4')">Phi-4</button>
        </div>
        <div id="ramHint" class="card" style="padding:12px;margin-bottom:16px;display:none"></div>
        <div id="searchResults" class="model-list">
          <div class="empty"><div class="icon">🔍</div>输入关键词搜索 GGUF 模型<br>
          <span style="font-size:13px;color:var(--text3)">推荐国内用户选「🪞 HF镜像」，速度更快</span></div>
        </div>
      </div>
      <div id="fileModal" class="modal-overlay" onclick="if(event.target===this)closeFileModal()">
        <div class="modal-box">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <div>
              <h3 id="fileModalTitle">选择模型文件</h3>
              <div id="fileModalSubtitle" style="font-size:12px;color:var(--text3);margin-top:2px"></div>
            </div>
            <button class="btn btn-ghost btn-sm" onclick="closeFileModal()">✕ 关闭</button>
          </div>
          <div id="fileModalTabs" style="display:flex;gap:6px;margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:10px">
            <button class="btn btn-sm" id="tabAll" onclick="filterFiles('all')" style="background:var(--primary);color:#fff">全部</button>
            <button class="btn btn-ghost btn-sm" id="tabMain" onclick="filterFiles('main')">主模型</button>
            <button class="btn btn-ghost btn-sm" id="tabMmproj" onclick="filterFiles('mmproj')">视觉模块</button>
            <span style="margin-left:auto;font-size:12px;color:var(--text3);align-self:center">
              ✅ 可同时勾选主模型+视觉模块一键下载
            </span>
          </div>
          <div id="fileModalContent"></div>
          <div id="fileModalFooter" style="display:none;margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">
            <div id="selectedSummary" style="font-size:13px;color:var(--text2);margin-bottom:10px"></div>
            <div style="display:flex;gap:8px">
              <button class="btn btn-primary" onclick="confirmMultiSelect()">⬇️ 设为待部署并下载</button>
              <button class="btn btn-ghost btn-sm" onclick="clearSelection()">✕ 取消选择</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 模型库 -->
    <div id="page-library" class="page">
      <div class="page-title">📚 模型库</div>
      <div class="card">
        <div class="card-title" style="display:flex;align-items:center">
          <span>📦 已下载的模型</span>
          <button class="btn btn-ghost btn-sm" onclick="refreshLibrary()" style="margin-left:auto">🔄 刷新</button>
        </div>
        <div id="libraryContent">加载中...</div>
      </div>
      <div class="card" style="padding:12px;font-size:13px;color:var(--text3)">
        💡 在「模型市场」下载新模型后，自动出现在这里。点「激活」切换当前使用的模型，需重启服务器生效。
      </div>
    </div>

    <!-- 配置编辑 -->
    <div id="page-config" class="page">
      <div class="page-title">⚙️ 配置编辑</div>
      <div class="card">
        <div class="card-title">📦 模型配置</div>
        <div class="form-group">
          <label class="form-label">仓库 ID (repo_id)</label>
          <input class="form-input" id="cfg-repo_id" placeholder="例如：unsloth/Qwen3.5-0.8B-GGUF">
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">模型文件 (model_file)</label>
            <input class="form-input" id="cfg-model_file" placeholder="例如：Qwen3.5-0.8B-Q4_K_M.gguf">
          </div>
          <div class="form-group">
            <label class="form-label">视觉模块 (mmproj_file)</label>
            <input class="form-input" id="cfg-mmproj_file" placeholder="留空则不启用视觉功能">
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">🌐 下载配置</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">HuggingFace 镜像</label>
            <input class="form-input" id="cfg-hf_mirror" placeholder="https://hf-mirror.com">
          </div>
          <div class="form-group">
            <label class="form-label">GitHub 镜像</label>
            <input class="form-input" id="cfg-github_mirror" placeholder="可选；留空使用 GitHub 官方直链">
          </div>
        </div>
      </div>
       <div class="card">
        <div class="card-title">🎮 GPU 配置</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">计算后端</label>
            <select class="form-input" id="cfg-gpu_backend">
              <option value="auto">auto — 自动检测（推荐）</option>
              <option value="cuda">cuda — NVIDIA CUDA</option>
              <option value="vulkan">vulkan — 通用 GPU</option>
              <option value="cpu">cpu — 仅 CPU</option>
            </select>
            <div class="form-hint">auto 自动检测：CUDA > Vulkan > CPU</div>
          </div>
          <div class="form-group">
            <label class="form-label">GPU 卸载层数</label>
            <select class="form-input" id="cfg-gpu_layers_preset" onchange="onGpuLayerPreset()">
              <option value="-1">🤖 自动（推荐）— 根据模型大小和显存自动计算</option>
              <option value="99">💪 全部卸载 — 强制全部放入 GPU（小模型用）</option>
              <option value="0">🚫 不使用 GPU — 纯 CPU 运行</option>
              <option value="custom">🔧 手动指定层数</option>
            </select>
            <input class="form-input" id="cfg-gpu_layers" type="number" value="-1"
                   style="display:none;margin-top:4px" placeholder="输入层数">
            <div class="form-hint" id="gpuLayersHint">自动模式：系统会根据模型大小和可用显存计算最佳层数，大模型自动部分卸载</div>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Flash Attention</label>
          <div class="form-toggle" onclick="toggleSwitch('cfg-flash_attn')">
            <div class="toggle-switch on" id="cfg-flash_attn"></div>
            <span>启用（减少显存占用，推荐开启）</span>
          </div>
        </div>
        <div id="gpuDetectInfo" style="margin-top:8px;padding:10px;background:var(--bg3);border-radius:6px;font-size:13px"></div>
        <div id="gpuGuide" style="margin-top:8px;padding:10px;background:var(--bg3);border-radius:6px;font-size:12px;color:var(--text3);display:none"></div>
      </div>
      <div class="card">
        <div class="card-title">🖥️ 服务器配置</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">监听端口</label>
            <input class="form-input" id="cfg-port" type="number" value="8080">
          </div>
          <div class="form-group">
            <label class="form-label">CPU 线程数 (0=自动)</label>
            <input class="form-input" id="cfg-threads" type="number" value="0">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">上下文长度 (ctx_size)</label>
            <input class="form-input" id="cfg-ctx_size" type="number" value="8192">
          </div>
          <div class="form-group">
            <label class="form-label">思考预算 (tokens)</label>
            <input class="form-input" id="cfg-reasoning_budget" type="number" value="512">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">思考模式</label>
            <div class="form-toggle" onclick="toggleSwitch('cfg-thinking')">
              <div class="toggle-switch" id="cfg-thinking"></div>
              <span>enable_thinking</span>
            </div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">🎛️ 采样参数</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Temperature</label>
            <input class="form-input" id="cfg-temperature" type="number" step="0.1" value="0.7">
          </div>
          <div class="form-group">
            <label class="form-label">Top-K</label>
            <input class="form-input" id="cfg-top_k" type="number" value="20">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Top-P</label>
            <input class="form-input" id="cfg-top_p" type="number" step="0.05" value="0.8">
          </div>
          <div class="form-group">
            <label class="form-label">Presence Penalty</label>
            <input class="form-input" id="cfg-presence_penalty" type="number" step="0.1" value="1.5">
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">⚡ 高级性能</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">性能策略</label>
            <select class="form-input" id="cfg-performance_profile">
              <option value="auto">自动适配</option>
              <option value="maximum">极限性能</option>
              <option value="compatible">兼容优先</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">GPU 预留显存 MB（0=自动）</label>
            <input class="form-input" id="cfg-fit_target_mb" type="number" min="0" value="0">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Speculative / MTP</label>
            <select class="form-input" id="cfg-spec_type">
              <option value="off">关闭</option>
              <option value="draft-mtp">draft-mtp</option>
              <option value="auto">自动检测</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">MTP 草稿长度</label>
            <input class="form-input" id="cfg-spec_draft_n_max" type="number" value="3">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">KV Cache K</label>
            <select class="form-input" id="cfg-cache_type_k">
              <option value="auto">自动</option><option value="f16">f16</option><option value="q8_0">q8_0</option><option value="q4_0">q4_0</option><option value="q4_1">q4_1</option><option value="iq4_nl">iq4_nl</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">KV Cache V</label>
            <select class="form-input" id="cfg-cache_type_v">
              <option value="auto">自动</option><option value="f16">f16</option><option value="q8_0">q8_0</option><option value="q4_0">q4_0</option><option value="q4_1">q4_1</option><option value="iq4_nl">iq4_nl</option>
            </select>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">前 N 层 MoE 放 CPU</label>
            <input class="form-input" id="cfg-n_cpu_moe" type="number" value="0">
          </div>
          <div class="form-group">
            <label class="form-label">统一 KV</label>
            <div class="form-toggle" onclick="toggleSwitch('cfg-kv_unified')">
              <div class="toggle-switch on" id="cfg-kv_unified"></div>
              <span>kv_unified</span>
            </div>
          </div>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:10px">
        <button class="btn btn-primary" onclick="saveConfig()">💾 保存配置</button>
        <button class="btn btn-ghost" onclick="loadConfig()">🔄 重新加载</button>
        <button class="btn btn-ghost" onclick="resetConfig()">↩️ 恢复默认</button>
      </div>
    </div>

    <!-- 部署管理 -->
    <div id="page-deploy" class="page">
      <div class="page-title">🚀 部署管理</div>
      <div class="card">
        <div class="card-title">📡 当前状态</div>
        <div id="statusContent">加载中...</div>
      </div>
      <div class="card">
        <div class="card-title">🔧 操作</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-primary" onclick="startDeploy()" id="btnDeploy">📦 一键部署</button>
          <button class="btn btn-ghost" onclick="upgradeLlama()" id="btnUpgradeLlama">⬆️ 升级 llama.cpp</button>
          <button class="btn btn-success" onclick="startServer(false)">▶️ 启动服务器</button>
          <button class="btn btn-success" onclick="startServer(true)">👁️ 启动视觉服务</button>
          <button class="btn btn-danger" onclick="stopServer()">⏹️ 停止服务器</button>
        </div>
      </div>
      <div class="card">
        <div class="card-title">🌐 局域网与兼容网关</div>
        <div id="lanContent">加载中...</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px">
          <button class="btn btn-primary" onclick="publishLan()">📡 发布到局域网</button>
          <button class="btn btn-success" onclick="startGateway()">🔌 启动兼容网关</button>
          <button class="btn btn-danger" onclick="stopGateway()">⏹️ 停止网关</button>
          <button class="btn btn-ghost" onclick="openFirewall()">🛡️ 放行防火墙</button>
        </div>
      </div>
      <div class="card">
        <div class="card-title">📋 部署日志</div>
        <div class="log-box" id="deployLog">等待操作...</div>
      </div>
    </div>

    <!-- 系统信息 -->
    <div id="page-system" class="page">
      <div class="page-title">📊 系统信息</div>
      <div id="systemInfo">加载中...</div>
    </div>
  </div>
</div>

<script>
var sysInfo={},currentConfig={},logPollTimer=null,logLineCount=0;

async function api(url,method,body){
  try{
    var opts={method:method||'GET',headers:{'Content-Type':'application/json'}};
    if(body)opts.body=JSON.stringify(body);
    var r=await fetch(url,opts);
    return await r.json();
  }catch(e){return{error:e.message}}
}

function switchPage(name,el){
  document.querySelectorAll('.page').forEach(function(p){p.classList.remove('active')});
  document.querySelectorAll('.nav-item').forEach(function(n){n.classList.remove('active')});
  document.getElementById('page-'+name).classList.add('active');
  if(el)el.classList.add('active');
  if(name==='library')refreshLibrary();
}

function formatNum(n){if(!n)return'0';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)}

function onGpuLayerPreset(){
  var sel=document.getElementById('cfg-gpu_layers_preset');
  var inp=document.getElementById('cfg-gpu_layers');
  var hint=document.getElementById('gpuLayersHint');
  var v=sel.value;
  if(v==='custom'){
    inp.style.display='block';
    inp.value=inp.value==='99'||inp.value==='-1'?'20':inp.value;
    hint.textContent='手动指定要卸载到 GPU 的层数，数字越大 GPU 占用越多';
  }else{
    inp.style.display='none';
    inp.value=v;
    if(v==='-1')hint.textContent='自动模式：系统会根据模型大小和可用显存计算最佳层数，大模型自动部分卸载';
    else if(v==='99')hint.textContent='全部层放入 GPU，如果显存不足会启动失败（适合小模型）';
    else if(v==='0')hint.textContent='完全不使用 GPU，仅用 CPU 运算（最慢但最稳定）';
  }
}

function toggleSwitch(id){document.getElementById(id).classList.toggle('on')}
function setVal(id,v){var e=document.getElementById(id);if(e)e.value=v!=null?v:''}
function getVal(id){var e=document.getElementById(id);return e?e.value.trim():''}
function getNum(id,d){var v=parseInt(getVal(id));return isNaN(v)?d:v}
function getFloat(id,d){var v=parseFloat(getVal(id));return isNaN(v)?d:v}
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function firstLanIp(st){return ((st.lan_ips&&st.lan_ips[0])||location.hostname||'127.0.0.1')}
function copyText(txt){navigator.clipboard.writeText(txt).then(function(){showToast('已复制')}).catch(function(){prompt('复制以下内容',txt)})}

async function init(){
  sysInfo=await api('/api/system');
  document.getElementById('sidebarInfo').innerHTML=
    '<div>💻 '+sysInfo.os+' '+sysInfo.arch+'</div>'+
    '<div>🧠 内存: '+sysInfo.ram_gb+'GB</div>'+
    '<div>💾 磁盘: '+sysInfo.disk_free_gb+'GB 可用</div>'+
    '<div>⚡ CPU: '+sysInfo.cpu_count+' 核</div>'+
    (sysInfo.gpu_name?'<div>🎮 '+sysInfo.gpu_name+'</div><div>   显存: '+(sysInfo.gpu_vram_mb/1024).toFixed(1)+'GB</div>':'<div>🎮 无独显</div>');
  loadConfig();pollStatus();updateSystemPage();
  var vram=sysInfo.gpu_vram_mb||0;
  var rec=await api('/api/recommend?ram='+sysInfo.ram_gb+'&vram='+vram);
  var hint=document.getElementById('ramHint');
  if(rec&&rec.tips){
    hint.style.display='block';
    hint.innerHTML='<div style="font-size:13px"><strong>💡 设备: '+sysInfo.ram_gb+'GB 内存'+
      (sysInfo.gpu_name?' + '+sysInfo.gpu_name+' '+(sysInfo.gpu_vram_mb/1024).toFixed(1)+'GB':'')+
      '</strong> · 推荐量化: <span class="tag tag-green">'+rec.recommended_quant+'</span>'+
      ' · 推荐 ctx: <span class="tag tag-yellow">'+rec.recommended_ctx+'</span>'+
      (rec.gpu_max_model_gb?' · GPU可载: <span class="tag tag-blue">≤'+rec.gpu_max_model_gb+'GB</span>':'')+
      '<br><span style="color:var(--text3)">'+(rec.tips||[]).join(' · ')+'</span></div>';
  }
  var gpuEl=document.getElementById('gpuDetectInfo');
  if(gpuEl){
    gpuEl.innerHTML=sysInfo.gpu_name
      ?'<span class="tag tag-green">✅ 检测到</span> <strong>'+sysInfo.gpu_name+'</strong> · 显存: '+(sysInfo.gpu_vram_mb/1024).toFixed(1)+'GB · 可用: '+(sysInfo.gpu_vram_free_mb/1024).toFixed(1)+'GB'
      :'<span class="tag tag-yellow">⚠️ 未检测到独立显卡</span> 将使用 CPU 模式';
  }
  var guideEl=document.getElementById('gpuGuide');
  if(guideEl&&sysInfo.gpu_name){
    var vg=(sysInfo.gpu_vram_mb/1024).toFixed(1);
    guideEl.style.display='block';
    guideEl.innerHTML='<strong>💡 显存 '+vg+'GB 参考：</strong><br>'+
      (vg>=16?'• 可流畅运行 14B 及以下模型（全部卸载到GPU）<br>• 32B 模型可部分卸载':
       vg>=8?'• 可流畅运行 7-8B 模型（全部卸载到GPU）<br>• 14B 模型可部分卸载':
       vg>=6?'• 可流畅运行 3-4B 模型（全部卸载到GPU）<br>• 7-8B 模型可部分卸载（约60%层）<br>• 更大模型建议自动模式':
       vg>=4?'• 可流畅运行 1-2B 模型（全部卸载到GPU）<br>• 3B+ 模型建议自动模式':
       '• 显存较小，建议使用自动模式')+
      '<br>• <strong>推荐使用「🤖 自动」模式</strong>，系统会智能计算最佳配置';
  }
}

// 各源的提示文字
var SOURCE_HINTS = {
  'huggingface': '🌍 HuggingFace 官方：资源最全，国内可能需要代理',
  'hf_mirror':   '🪞 HF镜像：HuggingFace 国内加速，速度快，与官方同步',
  'huggingface_mirror': '🪞 HF镜像：HuggingFace 国内加速，速度快，与官方同步',
  'modelscope':  '🏠 魔搭社区：阿里巴巴运营，中文模型丰富，国内速度极快',
  'gguf':        '🔥 GGUF精选：bartowski / unsloth / mradermacher 等知名量化发布者',
  'unsloth':     '⚡ Unsloth精选：高质量动态量化(Dynamic GGUF)，推理速度更快',
  'giteeai':     '🔴 Gitee AI：国内自研平台，中文模型收录完整，访问无障碍',
};

function onSourceChange(){
  var src = document.getElementById('searchSource').value;
  var hint = document.getElementById('sourceHint');
  if(hint) hint.textContent = SOURCE_HINTS[src] || '';
}

function quickSearch(kw){
  document.getElementById('searchInput').value = kw;
  doSearch();
}

// ===== 搜索 =====
async function doSearch(){
  var q = document.getElementById('searchInput').value.trim();
  var source = document.getElementById('searchSource').value;
  if(!q) return;
  var c = document.getElementById('searchResults');
  c.innerHTML = '<div class="empty"><div class="loading"></div><br>搜索中...</div>';
  var data = await api('/api/search?q='+encodeURIComponent(q)+'&source='+encodeURIComponent(source));
  if(data.error){c.innerHTML='<div class="empty"><div class="icon">❌</div>'+data.error+'</div>';return}
  var results = (data.results||[]).filter(function(m){
    return m&&m.id&&m.id!=='undefined'&&String(m.id).length>1;
  });
  if(!results.length){
    c.innerHTML='<div class="empty"><div class="icon">🔍</div>未找到相关 GGUF 模型<br>'+
      '<span style="color:var(--text3)">建议换一个关键词，或切换到其他源试试</span></div>';
    return;
  }
  var srcForFiles = data.source || source;
  c.innerHTML = results.map(function(m){
    var id = m.id || '';
    var escapedId = id.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    var escapedSrc = (srcForFiles||'').replace(/'/g,"\\'");
    // 提取关键标签：架构/大小/格式
    var keyTags = (m.tags||[]).filter(function(t){
      if(typeof t!=='string') return false;
      t = t.toLowerCase();
      return t==='gguf'||t.match(/^\d+b$/)||t.match(/^(text-generation|conversational|chat|instruct|vision)$/);
    }).slice(0,5);
    var tagHtml = keyTags.map(function(t){
      var cls = t==='gguf'?'tag-green':t.match(/^\d+b$/)?'tag-yellow':'tag-blue';
      return '<span class="tag '+cls+'">'+t+'</span>';
    }).join('');
    // 提取模型大小（从名字中匹配 7B/13B/70B 等）
    var sizeMatch = id.match(/[_\-](\d+\.?\d*[bBmMkK])[_\-\s.]/);
    var sizeTag = sizeMatch ? '<span class="tag tag-yellow">'+sizeMatch[1].toUpperCase()+'</span>' : '';
    // 发布者
    var author = m.author||id.split('/')[0]||'';
    return '<div class="model-card" onclick="showFiles(\''+escapedId+'\',\''+escapedSrc+'\')">'+
      '<div class="name" style="display:flex;align-items:center;gap:8px">'+
        '<span>'+id+'</span>'+sizeTag+
      '</div>'+
      '<div class="meta">'+
        (author?'<span>👤 '+author+'</span>':'')+
        '<span>⬇️ '+formatNum(m.downloads||0)+'</span>'+
        '<span>❤️ '+formatNum(m.likes||0)+'</span>'+
        (m.updated?'<span>📅 '+(m.updated||'').slice(0,10)+'</span>':'')+
        '<span style="color:var(--text3)">📦 '+(m.source||srcForFiles)+'</span>'+
      '</div>'+
      '<div class="tags">'+tagHtml+'</div>'+
    '</div>';
  }).join('');
}

// ===== 文件选择（支持多选 + 标签过滤）=====
var _allFiles = [], _visibleFiles = [], _curRepo = '', _curSrc = '', _selectedMain = null, _selectedMmproj = null, _selectedMmprojAuto = false;

async function showFiles(repoId, source){
  _curRepo = repoId; _curSrc = source;
  _selectedMain = null; _selectedMmproj = null; _selectedMmprojAuto = false;
  var modal = document.getElementById('fileModal');
  modal.classList.add('show');
  document.getElementById('fileModalTitle').textContent = repoId;
  var sub = document.getElementById('fileModalSubtitle');
  if(sub) sub.textContent = '来源：' + source;
  document.getElementById('fileModalContent').innerHTML =
    '<div class="empty"><div class="loading"></div><br>加载文件列表...</div>';
  var footer = document.getElementById('fileModalFooter');
  if(footer) footer.style.display = 'none';

  var data = await api('/api/files/'+encodeURIComponent(repoId)+'?source='+encodeURIComponent(source));
  if(!data || data.error){
    document.getElementById('fileModalContent').innerHTML =
      '<div class="empty"><div class="icon">❌</div>'+(data?data.error:'请求失败')+
      '<br><br><a href="https://huggingface.co/'+repoId+'" target="_blank" class="btn btn-ghost btn-sm">🔗 HuggingFace</a> '+
      '<a href="https://www.modelscope.cn/models/'+repoId+'" target="_blank" class="btn btn-ghost btn-sm">🔗 ModelScope</a>'+
      '<a href="https://ai.gitee.com/'+repoId+'" target="_blank" class="btn btn-ghost btn-sm">🔗 Gitee AI</a></div>';
    return;
  }
  _allFiles = (data.files||[]).filter(function(f){return f&&f.filename&&!f.error});
  if(!_allFiles.length){
    document.getElementById('fileModalContent').innerHTML =
      '<div class="empty"><div class="icon">📭</div>未找到 GGUF 文件<br>'+
      '<a href="https://huggingface.co/'+repoId+'" target="_blank" class="btn btn-ghost btn-sm" style="margin-top:8px">🔗 在 HuggingFace 查看</a></div>';
    return;
  }
  // 显示 tab 统计
  var mainCount = _allFiles.filter(function(f){return !f.is_mmproj}).length;
  var mmprojCount = _allFiles.filter(function(f){return f.is_mmproj}).length;
  document.getElementById('tabMain').textContent = '主模型(' + mainCount + ')';
  document.getElementById('tabMmproj').textContent = '视觉模块(' + mmprojCount + ')';
  if(mmprojCount===0) document.getElementById('tabMmproj').style.opacity='0.4';
  filterFiles('all');
}

function filterFiles(tab){
  // 更新 tab 按钮样式
  ['all','main','mmproj'].forEach(function(t){
    var btn = document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1));
    if(btn){ btn.style.background = t===tab?'var(--primary)':''; btn.style.color = t===tab?'#fff':''; btn.className = t===tab?'btn btn-sm':'btn btn-ghost btn-sm'; }
  });
  var files = _allFiles;
  if(tab==='main') files = _allFiles.filter(function(f){return !f.is_mmproj});
  if(tab==='mmproj') files = _allFiles.filter(function(f){return f.is_mmproj});
  renderFiles(files);
}

function renderFiles(files){
  _visibleFiles = files || [];
  var QUANT_COLOR = {
    'Q8_0':'tag-green','Q6_K':'tag-green','Q5_K_M':'tag-green','Q5_K_S':'tag-green',
    'Q4_K_M':'tag-yellow','Q4_K_S':'tag-yellow','Q4_0':'tag-yellow',
    'Q3_K_M':'tag-red','Q3_K_L':'tag-red','Q2_K':'tag-red',
    'F16':'tag-blue','BF16':'tag-blue','IQ4_XS':'tag-yellow','IQ4_NL':'tag-yellow',
  };
  var QUANT_TIP = {
    'Q8_0':'质量最高，文件较大','Q6_K':'质量极佳，推荐高显存用户',
    'Q5_K_M':'质量佳，均衡选择','Q4_K_M':'质量/大小均衡，最推荐',
    'Q4_K_S':'稍小一点，略有质量损失','Q4_0':'老格式，不如 Q4_K_M',
    'Q3_K_M':'质量损失明显，内存受限时用','Q2_K':'质量较差，仅限内存极小设备',
    'F16':'全精度，质量最佳，文件最大','BF16':'Brain Float16，适合训练',
    'IQ4_XS':'iQuant 系列，比 Q4_K_S 小10%','IQ4_NL':'iQuant NL，质量接近 Q4_K_M',
  };
  var vram = (sysInfo&&sysInfo.gpu_vram_mb||0)/1024;
  var c = document.getElementById('fileModalContent');
  if(!_visibleFiles.length){ c.innerHTML='<div class="empty">该分类下没有文件</div>'; return; }
  c.innerHTML = _visibleFiles.map(function(f, idx){
    var isMain = !f.is_mmproj;
    var checked = isMain ? (_selectedMain&&_selectedMain.filename===f.filename) :
                           (_selectedMmproj&&_selectedMmproj.filename===f.filename);
    var qColor = QUANT_COLOR[f.quant]||'tag-blue';
    var qTip = QUANT_TIP[f.quant]||f.quant;
    var sizeGb = f.size_gb||0;
    var ramOk = !f.ram_estimate_gb || sysInfo.ram_gb >= f.ram_estimate_gb;
    var vramOk = !sizeGb || vram<=0 || vram >= sizeGb * 0.9;
    var compat = ramOk&&vramOk ? '<span class="tag tag-green">✅ 兼容</span>' :
                 !ramOk ? '<span class="tag tag-red">⚠️ 内存不足</span>' :
                 '<span class="tag tag-yellow">⚠️ 显存可能不足</span>';
    var checkboxId = 'chk-' + f.filename.replace(/[^a-zA-Z0-9]/g,'_');
    return '<div class="model-card" style="cursor:default;'+(checked?'border-color:var(--primary);background:var(--primary-bg)':'')+'" id="card-'+checkboxId+'">'+
      '<div style="display:flex;align-items:flex-start;gap:10px">'+
        '<input type="checkbox" id="'+checkboxId+'" '+(checked?'checked':'')+
          ' onchange="toggleSelectByIndex(this,'+idx+')"'+
          ' style="margin-top:4px;width:16px;height:16px;cursor:pointer;flex-shrink:0">'+
        '<div style="flex:1;min-width:0">'+
          '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'+
            '<span style="font-weight:600;word-break:break-all">'+(f.is_shard?'📦 ':'📄 ')+f.filename+'</span>'+
            (f.is_mmproj?'<span class="tag tag-blue">视觉模块</span>':'')+
          '</div>'+
          '<div class="meta" style="margin-top:4px">'+
            (sizeGb?'<span>📦 '+sizeGb.toFixed(2)+'GB</span>':'')+
            (f.ram_estimate_gb?'<span>🧠 需约 '+f.ram_estimate_gb+'GB 内存</span>':'')+
            (f.shard_count&&f.shard_count>1?'<span>🔀 '+f.shard_count+' 分片</span>':'')+
          '</div>'+
          '<div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">'+
            '<span class="tag '+qColor+'" title="'+qTip+'">'+f.quant+'</span>'+
            compat+
            (f.quant&&QUANT_TIP[f.quant]?'<span style="font-size:11px;color:var(--text3)">'+qTip+'</span>':'')+
          '</div>'+
        '</div>'+
      '</div>'+
    '</div>';
  }).join('');
}

function toggleSelectByIndex(el, idx){
  var file = _visibleFiles[idx];
  if(!file){ return; }
  toggleSelect(el, file, !file.is_mmproj);
}

function toggleSelect(el, file, isMain){
  if(isMain){
    if(!el.checked){ _selectedMain=null; }
    else {
      _selectedMain=file;
      if(!_selectedMmproj || _selectedMmprojAuto){
        _selectedMmproj=findPairedMmproj(file);
        _selectedMmprojAuto=!!_selectedMmproj;
      }
    }
  } else {
    if(!el.checked){ _selectedMmproj=null; _selectedMmprojAuto=false; }
    else { _selectedMmproj=file; _selectedMmprojAuto=false; }
  }
  renderFiles(_visibleFiles);
  updateFooter();
}

function updateFooter(){
  var footer = document.getElementById('fileModalFooter');
  var summary = document.getElementById('selectedSummary');
  var hasAny = _selectedMain || _selectedMmproj;
  if(footer) footer.style.display = hasAny ? 'block' : 'none';
  if(summary && hasAny){
    var lines = [];
    if(_selectedMain) lines.push('📄 主模型：'+_selectedMain.filename+(_selectedMain.size_gb?' ('+_selectedMain.size_gb.toFixed(2)+'GB)':''));
    if(_selectedMmproj){
      var pairedName=_selectedMain?pairedMmprojName(_selectedMain.filename,_selectedMmproj.filename):fileBaseName(_selectedMmproj.filename);
      lines.push('👁️ 视觉模块'+(_selectedMmprojAuto?'（自动匹配）':'')+'：'+_selectedMmproj.filename+
        (pairedName!==fileBaseName(_selectedMmproj.filename)?' → '+pairedName:'')+
        (_selectedMmproj.size_gb?' ('+_selectedMmproj.size_gb.toFixed(2)+'GB)':''));
    }
    summary.innerHTML = lines.join('<br>');
  }
}

function clearSelection(){
  _selectedMain=null; _selectedMmproj=null; _selectedMmprojAuto=false;
  document.querySelectorAll('#fileModalContent input[type=checkbox]').forEach(function(cb){cb.checked=false;});
  document.querySelectorAll('#fileModalContent .model-card').forEach(function(card){
    card.style.borderColor=''; card.style.background='';
  });
  updateFooter();
}

async function confirmMultiSelect(){
  if(!_selectedMain && !_selectedMmproj){alert('请先勾选文件');return;}
  var names = [_selectedMain&&_selectedMain.filename, _selectedMmproj&&_selectedMmproj.filename].filter(Boolean);
  await applySelectedFilesForDeploy();
  closeFileModal();
  if(names.length){
    showToast('已选择：'+names.join(' + ')+'，请到「部署管理」下载');
  }
}

async function applySelectedFilesForDeploy(){
  if(!_selectedMain && !_selectedMmproj) return false;
  var main = _selectedMain, mmproj = _selectedMmproj;
  // 主模型
  if(main){
    await selectFile(_curRepo, main, _curSrc, !!mmproj);
  }
  // 视觉模块（bind_mmproj 方式写入）
  if(mmproj){
    var body = {
      repo_id: _curRepo,
      source: _curSrc,
      mmproj_file: main?pairedMmprojName(main.filename,mmproj.filename):fileBaseName(mmproj.filename),
      mmproj_repo_file: mmproj.filename,
      mmproj_download_url: mmproj.download_url||'',
      mmproj_size: mmproj.size||0,
      bind_mmproj: true,
    };
    await api('/api/config/model','POST',body);
  }
  _selectedMain=null; _selectedMmproj=null; _selectedMmprojAuto=false;
  updateFooter();
  await loadConfig();
  return true;
}

function closeFileModal(){document.getElementById('fileModal').classList.remove('show')}

function fileWords(name){
  return String(name||'').toLowerCase().split(/[^a-z0-9]+/).filter(function(w){
    return w.length>1&&!['gguf','mmproj','f16','q4','q5','q6','q8','k','m','it','instruct'].includes(w);
  });
}

function fileBaseName(name){
  return String(name||'').replace(/\\/g,'/').split('/').pop();
}

function pairedMmprojName(mainName, mmName){
  var main=fileBaseName(mainName), mm=fileBaseName(mmName);
  if(!main||!mm)return mm;
  var stem=main.replace(/\.gguf$/i,'').replace(/-\d{5}-of-\d{5}$/i,'');
  if(mm.toLowerCase().indexOf((stem+'.mmproj').toLowerCase())===0)return mm;
  var variants=mm.replace(/\.gguf$/i,'').match(/(?:^|[-_.])(bf16|f16|f32|q[2-8](?:_[0-9a-z]+)?|iq[1-4](?:_[0-9a-z]+)?)(?=$|[-_.])/ig)||[];
  var variant=variants.length?variants[variants.length-1].replace(/^[-_.]/,'').toLowerCase():'';
  return stem+'.mmproj'+(variant?'-'+variant:'')+'.gguf';
}

function modelFamily(name){
  var t=String(name||'').toLowerCase().replace(/_/g,'-');
  var specs=[[/qwen-?(\d+(?:[.-]\d+)?)/,'qwen'],[/gemma-?(\d+(?:[.-]\d+)?)/,'gemma'],
    [/llama-?(\d+(?:[.-]\d+)?)/,'llama'],[/minicpm-?v?-?(\d+(?:[.-]\d+)?)?/,'minicpm'],
    [/internvl-?(\d+(?:[.-]\d+)?)?/,'internvl'],[/glm-?(\d+(?:[.-]\d+)?)?/,'glm'],
    [/pixtral/,'pixtral'],[/mistral/,'mistral'],[/phi-?(\d+(?:[.-]\d+)?)?/,'phi']];
  for(var i=0;i<specs.length;i++){
    var m=t.match(specs[i][0]);
    if(m)return specs[i][1]+String(m[1]||'').replace(/-/g,'.');
  }
  return '';
}

function mmprojMatches(mainName, mmName){
  var mf=modelFamily(mainName),vf=modelFamily(mmName);
  if(mf&&vf)return mf===vf;
  var mw=fileWords(mainName),vw=fileWords(mmName);
  return mw.some(function(w){return vw.indexOf(w)>=0});
}

function findPairedMmproj(mainFile){
  var files=_allFiles||[];
  var matches=files.filter(function(f){return f.is_mmproj&&mmprojMatches(mainFile.filename,f.filename)});
  if(!matches.length){
    var all=files.filter(function(f){return f.is_mmproj});
    if(all.length===1)return all[0];
    return null;
  }
  matches.sort(function(a,b){return (b.size||0)-(a.size||0)});
  return matches[0];
}

// selectFile：新版，接受文件对象（供 confirmMultiSelect 调用）
async function selectFile(repoId, fileObj, source, skipClose){
  if(!fileObj) return;
  var data = {repo_id: repoId, source: source};
  if(fileObj.is_mmproj){
    var remoteMmproj = fileObj.is_shard ? fileObj.first_shard : fileObj.filename;
    data.mmproj_file = fileBaseName(remoteMmproj);
    data.mmproj_repo_file = remoteMmproj;
    data.mmproj_download_url = fileObj.download_url||'';
    data.mmproj_size = fileObj.size||0;
    data.mmproj_use_xet = true;
    data.bind_mmproj = true;
  } else {
    if(fileObj.is_shard && fileObj.shard_files){
      var sorted = fileObj.shard_files.slice().sort();
      data.model_file = fileBaseName(sorted[0]);
      data.model_repo_file = sorted[0];
      data.shard_files = sorted;
      data.shard_count = fileObj.shard_count;
      data.model_size = fileObj.size||0;
      data.shard_file_sizes = fileObj.shard_file_sizes||{};
      data.shard_download_urls = fileObj.shard_download_urls||{};
    } else {
      data.model_file = fileBaseName(fileObj.filename);
      data.model_repo_file = fileObj.filename;
      data.model_download_url = fileObj.download_url||'';
      data.model_size = fileObj.size||0;
    }
    data.auto_match_mmproj = true;
  }
  var result = await api('/api/config/model','POST',data);
  if(!skipClose) closeFileModal();
  await loadConfig();
  return result;
}

function showToast(msg, duration){
  var t = document.getElementById('toast');
  if(!t){ t=document.createElement('div'); t.id='toast';
    t.style.cssText='position:fixed;bottom:24px;left:50%;transform:translateX(-50%);'+
      'background:var(--bg3);border:1px solid var(--border);color:var(--text);'+
      'padding:10px 20px;border-radius:8px;font-size:13px;z-index:9999;max-width:80vw;text-align:center';
    document.body.appendChild(t); }
  t.textContent=msg; t.style.display='block';
  clearTimeout(t._timer);
  t._timer=setTimeout(function(){t.style.display='none'}, duration||3000);
}

// ===== 模型库 =====
async function refreshLibrary(){
  var el=document.getElementById('libraryContent');
  el.innerHTML='<div class="empty"><div class="loading"></div><br>扫描中...</div>';
  var data=await api('/api/models');
  var models=(data&&data.models)||[];
  if(!models.length){
    el.innerHTML='<div class="empty"><div class="icon">📭</div>还没有下载任何模型<br>请到「🔍 模型市场」搜索并下载</div>';
    return;
  }
  var mainModels=models.filter(function(m){return !m.is_mmproj});
  var visionModels=models.filter(function(m){return m.is_mmproj});
  var html='<table class="file-table"><thead><tr><th>模型</th><th>量化</th><th>大小</th><th>状态</th><th>操作</th></tr></thead><tbody>';
  mainModels.forEach(function(m){
    var ef=m.filename.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    html+='<tr><td><div style="font-weight:500">'+m.filename+'</div>'+
      '<div style="font-size:11px;color:var(--text3)">'+m.path+'</div></td>'+
      '<td><span class="tag tag-blue">'+m.quant+'</span></td>'+
      '<td>'+(m.size_mb>1024?m.size_gb+' GB':m.size_mb+' MB')+'</td>'+
      '<td>'+(m.is_active?'<span class="tag tag-green">🟢 使用中</span>':'<span class="tag tag-yellow">待命</span>')+'</td>'+
      '<td style="white-space:nowrap">'+(m.is_active?'<span style="font-size:12px;color:var(--text3)">当前模型</span>':
      '<button class="btn btn-primary btn-sm" onclick="activateModel(\''+ef+'\',false)">激活</button> '+
      '<button class="btn btn-danger btn-sm" onclick="deleteModel(\''+ef+'\')">删除</button>')+'</td></tr>';
  });
  html+='</tbody></table>';
  if(visionModels.length){
    html+='<div class="card-title" style="margin-top:16px">🔭 视觉模块</div>';
    html+='<table class="file-table"><thead><tr><th>文件</th><th>大小</th><th>状态</th><th>操作</th></tr></thead><tbody>';
    visionModels.forEach(function(m){
      var ef=m.filename.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
      html+='<tr><td>'+m.filename+'</td>'+
        '<td>'+(m.size_mb>1024?m.size_gb+' GB':m.size_mb+' MB')+'</td>'+
        '<td>'+(m.is_active?'<span class="tag tag-green">🟢 使用中</span>':'<span class="tag tag-yellow">待命</span>')+'</td>'+
        '<td>'+(m.is_active?'<span style="font-size:12px;color:var(--text3)">当前模块</span>':
        '<button class="btn btn-primary btn-sm" onclick="activateModel(\''+ef+'\',true)">激活</button>')+'</td></tr>';
    });
    html+='</tbody></table>';
  }
  el.innerHTML=html;
}

async function activateModel(filename,isMmproj){
  if(!confirm('切换为: '+filename+' ？\n需重启服务器生效'))return;
  var r=await api('/api/models/activate','POST',{filename:filename,is_mmproj:isMmproj});
  alert((r&&r.message)?r.message:'完成');
  refreshLibrary();loadConfig();
}

async function deleteModel(filename){
  if(!confirm('⚠️ 确定删除 '+filename+' ？\n不可恢复！'))return;
  var r=await api('/api/models/delete','POST',{filename:filename});
  alert((r&&r.message)?r.message:'完成');
  refreshLibrary();
}

// ===== 配置 =====
async function loadConfig(){
  currentConfig=await api('/api/config');
  if(!currentConfig||currentConfig.error||!currentConfig.model)currentConfig=defaultCfg();
  var m=currentConfig.model||{},d=currentConfig.download||{},s=currentConfig.server||{},
      sp=currentConfig.sampling||{},g=currentConfig.gpu||{},p=currentConfig.performance||{};
  setVal('cfg-repo_id',m.repo_id);setVal('cfg-model_file',m.model_file);
  setVal('cfg-mmproj_file',m.mmproj_file);setVal('cfg-hf_mirror',d.hf_mirror);
  setVal('cfg-github_mirror',d.github_mirror);setVal('cfg-port',s.port);
  setVal('cfg-threads',s.threads);setVal('cfg-ctx_size',s.ctx_size!=null?s.ctx_size:8192);
  setVal('cfg-reasoning_budget',s.reasoning_budget!=null?s.reasoning_budget:512);
  setVal('cfg-temperature',sp.temperature);setVal('cfg-top_k',sp.top_k);
  setVal('cfg-top_p',sp.top_p);setVal('cfg-presence_penalty',sp.presence_penalty);
  setVal('cfg-spec_type',p.spec_type||'off');setVal('cfg-spec_draft_n_max',p.spec_draft_n_max!=null?p.spec_draft_n_max:3);
  setVal('cfg-performance_profile',p.profile||'auto');setVal('cfg-fit_target_mb',p.fit_target_mb!=null?p.fit_target_mb:0);
  setVal('cfg-cache_type_k',p.cache_type_k||'auto');setVal('cfg-cache_type_v',p.cache_type_v||'auto');
  setVal('cfg-n_cpu_moe',p.n_cpu_moe!=null?p.n_cpu_moe:0);
  var bk=document.getElementById('cfg-gpu_backend');if(bk)bk.value=g.backend||'auto';
  var glVal=g.gpu_layers!=null?g.gpu_layers:-1;
  setVal('cfg-gpu_layers',glVal);
  var glPreset=document.getElementById('cfg-gpu_layers_preset');
  if(glPreset){
    if(glVal===-1)glPreset.value='-1';
    else if(glVal===99)glPreset.value='99';
    else if(glVal===0)glPreset.value='0';
    else{glPreset.value='custom';document.getElementById('cfg-gpu_layers').style.display='block';}
  }
  var fa=document.getElementById('cfg-flash_attn');
  if(fa){if(g.flash_attention!==false)fa.classList.add('on');else fa.classList.remove('on')}
  var tk=document.getElementById('cfg-thinking');
  if(tk){if(s.enable_thinking)tk.classList.add('on');else tk.classList.remove('on')}
  var kvu=document.getElementById('cfg-kv_unified');
  if(kvu){if(p.kv_unified!==false)kvu.classList.add('on');else kvu.classList.remove('on')}
}

function defaultCfg(){
  return{model:{repo_id:'',model_file:'',mmproj_file:'',mmproj_use_xet:true,mmproj_map:{}},
    download:{hf_mirror:'https://hf-mirror.com',github_mirror:'',timeout:300,retries:3},
    server:{host:'0.0.0.0',port:8080,threads:0,ctx_size:8192,enable_thinking:false,reasoning_budget:512},
    compat:{enabled:true,host:'0.0.0.0',port:11434,upstream_url:'http://127.0.0.1:8080',
      model_alias:'llama-deploy-local',api_key:'local-no-key-needed',claude_tool_mode:'repair',request_timeout:600},
    gpu:{backend:'auto',gpu_layers:-1,flash_attention:true},
    sampling:{temperature:0.7,top_k:20,top_p:0.8,presence_penalty:1.5,max_tokens:2048},
    performance:{profile:'auto',parallel:1,threads_batch:0,batch_size:0,ubatch_size:0,fit_target_mb:0,
      priority:2,priority_batch:2,cache_reuse:512,auto_gpu_layers:true,
      cache_type_k:'auto',cache_type_v:'auto',spec_type:'off',spec_draft_n_max:3,
      spec_draft_ngl:'auto',allow_experimental_mtp:false,kv_unified:true,ctx_checkpoints:32,cpu_moe:false,n_cpu_moe:0},
    build:{use_openblas:true,jobs:0},ui:{language:'zh',verbose:true}}
}

async function saveConfig(){
  var oldModel=(currentConfig&&currentConfig.model)||{};
  var oldPerf=(currentConfig&&currentConfig.performance)||defaultCfg().performance;
  var oldCompat=(currentConfig&&currentConfig.compat)||defaultCfg().compat;
  var serverPort=getNum('cfg-port',8080);
  oldCompat.upstream_url='http://127.0.0.1:'+serverPort;
  oldPerf.spec_type=getVal('cfg-spec_type')||'off';
  oldPerf.spec_draft_n_max=getNum('cfg-spec_draft_n_max',3);
  oldPerf.profile=getVal('cfg-performance_profile')||'auto';
  oldPerf.fit_target_mb=getNum('cfg-fit_target_mb',0);
  oldPerf.cache_type_k=getVal('cfg-cache_type_k')||'auto';
  oldPerf.cache_type_v=getVal('cfg-cache_type_v')||'auto';
  oldPerf.n_cpu_moe=getNum('cfg-n_cpu_moe',0);
  oldPerf.kv_unified=document.getElementById('cfg-kv_unified').classList.contains('on');
  var cfg={
    model:{repo_id:getVal('cfg-repo_id'),model_file:getVal('cfg-model_file'),
           mmproj_file:getVal('cfg-mmproj_file'),mmproj_use_xet:true,
           shard_files:oldModel.shard_files||undefined,shard_count:oldModel.shard_count||undefined,
           shard_file_sizes:oldModel.shard_file_sizes||undefined,mmproj_map:oldModel.mmproj_map||{}},
    download:{hf_mirror:getVal('cfg-hf_mirror'),github_mirror:getVal('cfg-github_mirror'),timeout:300,retries:3},
     server:{host:'0.0.0.0',port:serverPort,threads:getNum('cfg-threads',0),
             ctx_size:getNum('cfg-ctx_size',2048),
             enable_thinking:document.getElementById('cfg-thinking').classList.contains('on'),
             reasoning_budget:getNum('cfg-reasoning_budget',512)},
     compat:oldCompat,
     gpu:{backend:document.getElementById('cfg-gpu_backend').value||'auto',
         gpu_layers:getNum('cfg-gpu_layers',-1),
         flash_attention:document.getElementById('cfg-flash_attn').classList.contains('on')},
    sampling:{temperature:getFloat('cfg-temperature',0.7),top_k:getNum('cfg-top_k',20),
              top_p:getFloat('cfg-top_p',0.8),presence_penalty:getFloat('cfg-presence_penalty',1.5),max_tokens:2048},
    performance:oldPerf,
    build:{use_openblas:true,jobs:0},ui:{language:'zh',verbose:true}
  };
  var r=await api('/api/config','POST',cfg);
  alert((r&&r.message)?r.message:'已保存');
}

function resetConfig(){if(confirm('确定恢复默认配置？')){currentConfig=defaultCfg();loadConfig()}}

// ===== 部署 =====
async function startDeploy(){
  if(_selectedMain || _selectedMmproj){
    var pendingNames = [_selectedMain&&_selectedMain.filename, _selectedMmproj&&_selectedMmproj.filename].filter(Boolean);
    if(!confirm('检测到模型市场已勾选但尚未确认的文件：\n'+pendingNames.join('\n')+'\n\n是否先设为待部署模型并开始部署？'))return;
    await applySelectedFilesForDeploy();
    closeFileModal();
  }
  var cfg=await api('/api/config');
  if(!cfg.model||!cfg.model.model_file){alert('⚠️ 请先选择模型');return}
  if(!confirm('开始部署？'))return;
  document.getElementById('btnDeploy').disabled=true;
  var up=document.getElementById('btnUpgradeLlama');if(up)up.disabled=true;
  logLineCount=0;document.getElementById('deployLog').textContent='🚀 开始部署...\n';
  var r=await api('/api/deploy','POST');
  if(r&&r.status==='error'){
    alert(r.message||'启动部署失败');
    document.getElementById('btnDeploy').disabled=false;
    if(up)up.disabled=false;
    return;
  }
  startLogPoll();
}

async function upgradeLlama(){
  var st=await api('/api/status');
  if(st&&st.server_running){alert('请先停止服务器，再升级 llama.cpp');return}
  if(!confirm('升级 llama.cpp 到最新可用版本？\n失败时会自动回滚到当前版本。'))return;
  document.getElementById('btnUpgradeLlama').disabled=true;
  logLineCount=0;document.getElementById('deployLog').textContent='⬆️ 开始升级 llama.cpp...\n';
  var r=await api('/api/llama/update','POST');
  if(r&&r.status==='error'){
    alert(r.message||'启动升级失败');
    document.getElementById('btnUpgradeLlama').disabled=false;
    return;
  }
  startLogPoll();
}

function startLogPoll(){
  if(logPollTimer)clearInterval(logPollTimer);
  logPollTimer=setInterval(async function(){
    var data=await api('/api/deploy/log?since='+logLineCount);
    if(data.lines&&data.lines.length){
      var box=document.getElementById('deployLog');
      data.lines.forEach(function(line){
        var span=document.createElement('span');
        if(line.indexOf('✅')>=0)span.className='log-ok';
        else if(line.indexOf('❌')>=0)span.className='log-err';
        else if(line.indexOf('⚠')>=0)span.className='log-warn';
        else if(line.indexOf('📌')>=0||line.indexOf('🔍')>=0||line.indexOf('🚀')>=0)span.className='log-info';
        span.textContent=line+'\n';box.appendChild(span);
      });
      box.scrollTop=box.scrollHeight;logLineCount=data.total;
    }
    if(!data.running){clearInterval(logPollTimer);logPollTimer=null;
      document.getElementById('btnDeploy').disabled=false;
      var up=document.getElementById('btnUpgradeLlama');if(up)up.disabled=false;
      pollStatus();}
  },1000);
}

async function startServer(vision){
  var r=await api('/api/server/start','POST',{vision:vision});
  alert((r&&r.message)?r.message:'操作完成');setTimeout(pollStatus,2000);
}
async function stopServer(){
  var r=await api('/api/server/stop','POST');
  alert((r&&r.message)?r.message:'操作完成');setTimeout(pollStatus,1000);
}

async function publishLan(){
  var r=await api('/api/lan/publish','POST',{});
  alert((r&&r.message)?r.message:'操作完成');
  setTimeout(pollStatus,1200);
}
async function startGateway(){
  var r=await api('/api/gateway/start','POST',{});
  alert((r&&r.message)?r.message:'操作完成');
  setTimeout(pollStatus,1000);
}
async function stopGateway(){
  var r=await api('/api/gateway/stop','POST',{});
  alert((r&&r.message)?r.message:'操作完成');
  setTimeout(pollStatus,1000);
}
async function openFirewall(){
  if(!confirm('将尝试放行 llama-deploy 的服务端口和兼容网关端口。Windows 可能需要管理员权限，是否继续？'))return;
  var r=await api('/api/lan/firewall','POST',{});
  alert((r&&r.message)?r.message:'操作完成');
}

function renderLan(st){
  var box=document.getElementById('lanContent');if(!box)return;
  var ip=firstLanIp(st), apiPort=st.port||8080, gwPort=st.gateway_port||11434;
  var llamaUrl='http://'+ip+':'+apiPort;
  var openaiUrl='http://'+ip+':'+apiPort+'/v1';
  var compatBase='http://'+ip+':'+gwPort;
  var compatOpenAI=compatBase+'/v1';
  var managerUrl='http://'+ip+':'+location.port;
  var key=st.gateway_api_key||'local-no-key-needed';
  var alias=st.gateway_model_alias||'llama-deploy-local';
  var claudePs='$env:ANTHROPIC_BASE_URL="'+compatBase+'"\n$env:ANTHROPIC_API_KEY="'+key+'"\n$env:ANTHROPIC_MODEL="'+alias+'"\n$env:ANTHROPIC_SMALL_FAST_MODEL="'+alias+'"\n$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"\nclaude';
  var claudeCmd='set ANTHROPIC_BASE_URL='+compatBase+' && set ANTHROPIC_API_KEY='+key+' && set ANTHROPIC_MODEL='+alias+' && set ANTHROPIC_SMALL_FAST_MODEL='+alias+' && set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 && claude';
  var ollamaEnv='OLLAMA_HOST='+compatBase;
  var openaiEnv='OPENAI_BASE_URL='+compatOpenAI+'\nOPENAI_API_KEY=local-no-key-needed';
  var codexEnv='OPENAI_BASE_URL='+compatOpenAI+'\nOPENAI_API_KEY=local-no-key-needed\ncodex';
  var geminiPs='$env:GOOGLE_GEMINI_BASE_URL="'+compatBase+'"\n$env:GEMINI_BASE_URL="'+compatBase+'"\n$env:GEMINI_API_KEY="local-no-key-needed"\ngemini';
  box.innerHTML=
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;font-size:13px">'+
    '<div><div style="color:var(--text3)">局域网 IP</div><strong>'+esc((st.lan_ips||[]).join(' / '))+'</strong></div>'+
    '<div><div style="color:var(--text3)">llama-server</div><span class="status-dot '+(st.server_running?'green':'red')+'"></span> '+(st.server_running?'运行中':'未运行')+' · <a href="'+llamaUrl+'" target="_blank">'+esc(llamaUrl)+'</a></div>'+
    '<div><div style="color:var(--text3)">兼容网关</div><span class="status-dot '+(st.gateway_running?'green':'red')+'"></span> '+(st.gateway_running?'运行中':'未运行')+' · <a href="'+compatBase+'" target="_blank">'+esc(compatBase)+'</a></div>'+
    '<div><div style="color:var(--text3)">管理界面</div><a href="'+managerUrl+'" target="_blank">'+esc(managerUrl)+'</a></div>'+
    '</div>'+
    '<div style="margin-top:12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px">'+
    '<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px"><strong>Claude Code / Anthropic</strong><br><code style="word-break:break-all">'+esc(compatBase)+'</code><br><button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(claudePs))+')">复制 PowerShell</button> <button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(claudeCmd))+')">复制 CMD</button></div>'+
    '<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px"><strong>Ollama API</strong><br><code style="word-break:break-all">'+esc(compatBase)+'</code><br><button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(ollamaEnv))+')">复制 OLLAMA_HOST</button></div>'+
    '<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px"><strong>OpenAI API</strong><br><code style="word-break:break-all">'+esc(openaiUrl)+'</code><br><code style="word-break:break-all">'+esc(compatOpenAI)+'</code><br><button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(openaiEnv))+')">复制环境变量</button></div>'+
    '<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px"><strong>Codex / OpenAI Responses</strong><br><code style="word-break:break-all">'+esc(compatOpenAI)+'</code><br><button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(codexEnv))+')">复制 Codex 环境</button></div>'+
    '<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px"><strong>Gemini CLI</strong><br><code style="word-break:break-all">'+esc(compatBase)+'</code><br><button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="copyText('+esc(JSON.stringify(geminiPs))+')">复制 PowerShell</button></div>'+
    '</div>';
}

async function pollStatus(){
  var st=await api('/api/status');if(!st||st.error)return;
  var llamaInfo=st.llama_version?('版本 '+st.llama_version+(st.llama_build?'<br><span style="font-size:12px;color:var(--text3)">'+st.llama_build+'</span>':'')):'未知';
  var modelDot=st.model_deployed?'green':(st.model_exists?'yellow':'red');
  var modelText='未部署';
  if(st.model_deployed)modelText=st.model_file+' ('+st.model_size_mb+'MB)';
  if(st.model_size_mismatch)modelText+=' · 大小与市场记录不同';
  document.getElementById('statusContent').innerHTML=
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:14px">'+
    '<div><span style="color:var(--text3)">服务器</span><br>'+
    '<span class="status-dot '+(st.server_running?'green':'red')+'"></span> '+
    '<strong>'+(st.server_running?'运行中':'未运行')+'</strong>'+
    (st.server_pid?' (PID:'+st.server_pid+')':'')+
    (st.server_running?'<br><a href="http://localhost:'+st.port+'" target="_blank">🌐 打开 Web UI →</a>':'')+
    '</div><div><span style="color:var(--text3)">模型</span><br>'+
    '<span class="status-dot '+modelDot+'"></span> '+esc(modelText)+
    '</div><div><span style="color:var(--text3)">视觉模块</span><br>'+
    '<span class="status-dot '+(st.mmproj_deployed?'green':(st.mmproj_file?'red':'yellow'))+'"></span> '+
    (st.mmproj_deployed?st.mmproj_file:(st.mmproj_file?'未下载':'未配置'))+
    '</div><div><span style="color:var(--text3)">API 端口</span><br><strong>'+st.port+'</strong></div>'+
    '<div><span style="color:var(--text3)">兼容网关</span><br>'+
    '<span class="status-dot '+(st.gateway_running?'green':'red')+'"></span> '+
    '<strong>'+(st.gateway_running?'运行中':'未运行')+'</strong>'+
    (st.gateway_pid?' (PID:'+st.gateway_pid+')':'')+
    '</div><div><span style="color:var(--text3)">llama.cpp</span><br>'+llamaInfo+'</div></div>';
  renderLan(st);
}

// ===== 系统信息 =====
async function updateSystemPage(){
  var i=sysInfo;var vram=i.gpu_vram_mb||0;
  var rec=await api('/api/recommend?ram='+i.ram_gb+'&vram='+vram);
  document.getElementById('systemInfo').innerHTML=
    '<div class="card"><div class="card-title">🖥️ 硬件</div>'+
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'+
    '<div class="form-group"><div class="form-label">操作系统</div><div style="font-size:18px;font-weight:600">'+i.os+' '+i.arch+'</div></div>'+
    '<div class="form-group"><div class="form-label">CPU</div><div style="font-size:18px;font-weight:600">'+i.cpu_count+' 核'+(i.is_arm?' (ARM)':'')+'</div></div>'+
    '<div class="form-group"><div class="form-label">内存</div><div style="font-size:18px;font-weight:600">'+i.ram_gb+' GB</div>'+
    '<div class="form-hint">可用: '+(i.ram_avail_gb||'?')+' GB</div></div>'+
    '<div class="form-group"><div class="form-label">磁盘</div><div style="font-size:18px;font-weight:600">'+i.disk_free_gb+' GB 可用</div></div>'+
    (i.gpu_name?'<div class="form-group"><div class="form-label">GPU</div><div style="font-size:18px;font-weight:600">'+i.gpu_name+'</div>'+
    '<div class="form-hint">显存: '+(i.gpu_vram_mb/1024).toFixed(1)+' GB · 可用: '+(i.gpu_vram_free_mb/1024).toFixed(1)+' GB</div></div>':'')+
    '</div></div>'+
    '<div class="card"><div class="card-title">💡 AI 建议</div><div style="font-size:14px;line-height:2">'+
    '<div>📦 推荐量化: <span class="tag tag-green">'+(rec.recommended_quant||'Q4_K_M')+'</span></div>'+
    '<div>📐 推荐 ctx: <span class="tag tag-yellow">'+(rec.recommended_ctx||2048)+'</span></div>'+
    (rec.gpu_max_model_gb?'<div>🎮 GPU可载: <span class="tag tag-blue">≤'+rec.gpu_max_model_gb+' GB</span></div>':'')+
    '<hr style="border-color:var(--border);margin:8px 0">'+
    ((rec.tips||[]).map(function(t){return'<div>• '+t+'</div>'}).join(''))+
    '</div></div>';
}

init();setInterval(pollStatus,10000);
</script>
</body>
</html>"""


# ============================================================
#  主入口
# ============================================================

def main():
    port = DEFAULT_PORT
    host = "127.0.0.1"
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ("--port", "-p") and i + 2 <= len(sys.argv[1:]):
            port = int(sys.argv[i + 2])
        if arg == "--host" and i + 2 <= len(sys.argv[1:]):
            host = sys.argv[i + 2]

    if not CONFIG_FILE.exists():
        save_config(default_config())
        print(f"📝 已创建默认配置: {CONFIG_FILE}")

    server = http.server.ThreadingHTTPServer((host, port), APIHandler)

    print(f"""
╔══════════════════════════════════════════════╗
║  🦙 llama-deploy 智能管理器 v{VERSION}           ║
╠══════════════════════════════════════════════╣
║                                              ║
║  🌐 管理界面: http://localhost:{port}          ║
║  📡 监听地址:  http://{host}:{port}             ║
║                                              ║
║  Ctrl+C 退出                                 ║
╚══════════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 再见！")
        server.server_close()


if __name__ == "__main__":
    main()
