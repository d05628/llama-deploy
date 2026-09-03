#!/usr/bin/env python3
"""
llama-deploy: 一键部署 llama.cpp + Qwen 模型
支持 Windows / Linux / macOS / 树莓派
对中文用户友好，支持国内镜像加速

用法：python deploy.py
"""
# ============================================================
#  Windows UTF-8 修复（必须在最前面）
# ============================================================
import sys
import io
import os

# 强制 UTF-8 输出，解决 Windows GBK 无法显示 Emoji 的问题
if sys.platform == "win32":
    # 尝试启用 Windows 终端 UTF-8 模式
    try:
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass
    # 就地切换编码，避免多个模块导入时重复包装并意外关闭底层缓冲区。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import json
import csv
import hashlib
import html
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================
#  常量定义
# ============================================================

VERSION = "1.0.1"
BASE_DIR = Path(__file__).parent.resolve()
LLAMA_DIR = BASE_DIR / "llama.cpp"
MODELS_DIR = BASE_DIR / "models"
VENV_DIR = BASE_DIR / ".hf-venv"
CONFIG_FILE = BASE_DIR / "config.jsonc"
LLAMA_GITHUB_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp"

# ============================================================
#  JSONC 解析器（支持 // 和 /* */ 注释）
# ============================================================

def parse_jsonc(filepath: Path) -> dict:
    """解析 JSONC 文件（JSON with Comments）"""
    text = filepath.read_text(encoding="utf-8")
    # 移除单行注释 // ...
    text = re.sub(r'(?<!:)//.*', '', text)
    # 移除多行注释 /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # 移除行尾逗号（宽容解析）
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n❌ 配置文件解析失败: {e}")
        print(f"   请检查 {filepath} 的 JSON 格式是否正确")
        sys.exit(1)


def local_filename(remote_name: str) -> str:
    """将仓库相对路径转换为跨平台一致的本地文件名。"""
    return str(remote_name or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def paired_mmproj_filename(model_file: str, mmproj_file: str) -> str:
    """让视觉模型文件名携带对应主模型名，复制后无需原配置也能识别。"""
    model_name = local_filename(model_file)
    mmproj_name = local_filename(mmproj_file)
    if not model_name or not mmproj_name:
        return mmproj_name
    model_stem = re.sub(r"-\d{5}-of-\d{5}$", "", Path(model_name).stem, flags=re.I)
    if mmproj_name.lower().startswith((model_stem + ".mmproj").lower()):
        return mmproj_name
    variants = re.findall(
        r"(?:^|[-_.])(bf16|f16|f32|q[2-8](?:_[0-9a-z]+)?|iq[1-4](?:_[0-9a-z]+)?)(?=$|[-_.])",
        Path(mmproj_name).stem,
        flags=re.I,
    )
    suffix = "-" + variants[-1].lower() if variants else ""
    return f"{model_stem}.mmproj{suffix}.gguf"


# ============================================================
#  国际化（i18n）
# ============================================================

MESSAGES = {
    "zh": {
        "welcome":        "🚀 llama-deploy 一键部署工具 v{}",
        "detecting":      "🔍 检测系统环境...",
        "os_info":        "   系统: {} | 架构: {} | CPU: {} 核 | 内存: {:.1f}GB",
        "step":           "\n📌 步骤 {}/{}：{}",
        "check_deps":     "检查系统依赖",
        "install_deps":   "   正在安装依赖: {}",
        "dep_ok":         "   ✅ 系统依赖已就绪",
        "download_llama": "下载 llama.cpp",
        "llama_exists":   "   ✅ llama.cpp 已存在，跳过下载",
        "downloading":    "   ⬇️  正在下载: {}",
        "download_ok":    "   ✅ 下载完成",
        "download_fail":  "   ❌ 下载失败: {}",
        "build_llama":    "编译 llama.cpp",
        "building":       "   🔨 正在编译（约10-20分钟）...",
        "build_ok":       "   ✅ 编译完成",
        "build_skip":     "   ✅ 已有编译结果，跳过",
        "build_fail":     "   ❌ 编译失败",
        "download_model": "下载模型文件",
        "model_exists":   "   ✅ 模型已存在: {} ({:.0f}MB)",
        "model_small":    "   ⚠️  文件异常小（{}字节），可能是指针文件，重新下载...",
        "download_mmproj":"下载视觉模块（mmproj）",
        "mmproj_skip":    "   ⏭️  未配置视觉模块，跳过",
        "mmproj_exists":  "   ✅ 视觉模块已存在: {} ({:.0f}MB)",
        "xet_install":    "   📦 创建虚拟环境并安装 huggingface_hub...",
        "xet_download":   "   ⬇️  通过 Xet 协议下载视觉模块...",
        "verify":         "验证部署结果",
        "verify_ok":      "   ✅ {}",
        "verify_fail":    "   ❌ 缺失: {}",
        "done":           "\n🎉 部署完成！",
        "usage_cli":      "   聊天模式:  python run.py chat",
        "usage_server":   "   服务器:    python run.py server",
        "usage_server_v": "   视觉服务:  python run.py server --vision",
        "usage_stop":     "   停止服务:  python run.py stop",
        "error":          "\n❌ 错误: {}",
        "retry":          "   🔄 重试 ({}/{})",
        "mem_warn":       "   ⚠️  内存仅 {:.1f}GB，建议降低 ctx_size 至 1024",
        "progress":       "   进度: {:.1f}%",
        "win_download":   "   ⬇️  下载 Windows 预编译版本...",
        "swap_info":      "   💡 当前 Swap: {}MB，建议至少 2048MB",
        "swap_hint":      "   💡 增加 Swap 命令:\n"
                          "      sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile\n"
                          "      sudo dphys-swapfile setup && sudo dphys-swapfile swapon",
    },
    "en": {
        "welcome":        "🚀 llama-deploy v{}",
        "detecting":      "🔍 Detecting system...",
        "os_info":        "   OS: {} | Arch: {} | CPU: {} cores | RAM: {:.1f}GB",
        "step":           "\n📌 Step {}/{}：{}",
        "check_deps":     "Check dependencies",
        "install_deps":   "   Installing: {}",
        "dep_ok":         "   ✅ Dependencies OK",
        "download_llama": "Download llama.cpp",
        "llama_exists":   "   ✅ llama.cpp exists, skipping",
        "downloading":    "   ⬇️  Downloading: {}",
        "download_ok":    "   ✅ Download complete",
        "download_fail":  "   ❌ Download failed: {}",
        "build_llama":    "Build llama.cpp",
        "building":       "   🔨 Building (10-20 min)...",
        "build_ok":       "   ✅ Build complete",
        "build_skip":     "   ✅ Already built, skipping",
        "build_fail":     "   ❌ Build failed",
        "download_model": "Download model",
        "model_exists":   "   ✅ Model exists: {} ({:.0f}MB)",
        "model_small":    "   ⚠️  File too small ({}B), re-downloading...",
        "download_mmproj":"Download vision module (mmproj)",
        "mmproj_skip":    "   ⏭️  No mmproj configured, skipping",
        "mmproj_exists":  "   ✅ Vision module exists: {} ({:.0f}MB)",
        "xet_install":    "   📦 Setting up venv for huggingface_hub...",
        "xet_download":   "   ⬇️  Downloading via Xet...",
        "verify":         "Verify deployment",
        "verify_ok":      "   ✅ {}",
        "verify_fail":    "   ❌ Missing: {}",
        "done":           "\n🎉 Deployment complete!",
        "usage_cli":      "   Chat:      python run.py chat",
        "usage_server":   "   Server:    python run.py server",
        "usage_server_v": "   Vision:    python run.py server --vision",
        "usage_stop":     "   Stop:      python run.py stop",
        "error":          "\n❌ Error: {}",
        "retry":          "   🔄 Retry ({}/{})",
        "mem_warn":       "   ⚠️  Only {:.1f}GB RAM, consider ctx_size=1024",
        "progress":       "   Progress: {:.1f}%",
        "win_download":   "   ⬇️  Downloading pre-built Windows binary...",
        "swap_info":      "   💡 Swap: {}MB (recommend 2048MB+)",
        "swap_hint":      "   💡 To increase swap:\n"
                          "      sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile\n"
                          "      sudo dphys-swapfile setup && sudo dphys-swapfile swapon",
    }
}


class Printer:
    """国际化打印器"""
    def __init__(self, lang="zh"):
        self.lang = lang if lang in MESSAGES else "zh"
        self.msgs = MESSAGES[self.lang]

    def __call__(self, key, *args):
        tpl = self.msgs.get(key, key)
        try:
            msg = tpl.format(*args) if args else tpl
        except (IndexError, KeyError):
            msg = tpl
        print(msg)
        return msg


# ============================================================
#  系统检测
# ============================================================

class SystemInfo:
    """检测运行环境"""
    def __init__(self):
        self.os_name = platform.system()          # Windows / Linux / Darwin
        self.arch = platform.machine()             # x86_64 / aarch64 / armv7l / AMD64
        self.cpu_count = os.cpu_count() or 4
        self.ram_gb = self._get_ram_gb()
        self.is_windows = self.os_name == "Windows"
        self.is_linux = self.os_name == "Linux"
        self.is_mac = self.os_name == "Darwin"
        self.is_arm = self.arch.lower() in ("aarch64", "armv7l", "arm64")
        self.is_rpi = self.is_linux and self.is_arm and self._is_raspberry_pi()

    def _get_ram_gb(self) -> float:
        try:
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMSTAT(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]
                stat = MEMSTAT()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024 ** 3)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            pass
        return 0.0

    def _is_raspberry_pi(self) -> bool:
        try:
            with open("/proc/cpuinfo") as f:
                return "raspberry" in f.read().lower()
        except Exception:
            pass
        try:
            model = Path("/sys/firmware/devicetree/base/model").read_text()
            return "raspberry" in model.lower()
        except Exception:
            return False

    def get_swap_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("SwapTotal"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return -1


def running_llama_processes() -> list:
    """返回正在运行的 llama.cpp 引擎进程名，升级前用于避免文件占用。"""
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

# ============================================================
#  GPU 检测
# ============================================================

class GPUDetector:
    """检测 GPU 并推荐后端"""

    @staticmethod
    def detect() -> dict:
        info = {
            "has_nvidia": False,
            "has_amd": False,
            "nvidia_name": "",
            "nvidia_vram_mb": 0,
            "cuda_version": "",
            "compute_capability": "",
            "cuda_available": False,
            "vulkan_available": False,
            "recommended_backend": "cpu",
            "recommended_layers": 0,
        }

        # ── 检测 NVIDIA GPU ───────────────────────────────────────────────────
        # nvidia-smi 输出格式：name, memory.total, memory.free（逗号分隔）
        # GPU 名字可能含逗号（如 "Tesla T4, ..."），从右侧取两个数字字段更安全
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0 and result.stdout.strip():
                # 多卡时取第一行
                first_line = result.stdout.strip().splitlines()[0]
                # 从右侧切出最后一个数字字段（总显存），其余视为名字
                parts = [x.strip() for x in first_line.rsplit(",", 1)]
                if len(parts) == 2 and parts[1].isdigit():
                    info.update({
                        "has_nvidia": True,
                        "nvidia_name": parts[0],
                        "nvidia_vram_mb": int(parts[1]),
                        "cuda_available": True,
                        "recommended_backend": "cuda",
                        "recommended_layers": -1,   # 全部卸载
                    })
                    version_result = subprocess.run(
                        ["nvidia-smi"],
                        capture_output=True,
                        timeout=10,
                        encoding="utf-8",
                        errors="replace",
                    )
                    version_match = re.search(
                        r"CUDA Version:\s*(\d+(?:\.\d+)?)",
                        version_result.stdout or "",
                        flags=re.I,
                    )
                    if version_match:
                        info["cuda_version"] = version_match.group(1)

                    # RTX 50 / Blackwell 需要根据 compute capability 选择原生 CUDA 架构。
                    # 独立查询以兼容不支持 compute_cap 字段的旧版 nvidia-smi。
                    capability_result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        timeout=10,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if capability_result.returncode == 0:
                        capabilities = [
                            value.strip()
                            for value in capability_result.stdout.splitlines()
                            if re.fullmatch(r"\d+(?:\.\d+)?", value.strip())
                        ]
                        if capabilities:
                            # 异构多卡源码构建采用最低计算能力，避免生成部分设备不能加载的二进制。
                            info["compute_capability"] = min(
                                capabilities, key=lambda value: tuple(int(x) for x in value.split("."))
                            )
        except FileNotFoundError:
            pass   # nvidia-smi 未安装，静默跳过
        except Exception:
            pass

        # ── 检测 Vulkan ───────────────────────────────────────────────────────
        try:
            if platform.system() == "Windows":
                # 检查 Vulkan SDK 或系统目录
                vulkan_paths = [
                    Path(os.environ.get("VULKAN_SDK", "")) / "Bin" / "vulkaninfo.exe",
                    Path("C:/Windows/System32/vulkaninfo.exe"),
                    Path("C:/Windows/System32/vulkan-1.dll"),
                ]
                info["vulkan_available"] = any(p.exists() for p in vulkan_paths if p.parent != Path("Bin"))
            else:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    timeout=10,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    info["vulkan_available"] = True
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # ── 推荐后端（无 NVIDIA 时考虑 Vulkan）────────────────────────────────
        if not info["has_nvidia"] and info["vulkan_available"]:
            info["recommended_backend"] = "vulkan"
            info["recommended_layers"] = -1

        return info
# ============================================================
#  下载工具
# ============================================================

class InvalidDownloadResponse(RuntimeError):
    """下载地址返回了网页等非目标文件。"""


class Downloader:
    """文件下载器，支持进度显示和重试"""
    def __init__(self, config: dict, printer: Printer):
        self.hf_mirror = config.get("download", {}).get("hf_mirror", "").rstrip("/")
        self.gh_mirror = config.get("download", {}).get("github_mirror", "").rstrip("/")
        self.timeout = config.get("download", {}).get("timeout", 300)
        self.retries = config.get("download", {}).get("retries", 3)
        self.p = printer

    def _progress_hook(self, block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            size_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            # 每 5% 打印一行（而不是 \r 覆盖），兼容 Web 日志
            if not hasattr(self, '_last_pct'):
                self._last_pct = -5
            if pct - self._last_pct >= 5 or pct >= 100:
                print(f"   [{bar}] {pct:.1f}% ({size_mb:.0f}/{total_mb:.0f}MB)")
                self._last_pct = pct
            if pct >= 100:
                self._last_pct = -5  # 重置

    def _size_ok(self, path: Path, expected_size: int = 0, min_size: int = 1000) -> bool:
        if not path.exists():
            return False
        size = path.stat().st_size
        if expected_size and expected_size > 0:
            return size == expected_size
        return size > min_size

    def _print_progress(self, downloaded: int, total_size: int):
        if total_size <= 0:
            return
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        size_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        if not hasattr(self, "_last_pct"):
            self._last_pct = -5
        if pct - self._last_pct >= 5 or pct >= 100:
            print(f"   [{bar}] {pct:.1f}% ({size_mb:.0f}/{total_mb:.0f}MB)")
            self._last_pct = pct
        if pct >= 100:
            self._last_pct = -5

    @staticmethod
    def _content_range_info(content_range: str) -> tuple:
        """Return (start, end, total) from a Content-Range header."""
        m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range or "", re.I)
        if not m:
            return None, None, 0
        total = 0 if m.group(3) == "*" else int(m.group(3))
        return int(m.group(1)), int(m.group(2)), total

    def download_file(self, url: str, dest: Path, expected_size: int = 0) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")

        if expected_size and tmp.exists() and tmp.stat().st_size > expected_size:
            print(f"   ⚠️  发现异常大的临时文件，重新下载: {tmp.name}")
            tmp.unlink(missing_ok=True)

        if expected_size and self._size_ok(tmp, expected_size):
            os.replace(tmp, dest)
            self.p("download_ok")
            return True

        if expected_size and dest.exists() and not self._size_ok(dest, expected_size):
            dest_size = dest.stat().st_size
            if 0 < dest_size < expected_size:
                if (not tmp.exists()) or dest_size > tmp.stat().st_size:
                    missing_mb = (expected_size - dest_size) / (1024 * 1024)
                    print(f"   🔄 发现未完成文件，改为断点续传: {dest.name}（缺少 {missing_mb:.1f}MB）")
                    os.replace(dest, tmp)
                else:
                    print(f"   ⚠️  发现未完成文件但已有更大的临时文件，移除旧文件: {dest.name}")
                    dest.unlink(missing_ok=True)
            else:
                bad = dest.with_name(dest.name + f".bad-{int(time.time())}")
                print(f"   ⚠️  文件大小异常，已隔离为: {bad.name}")
                os.replace(dest, bad)

        max_attempts = self.retries
        if expected_size >= 1024 ** 3:
            # Large GGUF files often fail due to transient mirror/network drops.
            # Keep retries conservative for small files, but be more patient here.
            max_attempts = max(max_attempts, 10)

        self._last_pct = -5
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    self.p("retry", attempt, max_attempts)
                self.p("downloading", url.split("/")[-1])

                resume_from = tmp.stat().st_size if tmp.exists() else 0
                headers = {"User-Agent": "llama-deploy"}
                if resume_from > 0:
                    headers["Range"] = f"bytes={resume_from}-"
                    print(f"   🔁 断点续传: 已有 {resume_from // (1024 * 1024)}MB")

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = getattr(resp, "status", resp.getcode())
                    resp_headers = resp.headers
                    content_type = (resp_headers.get("Content-Type") or "").lower()
                    if "text/html" in content_type or "application/xhtml+xml" in content_type:
                        raise InvalidDownloadResponse(
                            f"服务器返回了 HTML 页面而不是目标文件（{content_type.split(';', 1)[0]}）"
                        )
                    content_len = 0
                    try:
                        content_len = int(resp_headers.get("Content-Length") or 0)
                    except Exception:
                        content_len = 0

                    mode = "wb"
                    downloaded = 0
                    total_size = expected_size or content_len

                    if resume_from > 0 and status == 206:
                        cr_start, _cr_end, cr_total = self._content_range_info(
                            resp_headers.get("Content-Range", "")
                        )
                        if cr_start is not None and cr_start != resume_from:
                            print("   ⚠️  服务器返回的续传位置不匹配，重新下载")
                            tmp.unlink(missing_ok=True)
                            raise RuntimeError("断点续传位置不匹配，已清理临时文件")
                        else:
                            mode = "ab"
                            downloaded = resume_from
                            total_size = expected_size or cr_total or (resume_from + content_len)
                    elif resume_from > 0:
                        print("   ⚠️  当前地址不支持断点续传，重新下载")
                        tmp.unlink(missing_ok=True)
                        resume_from = 0

                    if resume_from == 0:
                        mode = "wb"
                        downloaded = 0
                        total_size = expected_size or content_len

                    self._print_progress(downloaded, total_size)
                    with open(tmp, mode) as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            self._print_progress(downloaded, total_size)

                header_size = 0
                try:
                    header_size = int(resp_headers.get("Content-Length") or 0)
                except Exception:
                    header_size = 0
                needed = expected_size or total_size or header_size
                if self._size_ok(tmp, needed):
                    os.replace(tmp, dest)
                    self.p("download_ok")
                    return True
                else:
                    bad_size = tmp.stat().st_size if tmp.exists() else 0
                    self.p("model_small", bad_size)
                    if not expected_size or bad_size < 1024 * 1024:
                        tmp.unlink(missing_ok=True)
                    else:
                        print(f"   🔁 保留未完成文件，稍后继续: {tmp.name} ({bad_size // (1024 * 1024)}MB)")
            except InvalidDownloadResponse as e:
                self.p("download_fail", str(e))
                tmp.unlink(missing_ok=True)
                break
            except urllib.error.HTTPError as e:
                if e.code == 416 and expected_size and self._size_ok(tmp, expected_size):
                    os.replace(tmp, dest)
                    self.p("download_ok")
                    return True
                if e.code == 416 and expected_size and tmp.exists() and tmp.stat().st_size > expected_size:
                    print(f"   ⚠️  临时文件超过目标大小，重新下载: {tmp.name}")
                    tmp.unlink(missing_ok=True)
                    time.sleep(2)
                    continue
                self.p("download_fail", f"HTTP {e.code}: {e.reason}")
                if e.code in (401, 403, 404):
                    break
                time.sleep(2)
            except Exception as e:
                self.p("download_fail", str(e))
                if tmp.exists() and tmp.stat().st_size > 0:
                    print(f"   🔁 已保留临时文件，下一次继续: {tmp.name} ({tmp.stat().st_size // (1024 * 1024)}MB)")
                time.sleep(2)
        return False

    def hf_url(self, repo_id: str, filename: str) -> str:
        base = self.hf_mirror or "https://huggingface.co"
        safe_filename = urllib.parse.quote(filename.replace("\\", "/"), safe="/")
        return f"{base}/{repo_id}/resolve/main/{safe_filename}"

    def hf_urls(self, repo_id: str, filename: str) -> list:
        safe_filename = urllib.parse.quote(filename.replace("\\", "/"), safe="/")
        bases = []
        if self.hf_mirror:
            bases.append(self.hf_mirror)
        bases.append("https://huggingface.co")
        urls = []
        for base in bases:
            url = f"{base}/{repo_id}/resolve/main/{safe_filename}"
            if url not in urls:
                urls.append(url)
        return urls

    def github_urls(self, path_or_url: str) -> list:
        """返回 GitHub 镜像和官方直链；镜像失效时可自动回退。"""
        direct = path_or_url
        if not direct.startswith(("http://", "https://")):
            direct = f"https://github.com/{direct.lstrip('/')}"
        urls = []
        if self.gh_mirror:
            urls.append(f"{self.gh_mirror}/{direct}")
        urls.append(direct)
        return list(dict.fromkeys(urls))

    def github_url(self, path_or_url: str) -> str:
        """兼容旧调用：返回首选 GitHub 地址。"""
        return self.github_urls(path_or_url)[0]

    def download_from_urls(
        self, urls: list, dest: Path, expected_size: int = 0, validator=None
    ) -> bool:
        """依次尝试候选地址，并在每次下载后执行内容校验。"""
        for index, url in enumerate(dict.fromkeys(urls), 1):
            if index > 1:
                print("   🔄 镜像不可用，回退 GitHub 官方直链...")
            if not self.download_file(url, dest, expected_size):
                continue
            try:
                valid = validator(dest) if validator else True
            except Exception as e:
                print(f"   ⚠️  文件校验失败: {e}")
                valid = False
            if valid:
                return True
            print(f"   ⚠️  下载内容不是有效目标文件: {dest.name}")
            dest.unlink(missing_ok=True)
            dest.with_name(dest.name + ".part").unlink(missing_ok=True)
        return False


# ============================================================
#  核心部署逻辑
# ============================================================

class Deployer:
    """主部署类"""

    def __init__(self, upgrade_llama: bool = False, force_source: bool = False):
        # 加载配置
        if not CONFIG_FILE.exists():
            print(f"❌ 找不到配置文件: {CONFIG_FILE}")
            print(f"   请确保 config.jsonc 与 deploy.py 在同一目录")
            sys.exit(1)

        self.config = parse_jsonc(CONFIG_FILE)
        lang = self.config.get("ui", {}).get("language", "zh")
        self.p = Printer(lang)
        self.verbose = self.config.get("ui", {}).get("verbose", True)
        self.sys = SystemInfo()
        self.dl = Downloader(self.config, self.p)
        self.upgrade_llama = upgrade_llama
        self.force_source = force_source
        self.llama_backup_dir = None
        self.target_llama_tag = ""

        # 模型配置
        mc = self.config.get("model", {})
        self.repo_id = mc.get("repo_id", "")
        self.model_source = str(mc.get("source", "huggingface") or "huggingface").lower()
        self.model_repo_file = mc.get("model_repo_file") or mc.get("model_file", "")
        self.model_file = local_filename(mc.get("model_file") or self.model_repo_file)
        self.model_download_url = mc.get("model_download_url", "")
        self.mmproj_repo_file = mc.get("mmproj_repo_file") or mc.get("mmproj_file", "")
        raw_mmproj_file = local_filename(mc.get("mmproj_file") or self.mmproj_repo_file)
        self.mmproj_file = paired_mmproj_filename(self.model_file, raw_mmproj_file)
        self.mmproj_download_url = mc.get("mmproj_download_url", "")
        self.mmproj_use_xet = mc.get("mmproj_use_xet", True)

        # GPU 配置
        gc = self.config.get("gpu", {})
        self.gpu_backend = gc.get("backend", "auto")
        self.gpu_layers = gc.get("gpu_layers", -1)
        self.flash_attention = gc.get("flash_attention", True)

        # GPU 检测
        self.gpu_info = GPUDetector.detect()

        # 路径
        self.model_path = MODELS_DIR / self.model_file
        self.mmproj_dir = MODELS_DIR / "vision"
        self.mmproj_path = self.mmproj_dir / self.mmproj_file

        # 可执行文件路径（先按最常见位置初始化，启动时再 _find_llama_binary 精确定位）
        if self.sys.is_windows:
            self.cli_bin    = LLAMA_DIR / "llama-cli.exe"
            self.server_bin = LLAMA_DIR / "llama-server.exe"
        else:
            self.cli_bin    = LLAMA_DIR / "build" / "bin" / "llama-cli"
            self.server_bin = LLAMA_DIR / "build" / "bin" / "llama-server"

        # 尝试精确定位（覆盖上面的默认路径）
        self._relocate_bins()

    def run(self):
        """执行完整部署流程"""
        total_steps = 4 if self.upgrade_llama else 6
        self.p("welcome", VERSION)
        self._detect_system()

        self._step(1, total_steps, "check_deps",     self._check_deps)
        try:
            self._step(2, total_steps, "download_llama", self._download_llama)
            self._step(3, total_steps, "build_llama", self._build_llama)
            if self.upgrade_llama:
                self._step(4, total_steps, "verify", self._verify_llama_only)
        except BaseException:
            # 下载、编译或实际运行验证任一失败，都恢复旧引擎。
            if self.llama_backup_dir:
                self._restore_llama_backup(self.llama_backup_dir)
                self._relocate_bins()
            raise
        else:
            if self.llama_backup_dir:
                self._cleanup_llama_backup(self.llama_backup_dir)

        if not self.upgrade_llama:
            self._step(4, total_steps, "download_model", self._download_model)
            self._step(5, total_steps, "download_mmproj", self._download_mmproj)
            self._step(6, total_steps, "verify", self._verify)

        self._print_usage()

    def _step(self, num, total, name_key, func):
        name = self.p.msgs.get(name_key, name_key)
        self.p("step", num, total, name)
        try:
            func()
        except Exception as e:
            self.p("error", str(e))
            if self.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # ----- 系统检测 -----
    def _detect_system(self):
        self.p("detecting")
        self.p("os_info",
               f"{self.sys.os_name}{' (树莓派)' if self.sys.is_rpi else ''}",
               self.sys.arch,
               self.sys.cpu_count,
               self.sys.ram_gb)

        if self.sys.ram_gb < 3.5:
            self.p("mem_warn", self.sys.ram_gb)

        if self.sys.is_linux:
            swap = self.sys.get_swap_mb()
            if swap >= 0:
                self.p("swap_info", swap)
                if swap < 1024:
                    self.p("swap_hint")
        # GPU 信息
        gpu = self.gpu_info
        if gpu["has_nvidia"]:
            print(f"   GPU: {gpu['nvidia_name']} ({gpu['nvidia_vram_mb']}MB)")
            print(f"   CUDA: ✅ 可用")
            if gpu.get("compute_capability"):
                print(f"   Compute Capability: {gpu['compute_capability']}")
        elif gpu["vulkan_available"]:
            print(f"   GPU: Vulkan 可用")
        else:
            print(f"   GPU: 未检测到，将使用 CPU")

        # 确定实际使用的后端
        if self.gpu_backend == "auto":
            self.actual_backend = gpu["recommended_backend"]
        else:
            self.actual_backend = self.gpu_backend
        print(f"   计算后端: {self.actual_backend.upper()}")

    # ----- 依赖检查 -----
    def _check_deps(self):
        if self.sys.is_windows:
            # Windows 使用预编译版本，无需额外依赖
            self._check_cmd("curl", required=False)
            self.p("dep_ok")
            return

        # Linux / macOS
        missing = []
        for cmd in ["git", "cmake", "make", "gcc"]:
            if not shutil.which(cmd):
                missing.append(cmd)

        if missing and self.sys.is_linux:
            packages = [
                "build-essential", "cmake", "git", "libcurl4-openssl-dev",
                "libopenblas-dev", "wget",
            ]
            self.p("install_deps", " ".join(packages))
            self._exec(["sudo", "apt", "update"])
            self._exec(["sudo", "apt", "install", "-y"] + packages)

        elif missing and self.sys.is_mac:
            self.p("install_deps", "xcode-select, cmake")
            subprocess.run(["xcode-select", "--install"], check=False)
            if not shutil.which("cmake"):
                self._exec(["brew", "install", "cmake"])

        self.p("dep_ok")

    def _check_cmd(self, cmd, required=True):
        if shutil.which(cmd) is None and required:
            raise RuntimeError(f"缺少命令: {cmd}")

    def _find_llama_binary(self, stem: str) -> Optional[Path]:
        """
        在 LLAMA_DIR 及常见子目录中搜索 llama.cpp 可执行文件。
        支持新旧版本的各种命名：
          llama-server.exe / server.exe（旧版）/ llama-server（Linux）
        搜索顺序：根目录 → build/bin → bin → build/Release/bin
        """
        is_win = self.sys.is_windows
        # 候选文件名（按优先级）
        if is_win:
            names = [f"{stem}.exe", f"{stem.split('-')[-1]}.exe"]
        else:
            names = [stem, stem.split("-")[-1]]

        search_dirs = [
            LLAMA_DIR,
            LLAMA_DIR / "build" / "bin",
            LLAMA_DIR / "build" / "bin" / "Release",
            LLAMA_DIR / "bin",
            LLAMA_DIR / "build" / "Release",
            LLAMA_DIR / "build" / "Release" / "bin",
            LLAMA_DIR / "build" / "Debug" / "bin",
        ]

        for d in search_dirs:
            if not d.is_dir():
                continue
            for name in names:
                p = d / name
                if p.exists() and p.stat().st_size > 1024:   # 大于 1KB 排除空文件
                    return p
        return None

    def _relocate_bins(self):
        """
        精确定位 llama-server 和 llama-cli 的实际路径，
        覆盖 __init__ 中设置的默认路径。
        只在文件真实存在时覆盖，避免在首次部署前误判。
        """
        found_server = self._find_llama_binary("llama-server")
        found_cli    = self._find_llama_binary("llama-cli")
        if found_server:
            self.server_bin = found_server
        if found_cli:
            self.cli_bin = found_cli

    # ----- 下载 llama.cpp -----
    def _download_llama(self):
        if self.upgrade_llama and LLAMA_DIR.exists():
            running = running_llama_processes()
            if running:
                raise RuntimeError(
                    "请先停止所有 llama.cpp 进程再升级: " + ", ".join(running)
                )
            self._backup_llama_dir()

        if self.sys.is_windows and not self.force_source:
            self._download_llama_windows()
        else:
            self._download_llama_source()

    def _download_llama_windows(self):
        """Windows：下载预编译版本（自动选择 CUDA/Vulkan/CPU）"""
        if (
            not self.upgrade_llama
            and self.server_bin.exists()
            and self.cli_bin.exists()
            and self._check_existing_backend()
        ):
            self.p("llama_exists")
            print(f"   📍 llama-server: {self.server_bin}")
            print(f"   📍 llama-cli:    {self.cli_bin}")
            return

        # 清理旧的不完整下载
        if LLAMA_DIR.exists():
            shutil.rmtree(LLAMA_DIR, ignore_errors=True)

        self.p("win_download")

        release = self._resolve_windows_release()
        assets = release.get("assets", []) if release else []
        self.target_llama_tag = str(release.get("tag_name", "") or "") if release else ""
        if self.target_llama_tag:
            print(f"   🎯 目标版本: {self.target_llama_tag}")
        main_asset, cudart_asset = self._find_best_asset(assets)

        if main_asset and self.actual_backend == "cuda":
            asset_cuda = re.search(r"cuda-(\d+(?:\.\d+)?)", str(main_asset.get("name", "")), re.I)
            capability = self._version_tuple(self.gpu_info.get("compute_capability", ""))
            selected_cuda = self._version_tuple(asset_cuda.group(1)) if asset_cuda else ()
            if capability >= (12, 0) and selected_cuda and selected_cuda < (12, 8):
                print(
                    "   ⚠️  已检测到 RTX 50 / Blackwell，但当前驱动只允许选择 CUDA "
                    f"{asset_cuda.group(1)} 预编译包；它可以运行，但不含 sm_120 原生优化。"
                )
                print("   💡 更新 NVIDIA 驱动，或安装 CUDA Toolkit 12.8+ 后运行 python deploy.py --build-from-source。")

        if not main_asset:
            missing = [name for name in ("git", "cmake") if not shutil.which(name)]
            if missing:
                raise RuntimeError(
                    "未找到兼容的 Windows 预编译包，且源码构建工具缺失: "
                    + ", ".join(missing)
                )
            print(f"   ⚠️  未找到预编译版本，尝试从源码编译...")
            self._download_llama_source()
            return

        LLAMA_DIR.mkdir(parents=True, exist_ok=True)

        # 下载并解压主程序包
        print(f"   📦 下载主程序: {main_asset['name']}")
        main_zip = BASE_DIR / "llama-main.zip"
        if not self.dl.download_from_urls(
            self.dl.github_urls(main_asset["browser_download_url"]),
            main_zip,
            int(main_asset.get("size") or 0),
            lambda path: self._validate_release_asset(path, main_asset),
        ):
            raise RuntimeError("主程序下载失败")

        self._extract_zip_flat(main_zip, LLAMA_DIR)
        main_zip.unlink(missing_ok=True)

        # 下载并解压 CUDA 运行时（如果需要）
        if cudart_asset:
            print(f"   📦 下载 CUDA 运行时: {cudart_asset['name']}")
            cudart_zip = BASE_DIR / "llama-cudart.zip"
            if not self.dl.download_from_urls(
                self.dl.github_urls(cudart_asset["browser_download_url"]),
                cudart_zip,
                int(cudart_asset.get("size") or 0),
                lambda path: self._validate_release_asset(path, cudart_asset),
            ):
                raise RuntimeError("CUDA 运行时下载失败，已取消升级")
            self._extract_zip_flat(cudart_zip, LLAMA_DIR)
            cudart_zip.unlink(missing_ok=True)

        # 重新定位 bin 路径（解压后路径才确定）
        self._relocate_bins()

        # 验证
        if self.server_bin.exists():
            self.p("download_ok")
            exe_count = len(list(LLAMA_DIR.glob("**/*.exe")))
            dll_count = len(list(LLAMA_DIR.glob("**/*.dll")))
            print(f"   📁 已安装: {exe_count} 个可执行文件, {dll_count} 个 DLL")
            print(f"   📍 llama-server: {self.server_bin}")
            print(f"   📍 llama-cli:    {self.cli_bin}")
        else:
            # 列出目录内容帮助排查
            print(f"   ⚠️  未找到 llama-server（已搜索 LLAMA_DIR 及子目录）")
            print(f"   📁 LLAMA_DIR 内容（前30项）:")
            for f in sorted(LLAMA_DIR.rglob("*"))[:30]:
                if f.is_file():
                    print(f"       {f.relative_to(LLAMA_DIR)} ({f.stat().st_size // 1024}KB)")
            raise RuntimeError(
                "下载解压后未找到 llama-server.exe\n"
                "可能原因：GitHub Release 包结构变更，或下载不完整。\n"
                "请手动从 https://github.com/ggml-org/llama.cpp/releases 下载并解压到 llama.cpp/ 目录"
            )

    @staticmethod
    def _github_headers() -> dict:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "llama-deploy/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _fetch_github_json(self, url: str):
        last_error = None
        for candidate in self.dl.github_urls(url):
            try:
                req = urllib.request.Request(candidate, headers=self._github_headers())
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_error = e
        raise RuntimeError(f"GitHub JSON 读取失败: {last_error}")

    def _fetch_github_text(self, url: str) -> str:
        headers = {"User-Agent": "llama-deploy/1.0"}
        last_error = None
        for candidate in self.dl.github_urls(url):
            try:
                req = urllib.request.Request(candidate, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                if url.endswith("/releases.atom") and "<feed" not in text:
                    raise InvalidDownloadResponse("返回内容不是 Release Atom feed")
                if "/releases/expanded_assets/" in url and "/releases/download/" not in text:
                    raise InvalidDownloadResponse("返回内容不含 Release 资产")
                return text
            except Exception as e:
                last_error = e
        raise RuntimeError(f"GitHub 页面读取失败: {last_error}")

    def _release_tags_from_atom(self) -> list:
        """从官方 Atom feed 获取 nightly 标签，不消耗 GitHub API 配额。"""
        text = self._fetch_github_text(f"{LLAMA_GITHUB_URL}/releases.atom")
        root = ET.fromstring(text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        tags = []
        for entry in root.findall("atom:entry", namespace):
            entry_id = entry.findtext("atom:id", default="", namespaces=namespace)
            tag = entry_id.rsplit("/", 1)[-1]
            if re.fullmatch(r"b\d+", tag) and tag not in tags:
                tags.append(tag)
        return tags

    def _release_assets_from_html(self, tag: str) -> list:
        """API 限流时，从官方 expanded-assets 页面恢复资产直链。"""
        safe_tag = urllib.parse.quote(tag, safe="")
        text = self._fetch_github_text(
            f"{LLAMA_GITHUB_URL}/releases/expanded_assets/{safe_tag}"
        )
        hrefs = re.findall(
            r'href="([^"\s]*/ggml-org/llama\.cpp/releases/download/[^"\s]+)"',
            text,
            flags=re.I,
        )
        assets = []
        seen = set()
        for href in hrefs:
            url = urllib.parse.urljoin("https://github.com", html.unescape(href))
            name = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
            if url in seen:
                continue
            seen.add(url)
            assets.append({
                "name": name,
                "browser_download_url": url,
                "size": 0,
                "state": "uploaded",
            })
        return assets

    def _resolve_windows_release(self) -> dict:
        """查找最新且目标架构/后端资产完整的 nightly release。"""
        api_error = None
        try:
            releases = self._fetch_github_json(f"{LLAMA_GITHUB_API}/releases?per_page=20")
            if not isinstance(releases, list):
                raise RuntimeError("GitHub Releases API 返回格式异常")
            nightly = [
                item for item in releases
                if not item.get("draft") and re.fullmatch(r"b\d+", str(item.get("tag_name", "")))
            ]
            nightly.sort(
                key=lambda item: int(str(item.get("tag_name", ""))[1:]),
                reverse=True,
            )
            for release in nightly:
                main_asset, _runtime_asset = self._find_best_asset(
                    release.get("assets", []), announce=False
                )
                if main_asset:
                    return release
        except Exception as e:
            api_error = e

        if api_error:
            print(f"   ⚠️  GitHub API 不可用，改用官方 Release feed: {api_error}")
        else:
            print("   ⚠️  API 中暂无完整 nightly 资产，改用官方 Release feed...")

        try:
            for tag in self._release_tags_from_atom()[:10]:
                assets = self._release_assets_from_html(tag)
                main_asset, _runtime_asset = self._find_best_asset(assets, announce=False)
                if main_asset:
                    return {"tag_name": tag, "assets": assets, "prerelease": True}
        except Exception as e:
            print(f"   ⚠️  官方 Release feed 读取失败: {e}")
        return {}

    def _extract_zip_flat(self, zip_path: Path, dest_dir: Path):
        """解压 zip，将所有文件平铺到 dest_dir（不保留子目录结构）"""
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                # 跳过目录
                if info.is_dir():
                    continue
                # 取文件名（去掉路径前缀）
                filename = Path(info.filename).name
                if not filename:
                    continue
                # 解压到目标目录
                target = dest_dir / filename
                with z.open(info) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

    def _validate_release_asset(self, path: Path, asset: dict) -> bool:
        if not zipfile.is_zipfile(path):
            return False
        digest = str(asset.get("digest") or "")
        if not digest.lower().startswith("sha256:"):
            return True
        expected = digest.split(":", 1)[1].lower()
        hasher = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected

    def _find_best_asset(self, assets: list, announce: bool = True) -> tuple:
        """精确选择本机架构/后端资产，并为 CUDA 匹配同版本 runtime。"""
        if not assets:
            return (None, None)

        arch = "arm64" if getattr(self.sys, "is_arm", False) else "x64"
        main_assets = {"cpu": [], "vulkan": [], "cuda": []}
        runtimes = {}

        for asset in assets:
            if asset.get("state") not in (None, "uploaded"):
                continue
            name = str(asset.get("name", "")).lower()
            runtime_match = re.fullmatch(
                r"cudart-llama-bin-win-cuda-(\d+(?:\.\d+)?)-(x64|arm64)\.zip",
                name,
            )
            if runtime_match:
                if runtime_match.group(2) == arch:
                    runtimes[runtime_match.group(1)] = asset
                continue
            main_match = re.fullmatch(
                r"llama-.+?-bin-win-(cpu|vulkan|cuda-(\d+(?:\.\d+)?))-(x64|arm64)\.zip",
                name,
            )
            if not main_match or main_match.group(3) != arch:
                continue
            backend_name = main_match.group(1)
            if backend_name.startswith("cuda-"):
                cuda_text = main_match.group(2)
                main_assets["cuda"].append((cuda_text, asset))
            else:
                main_assets[backend_name].append(asset)

        backend = getattr(self, "actual_backend", "cpu")
        if backend == "cuda":
            paired = [
                (tuple(int(part) for part in version.split(".")), version, asset, runtimes[version])
                for version, asset in main_assets["cuda"]
                if version in runtimes
            ]
            supported_text = str(self.gpu_info.get("cuda_version", "") or "")
            supported_match = re.match(r"(\d+)(?:\.(\d+))?", supported_text)
            if supported_match:
                supported = (int(supported_match.group(1)), int(supported_match.group(2) or 0))
                paired = [item for item in paired if item[0] <= supported]
            elif any(item[0][0] == 12 for item in paired):
                # 无法检测驱动上限时，优先兼容面更广的 CUDA 12。
                paired = [item for item in paired if item[0][0] == 12]
            if not paired:
                return (None, None)
            chosen = max(paired, key=lambda item: item[0])
            if announce:
                print(f"   🎯 选择后端: CUDA {chosen[1]} ({arch})")
            return (chosen[2], chosen[3])

        candidates = main_assets.get(backend, [])
        if not candidates:
            return (None, None)
        chosen = sorted(candidates, key=lambda item: str(item.get("name", "")))[-1]
        if announce:
            print(f"   🎯 选择后端: {backend.upper()} ({arch})")
        return (chosen, None)

    @staticmethod
    def _version_tuple(value) -> tuple:
        match = re.search(r"(\d+)(?:\.(\d+))?", str(value or ""))
        return (int(match.group(1)), int(match.group(2) or 0)) if match else ()

    def _check_existing_backend(self) -> bool:
        """检查已有的 llama.cpp 是否匹配所需后端（CUDA/Vulkan/CPU）"""
        if not self.server_bin.exists():
            return False
        backend = getattr(self, "actual_backend", "cpu")
        if backend == "cuda":
            # rglob 搜索 CUDA DLL，支持解压到子目录的情况
            cuda_dlls = (
                list(LLAMA_DIR.rglob("cublas*.dll"))
                + list(LLAMA_DIR.rglob("cuda*.dll"))
                + list(LLAMA_DIR.rglob("ggml-cuda.dll"))
                + list(LLAMA_DIR.rglob("ggml_cuda.dll"))
            )
            return len(cuda_dlls) > 0
        elif backend == "vulkan":
            return len(list(LLAMA_DIR.rglob("*vulkan*"))) > 0
        return True

    def _backup_llama_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = BASE_DIR / f"llama.cpp.backup-{timestamp}"
        suffix = 1
        while backup_dir.exists():
            backup_dir = BASE_DIR / f"llama.cpp.backup-{timestamp}-{suffix}"
            suffix += 1
        print(f"   备份当前 llama.cpp 到 {backup_dir.name}")
        os.replace(LLAMA_DIR, backup_dir)
        self.llama_backup_dir = backup_dir
        return backup_dir

    def _restore_llama_backup(self, backup_dir: Path):
        if not backup_dir or not backup_dir.exists():
            raise RuntimeError(f"升级失败且备份目录不存在: {backup_dir}")
        print(f"   升级失败，正在回滚到 {backup_dir.name}")
        failed_dir = None
        if LLAMA_DIR.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            failed_dir = BASE_DIR / f"llama.cpp.failed-{timestamp}"
            suffix = 1
            while failed_dir.exists():
                failed_dir = BASE_DIR / f"llama.cpp.failed-{timestamp}-{suffix}"
                suffix += 1
            try:
                os.replace(LLAMA_DIR, failed_dir)
            except OSError as e:
                raise RuntimeError(
                    f"无法隔离失败的新引擎；旧备份仍保留在 {backup_dir}: {e}"
                ) from e
        try:
            os.replace(backup_dir, LLAMA_DIR)
        except OSError as e:
            if failed_dir and failed_dir.exists() and not LLAMA_DIR.exists():
                os.replace(failed_dir, LLAMA_DIR)
            raise RuntimeError(f"无法恢复旧引擎；备份仍保留在 {backup_dir}: {e}") from e
        if not LLAMA_DIR.exists() or backup_dir.exists():
            raise RuntimeError(f"旧引擎回滚验证失败；请手动恢复 {backup_dir}")
        self.llama_backup_dir = None
        if failed_dir and failed_dir.exists():
            try:
                shutil.rmtree(failed_dir)
            except OSError:
                print(f"   ⚠️  失败版本已隔离但无法清理: {failed_dir}")

    def _cleanup_llama_backup(self, backup_dir: Path):
        if backup_dir and backup_dir.exists():
            try:
                shutil.rmtree(backup_dir)
            except OSError as e:
                print(f"   ⚠️  新版已验证，但旧备份无法自动清理: {backup_dir} ({e})")
                return
        self.llama_backup_dir = None

    def _verify_llama_only(self):
        checks = {
            "llama-server": self.server_bin,
            "llama-cli": self.cli_bin,
        }
        all_ok = True
        for name, path in checks.items():
            if path.exists():
                self.p("verify_ok", f"{name} ({path})")
            else:
                self.p("verify_fail", name)
                all_ok = False
        if not all_ok:
            raise RuntimeError("llama.cpp 升级验证失败")
        versions = {}
        for name, binary in checks.items():
            try:
                result = subprocess.run(
                    [str(binary), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(binary.parent),
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as e:
                raise RuntimeError(f"{name} 无法运行: {e}") from e
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            if result.returncode != 0:
                raise RuntimeError(f"{name} --version 失败（退出码 {result.returncode}）: {output[-300:]}")
            build = self._parse_llama_build(output)
            if not build and self.target_llama_tag:
                raise RuntimeError(f"无法解析 {name} 版本: {output[-300:]}")
            versions[name] = build

        target = int(re.sub(r"\D", "", self.target_llama_tag) or "0")
        for name, got in versions.items():
            if target and got < target:
                raise RuntimeError(f"{name} 仍是 b{got}，低于目标 {self.target_llama_tag}")
        if versions["llama-server"]:
            print(f"   ✅ llama.cpp 当前版本: b{versions['llama-server']}")
        else:
            print("   ✅ llama.cpp 二进制运行验证通过（源码构建未提供 build number）")

    @staticmethod
    def _parse_llama_build(output: str) -> int:
        """兼容旧版 `version: 9934` 与新版 `build 10717` 输出。"""
        build_match = re.search(r"\bbuild\s+(\d+)\b", output or "", flags=re.I)
        if build_match:
            return int(build_match.group(1))
        legacy_match = re.search(r"version:\s*(\d+)\b", output or "", flags=re.I)
        return int(legacy_match.group(1)) if legacy_match else 0

    def _download_llama_source(self):
        """Linux/macOS/树莓派：克隆源码"""
        if not self.upgrade_llama and LLAMA_DIR.exists() and (LLAMA_DIR / "CMakeLists.txt").exists():
            self.p("llama_exists")
            return

        if not self.target_llama_tag:
            try:
                tags = self._release_tags_from_atom()
                if tags:
                    self.target_llama_tag = tags[0]
                    print(f"   🎯 目标版本: {self.target_llama_tag}")
            except Exception as e:
                print(f"   ⚠️  无法解析 nightly 标签，将使用 master: {e}")

        clone_ok = False
        clone_args = ["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1"]
        if self.target_llama_tag:
            clone_args.extend(["--branch", self.target_llama_tag])
        for url in self.dl.github_urls("ggml-org/llama.cpp.git"):
            if LLAMA_DIR.exists():
                shutil.rmtree(LLAMA_DIR, ignore_errors=True)
            try:
                self._exec(clone_args + [url, str(LLAMA_DIR)])
                clone_ok = True
                break
            except Exception as e:
                print(f"   ⚠️  git clone 失败 ({url}): {e}")

        if not clone_ok:
            print("   ⚠️  所有 git clone 地址均失败，尝试下载源码压缩包...")
            if LLAMA_DIR.exists():
                shutil.rmtree(LLAMA_DIR, ignore_errors=True)
            if self.target_llama_tag:
                archive_path = f"ggml-org/llama.cpp/archive/refs/tags/{self.target_llama_tag}.zip"
            else:
                archive_path = "ggml-org/llama.cpp/archive/refs/heads/master.zip"
            zip_path = BASE_DIR / "llama-src.zip"
            if not self.dl.download_from_urls(
                self.dl.github_urls(archive_path),
                zip_path,
                validator=zipfile.is_zipfile,
            ):
                raise RuntimeError("llama.cpp 源码压缩包下载失败")
            with zipfile.ZipFile(zip_path, "r") as archive:
                roots = {
                    Path(info.filename).parts[0]
                    for info in archive.infolist()
                    if info.filename and Path(info.filename).parts
                }
                if len(roots) != 1:
                    raise RuntimeError("llama.cpp 源码压缩包目录结构异常")
                archive.extractall(BASE_DIR)
            extracted_dir = BASE_DIR / roots.pop()
            extracted_dir.rename(LLAMA_DIR)
            zip_path.unlink(missing_ok=True)

        self.p("download_ok")

    # ----- 编译 llama.cpp -----
    def _build_llama(self):
        if self.sys.is_windows:
            if self.server_bin.exists():
                self.p("build_skip")
                return
            if not (LLAMA_DIR / "CMakeLists.txt").exists():
                self.p("build_skip")
                return

        # 检查是否已编译
        if self.server_bin.exists() and self.cli_bin.exists():
            self.p("build_skip")
            return

        self.p("building")

        if not shutil.which("cmake"):
            raise RuntimeError("缺少 cmake，无法从源码构建 llama.cpp")

        build_cfg = self.config.get("build", {})
        use_blas = build_cfg.get("use_openblas", True)
        jobs = build_cfg.get("jobs", 0) or self.sys.cpu_count
        backend = getattr(self, 'actual_backend', 'cpu')

        if backend == "cuda" and not (
            shutil.which("nvcc")
            or (os.environ.get("CUDA_PATH") and (Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc.exe").exists())
        ):
            print("   ⚠️  检测到 NVIDIA 驱动但未找到 CUDA Toolkit (nvcc)，源码构建回退 CPU")
            backend = "cpu"
            self.actual_backend = "cpu"
        if backend == "vulkan" and not shutil.which("glslc"):
            print("   ⚠️  未找到 Vulkan 编译器 glslc，源码构建回退 CPU")
            backend = "cpu"
            self.actual_backend = "cpu"

        cmake_args = ["-DCMAKE_BUILD_TYPE=Release"]
        target_build = int(re.sub(r"\D", "", self.target_llama_tag) or "0")
        if target_build:
            cmake_args += [
                f"-DLLAMA_BUILD_NUMBER={target_build}",
                f"-DLLAMA_BUILD_COMMIT={self.target_llama_tag}",
            ]

        if backend == "cuda":
            cmake_args.append("-DGGML_CUDA=ON")
            capability = self._version_tuple(self.gpu_info.get("compute_capability", ""))
            nvcc_path = shutil.which("nvcc")
            if not nvcc_path and os.environ.get("CUDA_PATH"):
                candidate = Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc.exe"
                nvcc_path = str(candidate) if candidate.exists() else None
            toolkit = ()
            if nvcc_path:
                try:
                    nvcc_result = subprocess.run(
                        [nvcc_path, "--version"], capture_output=True, timeout=10,
                        encoding="utf-8", errors="replace",
                    )
                    version_match = re.search(r"release\s+(\d+(?:\.\d+)?)", nvcc_result.stdout, re.I)
                    toolkit = self._version_tuple(version_match.group(1)) if version_match else ()
                except Exception:
                    pass
            if capability >= (12, 0):
                required = (12, 9) if capability >= (12, 1) else (12, 8)
                arch = "121a-real" if capability >= (12, 1) else "120a-real"
                if toolkit >= required:
                    cmake_args += [f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DGGML_NATIVE=ON"]
                    print(f"   🚀 Blackwell 原生 CUDA 架构: {arch} (Toolkit {toolkit[0]}.{toolkit[1]})")
                else:
                    print(
                        f"   ⚠️  Blackwell 原生构建需要 CUDA Toolkit {required[0]}.{required[1]}+；"
                        "当前将使用 CMake 的兼容架构设置"
                    )
            print(f"   🎯 编译后端: CUDA")
        elif backend == "vulkan":
            cmake_args.append("-DGGML_VULKAN=ON")
            print(f"   🎯 编译后端: Vulkan")
        elif use_blas and self.sys.is_linux:
            cmake_args.append("-DGGML_BLAS=ON")
            cmake_args.append("-DGGML_BLAS_VENDOR=OpenBLAS")
            print(f"   🎯 编译后端: CPU + OpenBLAS")

        build_dir = LLAMA_DIR / "build"
        self._exec(["cmake", "-B", str(build_dir)] + cmake_args, cwd=str(LLAMA_DIR))
        self._exec(
            ["cmake", "--build", str(build_dir), "--config", "Release", f"-j{jobs}"],
            cwd=str(LLAMA_DIR),
        )
        self._relocate_bins()

        if self.server_bin.exists():
            self.p("build_ok")
        else:
            raise RuntimeError(self.p.msgs["build_fail"])
    # ----- 下载模型 -----
    def _download_model(self):
        mc = self.config.get("model", {})
        shard_files = mc.get("shard_files", [])
        shard_count = mc.get("shard_count", 0)

        if shard_files and shard_count > 1:
            self._download_model_sharded(shard_files)
        else:
            self._download_model_single()

    def _file_complete(self, path: Path, expected_size: int = 0) -> bool:
        if not path or not path.exists() or path.suffix.lower() == ".part":
            return False
        size = path.stat().st_size
        if size <= 10000:
            return False
        if expected_size and expected_size > 0 and size != expected_size:
            print(
                f"   ⚠️  {path.name} 大小与市场记录不同，仍按已有模型处理 "
                f"({size // (1024 * 1024)}/{expected_size // (1024 * 1024)}MB)"
            )
        return True

    def _binary_complete(self, path: Path) -> bool:
        """llama.cpp Windows exe can be a small launcher; DLLs hold most code."""
        return bool(path and path.exists() and path.is_file() and path.stat().st_size > 1024)

    def _find_existing(self, filename: str, expected_size: int = 0) -> Path:
        """在 models/ 及子目录中搜索已存在的文件"""
        # 取纯文件名（去掉子目录前缀）
        pure_name = Path(filename).name

        # 1. 直接路径
        direct = MODELS_DIR / filename
        if self._file_complete(direct, expected_size):
            return direct

        # 2. 纯文件名在 models/ 根目录
        root = MODELS_DIR / pure_name
        if self._file_complete(root, expected_size):
            return root

        # 3. 递归搜索子目录
        for f in MODELS_DIR.rglob(pure_name):
            if f.is_file() and self._file_complete(f, expected_size):
                return f

        return None

    def _download_model_single(self):
        """下载单个模型文件"""
        mc = self.config.get("model", {})
        expected_size = int(mc.get("model_size", 0) or 0)
        # 先搜索是否已存在
        found = self._find_existing(self.model_file, expected_size)
        if found:
            size_mb = found.stat().st_size / (1024 * 1024)
            self.p("model_exists", found.name, size_mb)
            return

        # 不存在，下载到 models/ 根目录
        pure_name = Path(self.model_file).name
        dest = MODELS_DIR / pure_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        # HF 仓库中的文件路径可能包含子目录
        # 先试带子目录的路径，再试纯文件名
        urls_to_try = []
        if self.model_download_url:
            urls_to_try.append(self.model_download_url)
        if self.model_source == "huggingface" and self.model_repo_file:
            urls_to_try.extend(self.dl.hf_urls(self.repo_id, self.model_repo_file))
        if self.model_source == "huggingface":
            urls_to_try.extend(self.dl.hf_urls(self.repo_id, pure_name))
        urls_to_try = list(dict.fromkeys(urls_to_try))

        for url in urls_to_try:
            if self.dl.download_file(url, dest, expected_size):
                return

        raise RuntimeError(f"模型下载失败: {self.model_file}")

    def _download_model_sharded(self, shard_files: list):
        """下载分片模型"""
        mc = self.config.get("model", {})
        shard_sizes = mc.get("shard_file_sizes", {}) or {}
        shard_download_urls = mc.get("shard_download_urls", {}) or {}
        total = len(shard_files)
        downloaded = 0
        skipped = 0

        for i, shard_name in enumerate(sorted(shard_files)):
            pure_name = Path(shard_name).name
            expected_size = int(shard_sizes.get(shard_name, shard_sizes.get(pure_name, 0)) or 0)

            # 搜索是否已存在
            found = self._find_existing(shard_name, expected_size)
            if found:
                skipped += 1
                continue

            # 下载到 models/ 根目录
            dest = MODELS_DIR / pure_name
            print(f"   ⬇️  分片 {i+1}/{total}: {pure_name}")

            # 先试带子目录路径，再试纯文件名
            success = False
            urls_to_try = []
            direct_url = shard_download_urls.get(shard_name) or shard_download_urls.get(pure_name)
            if direct_url:
                urls_to_try.append(direct_url)
            if self.model_source == "huggingface" and ("/" in shard_name or "\\" in shard_name):
                urls_to_try.extend(self.dl.hf_urls(self.repo_id, shard_name))
            if self.model_source == "huggingface":
                urls_to_try.extend(self.dl.hf_urls(self.repo_id, pure_name))
            urls_to_try = list(dict.fromkeys(urls_to_try))

            for url in urls_to_try:
                if self.dl.download_file(url, dest, expected_size):
                    success = True
                    break

            if success:
                downloaded += 1
            else:
                raise RuntimeError(f"分片下载失败: {pure_name}")

        # 修正 config 中的 model_file 为第一个分片的纯文件名
        sorted_shards = sorted(shard_files)
        first_pure = Path(sorted_shards[0]).name
        mc = self.config.get("model", {})
        if mc.get("model_file") != first_pure:
            print(f"   🔄 修正 config: model_file → {first_pure}")
            import json as _json
            cfg = parse_jsonc(CONFIG_FILE)
            cfg.setdefault("model", {})["model_file"] = first_pure
            CONFIG_FILE.write_text(
                _json.dumps(cfg, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        if skipped == total:
            total_size = sum(
                f.stat().st_size for name in shard_files
                for f in [self._find_existing(name)] if f
            )
            print(f"   ✅ 所有 {total} 个分片已存在 ({total_size // (1024*1024)}MB)")
        else:
            print(f"   ✅ 分片完成: {downloaded} 新下载, {skipped} 已存在, 共 {total}")
    # ----- 下载视觉模块 -----
    def _persist_mmproj_pair(self):
        """把规范化配对名写回本机配置；远端原名仍保留用于再次下载。"""
        cfg = parse_jsonc(CONFIG_FILE)
        model_cfg = cfg.setdefault("model", {})
        bindings = model_cfg.setdefault("mmproj_bindings", {})
        changed = any((
            model_cfg.get("model_file") != self.model_file,
            model_cfg.get("mmproj_file") != self.mmproj_file,
            bindings.get(self.model_file) != self.mmproj_file,
        ))
        model_cfg["model_file"] = self.model_file
        model_cfg["mmproj_file"] = self.mmproj_file
        if self.mmproj_repo_file:
            model_cfg["mmproj_repo_file"] = self.mmproj_repo_file
        bindings[self.model_file] = self.mmproj_file
        if changed:
            CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   🔗 已按主模型规范命名: {self.mmproj_file}")

    def _download_mmproj(self):
        if not self.mmproj_file:
            self.p("mmproj_skip")
            return
        mc = self.config.get("model", {})
        expected_size = int(mc.get("mmproj_size", 0) or 0)

        if self._find_mmproj_file():
            size_mb = self.mmproj_path.stat().st_size / (1024 * 1024)
            if size_mb > 1:
                self._persist_mmproj_pair()
                self.p("mmproj_exists", self.mmproj_file, size_mb)
                return

        # 先尝试直接下载（最快）
        print("   ⬇️  尝试直接下载视觉模块...")
        direct_urls = []
        if self.mmproj_download_url:
            direct_urls.append(self.mmproj_download_url)
        if self.model_source == "huggingface":
            if self.mmproj_repo_file:
                direct_urls.extend(self.dl.hf_urls(self.repo_id, self.mmproj_repo_file))
            direct_urls.extend(self.dl.hf_urls(self.repo_id, self.mmproj_file))
        direct_urls = list(dict.fromkeys(direct_urls))

        for url in direct_urls:
            self.mmproj_dir.mkdir(parents=True, exist_ok=True)
            if self.dl.download_file(url, self.mmproj_path, expected_size):
                size = self.mmproj_path.stat().st_size
                if size > 10000:
                    size_mb = size / (1024 * 1024)
                    self._persist_mmproj_pair()
                    self.p("mmproj_exists", self.mmproj_file, size_mb)
                    return
                else:
                    print(f"   ⚠️  文件太小（{size}字节），可能是 Xet 指针文件")
                    self.mmproj_path.unlink(missing_ok=True)

        if self.model_source != "huggingface":
            raise RuntimeError(
                f"视觉模块下载失败: {self.mmproj_file}\n"
                f"下载源 {self.model_source} 未提供可用直链，请在模型管理页重新选择该文件。"
            )

        # Hugging Face 直链失败时，用 Xet 方式
        print("   📦 直接下载不可用，切换到 Xet 协议...")
        self._download_mmproj_xet()
    def _download_mmproj_xet(self):
        """通过 huggingface_hub（Xet 协议）下载 mmproj"""
        self.p("xet_install")

        # 确定虚拟环境中的 Python 路径
        if self.sys.is_windows:
            venv_python = VENV_DIR / "Scripts" / "python.exe"
        else:
            venv_python = VENV_DIR / "bin" / "python3"

        # 创建虚拟环境（不存在时）
        if not venv_python.exists():
            self._exec([sys.executable, "-m", "venv", str(VENV_DIR)])

        # 升级 pip 并安装 huggingface_hub（含 hf_xet 扩展）
        try:
            self._exec([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        except RuntimeError:
            pass   # pip 升级失败不阻断后续

        self._exec([str(venv_python), "-m", "pip", "install", "huggingface_hub[hf_xet]"])

        self.mmproj_dir.mkdir(parents=True, exist_ok=True)
        # snapshot_download 要求 POSIX 风格路径（Windows 也接受 /）
        mmproj_dir_posix = self.mmproj_dir.as_posix()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if self.sys.is_windows:
            env["PYTHONUTF8"] = "1"

        # 按优先级尝试：镜像 -> 直连 HuggingFace
        download_attempts = []
        if self.dl.hf_mirror:
            download_attempts.append(("镜像 " + self.dl.hf_mirror, self.dl.hf_mirror))
        download_attempts.append(("直连 HuggingFace", ""))

        max_seconds = 1800   # 30 分钟超时

        for attempt_name, hf_endpoint in download_attempts:
            print(f"   ⬇️  尝试: {attempt_name} ...")

            # 构建内联下载脚本：设置端点 -> snapshot_download -> 打印 OK
            repo_json = json.dumps(self.repo_id)
            local_dir_json = json.dumps(mmproj_dir_posix)
            remote_file_json = json.dumps(self.mmproj_repo_file or self.mmproj_file)
            endpoint_json = json.dumps(hf_endpoint)
            download_script = "".join([
                "import os, sys\n",
                f'os.environ["HF_ENDPOINT"] = {endpoint_json}\n' if hf_endpoint else "",
                'os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"\n',
                "from huggingface_hub import snapshot_download\n",
                "snapshot_download(\n",
                f"    repo_id={repo_json},\n",
                f"    local_dir={local_dir_json},\n",
                f"    allow_patterns=[{remote_file_json}],\n",
                ")\n",
                'print("DOWNLOAD_OK")\n',
            ])

            try:
                proc = subprocess.Popen(
                    [str(venv_python), "-c", download_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                output_lines = []
                import time as _time
                start_time = _time.time()

                # readline 是阻塞调用，在单独循环里读取
                # 超时通过检查已用时间 + proc.poll() 组合判断
                while True:
                    if _time.time() - start_time > max_seconds:
                        proc.kill()
                        raise TimeoutError(f"下载超时（超过 {max_seconds // 60} 分钟）")

                    raw = proc.stdout.readline()
                    if not raw:
                        # readline 返回空 bytes 表示 EOF（子进程已结束）
                        if proc.poll() is not None:
                            break
                        # 子进程还活着但暂时无输出，稍候重试
                        _time.sleep(0.1)
                        continue

                    line = raw.decode("utf-8", errors="replace").rstrip()
                    output_lines.append(line)
                    # 打印关键进度信息（百分比、下载速度、错误）
                    if any(k in line for k in ("%", "Download", "Fetch", "OK", "Error", "error", "warning")):
                        print(f"   {line}")

                proc.wait()

                if proc.returncode == 0 and any("DOWNLOAD_OK" in l for l in output_lines):
                    if self._find_mmproj_file():
                        size_mb = self.mmproj_path.stat().st_size / (1024 * 1024)
                        self._persist_mmproj_pair()
                        self.p("mmproj_exists", self.mmproj_file, size_mb)
                        return
                    else:
                        print(f"   ⚠️  进程成功但未找到文件，尝试下一种方式...")
                else:
                    tail = "\n".join(output_lines[-5:])
                    print(f"   ⚠️  {attempt_name} 失败:\n   {tail[:300]}")

            except TimeoutError as e:
                print(f"   ⚠️  {attempt_name} 超时: {e}")
            except Exception as e:
                print(f"   ⚠️  {attempt_name} 异常: {e}")

        # ── 所有 Xet 方式失败，最后尝试直接 URL 下载 ─────────────────────────
        print("   ⬇️  所有 Xet 方式失败，尝试直接 URL 下载...")
        direct_urls = []
        if self.mmproj_download_url:
            direct_urls.append(self.mmproj_download_url)
        if self.mmproj_repo_file:
            direct_urls.extend(self.dl.hf_urls(self.repo_id, self.mmproj_repo_file))
        direct_urls.extend(self.dl.hf_urls(self.repo_id, self.mmproj_file))
        direct_urls = list(dict.fromkeys(direct_urls))

        for url in direct_urls:
            print(f"   ⬇️  尝试: {url}")
            expected_size = int(self.config.get("model", {}).get("mmproj_size", 0) or 0)
            if self.dl.download_file(url, self.mmproj_path, expected_size):
                size = self.mmproj_path.stat().st_size
                if size > 10_000:
                    size_mb = size / (1024 * 1024)
                    self._persist_mmproj_pair()
                    self.p("mmproj_exists", self.mmproj_file, size_mb)
                    return
                else:
                    print(f"   ⚠️  文件太小（{size} 字节），是 Xet 指针文件")
                    self.mmproj_path.unlink(missing_ok=True)

        raise RuntimeError(
            "所有下载方式均失败。\n"
            "请手动下载视觉模块后放入 models/vision/ 目录，\n"
            "或在 config 中清空 mmproj_file 跳过视觉功能。\n"
            f"手动下载地址: https://huggingface.co/{self.repo_id}"
        )

    def _find_mmproj_file(self) -> bool:
        """按配置的远端原名精确定位 mmproj，避免多模型共存时误配。"""
        expected_size = int(self.config.get("model", {}).get("mmproj_size", 0) or 0)
        if self._file_complete(self.mmproj_path, expected_size):
            return True
        expected_names = {
            local_filename(self.mmproj_file).casefold(),
            local_filename(self.mmproj_repo_file).casefold(),
        } - {""}
        candidates = []
        if self.mmproj_repo_file:
            remote_rel = Path(*str(self.mmproj_repo_file).replace("\\", "/").split("/"))
            candidates.append(self.mmproj_dir / remote_rel)
        for root in (self.mmproj_dir, MODELS_DIR):
            if not root.exists():
                continue
            for f in root.rglob("*.gguf"):
                if f.name.casefold() in expected_names:
                    candidates.append(f)
        for f in dict.fromkeys(candidates):
            if self._file_complete(f, expected_size):
                if f.resolve() != self.mmproj_path.resolve():
                    self.mmproj_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, self.mmproj_path)
                return True
        return False


    # ----- 验证 -----
    def _verify(self):
        # 重新定位 bin，确保路径是最新的（避免 __init__ 时目录还不存在）
        self._relocate_bins()

        bin_checks = {
            "llama-server": self.server_bin,
            "llama-cli":    self.cli_bin,
        }
        file_checks = {}

        # 模型文件：智能搜索
        mc = self.config.get("model", {})
        model_found = self._find_existing(self.model_file, int(mc.get("model_size", 0) or 0))
        if model_found:
            file_checks[self.model_file] = model_found
        else:
            # 检查分片模型
            shard_files = mc.get("shard_files", [])
            shard_sizes = mc.get("shard_file_sizes", {}) or {}
            if shard_files:
                all_found = True
                for sf in shard_files:
                    pure = Path(sf).name
                    f = self._find_existing(sf, int(shard_sizes.get(sf, shard_sizes.get(pure, 0)) or 0))
                    if f:
                        file_checks[Path(sf).name] = f
                    else:
                        file_checks[Path(sf).name] = MODELS_DIR / sf  # 会显示缺失
                        all_found = False
            else:
                file_checks[self.model_file] = self.model_path

        # 视觉模块
        if self.mmproj_file:
            mmproj_found = None
            expected_size = int(self.config.get("model", {}).get("mmproj_size", 0) or 0)
            for search_dir in [MODELS_DIR / "vision", MODELS_DIR]:
                p = search_dir / self.mmproj_file
                if self._file_complete(p, expected_size):
                    mmproj_found = p
                    break
            if not mmproj_found:
                for f in MODELS_DIR.rglob(self.mmproj_file):
                    if self._file_complete(f, expected_size):
                        mmproj_found = f
                        break
            file_checks[self.mmproj_file] = mmproj_found or (MODELS_DIR / "vision" / self.mmproj_file)

        all_ok = True
        for name, path in bin_checks.items():
            if self._binary_complete(path):
                self.p("verify_ok", f"{name} ({path})")
            else:
                self.p("verify_fail", name)
                all_ok = False

        for name, path in file_checks.items():
            if path and self._file_complete(path):
                self.p("verify_ok", f"{name} ({path})")
            else:
                self.p("verify_fail", name)
                all_ok = False

        if not all_ok:
            raise RuntimeError("部署验证失败，请检查上述缺失项")
    # ----- 打印使用说明 -----
    def _print_usage(self):
        self.p("done")
        print()
        print("   使用方式:" if self.p.lang == "zh" else "   Usage:")
        print("   " + "─" * 45)
        self.p("usage_cli")
        self.p("usage_server")
        if self.mmproj_file:
            self.p("usage_server_v")
        self.p("usage_stop")
        if self.upgrade_llama:
            print("   升级引擎:  python deploy.py --upgrade-llama")
        print()

    # ----- 工具方法 -----
    def _exec(self, cmd, cwd=None):
        """
        执行系统命令。
        安全策略：优先使用列表模式（避免 shell 注入和空格路径问题）。
        如果 cmd 是字符串，用 shlex.split 分割（Windows 下用 posix=False）。
        """
        import shlex

        if self.verbose:
            print(f"   $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if self.sys.is_windows:
            env["PYTHONUTF8"] = "1"

        # 字符串命令 -> 列表（避免 shell=True 的空格/注入问题）
        if isinstance(cmd, str):
            try:
                cmd_list = shlex.split(cmd, posix=not self.sys.is_windows)
            except ValueError:
                # shlex 解析失败时退回 shell 模式（如含管道符的命令）
                result = subprocess.run(
                    cmd, shell=True, cwd=cwd, env=env,
                    stdout=None if self.verbose else subprocess.PIPE,
                    stderr=None if self.verbose else subprocess.PIPE,
                    encoding="utf-8", errors="replace",
                )
                if result.returncode != 0:
                    stderr = (result.stderr or "")[-400:] if not self.verbose else ""
                    raise RuntimeError(f"命令失败(shell模式): {cmd}\n{stderr}")
                return result
        else:
            cmd_list = cmd

        result = subprocess.run(
            cmd_list, cwd=cwd, env=env,
            stdout=None if self.verbose else subprocess.PIPE,
            stderr=None if self.verbose else subprocess.PIPE,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            # 非 verbose 模式下也输出 stderr，方便用户排错
            stderr = (result.stderr or "")[-400:]
            if not self.verbose and stderr:
                print(f"   STDERR: {stderr}")
            raise RuntimeError(
                f"命令失败: {cmd if isinstance(cmd, str) else ' '.join(cmd_list)}\n{stderr}"
            )
        return result


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    try:
        force_source = "--build-from-source" in sys.argv
        upgrade_llama = force_source or "--upgrade-llama" in sys.argv or "--update-llama" in sys.argv
        deployer = Deployer(upgrade_llama=upgrade_llama, force_source=force_source)
        deployer.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")
        sys.exit(1)
