"""Strata (github.com/Niko1221/Strata, MIT) as an alternative engine for Qwen3.8 Flash-Next.

Measured on RTX 5060 Ti + RTX 3060 Ti (PCIe 3.0 x4), Flash-Next IQ3_XXS, 64K context:
llama.cpp 24.2 t/s decode / 227 t/s prefill, Strata 56.6 / 491. Strata's helper card owns a
share of the experts and computes them in place, so only small activations cross the slow
slot, and its MTP drafting is tuned for this model.

Everything Strata writes stays under engines/: the launcher runs with APPDATA and LOCALAPPDATA
pointed at engines/strata-home, and prepared packs and the draft layer go to models/strata.
The model files themselves are used where they are.
"""
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONTROL_DIR = BASE_DIR / "engines" / "strata"
HOME_DIR = BASE_DIR / "engines" / "strata-home"
INSTALL_DIR = BASE_DIR / "engines" / "strata-install"
DOWNLOAD_DIR = BASE_DIR / "engines" / "strata-dl"
DATA_DIR = BASE_DIR / "models" / "strata"

# Pinned so the archive is checked against a hash in this repository, not one fetched beside it.
RELEASE = "v0.2.6"
CONTROL_ZIP = "strata-control-0.1.51-windows-x86_64.zip"
CONTROL_SHA256 = "9efdd6139e0eaab90fc4d8e431ac149ccda5aa3e4ca7e27549489699425094f5"
DEFAULT_CONTEXT = 65536
# A helper card needs room for its expert share beyond Strata's own 512 MiB allowance.
HELPER_MIN_FREE_MB = 2048


def available() -> bool:
    """Only the Windows terminal archive is pinned and tested here."""
    return os.name == "nt"


def installed() -> bool:
    return (CONTROL_DIR / "strata.py").exists() and (CONTROL_DIR / "python" / "python.exe").exists()


def environment(cfg: dict) -> dict:
    env = os.environ.copy()
    env.update(APPDATA=str(HOME_DIR / "Roaming"), LOCALAPPDATA=str(HOME_DIR / "Local"),
               STRATA_INSTALL_DIR=str(INSTALL_DIR), STRATA_NO_PROMPT="1",
               PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    mirror = str((cfg.get("download") or {}).get("hf_mirror") or "").strip().rstrip("/")
    if mirror:
        env["HF_ENDPOINT"] = mirror
    return env


def command(*args) -> list:
    return [str(CONTROL_DIR / "python" / "python.exe"), str(CONTROL_DIR / "strata.py"), *map(str, args)]


def install(cfg: dict, say=print) -> None:
    """Fetch the pinned terminal archive, check its SHA-256 and unpack it into engines/strata."""
    if not available():
        raise RuntimeError("Strata 目前只在 Windows 上接入；其他系统请使用 llama.cpp 引擎")
    if installed():
        return
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOAD_DIR / CONTROL_ZIP
    url = f"https://github.com/Niko1221/Strata/releases/download/{RELEASE}/{CONTROL_ZIP}"
    mirror = str((cfg.get("download") or {}).get("github_mirror") or "").strip().rstrip("/")
    if mirror:
        url = f"{mirror}/{url}"
    if not archive.exists() or _sha256(archive) != CONTROL_SHA256:
        say(f"正在下载 Strata {RELEASE}（约 70MB）...")
        part = archive.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        os.replace(part, archive)
    if _sha256(archive) != CONTROL_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError("Strata 安装包的 SHA-256 与固定值不符，已删除；请检查网络或镜像后重试")
    staging = CONTROL_DIR.with_name(CONTROL_DIR.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    shutil.rmtree(CONTROL_DIR, ignore_errors=True)
    os.replace(staging, CONTROL_DIR)
    say("Strata 已安装；首次启动会再下载引擎（约 580MB）和草稿层（约 5GB）")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_roles(gpus: list) -> str:
    """Main = the card with the most free memory, helpers = the others that have room (nvidia-smi numbering)."""
    usable = [g for g in gpus if g.get("index") is not None]
    if len(usable) < 2:
        return ""
    main = max(usable, key=lambda g: g.get("vram_free_mb", 0))
    roles = [f"{main['index']}=main"]
    for g in usable:
        if g is not main:
            role = "helper" if g.get("vram_free_mb", 0) >= HELPER_MIN_FREE_MB else "off"
            roles.append(f"{g['index']}={role}")
    return ",".join(roles)


def launch_args(cfg: dict, model: Path, vision: bool, gpus: list) -> list:
    sc = cfg.get("server") or {}
    try:
        context = int(sc.get("ctx_size") or 0) or DEFAULT_CONTEXT
    except (TypeError, ValueError):
        context = DEFAULT_CONTEXT
    # Strata's CPU image encoder crashes on a Ryzen 5600 (0xC0000005); the GPU one starts in ~3 s.
    vision_mode = "gpu" if vision else "off"
    args = ["launch", "--model-path", model, "--data-dir", DATA_DIR, "--context", context,
            "--vision", vision_mode, "--tune", "skip", "--access", "local",
            "--port", int(sc.get("port") or 8080), "--no-tui", "--yes"]
    roles = gpu_roles(gpus)
    if roles:
        args += ["--gpu-roles", roles]
    return command(*args)


def status(port: int, timeout: float = 2.0) -> dict:
    """Strata's /health, or {} when nothing (or something other than Strata) answers on the port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    return data if data.get("service") == "strata" else {}


def stop(cfg: dict, timeout: int = 240) -> str:
    if not installed():
        return ""
    r = subprocess.run(command("launch", "--stop"), env=environment(cfg), cwd=str(CONTROL_DIR),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return (r.stdout or r.stderr or "").strip()


def launcher_running(pid: int) -> bool:
    """The `strata launch` process stays alive while it downloads, prepares and loads, then exits once serving."""
    if not pid:
        return False
    if os.name == "nt":
        # Check the command line, not just "python": a reused PID must never make stop() kill another program.
        query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"
        r = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        return str(CONTROL_DIR / "strata.py").lower() in (r.stdout or "").lower()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
