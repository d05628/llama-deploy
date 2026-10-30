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
    if not archive.exists() or _sha256(archive) != CONTROL_SHA256:
        say(f"正在下载 Strata {RELEASE}（约 70MB）...")
        part = archive.with_suffix(".part")
        url = f"https://github.com/Niko1221/Strata/releases/download/{RELEASE}/{CONTROL_ZIP}"
        for candidate in github_urls(cfg, url):
            try:
                with urllib.request.urlopen(candidate, timeout=120) as resp, open(part, "wb") as out:
                    shutil.copyfileobj(resp, out, 1 << 20)
            except OSError as exc:
                say(f"⚠️ 下载失败（{exc}），换下一个地址")
                continue
            if _sha256(part) == CONTROL_SHA256:
                break
            say("⚠️ 下载内容的 SHA-256 不符，换下一个地址")
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


RELEASE_FILES = "https://github.com/Niko1221/Strata/releases/latest/download/"
FEED_URL = RELEASE_FILES + "strata-feed.json"
FEED_UNREADABLE = "update feed could not be read"
FEED_DIR = BASE_DIR / "engines" / "strata-feed"


# Used only for Strata, whose feed is signed and whose archive hash is pinned: GitHub drops TLS
# connections from here for minutes at a time while gh-proxy keeps answering. llama.cpp downloads
# are not hash-checked, so this list must not leak into download.github_mirror.
FALLBACK_MIRRORS = ("https://gh-proxy.com",)


def github_urls(cfg: dict, url: str) -> list:
    """The configured mirror, GitHub itself, then the fallback mirrors."""
    mirror = str((cfg.get("download") or {}).get("github_mirror") or "").strip().rstrip("/")
    prefixes = ([mirror] if mirror else []) + [""] + list(FALLBACK_MIRRORS)
    return list(dict.fromkeys(f"{m}/{url}" if m else url for m in prefixes))


def feed_urls(cfg: dict) -> list:
    return github_urls(cfg, FEED_URL)


def _curl() -> str:
    """curl.exe speaks TLS through Windows (Schannel). On some networks most handshakes from Python's
    OpenSSL are cut off (SSL UNEXPECTED_EOF) while Schannel ones get through, and Strata downloads
    with Python, so the release files are fetched with curl and handed to Strata as a local feed."""
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"
    return shutil.which("curl") or (str(system) if system.exists() else "")


def fetch_release_file(cfg: dict, name: str, dest: Path, sha256: str = "", say=print) -> bool:
    curl = _curl()
    if not curl:
        return False
    if sha256 and dest.exists() and _sha256(dest) == sha256:
        return True
    part = dest.with_name(dest.name + ".part")
    for url in github_urls(cfg, RELEASE_FILES + name):
        r = subprocess.run([curl, "-sSfL", "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
                            "--connect-timeout", "20", "-o", str(part), url],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            continue
        if sha256 and _sha256(part) != sha256:
            say(f"⚠️ {name} 的 SHA-256 与更新源不符，换下一个地址")
            continue
        os.replace(part, dest)
        return True
    part.unlink(missing_ok=True)
    return False


def _check(cfg: dict, source) -> dict:
    r = subprocess.run(command("update", "--check", "--feed", source, "--output", "json"),
                       env=environment(cfg), cwd=str(CONTROL_DIR), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=90)
    return json.loads((r.stdout or "").strip().splitlines()[-1])


def stage_feed(cfg: dict, packages: bool = False, say=print):
    """Copy the signed feed into engines/strata-feed, and with packages=True the files the update needs.
    Strata still verifies the signature and every hash itself. None when the files cannot be fetched."""
    FEED_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("strata-feed.json", "strata-feed.json.sig"):
        (FEED_DIR / name).unlink(missing_ok=True)  # never fall back to a stale copy
        if not fetch_release_file(cfg, name, FEED_DIR / name, say=say):
            return None
    if not packages:
        return FEED_DIR
    try:
        plan = _check(cfg, FEED_DIR)
        feed = json.loads((FEED_DIR / "strata-feed.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    if not plan.get("ok", True):
        return None
    wanted = []
    engine = plan.get("engine") or {}
    offered, installed = engine.get("offered") or {}, engine.get("installed") or {}
    if offered.get("url") and offered.get("version") != installed.get("version"):
        wanted.append((offered["url"], offered.get("sha256", "")))
    control = plan.get("control") or {}
    if control.get("offered") and control.get("offered") != control.get("installed"):
        platform = (plan.get("hardware") or {}).get("platform") or "windows-x86_64"
        pkg = next((p for p in (feed.get("control") or {}).get("packages", []) if p.get("platform") == platform), None)
        if pkg:
            wanted.append((pkg["url"], pkg.get("sha256", "")))
    for name, sha in wanted:
        say(f"正在下载 {name} ...")
        if not fetch_release_file(cfg, name, FEED_DIR / name, sha, say=say):
            say(f"⚠️ 无法下载 {name}")
            return None
    return FEED_DIR


def update(cfg: dict, attempts: int = 2, delay: float = 10, say=print) -> int:
    """Run `strata update` from a local copy of the feed, falling back to Strata's own download, and retry
    only when the feed could not be read (GitHub drops connections from here)."""
    import time
    rc = 1
    staged = stage_feed(cfg, packages=True, say=say)
    for url in ([str(staged)] if staged else []) + feed_urls(cfg):
        for attempt in range(1, attempts + 1):
            proc = subprocess.Popen(command("update", "--feed", url), env=environment(cfg), cwd=str(CONTROL_DIR),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            output = []
            for line in proc.stdout:
                output.append(line)
                say(line.rstrip())
            rc = proc.wait()
            if rc == 0 and staged:
                for leftover in list(FEED_DIR.glob("*.tar.gz")) + list(FEED_DIR.glob("*.zip")):
                    leftover.unlink(missing_ok=True)  # installed now; do not keep ~600MB around
            if rc == 0 or FEED_UNREADABLE not in "".join(output):
                return rc
            if attempt < attempts:
                say(f"⚠️ 读取更新源时网络中断，{delay:.0f} 秒后重试（{attempt}/{attempts}）")
                time.sleep(delay)
        say(f"⚠️ 无法通过 {url} 读取更新源")
    return rc


def update_check(cfg: dict) -> dict:
    """Installed and offered engine / control versions from Strata's signed update feed."""
    data = {"error": "无法读取 Strata 更新信息"}
    staged = stage_feed(cfg)
    for url in ([str(staged)] if staged else []) + feed_urls(cfg):
        try:
            data = _check(cfg, url)
        except Exception as exc:
            data = {"error": f"无法读取 Strata 更新信息: {exc}"}
            continue
        if data.get("ok", True):
            break
    if "error" in data and "engine" not in data:
        return {"error": str(data["error"])}
    if not data.get("ok", True):
        return {"error": str(data.get("error") or "Strata 更新检查失败")}
    engine, control = data.get("engine") or {}, data.get("control") or {}
    info = {"engine": (engine.get("installed") or {}).get("version"),
            "engine_offered": (engine.get("offered") or {}).get("version"),
            "engine_variant": (engine.get("installed") or {}).get("variant"),
            "control": control.get("installed"), "control_offered": control.get("offered")}
    info["update_available"] = any(
        offered and offered != current for current, offered in
        ((info["engine"], info["engine_offered"]), (info["control"], info["control_offered"])))
    return info


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


if __name__ == "__main__":
    # python strata_engine.py update  (the web manager's upgrade button runs this)
    import sys
    if sys.argv[1:] != ["update"]:
        sys.exit("用法: python strata_engine.py update")
    from run import CONFIG_FILE, parse_jsonc
    sys.stdout.reconfigure(line_buffering=True, errors="replace")
    sys.exit(update(parse_jsonc(CONFIG_FILE) or {}))
