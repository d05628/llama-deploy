"""Model-aware engine routing. Prism binaries never share stock llama.cpp DLLs."""
import json
import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path

PRISM_REPO = "PrismML-Eng/llama.cpp"
# The complete release pinned by the author's Bonsai-demo setup scripts.
PRISM_TAG = "prism-b10743-adfffbe"
ENGINE_INFO_NAME = ".llama-deploy.json"
PRISM_TYPES = {142: "PQ2_0", 143: "PTQ1_0"}


def _read(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("truncated GGUF header")
    return value


def _integer(stream, width):
    return int.from_bytes(_read(stream, width), "little")


def _string(stream, keep=False):
    length = _integer(stream, 8)
    if length > 64 * 1024 * 1024:
        raise ValueError("invalid GGUF string length")
    if keep:
        return _read(stream, length).decode("utf-8", errors="replace")
    stream.seek(length, 1)


def _skip(stream, kind):
    widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind in widths:
        stream.seek(widths[kind], 1)
    elif kind == 8:
        _string(stream)
    elif kind == 9:
        element, count = _integer(stream, 4), _integer(stream, 8)
        if count > 10_000_000 or element == 9:
            raise ValueError("invalid GGUF array")
        if element in widths:
            stream.seek(widths[element] * count, 1)
        else:
            for _ in range(count):
                _skip(stream, element)
    else:
        raise ValueError("invalid GGUF metadata type")


@lru_cache(maxsize=64)
def _tensor_types(path, size, modified):
    # Only read metadata/tensor descriptors, never mmap or read weight data.
    with open(path, "rb") as stream:
        if _read(stream, 4) != b"GGUF" or _integer(stream, 4) not in (2, 3):
            raise ValueError("invalid GGUF header")
        tensors, entries = _integer(stream, 8), _integer(stream, 8)
        if tensors > 1_000_000 or entries > 1_000_000:
            raise ValueError("invalid GGUF entry count")
        for _ in range(entries):
            _string(stream)
            _skip(stream, _integer(stream, 4))
        types = set()
        for _ in range(tensors):
            _string(stream)
            dimensions = _integer(stream, 4)
            if dimensions > 8:
                raise ValueError("invalid GGUF tensor dimensions")
            stream.seek(8 * dimensions, 1)
            types.add(_integer(stream, 4))
            _read(stream, 8)
        if stream.tell() > size:
            raise ValueError("truncated GGUF header")
        return frozenset(types)


def prism_quant(model):
    path = Path(model)
    if path.is_file():
        stat = path.stat()
        try:
            found = _tensor_types(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            special = sorted(found.intersection(PRISM_TYPES))
            if special:
                return "/".join(PRISM_TYPES[kind] for kind in special)
        except (OSError, ValueError, struct.error):
            pass
    # Covers remote selections before weights are downloaded, and failed headers.
    match = re.search(r"(?:^|[-_.])(PQ2_0|PTQ1_0)(?:[-_.]|$)", path.name, re.I)
    return match.group(1).upper() if match else ""


def provider_for(cfg, model):
    provider = str(cfg.get("engine", {}).get("provider", "auto")).lower()
    if provider not in ("auto", "upstream", "prism"):
        raise RuntimeError("engine.provider 必须是 auto、upstream 或 prism")
    quant = prism_quant(model)
    if quant and provider == "upstream":
        raise RuntimeError(
            f"unsupported: {quant} (GGML type 142/143) 需要 Prism llama.cpp 专用后端；"
            "当前选择 upstream。请设 engine.provider=auto/prism 并运行 python deploy.py --install-prism。"
        )
    return "prism" if quant or provider == "prism" else "upstream"


def engine_info(root):
    try:
        value = json.loads((Path(root) / ENGINE_INFO_NAME).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_file_hashes(root):
    return {path.name: file_sha256(path) for path in sorted(Path(root).iterdir())
            if path.is_file() and path.suffix.lower() == ".dll"}


def executable_file_hashes(root):
    return {path.name: file_sha256(path) for path in sorted(Path(root).iterdir())
            if path.is_file() and path.suffix.lower() == ".exe"}


def verify_prism_integrity(root, info=None, *, allow_legacy=False):
    """Verify every registered hash; only explicit registration can add fields."""
    root = Path(root)
    info = engine_info(root) if info is None else info
    binary = prism_binary(root, "llama-server")
    if (not binary or binary.parent != root
            or not re.fullmatch(r"[a-f0-9]{64}", str(info.get("binary_sha256", "")))):
        raise RuntimeError("Prism 完整性登记缺失，请对刚验证的安装运行 --register-prism-manifest")
    if file_sha256(binary) != info["binary_sha256"]:
        raise RuntimeError("Prism binary SHA256 不匹配，拒绝启动")
    for key, label, inventory in (
        ("runtime_files_sha256", "DLL", runtime_file_hashes),
        ("executable_files_sha256", "EXE", executable_file_hashes),
    ):
        if key not in info:
            if allow_legacy:
                continue
            raise RuntimeError(f"Prism {label} 完整性登记缺失，请对刚验证的安装运行 --register-prism-manifest")
        expected = info[key]
        if (not isinstance(expected, dict) or not expected
                or any(not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in expected.values())):
            raise RuntimeError(f"Prism {label} 哈希登记无效")
        if inventory(root) != expected:
            raise RuntimeError(f"Prism {label} 清单或 SHA256 不匹配，可能混入其他引擎文件；拒绝启动")
    return True


def prism_binary(root, name, windows=True):
    exe = name + (".exe" if windows else "")
    for relative in (exe, "bin/" + exe, "build/bin/" + exe, "build/bin/Release/" + exe):
        candidate = Path(root) / relative
        if candidate.is_file():
            return candidate
    return None


def select_prism_binary(base_dir, cfg, model, name, backend, windows=True):
    if not windows:
        raise RuntimeError("unsupported: 当前 Prism 安装和完整性登记仅支持 Windows；其它平台请使用作者独立安装说明。")
    quant = prism_quant(model)
    configured = str(cfg.get("engine", {}).get("directory", "") or "")
    if configured:
        root = Path(configured)
        roots = [root if root.is_absolute() else Path(base_dir) / root]
    else:
        parent = Path(base_dir) / "engines" / "prism" / PRISM_TAG
        roots = sorted(parent.iterdir()) if parent.exists() else []
    for root in roots:
        info = engine_info(root)
        if (info.get("provider") != "prism" or info.get("backend") != backend
                or not {"PQ2_0", "PTQ1_0"}.issubset(info.get("quant_types", []))):
            continue
        binary = prism_binary(root, name, windows)
        verify_prism_integrity(root, info)
        if binary:
            if binary.parent != root or binary.name not in info["executable_files_sha256"]:
                raise RuntimeError("Prism EXE 不在已登记的平铺发布目录内，拒绝启动")
            return binary
    raise RuntimeError(
        f"unsupported: {quant or 'Prism'} 模型缺少已登记的 Prism {backend} 引擎 ({name})。"
        "请运行 python deploy.py --install-prism；不会自动回退到 stock llama.cpp 或 CPU。"
    )
