"""Opt-in launch proof: python tools/probe_multi_gpu.py --run [--model INSTALLED.gguf --modes auto force].

Defaults to installed Bonsai PQ2_0, single then dual GPU. Never reads config.jsonc.
Use --output NEW_DIRECTORY to retain logs/results; default is a fresh ignored tmp/ folder.
"""
import argparse
import csv
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import manager
import run

MODEL = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
PORT = 18199
PUBLIC_FLAGS = ("--device", "--split-mode", "--main-gpu", "--tensor-split", "--ctx-size",
                "--fit-target", "-ngl", "-np", "--cache-type-k", "--cache-type-v")


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("probe reached its shared 300-second time limit")
    return value


def require_free_port():
    with socket.socket() as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", PORT))


def validate_model_name(name):
    if (not name.lower().endswith(".gguf") or any(char in name for char in '/\\:*?[]"<>|')
            or any(ord(char) < 32 for char in name)):
        raise ValueError("--model must be an installed GGUF basename, without paths or wildcards")
    root = run.MODELS_DIR.resolve()
    if not any(path.is_file() and path.name == name and path.resolve().is_relative_to(root)
               for path in run.MODELS_DIR.rglob(name)):
        raise ValueError("--model was not found inside the installed models directory")
    return name


def build_runtime(mode, ctx, model=MODEL):
    cfg = manager.default_config()
    cfg["model"]["model_file"] = model
    cfg["server"].update(host="127.0.0.1", port=PORT, ctx_size=ctx, enable_thinking=False)
    cfg["gpu"].update(backend="cuda", gpu_layers=-1, multi_gpu=mode, split_mode="layer", reserve_mb=1024)
    cfg["performance"].update(profile="balanced", parallel=1, fit_target_mb=1024,
                              cache_type_k="q8_0", cache_type_v="q8_0", spec_type="off")
    return run.runtime(cfg, "server")


def gpu_sample(phase, deadline):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            timeout=min(5, remaining(deadline)))
    return {"phase": phase, "time": time.time(), "devices": [dict(zip(
        ("index", "uuid", "used_mib", "free_mib", "utilization_percent"), map(str.strip, row)))
        for row in csv.reader(result.stdout.splitlines()) if len(row) == 5], "returncode": result.returncode}


def request(url, key, deadline, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + key,
                                                       "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=remaining(deadline))


def stream_probe(key, deadline, prompt, expected, max_tokens=64):
    started, first, answer, timings, provider_model = time.monotonic(), None, "", {}, None
    payload = {"messages": [{"role": "user", "content": prompt}], "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    with request(f"http://127.0.0.1:{PORT}/v1/chat/completions", key, deadline, payload) as response:
        for line in response:
            remaining(deadline)
            if not line.startswith(b"data:"):
                continue
            if line[5:].strip() == b"[DONE]":
                break
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError("provider returned a stream error")
            provider_model = event.get("model", provider_model)
            timings.update(event.get("timings") or {})
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content and first is None:
                    first = time.monotonic()
                answer += content
    return {"prompt": prompt, "answer": answer, "semantic_ok": bool(re.search(expected, answer, re.I)),
            "ttft_seconds": first - started if first is not None else None,
            "elapsed_seconds": time.monotonic() - started, "provider_model": provider_model,
            "provider_tps": timings.get("predicted_per_second"), "provider_timings": timings}


def stop_owned(process):
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            return  # The owned process exited between poll and terminate.
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def kill_owned(process):
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass  # The same owned process already exited.


def actual_contexts(text, args):
    values = {int(value) for value in re.findall(r"\bn_ctx\s*=\s*(\d+)", text)}
    single = any(flag in args and args[args.index(flag) + 1] == "1" for flag in ("-np", "--parallel"))
    if single:
        values.update(int(ctx) for slots, ctx in re.findall(
            r"\binitializing,\s*n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+)", text) if slots == "1")
    return sorted(values)


def mode_matches(mode, flags):
    devices = flags.get("--device", "").split(",")
    if len(set(devices)) != len(devices) or not all(re.fullmatch(r"CUDA\d+", name) for name in devices):
        return False
    single = len(devices) == 1 and flags.get("--split-mode") == "none"
    multi = len(devices) >= 2 and flags.get("--split-mode") == "layer" and bool(flags.get("--tensor-split"))
    return (mode == "off" and single) or (mode == "force" and multi) or (mode == "auto" and (single or multi))


def probe_mode(mode, ctx, folder, deadline, model=MODEL):
    remaining(deadline)
    require_free_port()
    rt = build_runtime(mode, ctx, model)
    args, key = list(rt["args"]), secrets.token_urlsafe(32)
    help_text = run.help_text(rt["binary"])
    if "--api-key" in args or "--api-key-file" in args or "LLAMA_API_KEY" not in help_text:
        raise RuntimeError("probe requires supported private environment authentication")
    logging_flags = ["--verbose"] if "--verbose" in help_text else (["--log-verbosity", "5"] if "--log-verbosity" in help_text else [])
    if not logging_flags:
        raise RuntimeError("probe requires supported verbose logging for actual model-offload evidence")
    args += logging_flags  # Probe diagnostics only; never change production config/runtime defaults.
    env = {name: value for name, value in os.environ.items() if not name.startswith("LLAMA_ARG_") and name != "LLAMA_API_KEY"}
    env["LLAMA_API_KEY"] = key  # Never in CLI, JSON results, or shared settings.
    result = {"mode": mode, "model_file": model, "requested_ctx": ctx,
              "probe_logging_flags": logging_flags,
              "generated_flags": {flag: args[args.index(flag) + 1] for flag in PUBLIC_FLAGS if flag in args},
              "gpu_samples": [gpu_sample("before_load", deadline)], "requests": []}
    log_path = folder / (mode + ".server.log")
    with log_path.open("xb") as log:
        process = subprocess.Popen(args, cwd=rt["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
        watchdog = threading.Timer(max(0, deadline - time.monotonic()), lambda: kill_owned(process))
        watchdog.daemon = True
        try:
            watchdog.start()
            while True:
                remaining(deadline)
                if process.poll() is not None:
                    raise RuntimeError("owned server exited before readiness; inspect its probe log")
                try:
                    with request(f"http://127.0.0.1:{PORT}/health", key, deadline) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(min(0.25, remaining(deadline)))
            result["gpu_samples"].append(gpu_sample("loaded", deadline))
            result["warmup"] = stream_probe(key, deadline, "只回复一个汉字：好", "好", 16)
            for prompt, expected in (("只输出17+25的结果，禁止解释。", r"\b42\b"),
                                     ("地球围绕哪颗恒星公转？用一句中文回答。", "太阳|sun")):
                result["requests"].append(stream_probe(key, deadline, prompt, expected))
                result["gpu_samples"].append(gpu_sample("after_request", deadline))
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc).replace(key, "[redacted]")
        finally:
            watchdog.cancel()
            stop_owned(process)
    text = log_path.read_text(encoding="utf-8", errors="replace").replace(key, "[redacted]")
    log_path.write_text(text, encoding="utf-8")  # Only this process's fresh owned log.
    result["actual_ctx"] = actual_contexts(text, args)
    result["context_evidence"] = [line for line in text.splitlines()
                                   if re.search(r"\bn_ctx\s*=|\binitializing,\s*n_slots\s*=", line)]
    result["offload_evidence"] = [line for line in text.splitlines()
                                   if re.search(r"offloaded \d+/\d+ layers|(?:model|KV) buffer size", line)]
    result["gpu_buffer_devices"] = sorted(set(re.findall(r"\b(CUDA\d+)\b[^\n]*(?:model|KV) buffer size", text)))
    result["gpu_model_buffers_mib"] = {name: float(size) for name, size in re.findall(
        r"\b(CUDA\d+)\b[^\n]*?model buffer size\s*=\s*([\d.]+)\s*MiB", text)}
    counts = re.findall(r"offloaded (\d+)/(\d+) layers", text)
    result["all_layers_on_gpu"] = bool(counts and counts[-1][0] == counts[-1][1])
    selected = result["generated_flags"].get("--device", "").split(",")
    result["generated_mode_verified"] = mode_matches(mode, result["generated_flags"])
    result["gpu_offload_verified"] = result["generated_mode_verified"] and result["all_layers_on_gpu"] and all(
        result["gpu_model_buffers_mib"].get(name, 0) > 0 for name in selected)
    rates = re.findall(r"(?<!prompt )eval time\s*=.*?([\d.]+) tokens per second", text)
    for item, rate in zip([result.get("warmup", {})] + result["requests"], rates):
        if item.get("provider_tps") is None:
            item["provider_tps"] = float(rate)
    result["provider_tps_verified"] = len(result["requests"]) == 2 and all(
        isinstance(item.get("provider_tps"), (int, float)) and math.isfinite(item["provider_tps"]) and item["provider_tps"] > 0
        for item in result["requests"])
    result["context_verified"] = result["actual_ctx"] == [ctx]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly load GPU models and run requests")
    parser.add_argument("--ctx", type=int, choices=(16384, 32768, 65536), default=32768)
    parser.add_argument("--model", default=MODEL, help="installed GGUF basename (no path or wildcard)")
    parser.add_argument("--modes", nargs="+", choices=("off", "auto", "force"), default=["off", "force"])
    parser.add_argument("--output", type=Path, help="fresh output directory; must not already exist")
    options = parser.parse_args(argv)
    if not options.run:
        parser.print_help()
        return 0
    if len(set(options.modes)) != len(options.modes):
        parser.error("--modes must not contain duplicates")
    try:
        model = validate_model_name(options.model)
    except ValueError as exc:
        parser.error(str(exc))
    deadline = time.monotonic() + 297  # Reserve three seconds for owned-process cleanup.
    if options.output:
        options.output.mkdir(parents=True, exist_ok=False)
        folder = options.output
    else:
        parent = run.BASE_DIR / "tmp"
        parent.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="multi-gpu-probe-", dir=parent))
    results = []
    try:
        for mode in options.modes:
            results.append(probe_mode(mode, options.ctx, folder, deadline, model))
    except Exception as exc:
        results.append({"error": type(exc).__name__ + ": " + str(exc)})
    with (folder / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(str(folder.resolve()))
    return 0 if len(results) == len(options.modes) and all(not item.get("error") and item.get("context_verified")
               and item.get("gpu_offload_verified") and item.get("provider_tps_verified")
               and all(req["semantic_ok"] for req in item["requests"]) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
