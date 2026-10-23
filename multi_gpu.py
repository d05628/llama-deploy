"""Pure, per-device CUDA planning shared by the launcher and manager.

All capacities are MiB. NVIDIA indices identify inventory; llama device names
identify launch targets. They must be matched before generating arguments.
"""
import csv
import math
import re


def parse_nvidia_inventory(output):
    devices = []
    for row in csv.reader(output.splitlines()):
        row = [part.strip() for part in row]
        try:
            if len(row) >= 5 and row[0].isdigit() and row[1].startswith("GPU-"):
                # Names may contain commas even on drivers that omit CSV quotes.
                has_cap = len(row) >= 6 and row[-3].isdigit()
                cap = row[-1] if has_cap and re.fullmatch(r"\d+(?:\.\d+)?", row[-1]) else ""
                end = -3 if has_cap else -2
                total, free = row[-3:-1] if has_cap else row[-2:]
                name = ", ".join(row[2:end])
                index, uuid = int(row[0]), row[1]
            elif len(row) >= 3:
                name, total, free = ", ".join(row[:-2]), row[-2], row[-1]
                index, uuid, cap = len(devices), "", ""
            else:
                continue
            total_mb = int(total)
            if total_mb <= 0:
                continue
            known = bool(re.fullmatch(r"\d+", free))
            devices.append({"index": index, "uuid": uuid, "name": name,
                            "vram_mb": total_mb,
                            "vram_free_mb": min(total_mb, int(free)) if known else 0,
                            "free_known": known, "compute_capability": cap})
        except (ValueError, TypeError):
            continue
    return devices


def parse_engine_devices(output):
    found = []
    for line in output.splitlines():
        match = re.match(r"\s*(CUDA\d+):\s*(.*?)\s*\((\d+) MiB,\s*(\d+) MiB free\)", line)
        if match:
            found.append({"device": match[1], "name": match[2],
                          "vram_mb": int(match[3]), "vram_free_mb": int(match[4]),
                          "free_known": True})
    return found


def match_engine_devices(inventory, engine_devices):
    """Match by name/capacity, never equate nvidia-smi and CUDA ordinals.

    Identical cards are safe for automatic selection using engine free memory;
    their physical identity stays unknown so manual UUID/index choices fail
    explicitly rather than accidentally selecting another card.
    """
    result = []
    for device in engine_devices:
        matches = [item for item in inventory if item["name"] == device["name"]
                   and abs(item["vram_mb"] - device["vram_mb"]) <= 64]
        value = dict(device)
        if len(matches) == 1:
            value.update({key: matches[0].get(key) for key in
                          ("index", "uuid", "compute_capability")})
            # CUDA's allocatable free memory is often lower under WDDM.
            if matches[0].get("free_known", True):
                value["vram_free_mb"] = min(value["vram_free_mb"], matches[0]["vram_free_mb"])
        else:
            value.update(index=None, uuid="", compute_capability="")
        result.append(value)
    return result


def available_mb(device):
    if device.get("free_known", True):
        return max(0.0, min(float(device.get("vram_free_mb", 0)), float(device.get("vram_mb", 0))))
    # Unknown free memory is not a claim that the whole card is available.
    return 0.0


def choose_primary(devices, selector="auto"):
    if not devices:
        return None
    if str(selector).strip().lower() in ("", "auto"):
        return max(devices, key=lambda item: (available_mb(item), item.get("vram_mb", 0)))
    matches = [item for item in devices if str(selector) == str(item.get("index"))
               or str(selector) == item.get("uuid")]
    if len(matches) != 1:
        raise ValueError(f"gpu.main_gpu={selector} 无法唯一映射到当前可见 CUDA 设备；请检查 UUID、索引或 CUDA_VISIBLE_DEVICES")
    return matches[0]


def _weights(value, count):
    if value is None or value == "" or value == []:
        return None
    try:
        parts = value.split(",") if isinstance(value, str) else list(value)
        weights = [float(part) for part in parts]
    except (TypeError, ValueError):
        raise ValueError("gpu.tensor_split 必须是逗号分隔的非负有限数字") from None
    if len(weights) != count or any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0:
        raise ValueError(f"gpu.tensor_split 需要 {count} 个非负有限数字，且至少一个大于 0（按引擎设备顺序）")
    return weights


def configured_reserve(config, minimum=0):
    try:
        reserve = float(config.get("reserve_mb", 1024))
    except (TypeError, ValueError):
        raise ValueError("gpu.reserve_mb 必须是非负有限数字") from None
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("gpu.reserve_mb 必须是非负有限数字")
    return max(reserve, minimum)


def validate_row_devices(device_names, engine_device_names):
    # CUDA row-split callbacks index the complete backend device array. A
    # compact/reordered --device list cannot safely index compact split ratios.
    natural = [f"CUDA{index}" for index in range(len(engine_device_names))]
    if (len(engine_device_names) < 2 or engine_device_names != natural
            or device_names != engine_device_names):
        raise ValueError("row 多卡模式必须使用引擎完整的 CUDA 自然顺序（CUDA0,CUDA1,…），不能筛选或重排设备；请改用 layer 或释放所有显卡余量")


def build_plan(devices, config, weights_mb=0.0, kv_mb=0.0, extra_mb=0.0, reserve_mb=None):
    mode = str(config.get("multi_gpu", "auto")).lower()
    split_mode = str(config.get("split_mode", "layer")).lower()
    if mode not in ("auto", "off", "force"):
        raise ValueError("gpu.multi_gpu 必须是 auto、off 或 force")
    if split_mode not in ("layer", "row"):
        raise ValueError("gpu.split_mode 必须是 layer 或 row；tensor 模式不自动启用")
    reserve = configured_reserve(config, reserve_mb or 0)
    primary = choose_primary(devices, config.get("main_gpu", "auto"))
    plan = {"mode": mode, "use_multi_gpu": False, "split_mode": "none", "main_gpu": None,
            "devices": [], "device_names": [], "tensor_split": [], "notes": [],
            "reserve_mb": reserve, "payload_capacity_mb": 0.0}
    if primary is None:
        if mode == "force":
            raise ValueError("强制多卡需要至少两张可见 CUDA 显卡")
        return plan
    candidates = [item for item in devices if item is primary or available_mb(item) > reserve + 600 + 256]
    # An explicit split is validated even when auto ultimately stays on one GPU.
    custom = _weights(config.get("tensor_split", ""), len(devices))
    if custom and len(candidates) != len(devices):
        if any(value > 0 and item not in candidates for item, value in zip(devices, custom)):
            raise ValueError("gpu.tensor_split 为显存余量不足的设备分配了正权重，请增加空闲显存或将该设备比例设为 0")
        custom = [value for item, value in zip(devices, custom) if item in candidates]
    required = max(0.0, weights_mb) + max(0.0, kv_mb)
    primary_budget = max(0.0, available_mb(primary) - reserve - 600 - extra_mb)
    multi = mode == "force" or (mode == "auto" and weights_mb > 0 and required > primary_budget)
    if mode == "force" and len(candidates) < 2:
        raise ValueError("强制多卡需要至少两张有足够空闲显存的可见 CUDA 显卡")
    selected = candidates if multi and len(candidates) > 1 else [primary]
    multi = len(selected) > 1
    budgets = [max(0.0, available_mb(item) - reserve - 600 - (extra_mb if item is primary else 0))
               for item in selected]
    main_index = selected.index(primary)
    actual_mode = split_mode if multi else "none"
    engine_names = [item.get("device", "") for item in devices]
    selected_names = [item.get("device", str(item.get("index"))) for item in selected]
    if actual_mode == "row":
        validate_row_devices(selected_names, engine_names)
    # Row split retains all KV on the main GPU. Layer split distributes it.
    weight_budgets = list(budgets)
    if actual_mode == "row":
        weight_budgets[main_index] = max(0.0, weight_budgets[main_index] - kv_mb)
    proportions = custom if multi and custom else weight_budgets
    if not sum(proportions):
        proportions = [1.0] * len(selected)
    shares = [value / sum(proportions) for value in proportions]
    capacity = min((budget / share for budget, share in zip(weight_budgets, shares) if share > 0), default=0.0)
    if actual_mode == "row":
        capacity += kv_mb
        if kv_mb > budgets[main_index]:
            # A zero weight share does not remove the main GPU's whole KV.
            capacity = 0.0
    plan.update(use_multi_gpu=multi, split_mode=actual_mode, main_gpu=main_index,
                primary_index=primary.get("index"), primary_uuid=primary.get("uuid", ""),
                devices=[dict(item, payload_budget_mb=budget) for item, budget in zip(selected, budgets)],
                device_names=selected_names, engine_device_names=engine_names,
                tensor_split=shares if multi else [], tensor_split_explicit=bool(multi and custom),
                payload_capacity_mb=capacity,
                # Cards left out of the split can still host small side models.
                spare_devices=[dict(item, payload_budget_mb=max(0.0, available_mb(item) - reserve - 600))
                               for item in devices if item not in selected],
                model_fits=bool(weights_mb > 0 and required <= capacity))
    if multi:
        plan["notes"].append("逐卡扣除预留和运行开销后分配权重；多卡可减少 CPU 卸载，但 PCIe 通信可能降低速度，需实测")
        if actual_mode == "row":
            plan["notes"].append("row 模式的 KV cache 和中间结果集中在主卡，上下文仍受主卡余量限制")
    elif len(devices) > 1:
        plan["notes"].append("当前使用单卡；auto 仅在模型与目标上下文超出主卡预算时启用多卡")
    if weights_mb > 0 and required > capacity:
        plan["notes"].append("当前逐卡预算无法完整容纳权重和目标上下文，仍可能卸载到 CPU；可缩短上下文或减小模型")
    return plan


def context_limit(plan, weights_mb, kv_per_token_mb, native_ctx=0):
    if not plan["devices"] or kv_per_token_mb <= 0 or weights_mb <= 0:
        return None
    shares = plan["tensor_split"] or [1.0]
    limits = []
    for index, (device, share) in enumerate(zip(plan["devices"], shares)):
        remaining = device["payload_budget_mb"] - weights_mb * share
        if remaining < 0:
            return 0
        kv_share = (1.0 if index == plan["main_gpu"] else 0.0) if plan["split_mode"] == "row" else share
        if kv_share > 0:
            limits.append(remaining / (kv_per_token_mb * kv_share))
    value = max(0, int(min(limits, default=0))) // 1024 * 1024
    return min(value, int(native_ctx)) if native_ctx and int(native_ctx) > 0 else value


def launch_args(plan, help_text):
    if plan["split_mode"] == "row" and plan["use_multi_gpu"]:
        validate_row_devices(plan["device_names"], plan.get("engine_device_names", []))
    required = ["--device"]
    if plan["use_multi_gpu"]:
        required += ["--split-mode", "--main-gpu"]
        if plan.get("tensor_split_explicit"):
            required.append("--tensor-split")
    if any(flag not in help_text for flag in required):
        raise ValueError("当前引擎缺少可靠的多显卡选择/分配参数，请升级引擎或使用单卡环境")
    # llama-bench uses '/' within one device/split configuration, reserving
    # commas for separate benchmark cases; server/cli use commas within lists.
    device_separator = "/" if re.search(r"--device\s+<[^>\n]*/", help_text) else ","
    split_separator = "/" if re.search(r"--tensor-split\s+<[^>\n]*/", help_text) else ","
    args = ["--device", device_separator.join(plan["device_names"])]
    if "--split-mode" in help_text:
        args += ["--split-mode", plan["split_mode"]]
    if "--main-gpu" in help_text:
        args += ["--main-gpu", str(plan["main_gpu"])]
    if plan["use_multi_gpu"] and plan.get("tensor_split_explicit"):
        args += ["--tensor-split", split_separator.join(f"{value:.8g}" for value in plan["tensor_split"])]
    return args
