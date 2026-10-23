import unittest
from pathlib import Path
from unittest import mock

import manager
import multi_gpu
import run


def inventory():
    return multi_gpu.parse_nvidia_inventory(
        "0, GPU-small, NVIDIA GeForce RTX 3060 Ti, 8192, 7000, 8.6\n"
        "1, GPU-large, NVIDIA GeForce RTX 5060 Ti, 16311, 15000, 12.0\n")


def engine_inventory():
    # CUDA order deliberately differs from nvidia-smi indices.
    output = ("Available devices:\n"
              " CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 14500 MiB free)\n"
              " CUDA1: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 6500 MiB free)\n")
    return multi_gpu.match_engine_devices(inventory(), multi_gpu.parse_engine_devices(output))


class MultiGpuPlanningTests(unittest.TestCase):
    def test_prism_row_cross_card_is_rejected_even_if_flags_exist(self):
        import subprocess
        gpu = {'selected_backend':'cuda','gpus':inventory()}
        initial = {'fit_target_mb':1024,'gpu_weight_mb':14000,'mmproj_offload':False,
                   'cache_type_k':'q8_0','cache_type_v':'q8_0','min_ctx':8192}
        engine = 'CUDA0: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 6500 MiB free)\nCUDA1: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 14500 MiB free)'
        with mock.patch.object(run,'supports',return_value=True), \
                mock.patch.object(run,'rc',return_value=subprocess.CompletedProcess([],0,engine,'')), \
                mock.patch.object(run,'performance_tuning',return_value=initial), \
                mock.patch.object(run,'kv_cache_mb',return_value=1000), \
                mock.patch.object(run.engines,'prism_quant',return_value='PQ2_0'):
            with self.assertRaisesRegex(RuntimeError,'Bonsai Prism'):
                run.cuda_runtime_plan(Path('server.exe'),Path('model.gguf'),{},gpu,
                    {'multi_gpu':'force','split_mode':'row'}, {},32768)

    def test_inventory_keeps_identity_and_free_per_device(self):
        devices = inventory()
        self.assertEqual([item["index"] for item in devices], [0, 1])
        self.assertEqual(devices[1]["uuid"], "GPU-large")
        self.assertEqual(devices[1]["compute_capability"], "12.0")
        self.assertEqual(multi_gpu.choose_primary(devices)["index"], 1)

    def test_legacy_driver_unknown_free_and_malformed_rows(self):
        devices = multi_gpu.parse_nvidia_inventory("broken\nNVIDIA, with comma, 8192, N/A\ninvalid, nope, 0\n")
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "NVIDIA, with comma")
        self.assertFalse(devices[0]["free_known"])
        self.assertEqual(multi_gpu.available_mb(devices[0]), 0)

    def test_cuda_mapping_does_not_assume_physical_ordinals(self):
        devices = engine_inventory()
        self.assertEqual([item["index"] for item in devices], [1, 0])
        self.assertEqual(multi_gpu.choose_primary(devices, "GPU-large")["device"], "CUDA0")
        self.assertEqual(multi_gpu.choose_primary(devices, "0")["device"], "CUDA1")

    def test_identical_cards_require_explicit_identity_match(self):
        physical = inventory()
        physical[1].update(name=physical[0]["name"], vram_mb=physical[0]["vram_mb"])
        engine = [{"device": "CUDA0", "name": physical[0]["name"], "vram_mb": 8192, "vram_free_mb": 7000}]
        matched = multi_gpu.match_engine_devices(physical, engine)
        with self.assertRaisesRegex(ValueError, "无法唯一映射"):
            multi_gpu.choose_primary(matched, "GPU-small")

    def test_auto_stays_on_larger_card_for_small_model(self):
        plan = multi_gpu.build_plan(engine_inventory(), {}, 8000, 500)
        self.assertFalse(plan["use_multi_gpu"])
        self.assertEqual(plan["device_names"], ["CUDA0"])
        self.assertEqual(plan["primary_index"], 1)
        self.assertTrue(plan["model_fits"])

    def test_auto_uses_two_cards_only_when_needed(self):
        plan = multi_gpu.build_plan(engine_inventory(), {}, 14000, 1000)
        self.assertTrue(plan["use_multi_gpu"])
        self.assertTrue(plan["model_fits"])
        self.assertFalse(plan["tensor_split_explicit"])
        self.assertGreater(plan["tensor_split"][0], plan["tensor_split"][1])
        self.assertAlmostEqual(sum(plan["tensor_split"]), 1)

    def test_off_honors_manual_primary_and_force_requires_capacity(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "off", "main_gpu": "GPU-small"}, 14000)
        self.assertEqual(plan["device_names"], ["CUDA1"])
        self.assertFalse(plan["model_fits"])
        with self.assertRaisesRegex(ValueError, "至少两张"):
            multi_gpu.build_plan(engine_inventory()[:1], {"multi_gpu": "force"}, 14000)

    def test_custom_split_checks_each_card_not_total_vram(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force", "tensor_split": "1,9"}, 14000)
        self.assertFalse(plan["model_fits"])
        self.assertTrue(plan["tensor_split_explicit"])
        self.assertLess(plan["payload_capacity_mb"], 7000)
        for split in ("1", "-1,2", "nan,2", "inf,2", "0,0", "bad,2"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force", "tensor_split": split})

    def test_zero_free_is_never_counted_as_full_capacity(self):
        devices = engine_inventory()
        for item in devices:
            item["vram_free_mb"] = 0
        plan = multi_gpu.build_plan(devices, {}, 1000)
        self.assertEqual(plan["payload_capacity_mb"], 0)
        self.assertFalse(plan["model_fits"])

    def test_row_context_is_limited_by_primary_kv_capacity(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force", "split_mode": "row", "tensor_split": "1,1"}, 8000, 1000)
        limit = multi_gpu.context_limit(plan, 8000, 0.5, 262144)
        expected = int((plan["devices"][0]["payload_budget_mb"] - 4000) / 0.5) // 1024 * 1024
        self.assertEqual(limit, expected)
        self.assertLessEqual(multi_gpu.context_limit(plan, 8000, 0.5, 4096), 4096)

    def test_row_cannot_fit_kv_larger_than_primary_with_zero_weight_share(self):
        plan = multi_gpu.build_plan(engine_inventory(),
                                    {"multi_gpu": "force", "split_mode": "row", "tensor_split": "0,1"},
                                    1000, 14000)
        self.assertFalse(plan["model_fits"])
        self.assertEqual(plan["payload_capacity_mb"], 0)

    def test_row_rejects_reordered_or_incomplete_backend_enumeration(self):
        devices = engine_inventory()
        incomplete = [dict(devices[0]), dict(devices[1], device="CUDA2")]
        for value in (list(reversed(devices)), incomplete, inventory()):
            with self.subTest(devices=value), self.assertRaisesRegex(ValueError, "完整.*自然顺序"):
                multi_gpu.build_plan(value, {"multi_gpu": "force", "split_mode": "row"}, 8000)

    def test_row_rejects_candidate_filter_but_layer_still_supports_it(self):
        devices = engine_inventory()
        devices.append(dict(devices[1], device="CUDA2", index=2, uuid="GPU-busy", vram_free_mb=300))
        with self.assertRaisesRegex(ValueError, "不能筛选或重排"):
            multi_gpu.build_plan(devices, {"multi_gpu": "force", "split_mode": "row"}, 8000)
        plan = multi_gpu.build_plan(devices, {"multi_gpu": "force", "split_mode": "layer"}, 8000)
        self.assertEqual(plan["device_names"], ["CUDA0", "CUDA1"])

    def test_row_main_index_matches_global_backend_index(self):
        plan = multi_gpu.build_plan(engine_inventory(),
                                    {"multi_gpu": "force", "split_mode": "row", "main_gpu": "GPU-small"}, 8000)
        self.assertEqual(plan["main_gpu"], 1)
        self.assertEqual(plan["device_names"][plan["main_gpu"]], "CUDA1")
        args = multi_gpu.launch_args(plan, "--device --main-gpu --split-mode --tensor-split")
        self.assertEqual(args[args.index("--main-gpu") + 1], "1")
        plan["device_names"].reverse()
        with self.assertRaisesRegex(ValueError, "不能筛选或重排"):
            multi_gpu.launch_args(plan, "--device --main-gpu --split-mode --tensor-split")

    def test_cuda1_only_single_none_keeps_selected_relative_main_zero(self):
        plan = multi_gpu.build_plan(engine_inventory()[1:], {"multi_gpu": "off", "split_mode": "row"}, 1000)
        self.assertEqual(plan["device_names"], ["CUDA1"])
        self.assertEqual(plan["split_mode"], "none")
        self.assertEqual(plan["main_gpu"], 0)

    def test_layer_context_fits_every_card_and_caps_native(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force"}, 14000, 1000)
        limit = multi_gpu.context_limit(plan, 14000, 0.1, 262144)
        for device, share in zip(plan["devices"], plan["tensor_split"]):
            self.assertLessEqual((14000 + limit * 0.1) * share, device["payload_budget_mb"] + 0.001)
        self.assertIsNone(multi_gpu.context_limit(plan, 14000, 0))

    def test_launch_only_uses_advertised_flags(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force"}, 8000)
        with self.assertRaisesRegex(ValueError, "缺少"):
            multi_gpu.launch_args(plan, "--device --main-gpu")
        flags = multi_gpu.launch_args(plan, "--device --main-gpu --split-mode --tensor-split")
        self.assertEqual(flags[flags.index("--device") + 1], "CUDA0,CUDA1")
        self.assertNotIn("--tensor-split", flags)
        self.assertNotIn("tensor-parallel", " ".join(flags))

    def test_single_card_plan_reports_spare_cards_with_their_budget(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "off", "reserve_mb": 512}, 4000)
        self.assertFalse(plan["use_multi_gpu"])
        self.assertEqual(len(plan["spare_devices"]), 1)
        spare = plan["spare_devices"][0]
        self.assertNotEqual(spare["device"], plan["device_names"][0])
        self.assertEqual(spare["payload_budget_mb"], multi_gpu.available_mb(spare) - 512 - 600)
        forced = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force"}, 4000)
        self.assertEqual(forced["spare_devices"], [])

    def test_manual_split_is_preserved_but_auto_split_is_left_to_engine(self):
        help_text = "--device --main-gpu --split-mode --tensor-split"
        automatic = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force"}, 14000)
        custom = multi_gpu.build_plan(engine_inventory(),
                                      {"multi_gpu": "force", "tensor_split": "1,2"}, 14000)
        self.assertNotIn("--tensor-split", multi_gpu.launch_args(automatic, help_text))
        args = multi_gpu.launch_args(custom, help_text)
        self.assertIn("--tensor-split", args)
        self.assertEqual(args[args.index("--tensor-split") + 1], "0.33333333,0.66666667")

    def test_benchmark_uses_slashes_from_advertised_option_syntax(self):
        plan = multi_gpu.build_plan(engine_inventory(), {"multi_gpu": "force"}, 8000)
        help_text = "--device <dev0/dev1/...>\n--tensor-split <ts0/ts1/..>\n--main-gpu <i>\n--split-mode <none|layer|row>"
        args = multi_gpu.launch_args(plan, help_text)
        self.assertEqual(args[args.index("--device") + 1], "CUDA0/CUDA1")
        self.assertNotIn("--tensor-split", args)
        manual = multi_gpu.build_plan(engine_inventory(),
                                      {"multi_gpu": "force", "tensor_split": "1,2"}, 8000)
        manual_args = multi_gpu.launch_args(manual, help_text)
        self.assertIn("/", manual_args[manual_args.index("--tensor-split") + 1])

    def test_invalid_reserve_and_mode_are_rejected(self):
        for config in ({"reserve_mb": -1}, {"reserve_mb": "nan"}, {"multi_gpu": "maybe"}, {"split_mode": "tensor"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                multi_gpu.build_plan(engine_inventory(), config)


class MultiGpuRuntimeTests(unittest.TestCase):
    def test_default_single_gpu_does_not_add_launch_flags_or_probe(self):
        gpu = {"selected_backend": "cuda", "gpus": inventory()[:1], "vram_mb": 8192, "vram_free_mb": 7000}
        with mock.patch.object(run, "performance_tuning", return_value={"same": True}), mock.patch.object(run, "rc") as probe:
            projected, tuning, flags = run.cuda_runtime_plan(Path("binary"), Path("model"), {}, gpu, {}, {}, 8192)
        self.assertIs(projected, gpu)
        self.assertEqual(tuning, {"same": True})
        self.assertEqual(flags, [])
        probe.assert_not_called()

    def test_runtime_maps_primary_and_passes_real_split_flags(self):
        output = ("CUDA0: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 6500 MiB free)\n"
                  "CUDA1: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 14500 MiB free)\n")
        gpu = {"selected_backend": "cuda", "gpus": inventory(), "vram_mb": 24503, "vram_free_mb": 22000}
        meta = {"block_count": 32, "attention.head_count_kv": 8, "attention.key_length": 128, "attention.value_length": 128, "context_length": 32768}
        with mock.patch.object(run, "model_size_mb", return_value=14000), \
                mock.patch.object(run, "gpu_weight_mb", return_value=14000), \
                mock.patch.object(run, "meminfo", return_value={"avail_gb": 30}), \
                mock.patch.object(run, "supports", return_value=True), \
                mock.patch.object(run, "help_text", return_value="--device --main-gpu --split-mode --tensor-split"), \
                mock.patch.object(run, "rc", return_value=mock.Mock(returncode=0, stdout=output, stderr="")):
            projected, tuning, flags = run.cuda_runtime_plan(Path("binary"), Path("model"), meta, gpu, {}, {}, 8192)
        self.assertEqual(projected["active_gpu_count"], 2)
        self.assertEqual(flags[flags.index("--main-gpu") + 1], "1")
        self.assertGreaterEqual(tuning["fit_target_mb"], 1024)
        self.assertEqual(projected["compute_capability"], "8.6")
        self.assertGreater(tuning["recommended_ctx"], 0)

    def test_force_rejects_cpu_instead_of_silently_ignoring_setting(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            run.cuda_runtime_plan(Path("binary"), Path("model"), {}, {"selected_backend": "cpu"}, {"multi_gpu": "force"}, {}, 8192)

    def test_gpu_vision_module_targets_the_budgeted_primary(self):
        output = ("CUDA0: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 6500 MiB free)\n"
                  "CUDA1: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 14500 MiB free)\n")
        gpu = {"selected_backend": "cuda", "gpus": inventory(), "vram_mb": 24503, "vram_free_mb": 22000}
        with mock.patch.object(run, "model_size_mb", return_value=14000), \
                mock.patch.object(run, "gpu_weight_mb", return_value=14000), \
                mock.patch.object(run, "meminfo", return_value={"avail_gb": 30}), \
                mock.patch.object(run, "supports", return_value=True), \
                mock.patch.object(run, "help_text", return_value="--device --main-gpu --split-mode --tensor-split"), \
                mock.patch.object(run, "rc", return_value=mock.Mock(returncode=0, stdout=output, stderr="")):
            projected, tuning, flags = run.cuda_runtime_plan(Path("binary"), Path("model"), {}, gpu,
                                                            {"multi_gpu": "force"}, {}, 8192, True, 1000)
        self.assertEqual(flags[flags.index("--mmproj-device") + 1], "CUDA1")
        primary = projected["gpu_plan"]["devices"][1]
        self.assertLess(primary["payload_budget_mb"], 14500 - 1024 - 600)

    def test_manager_recommendation_has_separate_single_and_multi_budgets(self):
        rec = manager.get_recommendations(64, 16311, 15000, inventory(), {"gpu": {"multi_gpu": "auto"}})
        self.assertEqual(rec["gpu_plan"]["primary_index"], 1)
        self.assertGreater(rec["multi_gpu_max_model_gb"], rec["single_gpu_max_model_gb"])
        self.assertEqual(rec["context_basis"], "generic")

    def test_manager_custom_ratios_follow_engine_order(self):
        output = ("CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 14500 MiB free)\n"
                  "CUDA1: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 6500 MiB free)\n")
        cfg = {"model": {"model_file": "m.gguf"}, "gpu": {"multi_gpu": "force", "tensor_split": "1,9"}}
        with mock.patch.object(run, "resolve_model", return_value=Path("m.gguf")), \
                mock.patch.object(run, "gguf_meta", return_value={}), \
                mock.patch.object(run, "model_size_mb", return_value=14000), \
                mock.patch.object(run, "gpu_weight_mb", return_value=14000), \
                mock.patch.object(run, "meminfo", return_value={"avail_gb": 30}), \
                mock.patch.object(run, "supports", return_value=True), \
                mock.patch.object(run, "rc", return_value=mock.Mock(returncode=0, stdout=output, stderr="")):
            rec = manager.get_recommendations(64, 16311, 15000, inventory(), cfg)
        self.assertEqual(rec["gpu_plan"]["device_names"], ["CUDA0", "CUDA1"])
        self.assertFalse(rec["gpu_plan"]["model_fits"])
        self.assertLess(rec["gpu_max_model_gb"], 6)


if __name__ == "__main__":
    unittest.main()
