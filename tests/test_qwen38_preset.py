import unittest
from pathlib import Path
from unittest import mock

import run


class Qwen38PresetTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path("llama-server.exe")
        self.meta = {"general.architecture": "qwen4exp"}
        self.sc = {"ctx_size": 32768, "enable_thinking": False}
        self.gc = {"multi_gpu": "auto", "tensor_split": "", "main_gpu": "auto"}
        self.pc = {"qwen38_preset": "fast_quality", "qwen38_moe_cache_mib": 4096,
                   "spec_type": "draft-mtp", "cache_type_k": "q8_0"}

    def test_opt_in_profile_is_qwen_only_and_does_not_mutate_config(self):
        with mock.patch.object(run, "supports", return_value=True), \
                mock.patch.object(run, "supports_value", return_value=True):
            sc, gc, pc, active, notes = run.qwen38_launch_overrides(
                self.binary, Path("Qwen3.8-Flash-Next-Q2_0.gguf"), self.meta, "server",
                self.sc, self.gc, self.pc)

        self.assertTrue(active)
        self.assertEqual(gc["multi_gpu"], "off")
        self.assertEqual(gc["tensor_split"], "")
        self.assertTrue(pc["cpu_moe"])
        self.assertEqual(pc["moe_cache_mib"], 4096)
        self.assertEqual(pc["spec_type"], "off")
        self.assertEqual(pc["cache_type_k"], "f16")
        self.assertEqual((pc["batch_size"], pc["ubatch_size"]), (2048, 1024))
        self.assertEqual(sc["ctx_size"], 32768)
        self.assertTrue(sc["enable_thinking"])
        self.assertIn("MTP 保持关闭", notes[0])
        self.assertEqual(self.gc["multi_gpu"], "auto")
        self.assertEqual(self.pc["spec_type"], "draft-mtp")

    def test_other_models_and_non_server_modes_keep_existing_settings(self):
        for model, meta, mode in (
            (Path("Qwen3.6-35B.gguf"), {"general.architecture": "qwen35"}, "server"),
            (Path("Qwen3.8-Flash-Next-Q2_0.gguf"), self.meta, "chat"),
        ):
            with mock.patch.object(run, "supports", return_value=True) as supports, \
                    mock.patch.object(run, "supports_value", return_value=True):
                sc, gc, pc, active, notes = run.qwen38_launch_overrides(
                    self.binary, model, meta, mode, self.sc, self.gc, self.pc)
            self.assertFalse(active)
            self.assertEqual((sc, gc, pc), (self.sc, self.gc, self.pc))
            self.assertEqual(notes, [])
            supports.assert_not_called()

    def test_missing_engine_feature_falls_back_without_rejecting_model(self):
        def supports(_binary, flag):
            return flag != "--moe-cache-mib"

        with mock.patch.object(run, "supports", side_effect=supports), \
                mock.patch.object(run, "supports_value", return_value=True):
            sc, gc, pc, active, notes = run.qwen38_launch_overrides(
                self.binary, Path("Qwen3.8-Flash-Next-Q2_0.gguf"), self.meta, "server",
                self.sc, self.gc, self.pc)

        self.assertFalse(active)
        self.assertEqual((sc, gc, pc), (self.sc, self.gc, self.pc))
        self.assertIn("--moe-cache-mib", notes[0])
        self.assertIn("未阻止模型加载", notes[0])

    def test_cache_budget_is_clamped_to_safe_supported_range(self):
        self.pc["qwen38_moe_cache_mib"] = 99999
        with mock.patch.object(run, "supports", return_value=True), \
                mock.patch.object(run, "supports_value", return_value=True):
            _, _, pc, active, _ = run.qwen38_launch_overrides(
                self.binary, Path("Qwen3.8-Flash-Next-Q2_0.gguf"), self.meta, "server",
                self.sc, self.gc, self.pc)
        self.assertTrue(active)
        self.assertEqual(pc["moe_cache_mib"], run.QWEN38_CACHE_MAX_MIB)

    def test_engine_default_context_is_replaced_by_bounded_context(self):
        self.sc["ctx_size"] = 0
        self.pc["qwen38_moe_cache_mib"] = 0
        with mock.patch.object(run, "supports", return_value=True),                 mock.patch.object(run, "supports_value", return_value=True):
            sc, _, pc, active, _ = run.qwen38_launch_overrides(
                self.binary, Path("Qwen3.8-Flash-Next-Q2_0.gguf"), self.meta, "server",
                self.sc, self.gc, self.pc)
        self.assertTrue(active)
        self.assertEqual(sc["ctx_size"], run.QWEN38_DEFAULT_CTX)
        self.assertEqual(pc["moe_cache_mib"], 0)


class Qwen38CacheSizingTests(unittest.TestCase):
    meta = {"block_count": 48, "attention.head_count_kv": 2, "attention.key_length": 256,
            "attention.value_length": 256, "full_attention_interval": 4}
    tuning = {"cache_type_k": "f16", "cache_type_v": "f16", "fit_target_mb": 512}

    def plan_gpu(self, free_mb=15057):
        main = {"device": "CUDA1", "name": "RTX 5060 Ti", "vram_mb": 16311, "vram_free_mb": free_mb}
        return {"gpu_plan": {"devices": [main], "main_gpu": 0, "reserve_mb": 512, "spare_devices": []}}

    def size(self, gpu, cache=0, ctx=65536, mmproj=0.0):
        pc = {"moe_cache_mib": cache, "ubatch_size": 1024}
        with mock.patch.object(run, "gpu_dense_mb", return_value=3210.0):
            return run.qwen38_cache_mib(Path("m.gguf"), self.meta, gpu, self.tuning, pc, ctx, mmproj)

    def test_auto_cache_fills_main_card_after_dense_kv_and_compute(self):
        # 15057 free - 512 reserve - 600 overhead - 3210 dense - 1536 KV - 1051 compute
        cache, note = self.size(self.plan_gpu())
        self.assertEqual(cache, 7936)
        self.assertIn("自动", note)
        shorter, _ = self.size(self.plan_gpu(), ctx=16384)
        self.assertGreater(shorter, cache)

    def test_manual_cache_is_capped_to_avoid_first_inference_oom(self):
        self.assertEqual(self.size(self.plan_gpu(), cache=4096)[0], 4096)
        cache, note = self.size(self.plan_gpu(), cache=16384)
        self.assertEqual(cache, 7936)
        self.assertIn("OOM", note)

    def test_vision_on_main_card_and_tiny_cards_reduce_cache(self):
        self.assertLess(self.size(self.plan_gpu(), mmproj=900)[0], 7936)
        cache, note = self.size(self.plan_gpu(free_mb=5000))
        self.assertEqual(cache, 0)
        self.assertIn("关闭专家缓存", note)

    def test_single_gpu_host_without_plan_uses_detected_free_memory(self):
        cache, _ = self.size({"vram_free_mb": 15057})
        self.assertEqual(cache, 7936)


if __name__ == "__main__":
    unittest.main()
