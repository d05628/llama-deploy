import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import compat
import deploy
import manager
import run


class ModelPairingTests(unittest.TestCase):
    def test_remote_paths_become_portable_local_names(self):
        self.assertEqual(manager.local_model_name("quant/Qwen3-VL-Q4_K_M.gguf"), "Qwen3-VL-Q4_K_M.gguf")
        self.assertEqual(deploy.local_filename(r"vision\mmproj-Qwen3-VL-f16.gguf"), "mmproj-Qwen3-VL-f16.gguf")

    def test_mmproj_name_contains_the_main_model_name(self):
        expected = "Qwen3-VL-8B-Q4_K_M.mmproj-f16.gguf"
        self.assertEqual(
            manager.paired_mmproj_name("Qwen3-VL-8B-Q4_K_M.gguf", "vision/mmproj-model-f16.gguf"),
            expected,
        )
        self.assertEqual(
            deploy.paired_mmproj_filename(
                "Qwen3-VL-8B-Q4_K_M-00001-of-00003.gguf", "mmproj-model-F16.gguf"
            ),
            expected,
        )

    def test_family_match_rejects_other_qwen_generation(self):
        self.assertTrue(manager.mmproj_matches_model(
            "Qwen2.5-VL-7B-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-f16.gguf"
        ))
        self.assertFalse(manager.mmproj_matches_model(
            "Qwen2.5-VL-7B-Q4_K_M.gguf", "mmproj-Qwen3-VL-f16.gguf"
        ))

    def test_exact_binding_accepts_generic_official_mmproj_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "Qwen3-VL-Q4_K_M.gguf"
            mmproj = root / "mmproj-model-f16.gguf"
            model.write_bytes(b"x" * 2000)
            mmproj.write_bytes(b"x" * 2000)
            with mock.patch.object(run, "gguf_meta", return_value={"general.type": "mmproj"}):
                found, automatic_name = run.find_compatible_mmproj(model, mmproj, {})
            self.assertEqual(found, mmproj)
            self.assertEqual(automatic_name, "")

    def test_downloaded_mmproj_search_does_not_pick_unrelated_first_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vision = root / "vision"
            nested = vision / "remote"
            nested.mkdir(parents=True)
            (vision / "mmproj-wrong.gguf").write_bytes(b"w" * 20000)
            expected = nested / "mmproj-right.gguf"
            expected.write_bytes(b"r" * 20000)

            subject = deploy.Deployer.__new__(deploy.Deployer)
            subject.config = {"model": {"mmproj_size": 20000}}
            subject.mmproj_dir = vision
            subject.mmproj_file = "mmproj-right.gguf"
            subject.mmproj_repo_file = "remote/mmproj-right.gguf"
            subject.mmproj_path = vision / subject.mmproj_file
            with mock.patch.object(deploy, "MODELS_DIR", root):
                self.assertTrue(subject._find_mmproj_file())
            self.assertEqual(subject.mmproj_path.read_bytes(), b"r" * 20000)


class SecurityTests(unittest.TestCase):
    def test_model_name_rejects_traversal_and_globs(self):
        for bad in ["../etc/passwd", "a/b.gguf", "a\\b.gguf", "*.gguf",
                    "model?.gguf", "m[0-9].gguf", "", ".", ".."]:
            self.assertFalse(manager.safe_model_name(bad), bad)
        for good in ["Qwen3.8-27B-UD-IQ4_XS.gguf", "a.gguf", "模型-Q4_K_M.gguf"]:
            self.assertTrue(manager.safe_model_name(good), good)

    def test_delete_refuses_glob_pattern(self):
        # rglob 会把 "*.gguf" 当通配符匹配到任意模型并删除
        result = manager.ModelLibrary.delete("*.gguf")
        self.assertEqual(result["status"], "error")
        self.assertIn("非法文件名", result["message"])

    def test_activate_refuses_unknown_file(self):
        result = manager.ModelLibrary.activate("definitely-not-here-xyz.gguf")
        self.assertEqual(result["status"], "error")

    def test_session_token_is_unpredictable(self):
        self.assertGreaterEqual(len(manager.SESSION_TOKEN), 32)
        self.assertNotEqual(manager.SESSION_TOKEN, "__MANAGER_TOKEN__")

    def test_manager_sends_no_wildcard_cors(self):
        source = (manager.BASE_DIR / "manager.py").read_text(encoding="utf-8")
        self.assertNotIn('"Access-Control-Allow-Origin", "*"', source)

    def test_compat_auth_required_only_for_real_keys(self):
        handler = compat.CompatHandler
        for disabled in ["", "local-no-key-needed", "none", "disabled", "no-key", "  NONE  "]:
            self.assertFalse(handler.auth_required(disabled), disabled)
        for enabled in ["my-secret", "sk-abc123", " Real-Key "]:
            self.assertTrue(handler.auth_required(enabled), enabled)

    def test_zip_extraction_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evil = root / "evil.zip"
            with zipfile.ZipFile(evil, "w") as archive:
                archive.writestr("../escaped.txt", "pwned")
            dest = root / "out"
            dest.mkdir()
            with zipfile.ZipFile(evil) as archive:
                with self.assertRaises(RuntimeError):
                    deploy.Deployer._safe_extractall(archive, dest)
            self.assertFalse((root / "escaped.txt").exists())

    def test_zip_extraction_allows_normal_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = root / "ok.zip"
            with zipfile.ZipFile(good, "w") as archive:
                archive.writestr("sub/a.txt", "hello")
            dest = root / "out"
            dest.mkdir()
            with zipfile.ZipFile(good) as archive:
                deploy.Deployer._safe_extractall(archive, dest)
            self.assertEqual((dest / "sub" / "a.txt").read_text(), "hello")


class ProcessIdentityTests(unittest.TestCase):
    def test_pid_running_checks_process_identity(self):
        # PID 会被系统复用；只判断"存在这个 PID"就 kill 会误杀无关进程
        with mock.patch.object(manager, "process_name", return_value="notepad.exe"):
            self.assertFalse(manager.pid_running(1234, "llama-server"))
            self.assertTrue(manager.pid_running(1234, "notepad"))
            self.assertTrue(manager.pid_running(1234))

    def test_pid_running_rejects_invalid_pid(self):
        self.assertFalse(manager.pid_running(0, "llama-server"))
        self.assertFalse(manager.pid_running(-1))


class FrontendHygieneTests(unittest.TestCase):
    """内嵌前端的回归测试：这些是无法靠 Python 测试覆盖的部分。"""

    @classmethod
    def setUpClass(cls):
        cls.source = (manager.BASE_DIR / "manager.py").read_text(encoding="utf-8")

    def test_escaping_helpers_exist(self):
        self.assertIn("function esc(v)", self.source)
        self.assertIn("function escArg(v)", self.source)

    def test_api_helper_checks_http_status(self):
        self.assertIn("if(!r.ok)", self.source)

    def test_no_blocking_alert_dialogs(self):
        self.assertNotIn("alert('", self.source)
        self.assertNotIn("alert((r", self.source)

    def test_labels_are_bound_to_controls(self):
        unbound = self.source.count('<label class="form-label">')
        self.assertEqual(unbound, 0, "仍有 label 未通过 for= 绑定控件")

    def test_narrow_screens_keep_device_info(self):
        # 旧样式在 768px 断点直接 .sys-info{display:none}，
        # 把显存预算这类最该看的信息藏掉了
        block = self.source.split("@media(max-width:768px)")[1].split("}")[0]
        self.assertNotIn("display:none", block.replace(".nav-text,.sidebar .logo span{display:none", ""))


class ConfigDefaultsTests(unittest.TestCase):
    """默认配置曾有三份副本（Python / 前端 JS / config.example.jsonc）。
    前端那份已改为向后端拉取，示例文件用本测试锁住，防止再次漂移。"""

    @staticmethod
    def _keys(obj, prefix=""):
        found = set()
        for key, value in obj.items():
            found.add(prefix + key)
            if isinstance(value, dict):
                found |= ConfigDefaultsTests._keys(value, prefix + key + ".")
        return found

    def test_example_config_matches_default_config(self):
        example = run.parse_jsonc(manager.BASE_DIR / "config.example.jsonc")
        self.assertTrue(example, "config.example.jsonc 解析失败")
        defaults = manager.default_config()
        missing = self._keys(defaults) - self._keys(example)
        extra = self._keys(example) - self._keys(defaults)
        self.assertEqual(missing, set(), "config.example.jsonc 缺少这些字段")
        self.assertEqual(extra, set(), "config.example.jsonc 多出这些字段")

    def test_frontend_has_no_second_copy_of_defaults(self):
        # 前端只能通过 /api/config/default 取默认值，不能再自带一份字面量
        source = (manager.BASE_DIR / "manager.py").read_text(encoding="utf-8")
        self.assertIn("/api/config/default", source)
        self.assertNotIn("local-no-key-needed',claude_tool_mode", source)


class QuantDetectionTests(unittest.TestCase):
    def test_bf16_is_not_mistaken_for_f16(self):
        # "bf16" 里含有 "f16"，短名先匹配就会把 BF16 误标成 F16
        self.assertEqual(manager.detect_quant("Qwen3.5-2B-BF16.gguf"), "BF16")
        self.assertEqual(manager.detect_quant("model-F16.gguf"), "F16")

    def test_unsloth_dynamic_variants_are_recognised(self):
        for name, expected in [
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", "Q4_K_M"),
            ("Qwen3.8-27B-UD-IQ4_XS.gguf", "IQ4_XS"),
            ("Qwen3.8-27B-UD-Q3_K_XL.gguf", "Q3_K_XL"),
            ("Qwen3.8-27B-UD-Q4_K_XL.gguf", "Q4_K_XL"),
            ("Qwen3.8-27B-UD-Q8_K_XL.gguf", "Q8_K_XL"),
            ("Qwen3.8-27B-UD-IQ3_S.gguf", "IQ3_S"),
            ("GLM-4.7-Flash-MXFP4_MOE.gguf", "MXFP4_MOE"),
        ]:
            self.assertEqual(manager.detect_quant(name), expected, name)

    def test_longer_name_wins_over_its_own_prefix(self):
        self.assertEqual(manager.detect_quant("m-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(manager.detect_quant("m-Q4_K_S.gguf"), "Q4_K_S")
        self.assertEqual(manager.detect_quant("m-Q4_K.gguf"), "Q4_K")

    def test_mmproj_reports_its_own_precision_not_the_main_model(self):
        # 配对命名会把主模型量化写进视觉模块的文件名
        self.assertEqual(
            manager.detect_quant("Qwen3.8-27B-UD-Q4_K_M-mmproj-BF16.gguf"), "BF16"
        )
        self.assertEqual(manager.detect_quant("mmproj-model-f16.gguf"), "F16")

    def test_unparseable_name_is_unknown(self):
        self.assertEqual(manager.detect_quant("some-model.gguf"), "unknown")
        self.assertEqual(manager.detect_quant(""), "unknown")


class GpuBudgetTests(unittest.TestCase):
    def test_budget_is_well_below_total_vram(self):
        # 15.9GB 的卡装不下 15.3GB 的权重：KV cache、计算缓冲和桌面都要占显存
        budget = manager.gpu_weight_budget_gb(16311, 15063)
        self.assertLess(budget, 14.0)
        self.assertGreater(budget, 13.0)      # 实测预算 13.45GB
        self.assertLess(budget, 15.33)        # 必须判定现用的 Q4_K_M 装不下
        self.assertGreater(budget, 13.28)     # 且判定 IQ4_XS 装得下

    def test_budget_falls_back_to_total_when_free_is_unknown(self):
        self.assertGreater(manager.gpu_weight_budget_gb(16311, 0), 12.0)
        self.assertEqual(manager.gpu_weight_budget_gb(0, 0), 0.0)

    def test_bogus_free_value_is_ignored(self):
        # 空闲值大于总量时不可信，退回按总量估算
        self.assertEqual(
            manager.gpu_weight_budget_gb(16311, 99999),
            manager.gpu_weight_budget_gb(16311, 0),
        )


class RecommendationTests(unittest.TestCase):
    def test_gpu_advice_does_not_contradict_the_budget(self):
        # 旧实现在 vram>=6GB 时一律提示"可使用 Q5_K_M 或 Q8_0"，
        # 完全不看模型多大——27B 的 Q5_K_M 是 19GB，根本装不下
        recs = manager.get_recommendations(63.9, 16311, 15063)
        joined = " ".join(recs["tips"])
        self.assertNotIn("Q8_0", joined)
        self.assertNotIn("Q5_K_M", joined)
        self.assertIn(str(recs["gpu_max_model_gb"]), joined)

    def test_context_recommendation_follows_vram_not_system_ram(self):
        big_gpu = manager.get_recommendations(16, 16311, 15063)
        small_gpu = manager.get_recommendations(64, 4096, 0)
        self.assertGreater(big_gpu["recommended_ctx"], small_gpu["recommended_ctx"])

    def test_gpu_layers_tip_does_not_promise_full_offload(self):
        recs = manager.get_recommendations(63.9, 16311, 15063)
        joined = " ".join(recs["tips"])
        self.assertNotIn("将全部层卸载", joined)


class PerformanceDetectionTests(unittest.TestCase):
    def test_multi_gpu_memory_is_aggregated(self):
        output = "NVIDIA A, 8192, 7000\nNVIDIA B, 12288, 11000\n"
        completed = mock.Mock(returncode=0, stdout=output)
        with mock.patch.object(run, "rc", return_value=completed), mock.patch.object(run, "has_vulkan", return_value=False):
            result = run.accel("auto")
        self.assertEqual(result["gpu_count"], 2)
        self.assertEqual(result["vram_mb"], 20480)
        self.assertEqual(result["vram_free_mb"], 18000)
        self.assertEqual(result["selected_backend"], "cuda")

    def test_blackwell_compute_capability_is_detected(self):
        gpu_output = "NVIDIA GeForce RTX 5060 Ti, 16311, 15500\n"
        cap_output = "12.0\n"
        responses = [
            mock.Mock(returncode=0, stdout=gpu_output),
            mock.Mock(returncode=0, stdout=cap_output),
        ]
        with mock.patch.object(run, "rc", side_effect=responses), mock.patch.object(
            run, "has_vulkan", return_value=False
        ):
            result = run.accel("auto")
        self.assertEqual(result["compute_capability"], "12.0")
        self.assertTrue(run.is_blackwell(result))

    def test_qwen38_27b_uses_16gb_high_pressure_profile(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 15500,
            "compute_capability": "12.0",
        }
        meta = {"general.architecture": "qwen35", "block_count": 64}
        with mock.patch.object(run, "model_size_mb", return_value=15.65 * 1024), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-Q4_K_M.gguf"), meta, gpu,
                {"profile": "auto", "cache_type_k": "auto", "cache_type_v": "auto"},
                8192,
            )
        # auto 对齐到 balanced：留 64MiB 余量、KV 用 q8_0
        self.assertEqual(tuning["fit_target_mb"], 64)
        self.assertEqual(tuning["cache_type_k"], "q8_0")
        self.assertEqual(tuning["cache_type_v"], "q8_0")

    def _high_pressure(self, profile):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 15500,
            "compute_capability": "12.0",
        }
        meta = {"general.architecture": "qwen35", "block_count": 64}
        with mock.patch.object(run, "model_size_mb", return_value=15.65 * 1024), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            return run.performance_tuning(
                Path("Qwen3.8-27B-Q4_K_M.gguf"), meta, gpu,
                {"profile": profile, "cache_type_k": "auto", "cache_type_v": "auto"},
                8192,
            )

    def test_quality_profile_keeps_kv_at_full_precision(self):
        self.assertEqual(self._high_pressure("quality")["cache_type_k"], "f16")
        self.assertEqual(self._high_pressure("balanced")["cache_type_k"], "q8_0")
        self.assertEqual(self._high_pressure("speed")["cache_type_k"], "q8_0")

    def test_quality_profile_leaves_more_headroom(self):
        self.assertGreater(self._high_pressure("quality")["fit_target_mb"],
                           self._high_pressure("speed")["fit_target_mb"])

    def test_legacy_profile_names_still_work(self):
        # 老配置里的 auto / maximum 必须继续可用，且不得因改名而改变输出
        for legacy in ("auto", "maximum"):
            self.assertEqual(self._high_pressure(legacy)["profile"], "balanced")
        self.assertEqual(self._high_pressure("compatible")["profile"], "compatible")

    def test_unknown_profile_falls_back_to_balanced(self):
        self.assertEqual(self._high_pressure("nonsense")["profile"], "balanced")

    def test_explicit_cache_setting_overrides_profile(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 15500,
            "compute_capability": "12.0",
        }
        with mock.patch.object(run, "model_size_mb", return_value=15000), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-Q4_K_M.gguf"), {"block_count": 64}, gpu,
                {"profile": "maximum", "cache_type_k": "f16", "cache_type_v": "q4_0"},
                8192,
            )
        self.assertEqual(tuning["cache_type_k"], "f16")
        self.assertEqual(tuning["cache_type_v"], "q4_0")
        # maximum 档把显存余量压到 64MiB：权重比显存大时，余量每多一点
        # 就少一点权重驻留 GPU，实测 16GB 卡上比 256 快约 15%。
        self.assertEqual(tuning["fit_target_mb"], 64)

    def test_kv_estimate_honours_hybrid_attention_interval(self):
        # qwen35 每 4 层才有 1 层全注意力，其余是常数大小的循环状态。
        # 按 64 层全算会把 KV 高估 4 倍，进而误判上下文预算。
        meta = {
            "block_count": 65, "nextn_predict_layers": 1,
            "attention.head_count_kv": 4,
            "attention.key_length": 256, "attention.value_length": 256,
            "full_attention_interval": 4,
        }
        # 16 个注意力层 × 4 头 × (256+256) × 1.0625 字节 × 16384 token
        self.assertAlmostEqual(run.kv_cache_mb(meta, 16384, "q8_0", "q8_0"), 544.0, places=1)
        self.assertAlmostEqual(run.kv_cache_mb(meta, 16384, "q4_0", "q4_0"), 288.0, places=1)
        dense = dict(meta, full_attention_interval=1)
        self.assertAlmostEqual(run.kv_cache_mb(dense, 16384, "q8_0", "q8_0"), 544.0 * 4, places=1)

    def test_kv_estimate_returns_zero_without_metadata(self):
        self.assertEqual(run.kv_cache_mb({"block_count": 64}, 8192, "q8_0", "q8_0"), 0.0)

    def test_oversized_weights_are_reported_as_the_real_bottleneck(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
            "compute_capability": "12.0",
        }
        meta = {
            "general.architecture": "qwen35", "block_count": 65,
            "nextn_predict_layers": 1, "attention.head_count_kv": 4,
            "attention.key_length": 256, "attention.value_length": 256,
            "full_attention_interval": 4,
        }
        with mock.patch.object(run, "model_size_mb", return_value=15701), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-Q4_K_M.gguf"), meta, gpu,
                {"profile": "maximum", "cache_type_k": "q4_0", "cache_type_v": "q4_0"},
                65536,
            )
        note = " ".join(tuning["notes"])
        self.assertIn("超出可用显存", note)
        self.assertIn("ctx_size=65536", note)

    def _qwen38_meta(self):
        return {
            "general.architecture": "qwen35", "block_count": 65,
            "nextn_predict_layers": 1, "attention.head_count_kv": 4,
            "attention.key_length": 256, "attention.value_length": 256,
            "full_attention_interval": 4,
        }

    def test_vision_mode_reports_the_memory_it_costs(self):
        # 视觉模式要常驻 mmproj 并抬高 fit_target，足以把本来装得下的权重挤回 CPU。
        # 此前这段开销完全不参与预算计算，用户开了视觉只会觉得"莫名其妙变慢"。
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14849,
            "compute_capability": "12.0",
        }
        params = {"profile": "maximum", "cache_type_k": "q4_0", "cache_type_v": "q4_0"}
        with mock.patch.object(run, "model_size_mb", return_value=13590), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            without = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                params, 16384, vision=False, mmproj_mb=888,
            )
            with_vision = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                params, 16384, vision=True, mmproj_mb=888,
            )
        # 普通模式下这个模型装得下，不该有显存告警
        self.assertNotIn("留在 CPU", " ".join(without["notes"]))
        # 视觉模式下必须点明代价，并给出"改用普通模式"这个可执行建议
        note = " ".join(with_vision["notes"])
        self.assertIn("视觉模式", note)
        self.assertIn("888", note)
        self.assertIn("普通模式", note)

    def test_vision_reserves_more_than_text_mode(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14849,
            "compute_capability": "12.0",
        }
        params = {"profile": "maximum", "cache_type_k": "q4_0", "cache_type_v": "q4_0"}
        with mock.patch.object(run, "model_size_mb", return_value=13590), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            text = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                params, 16384, vision=False, mmproj_mb=888,
            )["fit_target_mb"]
            vis = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                params, 16384, vision=True, mmproj_mb=888,
            )["fit_target_mb"]
        self.assertGreater(vis, text)
        # 早先假设"预留必须覆盖 mmproj 自身体积"，实测证明这个假设是错的：
        # 预留 1488MiB 时文本 10.5 t/s、单图 9.9s；收到 768MiB 后
        # 变成 17.7 t/s、单图 7.5s —— 预留过头会把权重挤到 CPU，
        # 反过来拖慢视觉编码本身。所以只要求比文本模式多留，不要求覆盖 mmproj。
        self.assertLess(vis, 888)

    def test_explicit_fit_target_still_wins_in_vision_mode(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14849,
            "compute_capability": "12.0",
        }
        with mock.patch.object(run, "model_size_mb", return_value=13590), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                {"profile": "maximum", "cache_type_k": "q4_0",
                 "cache_type_v": "q4_0", "fit_target_mb": 333},
                16384, vision=True, mmproj_mb=888,
            )
        self.assertEqual(tuning["fit_target_mb"], 333)

    def test_no_free_vram_points_at_other_processes_not_at_the_quant(self):
        # 显存被别的进程占满时，"换用 ≤0.0GB 的量化档位" 是无意义的建议
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 143,
            "compute_capability": "12.0",
        }
        meta = {
            "general.architecture": "qwen35", "block_count": 65,
            "nextn_predict_layers": 1, "attention.head_count_kv": 4,
            "attention.key_length": 256, "attention.value_length": 256,
            "full_attention_interval": 4,
        }
        with mock.patch.object(run, "model_size_mb", return_value=15701), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-Q4_K_M.gguf"), meta, gpu,
                {"profile": "maximum", "cache_type_k": "q4_0", "cache_type_v": "q4_0"},
                16384,
            )
        note = " ".join(tuning["notes"])
        self.assertIn("其它进程", note)
        self.assertNotIn("量化档位", note)

    def test_context_advisory_names_a_ctx_that_actually_fits(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
            "compute_capability": "12.0",
        }
        meta = {
            "general.architecture": "qwen35", "block_count": 65,
            "nextn_predict_layers": 1, "attention.head_count_kv": 4,
            "attention.key_length": 256, "attention.value_length": 256,
            "full_attention_interval": 4,
        }
        # 权重装得下，但 65536 的 KV（2176MiB）会把它挤出去
        with mock.patch.object(run, "model_size_mb", return_value=13500), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-IQ4_XS.gguf"), meta, gpu,
                {"profile": "maximum", "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
                65536,
            )
        note = " ".join(tuning["notes"])
        self.assertIn("建议 ctx_size", note)
        advised = int(note.split("建议 ctx_size ≤")[1].split()[0])
        self.assertLess(advised, 65536)
        # 建议值必须真的装得下：权重 + 该上下文的 KV + 开销 ≤ 可用显存
        self.assertLessEqual(
            13500 + run.kv_cache_mb(meta, advised, "q8_0", "q8_0") + 600, 14919
        )

    @staticmethod
    def _write_gguf(path, tensors, alignment=32):
        """写一个最小 GGUF v3：只有 general.alignment 和按顺序排列的 F32 一维张量。"""
        def s(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw
        header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
        header += s("general.alignment") + struct.pack("<II", 4, alignment)
        offset, data = 0, b""
        for name, n_bytes in tensors:
            header += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n_bytes // 4)
            header += struct.pack("<IQ", 0, offset)
            padded = -(-n_bytes // alignment) * alignment
            data += b"\0" * padded
            offset += padded
        header += b"\0" * (-len(header) % alignment)
        path.write_bytes(header + data)

    def test_gpu_weight_excludes_cpu_embedding_and_idle_mtp_layer(self):
        # 文件体积会高估显存需求：token_embd 始终留在 CPU，MTP 层不开推测解码时不加载。
        # 实测 NVFP4 文件 16332MiB，真正进显存 15098MiB；按文件体积算会误判差距。
        mib = 1024 * 1024
        meta = {"block_count": 2, "nextn_predict_layers": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            self._write_gguf(path, [
                ("token_embd.weight", 3 * mib),
                ("blk.0.ffn_up.weight", 5 * mib),
                ("output.weight", 2 * mib),
                ("blk.1.nextn.eh_proj.weight", 1 * mib),
            ])
            run.TENSOR_SIZE_CACHE.clear()
            self.assertAlmostEqual(run.gpu_weight_mb(path, meta), 7.0, places=3)
            self.assertAlmostEqual(run.gpu_weight_mb(path, meta, mtp_enabled=True), 8.0, places=3)

    def test_auto_context_is_budgeted_at_its_floor(self):
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
            "compute_capability": "12.0",
        }
        with mock.patch.object(run, "model_size_mb", return_value=12726), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                {"profile": "balanced"}, 0,
            )
        self.assertTrue(tuning["gpu_fits"])
        self.assertEqual(tuning["min_ctx"], 8192)
        self.assertIn("自动适配", " ".join(tuning["notes"]))

    def test_mtp_is_not_offered_when_weights_already_spill(self):
        # 实测：权重已有层留在 CPU 时再开 MTP 反而更慢（12.8 -> 7.0 t/s）
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
            "compute_capability": "12.0",
        }
        with mock.patch.object(run, "model_size_mb", return_value=15099), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf"), self._qwen38_meta(), gpu,
                {"profile": "speed"}, 0,
            )
        self.assertFalse(tuning["gpu_fits"])
        self.assertFalse(tuning["gpu_fits_mtp"])
        self.assertIn("差约", " ".join(tuning["notes"]))

    def test_mtp_is_offered_when_everything_fits(self):
        # 实测 IQ4_XS 全驻留 + MTP：代码 63 t/s、散文 39 t/s（不开 MTP 为 27 t/s）
        gpu = {
            "selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
            "compute_capability": "12.0",
        }
        with mock.patch.object(run, "model_size_mb", return_value=12726), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 60}
        ):
            tuning = run.performance_tuning(
                Path("Qwen3.8-27B-UD-IQ4_XS.gguf"), self._qwen38_meta(), gpu,
                {"profile": "speed"}, 0,
            )
        self.assertTrue(tuning["gpu_fits_mtp"])

    def test_auto_context_never_passes_an_explicit_zero(self):
        # llama.cpp 把显式的 --ctx-size 0 当成"要求模型完整上下文"（262144），
        # 不缩上下文而是卸层，实测 27B 掉到 2.3 t/s。自动模式只能不传该参数。
        help_text = "-ngl, --gpu-layers N  max layers: 'auto', 'all'\n--fit on\n--fit-target\n--fit-ctx\n-fa\n-b\n-ub"
        gpu = {"selected_backend": "cuda", "vram_mb": 16311, "vram_free_mb": 14919,
               "compute_capability": "12.0"}
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server.exe"
            binary.write_bytes(b"")
            model = Path(tmp) / "Qwen3.8-27B-UD-IQ4_XS.gguf"
            model.write_bytes(b"")
            cfg = {"model": {"model_file": model.name}, "server": {"ctx_size": 0},
                   "performance": {"profile": "balanced"}}
            with mock.patch.object(run, "find_binary", return_value=binary), \
                    mock.patch.object(run, "resolve_model", return_value=model), \
                    mock.patch.object(run, "gguf_meta", return_value=self._qwen38_meta()), \
                    mock.patch.object(run, "accel", return_value=gpu), \
                    mock.patch.object(run, "help_text", return_value=help_text), \
                    mock.patch.object(run, "model_size_mb", return_value=12726), \
                    mock.patch.object(run, "meminfo", return_value={"avail_gb": 60}):
                args = run.runtime(cfg, "server")["args"]
        self.assertNotIn("--ctx-size", args)
        self.assertEqual(args[args.index("--fit-ctx") + 1], "8192")

    def test_vram_hog_note_skips_desktop_and_llama_and_converts_to_context(self):
        consumers = [("llama-server.exe", 15022), ("dwm.exe", 274), ("NVIDIA Overlay.exe", 195),
                     ("Code.exe", 153), ("csrss.exe", 65)]
        with mock.patch.object(run, "gpu_memory_consumers", return_value=consumers):
            note = run.vram_hog_note(self._qwen38_meta(), "q4_0")
        self.assertIn("348MiB", note)
        self.assertIn("NVIDIA Overlay", note)
        self.assertNotIn("dwm", note)
        self.assertNotIn("llama-server", note)
        self.assertIn("K 上下文", note)

    def test_moe_layer_estimate_uses_resident_weight_size(self):
        gpu = {"vram_free_mb": 6144}
        meta = {"general.architecture": "qwen3moe", "block_count": 40}
        with mock.patch.object(run, "model_size_mb", return_value=20 * 1024), mock.patch.object(
            run, "meminfo", return_value={"avail_gb": 64}
        ):
            layers = run.auto_layers(Path("model.gguf"), gpu, False, None, meta)
        self.assertGreater(layers, 0)
        self.assertLess(layers, 40)

    def test_gpu_auto_value_must_appear_near_option(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "llama-server"
            binary.write_bytes(b"x")
            key = str(binary.resolve())
            run.HELP_CACHE[key] = "-ngl, --gpu-layers N  exact, 'auto', or 'all'\n--color auto"
            self.assertTrue(run.supports_value(binary, "-ngl", "auto"))
            run.HELP_CACHE[key] = "-ngl, --gpu-layers N  exact number only\n--color auto"
            self.assertFalse(run.supports_value(binary, "-ngl", "auto"))
            run.HELP_CACHE.pop(key, None)


if __name__ == "__main__":
    unittest.main()
