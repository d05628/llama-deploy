import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
