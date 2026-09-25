import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import deploy


def asset(name, size=123, digest=""):
    value = {
        "name": name,
        "size": size,
        "state": "uploaded",
        "browser_download_url": (
            "https://github.com/ggml-org/llama.cpp/releases/download/b10717/" + name
        ),
    }
    if digest:
        value["digest"] = digest
    return value


class LlamaUpdateTests(unittest.TestCase):
    def make_deployer(self, backend="cpu", cuda_version=""):
        instance = object.__new__(deploy.Deployer)
        instance.sys = SimpleNamespace(is_arm=False, arch="AMD64")
        instance.actual_backend = backend
        instance.gpu_info = {"cuda_version": cuda_version}
        return instance

    def test_cuda_asset_matches_driver_and_runtime_version(self):
        instance = self.make_deployer("cuda", "12.9")
        assets = [
            asset("llama-b10717-bin-win-cuda-13.3-x64.zip"),
            asset("cudart-llama-bin-win-cuda-13.3-x64.zip"),
            asset("llama-b10717-bin-win-cuda-12.4-x64.zip"),
            asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ]

        main, runtime = instance._find_best_asset(assets, announce=False)

        self.assertIn("cuda-12.4-x64", main["name"])
        self.assertIn("cuda-12.4-x64", runtime["name"])

    def test_blackwell_driver_selects_newest_compatible_cuda_pair(self):
        instance = self.make_deployer("cuda", "13.3")
        instance.gpu_info["compute_capability"] = "12.0"
        assets = [
            asset("llama-b10717-bin-win-cuda-13.3-x64.zip"),
            asset("cudart-llama-bin-win-cuda-13.3-x64.zip"),
            asset("llama-b10717-bin-win-cuda-12.4-x64.zip"),
            asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ]
        main, runtime = instance._find_best_asset(assets, announce=False)
        self.assertIn("cuda-13.3-x64", main["name"])
        self.assertIn("cuda-13.3-x64", runtime["name"])

    def test_driver_minor_version_does_not_block_same_major_package(self):
        # CUDA 次版本兼容：驱动 581.80 报告 13.0，实测可正常运行 cuda-13.4 包。
        # 此前按次版本过滤会退回 12.4，丢掉 RTX 50 的 sm_120 原生内核。
        instance = self.make_deployer("cuda", "13.0")
        instance.gpu_info["compute_capability"] = "12.0"
        assets = [
            asset("llama-b11160-bin-win-cuda-13.4-x64.zip"),
            asset("cudart-llama-bin-win-cuda-13.4-x64.zip"),
            asset("llama-b11160-bin-win-cuda-12.4-x64.zip"),
            asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ]
        main, runtime = instance._find_best_asset(assets, announce=False)
        self.assertIn("cuda-13.4-x64", main["name"])
        self.assertIn("cuda-13.4-x64", runtime["name"])

    def test_incomplete_cuda_release_is_rejected(self):
        instance = self.make_deployer("cuda", "12.9")
        assets = [asset("llama-b10718-bin-win-cuda-12.4-x64.zip")]

        self.assertEqual((None, None), instance._find_best_asset(assets, announce=False))

    def test_cpu_match_is_exact_and_arch_specific(self):
        instance = self.make_deployer("cpu")
        assets = [
            asset("llama-b10717-bin-win-openvino-2026.3-x64.zip"),
            asset("llama-b10717-bin-win-cpu-arm64.zip"),
            asset("llama-b10717-bin-win-cpu-x64.zip"),
        ]

        main, runtime = instance._find_best_asset(assets, announce=False)

        self.assertEqual("llama-b10717-bin-win-cpu-x64.zip", main["name"])
        self.assertIsNone(runtime)

    def test_release_resolver_skips_newest_incomplete_build(self):
        instance = self.make_deployer("cuda", "12.9")
        releases = [
            {
                "tag_name": "b10718",
                "draft": False,
                "assets": [asset("llama-b10718-bin-win-cuda-12.4-x64.zip")],
            },
            {
                "tag_name": "b10717",
                "draft": False,
                "assets": [
                    asset("llama-b10717-bin-win-cuda-12.4-x64.zip"),
                    asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
                ],
            },
        ]
        instance._fetch_github_json = mock.Mock(return_value=releases)

        release = instance._resolve_windows_release()

        self.assertEqual("b10717", release["tag_name"])

    def test_release_resolver_falls_back_to_official_feed(self):
        instance = self.make_deployer("cpu")
        instance._fetch_github_json = mock.Mock(side_effect=RuntimeError("rate limited"))
        instance._release_tags_from_atom = mock.Mock(return_value=["b10717"])
        instance._release_assets_from_html = mock.Mock(
            return_value=[asset("llama-b10717-bin-win-cpu-x64.zip")]
        )

        release = instance._resolve_windows_release()

        self.assertEqual("b10717", release["tag_name"])

    def test_mirror_download_rejects_html_then_uses_direct_url(self):
        printer = lambda *args: None
        downloader = deploy.Downloader(
            {"download": {"github_mirror": "https://mirror.invalid", "retries": 1}},
            printer,
        )
        urls = downloader.github_urls(
            "https://github.com/ggml-org/llama.cpp/releases/download/b10717/test.zip"
        )
        calls = []

        def fake_download(url, destination, expected_size=0):
            calls.append(url)
            if len(calls) == 1:
                destination.write_text("<html>mirror landing page</html>", encoding="utf-8")
            else:
                with zipfile.ZipFile(destination, "w") as archive:
                    archive.writestr("llama-server.exe", b"test")
            return True

        downloader.download_file = fake_download
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "test.zip"
            ok = downloader.download_from_urls(
                urls, destination, validator=zipfile.is_zipfile
            )

        self.assertTrue(ok)
        self.assertEqual(2, len(calls))
        self.assertEqual("https://github.com", calls[-1][:18])

    def test_release_asset_sha256_is_verified(self):
        instance = self.make_deployer()
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "asset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("llama-server.exe", b"test")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

            self.assertTrue(
                instance._validate_release_asset(
                    archive_path, {"digest": "sha256:" + digest}
                )
            )
            self.assertFalse(
                instance._validate_release_asset(
                    archive_path, {"digest": "sha256:" + ("0" * 64)}
                )
            )

    def test_version_parser_supports_old_and_new_output(self):
        instance = self.make_deployer()
        self.assertEqual(9934, instance._parse_llama_build("version: 9934 (32e41fa5b)"))
        self.assertEqual(
            10717,
            instance._parse_llama_build(
                "version: 0.3.0-dev (build 10717, commit a32af33de)"
            ),
        )

    def test_upgrade_restores_backup_when_verification_fails(self):
        instance = self.make_deployer()
        backup = Path("llama.cpp.backup-test")
        instance.upgrade_llama = True
        instance.llama_backup_dir = backup
        instance.p = lambda *args: None
        instance._detect_system = mock.Mock()
        instance._check_deps = mock.Mock()
        instance._download_llama = mock.Mock()
        instance._build_llama = mock.Mock()
        instance._verify_llama_only = mock.Mock(
            side_effect=RuntimeError("verification failed")
        )
        instance._step = lambda _num, _total, _name, function: function()
        instance._restore_llama_backup = mock.Mock()
        instance._relocate_bins = mock.Mock()
        instance._cleanup_llama_backup = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            instance.run()

        instance._restore_llama_backup.assert_called_once_with(backup)
        instance._cleanup_llama_backup.assert_not_called()

    def _engine_dirs(self, tmp):
        base = Path(tmp)
        current, previous = base / "llama.cpp", base / "llama.cpp.previous"
        patches = [
            mock.patch.object(deploy, "BASE_DIR", base),
            mock.patch.object(deploy, "LLAMA_DIR", current),
            mock.patch.object(deploy, "LLAMA_PREVIOUS_DIR", previous),
            mock.patch.object(deploy, "running_llama_processes", return_value=[]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return base, current, previous

    def test_successful_upgrade_keeps_previous_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, current, previous = self._engine_dirs(tmp)
            backup = base / "llama.cpp.backup-1"
            backup.mkdir()
            (backup / "marker").write_text("old")
            previous.mkdir()
            (previous / "marker").write_text("older")
            instance = self.make_deployer()
            instance.llama_backup_dir = backup
            instance._keep_previous_engine(backup)
            self.assertFalse(backup.exists())
            # 只保留紧挨着的上一版，更早的那份被替换
            self.assertEqual((previous / "marker").read_text(), "old")

    def test_rollback_swaps_engines_and_can_roll_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            _base, current, previous = self._engine_dirs(tmp)
            current.mkdir()
            (current / "marker").write_text("new")
            previous.mkdir()
            (previous / "marker").write_text("old")
            with mock.patch.object(deploy, "engine_version", return_value="b1"):
                self.assertEqual(deploy.rollback_llama(), 0)
                self.assertEqual((current / "marker").read_text(), "old")
                self.assertEqual((previous / "marker").read_text(), "new")
                self.assertEqual(deploy.rollback_llama(), 0)
                self.assertEqual((current / "marker").read_text(), "new")

    def test_rollback_restores_current_when_previous_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _base, current, previous = self._engine_dirs(tmp)
            current.mkdir()
            (current / "marker").write_text("new")
            previous.mkdir()
            (previous / "marker").write_text("broken")
            versions = iter(["b2", ""])  # 回退前能跑，回退后的旧版跑不起来
            with mock.patch.object(deploy, "engine_version", side_effect=lambda _d: next(versions)):
                self.assertEqual(deploy.rollback_llama(), 1)
            self.assertEqual((current / "marker").read_text(), "new")
            self.assertEqual((previous / "marker").read_text(), "broken")

    def test_upgrade_fails_when_new_engine_cannot_see_the_gpu(self):
        # 驱动不够新时 llama.cpp 会静默退回 CPU，速度差一个数量级却不报错
        instance = self.make_deployer("cuda")
        instance.server_bin = Path("llama-server.exe")
        cpu_only = SimpleNamespace(returncode=0, stdout="Available devices:\n", stderr="")
        with mock.patch.object(deploy.subprocess, "run", return_value=cpu_only):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                instance._verify_gpu_devices()
        gpu = SimpleNamespace(returncode=0, stderr="",
                              stdout="Available devices:\n  CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB)\n")
        with mock.patch.object(deploy.subprocess, "run", return_value=gpu):
            instance._verify_gpu_devices()


if __name__ == "__main__":
    unittest.main()
