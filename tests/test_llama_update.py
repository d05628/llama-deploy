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


if __name__ == "__main__":
    unittest.main()
