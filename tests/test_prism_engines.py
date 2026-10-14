import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import deploy
import engines
import manager
import run


def gguf(path, kind):
    name = b"weight"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
                     + struct.pack("<Q", len(name)) + name
                     + struct.pack("<IQIQ", 1, 128, kind, 0) + b"\0" * 64)


class PrismEngineTests(unittest.TestCase):
    def test_real_tensor_types_detect_renamed_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind, quant in engines.PRISM_TYPES.items():
                path = Path(temporary) / f"renamed-{kind}.gguf"
                gguf(path, kind)
                self.assertEqual(engines.prism_quant(path), quant)
                self.assertEqual(engines.provider_for({}, path), "prism")

    def test_normal_gguf_and_original_bonsai_remain_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Bonsai-27B-Q1_0.gguf"
            gguf(path, 41)
            self.assertEqual(engines.provider_for({}, path), "upstream")

    def test_both_quant_labels_are_precise(self):
        for quant in ("PQ2_0", "PTQ1_0"):
            self.assertEqual(manager.detect_quant(f"Ternary-Bonsai-2-27B-{quant}.gguf"), quant)

    def test_forced_upstream_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported.*PQ2_0"):
            engines.provider_for({"engine": {"provider": "upstream"}}, "model-PQ2_0.gguf")

    def test_missing_prism_does_not_search_stock_or_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                engines.select_prism_binary(temporary, {}, "model-PTQ1_0.gguf", "llama-server", "cuda")

    def test_selection_checks_provenance_backend_and_explicit_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chosen"
            root.mkdir()
            (root / "llama-server.exe").write_bytes(b"exe")
            (root / "ggml-cuda.dll").write_bytes(b"dll")
            info = {"provider": "prism", "backend": "cuda", "quant_types": ["PQ2_0", "PTQ1_0"],
                    "binary_sha256": engines.file_sha256(root / "llama-server.exe"),
                    "runtime_files_sha256": engines.runtime_file_hashes(root),
                    "executable_files_sha256": engines.executable_file_hashes(root)}
            (root / engines.ENGINE_INFO_NAME).write_text(json.dumps(info))
            cfg = {"engine": {"directory": "chosen"}}
            self.assertEqual(engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", "llama-server", "cuda"),
                             root / "llama-server.exe")
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", "llama-server", "cpu")
            (root / "ggml-cuda.dll").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "DLL"):
                engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", "llama-server", "cuda")

    def test_selection_rejects_changed_missing_and_extra_executables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chosen"
            root.mkdir()
            for name in ("llama-server", "llama-cli", "llama-bench"):
                (root / (name + ".exe")).write_bytes(name.encode())
            (root / "ggml.dll").write_bytes(b"dll")
            info = {"provider": "prism", "backend": "cpu", "quant_types": ["PQ2_0", "PTQ1_0"],
                    "binary_sha256": engines.file_sha256(root / "llama-server.exe"),
                    "runtime_files_sha256": engines.runtime_file_hashes(root),
                    "executable_files_sha256": engines.executable_file_hashes(root)}
            (root / engines.ENGINE_INFO_NAME).write_text(json.dumps(info))
            cfg = {"engine": {"directory": "chosen"}}
            for name in ("llama-cli", "llama-bench"):
                binary = root / (name + ".exe")
                self.assertEqual(engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", name, "cpu"), binary)
                binary.write_bytes(b"stock engine replacement")
                with self.assertRaisesRegex(RuntimeError, "EXE"):
                    engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", name, "cpu")
                binary.unlink()
                with self.assertRaisesRegex(RuntimeError, "EXE"):
                    engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", name, "cpu")
                # A missing registered CLI/bench also blocks selection of server.
                with self.assertRaisesRegex(RuntimeError, "EXE"):
                    engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", "llama-server", "cpu")
                binary.write_bytes(name.encode())
            (root / "unexpected.exe").write_bytes(b"another engine")
            with self.assertRaisesRegex(RuntimeError, "EXE"):
                engines.select_prism_binary(temporary, cfg, "m-PQ2_0.gguf", "llama-server", "cpu")

    def test_nonwindows_prism_selection_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported.*Windows"):
            engines.select_prism_binary("unused", {}, "m-PQ2_0.gguf", "llama-server", "cpu", windows=False)

    def test_runtime_fails_before_stock_binary_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "m.gguf"
            gguf(model, 142)
            with mock.patch.object(run, "resolve_model", return_value=model), \
                    mock.patch.object(run, "BASE_DIR", Path(temporary)), \
                    mock.patch.object(run, "accel", return_value={"selected_backend": "cuda"}), \
                    mock.patch.object(run, "find_binary") as stock:
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    run.runtime({"model": {"model_file": model.name}}, "server")
                stock.assert_not_called()

    def test_bonsai_template_gets_explicit_thinking_switch_and_effort(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "llama-server.exe"
            binary.write_bytes(b"exe")
            model = Path(temporary) / "model-PQ2_0.gguf"
            gguf(model, 142)
            cfg = {"model": {"model_file": model.name}, "server": {"ctx_size": 1024}}
            meta = {"general.architecture": "qwen35", "block_count": 64,
                    "tokenizer.chat_template": "{% if enable_thinking %}<think>{% endif %}"}
            gpu = {"selected_backend": "cpu", "vram_mb": 0}
            with mock.patch.object(run, "resolve_model", return_value=model), \
                    mock.patch.object(engines, "select_prism_binary", return_value=binary), \
                    mock.patch.object(run, "gguf_meta", return_value=meta), \
                    mock.patch.object(run, "accel", return_value=gpu), \
                    mock.patch.object(run, "help_text", return_value="--chat-template-kwargs --reasoning --jinja"), \
                    mock.patch.object(run, "model_size_mb", return_value=7000), \
                    mock.patch.object(run, "meminfo", return_value={"avail_gb": 32}):
                args = run.runtime(cfg, "server")["args"]
                self.assertEqual(json.loads(args[args.index("--chat-template-kwargs") + 1]), {"enable_thinking": False})
                cfg["server"].update(enable_thinking=True, reasoning_effort="medium")
                args = run.runtime(cfg, "server")["args"]
                self.assertEqual(json.loads(args[args.index("--chat-template-kwargs") + 1]),
                                 {"enable_thinking": True, "reasoning_effort": "medium"})
                cfg["server"]["reasoning_effort"] = "low"
                with self.assertRaisesRegex(RuntimeError, "low"):
                    run.runtime(cfg, "server")


class PrismInstallTests(unittest.TestCase):
    def make_release(self, source):
        files = {
            "llama-prism-b10743-adfffbe-bin-win-cuda-12.4-x64.zip":
                ["llama-server.exe", "llama-cli.exe", "llama-bench.exe", "llama.dll", "ggml.dll", "ggml-cuda.dll"],
            "cudart-llama-bin-win-cuda-12.4-x64.zip": ["cudart64_12.dll", "cublas64_12.dll"],
        }
        assets = []
        for name, members in files.items():
            archive = source / name
            with zipfile.ZipFile(archive, "w") as z:
                for member in members:
                    z.writestr("release/" + member, b"x" * 2048)
            assets.append({"name": name, "size": archive.stat().st_size,
                           "digest": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
                           "browser_download_url": f"https://github.com/{engines.PRISM_REPO}/releases/download/{engines.PRISM_TAG}/{name}"})
        return {"tag_name": engines.PRISM_TAG, "assets": assets}

    def install(self, base, source, release, fail_runtime=False):
        def download(downloader, urls, destination, size, validate):
            name = urls[-1].rsplit("/", 1)[-1]
            destination.write_bytes((source / name).read_bytes())
            if fail_runtime and name.startswith("cudart"):
                return False
            return destination.stat().st_size == size and validate(destination)
        with mock.patch.object(deploy, "BASE_DIR", base), \
                mock.patch.object(deploy, "SystemInfo", return_value=SimpleNamespace(is_windows=True, is_arm=False)), \
                mock.patch.object(deploy.GPUDetector, "detect", return_value={"recommended_backend": "cuda", "cuda_version": "13.0"}), \
                mock.patch.object(deploy.Deployer, "_fetch_github_json", return_value=release), \
                mock.patch.object(deploy.Downloader, "download_from_urls", autospec=True, side_effect=download), \
                mock.patch.object(deploy.subprocess, "run") as execution:
            result = deploy.install_prism({}, cuda_version="12.4")
            execution.assert_not_called()
            return result

    def test_complete_install_preserves_stock_and_config_without_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            stock = base / "llama.cpp"
            stock.mkdir()
            sentinel = stock / "keep.txt"
            sentinel.write_text("original")
            cfg = base / "config.jsonc"
            cfg.write_text("original config")
            release = self.make_release(source)
            installed = self.install(base, source, release)
            self.assertEqual(engines.engine_info(installed)["quant_types"], ["PQ2_0", "PTQ1_0"])
            self.assertEqual(engines.engine_info(installed)["commit"], "adfffbe")
            self.assertEqual(engines.engine_info(installed)["binary_sha256"],
                             hashlib.sha256((installed / "llama-server.exe").read_bytes()).hexdigest())
            self.assertEqual(engines.engine_info(installed)["runtime_files_sha256"],
                             engines.runtime_file_hashes(installed))
            self.assertEqual(engines.engine_info(installed)["executable_files_sha256"],
                             engines.executable_file_hashes(installed))
            self.assertIn("llama-bench.exe", engines.engine_info(installed)["executable_files_sha256"])
            self.assertIn("cuda-12.4", str(installed))
            self.assertEqual(sentinel.read_text(), "original")
            self.assertEqual(cfg.read_text(), "original config")
            self.assertEqual(self.install(base, source, release), installed)
            (installed / "ggml-cuda.dll").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "DLL"):
                self.install(base, source, release)

    def test_legacy_server_and_dll_manifests_need_explicit_migration_without_download(self):
        for missing in (("executable_files_sha256",), ("runtime_files_sha256", "executable_files_sha256")):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                source = base / "source"
                source.mkdir()
                release = self.make_release(source)
                installed = self.install(base, source, release)
                manifest = installed / engines.ENGINE_INFO_NAME
                old_info = engines.engine_info(installed)
                for key in missing:
                    old_info.pop(key)
                manifest.write_text(json.dumps(old_info))
                with self.assertRaisesRegex(RuntimeError, "登记缺失"):
                    engines.select_prism_binary(base, {}, "m-PQ2_0.gguf", "llama-server", "cuda")
                with mock.patch.object(deploy, "BASE_DIR", base), \
                        mock.patch.object(deploy.Deployer, "_fetch_github_json", return_value=release), \
                        mock.patch.object(deploy.Downloader, "download_from_urls") as download, \
                        mock.patch.object(deploy.subprocess, "run") as execution:
                    self.assertEqual(deploy.register_prism_manifest(installed), manifest)
                    download.assert_not_called()
                    execution.assert_not_called()
                    engines.verify_prism_integrity(installed)
                    self.assertEqual(engines.engine_info(installed)["asset_digest"], old_info["asset_digest"])
                    self.assertEqual(engines.engine_info(installed)["binary_sha256"], old_info["binary_sha256"])
                    self.assertEqual(engines.engine_info(installed)["executable_files_sha256"],
                                     engines.executable_file_hashes(installed))
                    # Once hashes exist, this command cannot bless altered libraries.
                    (installed / "ggml-cuda.dll").write_bytes(b"changed")
                    with self.assertRaisesRegex(RuntimeError, "DLL"):
                        deploy.register_prism_manifest(installed)

    def test_registration_never_resigns_registered_executable_or_dll_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            installed = self.install(base, source, self.make_release(source))
            manifest = installed / engines.ENGINE_INFO_NAME
            complete = engines.engine_info(installed)
            cases = [((), "llama-cli.exe", "replace"), ((), "llama-bench.exe", "replace"),
                     ((), "llama-cli.exe", "delete"), ((), "llama-bench.exe", "delete"),
                     ((), "extra.exe", "add"),
                     (("executable_files_sha256",), "ggml-cuda.dll", "replace"),
                     (("executable_files_sha256", "runtime_files_sha256"), "llama-server.exe", "replace"),
                     (("runtime_files_sha256",), "llama-cli.exe", "replace")]
            with mock.patch.object(deploy, "BASE_DIR", base), \
                    mock.patch.object(deploy.Deployer, "_fetch_github_json") as network:
                before = manifest.read_bytes()
                self.assertEqual(deploy.register_prism_manifest(installed), manifest)
                self.assertEqual(manifest.read_bytes(), before, "complete manifest must not be rewritten")
                for missing, filename, mutation in cases:
                    with self.subTest(missing=missing, filename=filename, mutation=mutation):
                        info = {key:value for key,value in complete.items() if key not in missing}
                        manifest.write_text(json.dumps(info))
                        before = manifest.read_bytes()
                        target = installed / filename
                        original = target.read_bytes() if target.exists() else None
                        if mutation == "delete":
                            target.unlink()
                        else:
                            target.write_bytes(b"changed or extra executable/library")
                        with self.assertRaisesRegex(RuntimeError, "EXE|DLL|binary"):
                            deploy.register_prism_manifest(installed)
                        self.assertEqual(manifest.read_bytes(), before)
                        if original is None:
                            target.unlink()
                        else:
                            target.write_bytes(original)
                network.assert_not_called()

    def test_integrity_rejects_missing_added_and_changed_dll_or_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, dll = root / "llama-server.exe", root / "ggml.dll"
            binary.write_bytes(b"exe")
            dll.write_bytes(b"dll")
            info = {"binary_sha256": engines.file_sha256(binary),
                    "runtime_files_sha256": engines.runtime_file_hashes(root),
                    "executable_files_sha256": engines.executable_file_hashes(root)}
            self.assertTrue(engines.verify_prism_integrity(root, info))
            added = root / "extra.dll"
            added.write_bytes(b"unexpected")
            with self.assertRaisesRegex(RuntimeError, "DLL"):
                engines.verify_prism_integrity(root, info)
            added.unlink()
            dll.unlink()
            with self.assertRaisesRegex(RuntimeError, "DLL"):
                engines.verify_prism_integrity(root, info)
            dll.write_bytes(b"dll")
            binary.write_bytes(b"altered")
            with self.assertRaisesRegex(RuntimeError, "binary"):
                engines.verify_prism_integrity(root, info)

    def test_legacy_migration_rechecks_original_hashes_after_release_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            release = self.make_release(source)
            installed = self.install(base, source, release)
            manifest = installed / engines.ENGINE_INFO_NAME
            complete = engines.engine_info(installed)
            for missing, filename in (("executable_files_sha256", "ggml-cuda.dll"),
                                      ("runtime_files_sha256", "llama-cli.exe")):
                with self.subTest(missing=missing):
                    info = {key:value for key,value in complete.items() if key != missing}
                    manifest.write_text(json.dumps(info))
                    before = manifest.read_bytes()
                    target = installed / filename
                    original = target.read_bytes()

                    def metadata_lookup(_):
                        target.write_bytes(b"changed during metadata lookup")
                        return release

                    with mock.patch.object(deploy, "BASE_DIR", base), \
                            mock.patch.object(deploy.Deployer, "_fetch_github_json", side_effect=metadata_lookup), \
                            mock.patch.object(deploy.subprocess, "run") as execution:
                        with self.assertRaisesRegex(RuntimeError, "EXE|DLL"):
                            deploy.register_prism_manifest(installed)
                        execution.assert_not_called()
                    self.assertEqual(manifest.read_bytes(), before)
                    target.write_bytes(original)

    def test_runtime_download_failure_leaves_no_published_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                self.install(base, source, self.make_release(source), fail_runtime=True)
            self.assertEqual(list((base / "engines" / "prism" / engines.PRISM_TAG).iterdir()), [])

    def test_missing_digest_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            release = self.make_release(base)
            release["assets"][0]["digest"] = ""
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                self.install(base, base, release)
            self.assertFalse((base / "engines").exists())

    def test_incomplete_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            release = self.make_release(base)
            destination = base / "engines" / "prism" / engines.PRISM_TAG / Path(release["assets"][0]["name"]).stem
            destination.mkdir(parents=True)
            (destination / "keep").write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "未覆盖"):
                self.install(base, base, release)
            self.assertEqual((destination / "keep").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
