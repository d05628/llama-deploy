import io
import json
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import probe_multi_gpu as probe


class ProbeTests(unittest.TestCase):
    def test_default_invocation_never_loads_or_creates_artifacts(self):
        with mock.patch.object(probe, "build_runtime") as build, mock.patch.object(probe.Path, "mkdir") as mkdir, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(probe.main([]), 0)
        build.assert_not_called()
        mkdir.assert_not_called()

    def test_runtime_uses_defaults_and_never_reads_actual_config(self):
        with mock.patch.object(probe.run, "parse_jsonc", side_effect=AssertionError("must not read config")), \
                mock.patch.object(probe.run, "runtime", return_value={"owned": True}) as runtime:
            self.assertEqual(probe.build_runtime("force", 65536), {"owned": True})
        cfg, mode = runtime.call_args.args
        self.assertEqual(mode, "server")
        self.assertEqual(cfg["model"]["model_file"], probe.MODEL)
        self.assertEqual((cfg["server"]["host"], cfg["server"]["port"], cfg["server"]["ctx_size"]),
                         ("127.0.0.1", 18199, 65536))
        self.assertFalse(cfg["server"]["enable_thinking"])
        self.assertEqual(cfg["gpu"]["multi_gpu"], "force")
        self.assertEqual(cfg["performance"]["cache_type_k"], "q8_0")
        self.assertEqual(cfg["performance"]["fit_target_mb"], 1024)

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(probe, "probe_mode") as run_mode, \
                mock.patch.object(probe, "validate_model_name", return_value=probe.MODEL):
            with self.assertRaises(FileExistsError):
                probe.main(["--run", "--output", folder])
            run_mode.assert_not_called()

    def test_missing_speed_evidence_cannot_report_a_successful_proof(self):
        incomplete = {"context_verified": True, "gpu_offload_verified": True,
                      "provider_tps_verified": False, "requests": [{"semantic_ok": True}]}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(probe, "probe_mode", return_value=incomplete), \
                mock.patch.object(probe, "validate_model_name", return_value=probe.MODEL), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(probe.main(["--run", "--output", str(Path(folder) / "fresh")]), 1)

    def test_generic_installed_model_and_modes_reach_same_runtime(self):
        name = "Qwen3.8-27B-UD-Q4_K_M.gguf"
        with mock.patch.object(probe.run, "runtime", return_value={}) as runtime:
            probe.build_runtime("auto", 16384, name)
        cfg = runtime.call_args.args[0]
        self.assertEqual(cfg["model"]["model_file"], name)
        self.assertEqual(cfg["gpu"]["multi_gpu"], "auto")
        self.assertEqual(cfg["server"]["ctx_size"], 16384)
        complete = {"context_verified": True, "gpu_offload_verified": True,
                    "provider_tps_verified": True, "requests": [{"semantic_ok": True}]}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(probe, "probe_mode", return_value=complete) as run_mode, \
                mock.patch.object(probe, "validate_model_name", return_value=name), mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(probe.main(["--run", "--model", name, "--ctx", "16384", "--modes", "auto", "force",
                                         "--output", str(Path(folder) / "fresh")]), 0)
        self.assertEqual([call.args[0] for call in run_mode.call_args_list], ["auto", "force"])
        self.assertTrue(all(call.args[-1] == name for call in run_mode.call_args_list))

    def test_model_must_be_an_existing_in_root_gguf_basename(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(probe.run, "MODELS_DIR", Path(folder)):
            path = Path(folder) / "nested"
            path.mkdir()
            (path / "model.gguf").write_bytes(b"test fixture")
            self.assertEqual(probe.validate_model_name("model.gguf"), "model.gguf")
            for name in ("../model.gguf", "C:\\model.gguf", "/model.gguf", "*.gguf", "model[1].gguf", "missing.gguf", "model.bin"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    probe.validate_model_name(name)

    def test_auto_accepts_real_single_or_layer_but_force_requires_two_cards(self):
        single = {"--device": "CUDA1", "--split-mode": "none"}
        multi = {"--device": "CUDA0,CUDA1", "--split-mode": "layer", "--tensor-split": "0.3,0.7"}
        self.assertTrue(probe.mode_matches("auto", single))
        self.assertTrue(probe.mode_matches("auto", multi))
        self.assertTrue(probe.mode_matches("force", multi))
        self.assertFalse(probe.mode_matches("force", single))
        self.assertFalse(probe.mode_matches("off", multi))
        self.assertFalse(probe.mode_matches("auto", {"--device": "CUDA0,CUDA1", "--split-mode": "layer"}))
        self.assertFalse(probe.mode_matches("auto", {"--device": "CUDA1,CUDA1", "--split-mode": "none"}))

    def test_stream_reports_provider_timing_and_actual_answer(self):
        events = [{"model": "bonsai", "choices": [{"delta": {"content": "42"}}]},
                  {"choices": [], "timings": {"predicted_per_second": 23.5}}]
        data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n"
        with mock.patch.object(probe.urllib.request, "urlopen", return_value=io.BytesIO(data)) as opened:
            result = probe.stream_probe("private-test-key", time.monotonic() + 10, "compute", r"\b42\b")
        self.assertEqual(result["provider_tps"], 23.5)
        self.assertEqual(result["answer"], "42")
        self.assertTrue(result["semantic_ok"])
        self.assertGreaterEqual(result["ttft_seconds"], 0)
        request = opened.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
        self.assertNotIn("private-test-key", json.dumps(result))

    def test_cleanup_terminates_only_the_owned_handle_and_waits(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 2), 0]
        probe.stop_owned(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=2), mock.call(timeout=1)])

    def test_slot_context_requires_generated_and_observed_single_parallel(self):
        line = "0.04 I srv load_model: initializing, n_slots = 1, n_ctx_slot = 32768, kv_unified = 'true'"
        self.assertEqual(probe.actual_contexts(line, ["-np", "1"]), [32768])
        self.assertEqual(probe.actual_contexts(line, ["--parallel", "1"]), [32768])
        self.assertEqual(probe.actual_contexts(line, ["-np", "2"]), [])
        self.assertEqual(probe.actual_contexts(line, []), [])
        self.assertEqual(probe.actual_contexts(line.replace("n_slots = 1", "n_slots = 2"), ["-np", "1"]), [])
        self.assertEqual(probe.actual_contexts("n_ctx = 16384\n" + line, ["-np", "1"]), [16384, 32768])

    def test_probe_uses_private_environment_and_validates_actual_load_log(self):
        deadline = time.monotonic() + 30
        flags = ["server.exe", "--host", "127.0.0.1", "--port", "18199", "--device", "CUDA0,CUDA1", "--ctx-size", "32768",
                 "--split-mode", "layer", "--tensor-split", "0.3,0.7"]
        rt = {"args": flags, "binary": Path("server.exe"), "cwd": "."}
        process = mock.Mock()
        process.poll.return_value = None
        def launch(args, **kwargs):
            kwargs["stdout"].write(b"llama_context: n_ctx = 32768\noffloaded 65/65 layers to GPU\nCUDA0 model buffer size = 2000 MiB\nCUDA1 model buffer size = 4000 MiB\n")
            return process
        health = mock.MagicMock()
        health.__enter__.return_value.status = 200
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(probe, "require_free_port"), \
                mock.patch.object(probe, "build_runtime", return_value=rt), \
                mock.patch.object(probe.run, "help_text", return_value="--api-key (env: LLAMA_API_KEY) --verbose"), \
                mock.patch.object(probe, "gpu_sample", return_value={"devices": []}), \
                mock.patch.object(probe, "request", return_value=health), \
                mock.patch.object(probe, "stream_probe", return_value={"semantic_ok": True, "provider_tps": 20}), \
                mock.patch.object(probe.subprocess, "Popen", side_effect=launch) as popen:
            result = probe.probe_mode("force", 32768, Path(folder), deadline)
        call = popen.call_args
        self.assertEqual(call.args[0], flags + ["--verbose"])
        self.assertEqual(result["probe_logging_flags"], ["--verbose"])
        key = call.kwargs["env"]["LLAMA_API_KEY"]
        self.assertGreater(len(key), 20)
        self.assertNotIn(key, " ".join(call.args[0]))
        self.assertNotIn(key, json.dumps(result))
        self.assertNotIn("--host", result["generated_flags"])
        self.assertEqual(result["actual_ctx"], [32768])
        self.assertTrue(result["gpu_offload_verified"])
        self.assertTrue(result["context_verified"])
        self.assertTrue(result["provider_tps_verified"])
        self.assertEqual(len(result["requests"]), 2)
        process.terminate.assert_called_once_with()

    def test_force_proof_rejects_incomplete_gpu_evidence(self):
        for device_names, size, counts in (("CUDA1", 2000, True), ("CUDA0,CUDA1", 0, True), ("CUDA0,CUDA1", 2000, False)):
            flags = ["server.exe", "--device", device_names, "--split-mode", "layer", "--tensor-split", "0.3,0.7", "-np", "1"]
            rt = {"args": flags, "binary": Path("server.exe"), "cwd": "."}
            process = mock.Mock()
            process.poll.return_value = None
            def launch(args, **kwargs):
                kwargs["stdout"].write(("initializing, n_slots = 1, n_ctx_slot = 32768\n"
                    + ("offloaded 65/65 layers to GPU\n" if counts else "") + "CUDA0 model buffer size = "
                    + str(size) + " MiB\nCUDA0 KV buffer size = 100 MiB\nCUDA1 model buffer size = 4000 MiB\n").encode())
                return process
            health = mock.MagicMock()
            health.__enter__.return_value.status = 200
            with self.subTest(devices=device_names, size=size), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(probe, "require_free_port"), mock.patch.object(probe, "build_runtime", return_value=rt), \
                    mock.patch.object(probe.run, "help_text", return_value="LLAMA_API_KEY --verbose"), \
                    mock.patch.object(probe, "gpu_sample", return_value={}), mock.patch.object(probe, "request", return_value=health), \
                    mock.patch.object(probe, "stream_probe", return_value={"semantic_ok": True, "provider_tps": 20}), \
                    mock.patch.object(probe.subprocess, "Popen", side_effect=launch):
                result = probe.probe_mode("force", 32768, Path(folder), time.monotonic() + 30)
            self.assertFalse(result["gpu_offload_verified"])
            self.assertTrue(result["context_verified"])

    def test_missing_verbose_support_fails_before_starting_a_server(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(probe, "require_free_port"), \
                mock.patch.object(probe, "build_runtime", return_value={"args": [], "binary": Path("server")}), \
                mock.patch.object(probe.run, "help_text", return_value="LLAMA_API_KEY"), \
                mock.patch.object(probe.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "verbose logging"):
                probe.probe_mode("force", 32768, Path(folder), time.monotonic() + 10)
            popen.assert_not_called()

    def test_budget_expiration_and_occupied_port_fail_before_launch(self):
        with self.assertRaises(TimeoutError):
            probe.remaining(time.monotonic() - 1)
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            with mock.patch.object(probe, "PORT", occupied.getsockname()[1]), self.assertRaises(OSError):
                probe.require_free_port()


if __name__ == "__main__":
    unittest.main()
