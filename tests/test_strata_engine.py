import unittest
from pathlib import Path
from unittest import mock

import run
import strata_engine


GPUS = [{"index": 0, "name": "RTX 3060 Ti", "vram_free_mb": 8024},
        {"index": 1, "name": "RTX 5060 Ti", "vram_free_mb": 15057}]


class StrataLaunchTests(unittest.TestCase):
    def test_main_card_is_the_one_with_most_free_memory(self):
        self.assertEqual(strata_engine.gpu_roles(GPUS), "1=main,0=helper")
        # a nearly full card is left out instead of over-committing it
        busy = [dict(GPUS[0], vram_free_mb=900), GPUS[1]]
        self.assertEqual(strata_engine.gpu_roles(busy), "1=main,0=off")
        self.assertEqual(strata_engine.gpu_roles(GPUS[1:]), "")

    def test_launch_uses_existing_files_local_access_and_configured_port(self):
        cfg = {"server": {"port": 8080, "ctx_size": 0}}
        args = strata_engine.launch_args(cfg, Path("m.gguf"), False, GPUS)
        def value(flag):
            return args[args.index(flag) + 1]
        self.assertEqual(value("--model-path"), "m.gguf")
        self.assertEqual(value("--context"), str(strata_engine.DEFAULT_CONTEXT))
        self.assertEqual(value("--port"), "8080")
        self.assertEqual(value("--access"), "local")
        self.assertEqual(value("--vision"), "off")
        self.assertEqual(value("--gpu-roles"), "1=main,0=helper")
        self.assertIn("--no-tui", args)
        self.assertEqual(Path(args[1]).name, "strata.py")

    def test_vision_runs_on_the_gpu_and_context_follows_the_server_setting(self):
        args = strata_engine.launch_args({"server": {"ctx_size": 32768}}, Path("m.gguf"), True, GPUS)
        self.assertEqual(args[args.index("--vision") + 1], "gpu")
        self.assertEqual(args[args.index("--context") + 1], "32768")

    def test_all_strata_files_stay_in_the_project(self):
        env = strata_engine.environment({"download": {"hf_mirror": "https://hf-mirror.com/"}})
        for key in ("APPDATA", "LOCALAPPDATA", "STRATA_INSTALL_DIR"):
            Path(env[key]).relative_to(strata_engine.BASE_DIR)
        self.assertEqual(env["HF_ENDPOINT"], "https://hf-mirror.com")
        self.assertNotIn("PYTHONPATH", env)

    def test_status_only_recognises_strata(self):
        class Resp:
            def __init__(self, body): self.body = body
            def read(self): return self.body
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch("urllib.request.urlopen", return_value=Resp(b'{"status":"ok","service":"strata"}')):
            self.assertTrue(strata_engine.status(8080))
        with mock.patch("urllib.request.urlopen", return_value=Resp(b'{"status":"ok"}')):
            self.assertEqual(strata_engine.status(8080), {})
        with mock.patch("urllib.request.urlopen", side_effect=OSError):
            self.assertEqual(strata_engine.status(8080), {})


class StrataUpdateCheckTests(unittest.TestCase):
    def check(self, engine_offered, control_offered="0.1.51"):
        payload = {"ok": True, "engine": {"installed": {"version": "0.1.50", "variant": "cuda13"},
                                          "offered": {"version": engine_offered}},
                   "control": {"installed": "0.1.51", "offered": control_offered}}
        done = mock.Mock(stdout="log line\n" + __import__("json").dumps(payload) + "\n")
        with mock.patch("subprocess.run", return_value=done):
            return strata_engine.update_check({})

    def test_versions_are_compared_to_decide_whether_to_offer_an_upgrade(self):
        self.assertFalse(self.check("0.1.50")["update_available"])
        self.assertTrue(self.check("0.1.52")["update_available"])
        self.assertTrue(self.check("0.1.50", "0.1.53")["update_available"])
        self.assertEqual(self.check("0.1.50")["engine_variant"], "cuda13")

    def test_unreadable_output_is_reported_not_raised(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout="")):
            self.assertIn("error", strata_engine.update_check({}))


class EngineSelectionTests(unittest.TestCase):
    def pick(self, engine, flash_next=True, available=True):
        cfg = {"server": {"engine": engine}, "model": {"model_file": "m.gguf"}}
        with mock.patch.object(run, "resolve_model", return_value=Path("m.gguf")), \
                mock.patch.object(run, "gguf_meta", return_value={}), \
                mock.patch.object(run, "is_qwen38_flash_next", return_value=flash_next), \
                mock.patch.object(strata_engine, "available", return_value=available):
            return run.strata_model(cfg)

    def test_strata_only_takes_flash_next(self):
        self.assertEqual(self.pick("strata"), Path("m.gguf"))
        self.assertIsNone(self.pick("llama.cpp"))
        self.assertIsNone(self.pick("strata", flash_next=False))
        self.assertIsNone(self.pick("strata", available=False))


if __name__ == "__main__":
    unittest.main()
