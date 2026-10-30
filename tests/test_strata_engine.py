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
        with mock.patch("subprocess.run", return_value=done), mock.patch.object(strata_engine, "stage_feed", return_value=None):
            return strata_engine.update_check({})

    def test_versions_are_compared_to_decide_whether_to_offer_an_upgrade(self):
        self.assertFalse(self.check("0.1.50")["update_available"])
        self.assertTrue(self.check("0.1.52")["update_available"])
        self.assertTrue(self.check("0.1.50", "0.1.53")["update_available"])
        self.assertEqual(self.check("0.1.50")["engine_variant"], "cuda13")

    def test_unreadable_output_is_reported_not_raised(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout="")), \
                mock.patch.object(strata_engine, "stage_feed", return_value=None):
            self.assertIn("error", strata_engine.update_check({}))


class StrataUpdateRetryTests(unittest.TestCase):
    def run_update(self, outputs, cfg=None):
        calls = []
        def fake_popen(args, **kwargs):
            out, rc = outputs[len(calls)]
            calls.append(args[args.index("--feed") + 1])
            return mock.Mock(stdout=iter([out]), wait=mock.Mock(return_value=rc))
        with mock.patch("subprocess.Popen", side_effect=fake_popen), mock.patch("time.sleep"), \
                mock.patch.object(strata_engine, "stage_feed", return_value=None):
            rc = strata_engine.update(cfg or {}, say=lambda *_: None)
        return rc, calls

    def test_feed_read_errors_are_retried_then_succeed(self):
        unreadable = ("strata: the update feed could not be read: SSL EOF\n", 1)
        rc, calls = self.run_update([unreadable, ("Engine 0.1.51 is installed\n", 0)])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [strata_engine.FEED_URL] * 2)

    def test_other_failures_are_not_retried(self):
        rc, calls = self.run_update([("test start failed, kept 0.1.50\n", 3)])
        self.assertEqual((rc, len(calls)), (3, 1))

    def test_configured_github_mirror_is_tried_first(self):
        cfg = {"download": {"github_mirror": "https://ghproxy.example/"}}
        self.assertEqual(strata_engine.feed_urls(cfg)[:2],
                         ["https://ghproxy.example/" + strata_engine.FEED_URL, strata_engine.FEED_URL])
        # without one, GitHub first and the built-in mirror only as a fallback
        urls = strata_engine.feed_urls({})
        self.assertEqual(urls[0], strata_engine.FEED_URL)
        self.assertEqual(urls[1:], [f"{m}/{strata_engine.FEED_URL}" for m in strata_engine.FALLBACK_MIRRORS])

    def test_unreadable_feed_moves_on_to_the_next_address(self):
        unreadable = ("strata: the update feed could not be read: SSL EOF\n", 1)
        rc, calls = self.run_update([unreadable, unreadable, ("Engine 0.1.51 is installed\n", 0)])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [strata_engine.FEED_URL] * 2 + [strata_engine.feed_urls({})[1]])


class StrataStagedFeedTests(unittest.TestCase):
    PLAN = {"ok": True, "hardware": {"platform": "windows-x86_64"},
            "engine": {"installed": {"version": "0.1.50"},
                       "offered": {"version": "0.1.52", "url": "strata-engine-0.1.52-win.tar.gz", "sha256": "e" * 64}},
            "control": {"installed": "0.1.51", "offered": "0.1.53"}}
    FEED = {"control": {"packages": [{"platform": "linux-x86_64", "url": "c-linux.tar.gz", "sha256": "l" * 64},
                                     {"platform": "windows-x86_64", "url": "c-win.zip", "sha256": "w" * 64}]}}

    def stage(self, plan):
        import json, tempfile
        fetched = []
        def fake_fetch(cfg, name, dest, sha256="", say=print):
            fetched.append((name, sha256))
            if name == "strata-feed.json":
                dest.write_text(json.dumps(self.FEED), encoding="utf-8")
            return True
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch.object(strata_engine, "FEED_DIR", Path(tmp)),                 mock.patch.object(strata_engine, "fetch_release_file", side_effect=fake_fetch),                 mock.patch.object(strata_engine, "_check", return_value=plan):
            staged = strata_engine.stage_feed({}, packages=True, say=lambda *_: None)
        return staged, fetched

    def test_fetches_the_feed_signature_and_only_the_packages_this_pc_needs(self):
        staged, fetched = self.stage(self.PLAN)
        self.assertIsNotNone(staged)
        self.assertEqual(fetched, [("strata-feed.json", ""), ("strata-feed.json.sig", ""),
                                   ("strata-engine-0.1.52-win.tar.gz", "e" * 64), ("c-win.zip", "w" * 64)])

    def test_nothing_to_download_when_current(self):
        current = {**self.PLAN, "engine": {"installed": {"version": "0.1.52"}, "offered": {"version": "0.1.52", "url": "x"}},
                   "control": {"installed": "0.1.53", "offered": "0.1.53"}}
        _, fetched = self.stage(current)
        self.assertEqual([n for n, _ in fetched], ["strata-feed.json", "strata-feed.json.sig"])


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
