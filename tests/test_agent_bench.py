import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent_bench"))
import tasks  # noqa: E402


class BugfixTaskTests(unittest.TestCase):
    def test_fixture_starts_broken_and_catches_edited_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            tasks.bugfix_setup(work)
            self.assertFalse(any(work.rglob("__pycache__")))
            result = tasks.bugfix_check(work)
            self.assertFalse(result["checks"]["8 个测试全部通过"])     # 题目本身有 bug
            self.assertTrue(result["checks"]["没有修改测试文件"])
            test_file = work / "tests" / "test_invoice.py"
            test_file.write_text(test_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
            self.assertFalse(tasks.bugfix_check(work)["checks"]["没有修改测试文件"])


class PromptPassingTests(unittest.TestCase):
    def test_every_task_prompt_survives_windows_cmd_wrappers(self):
        # npm 的 .CMD 包装会在第一个换行处截断参数，连后面的权限参数一起丢
        import run_bench
        for task_id, task in tasks.TASKS.items():
            line = run_bench.one_line(task["prompt"])
            self.assertNotIn("\n", line, task_id)
            self.assertNotIn("%", line, task_id)
            self.assertNotIn('"', line, task_id)

    def test_permission_flags_come_through_for_claude(self):
        import run_bench
        cmd = run_bench.build_command("claude", tasks.TASKS["bugfix"], tasks.TASKS["bugfix"]["prompt"],
                                      Path("."), Path("."))
        prompt = cmd[cmd.index("-p") + 1]
        self.assertNotIn("\n", prompt)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")


class SvgTaskTests(unittest.TestCase):
    def test_valid_svg_passes_and_garbage_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shapes = "".join(f'<circle cx="{i}" cy="{i}" r="2"/>' for i in range(12))
            (work / "pelican.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg">{shapes}</svg>', encoding="utf-8")
            self.assertTrue(all(tasks.svg_check(work)["checks"].values()))
            (work / "pelican.svg").write_text("not svg at all", encoding="utf-8")
            self.assertFalse(tasks.svg_check(work)["checks"]["是合法的 SVG 文档"])


if __name__ == "__main__":
    unittest.main()


class ReportTests(unittest.TestCase):
    def test_report_names_the_model_and_engine_it_ran_on(self):
        import run_bench
        llama = {"default_generation_settings": {"n_ctx": 65536}, "build_info": "b11836-f5027fb21",
                 "model_path": "I:\models\Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf", "modalities": {"vision": True}}
        strata = {"default_generation_settings": {"n_ctx": 65536}, "build_info": "Strata 0.1.50",
                  "model_alias": "qwen3.8-flash-next-iq3_xxs", "modalities": {"vision": False}}
        self.assertEqual(run_bench.server_info(llama)["model"], "Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf")
        info = run_bench.server_info(strata)
        self.assertEqual((info["model"], info["engine"], info["vision"]), ("qwen3.8-flash-next-iq3_xxs", "Strata 0.1.50", False))
        result = {"agent": "claude", "task": "svg", "title": "SVG", "passed": 3, "total": 3, "minutes": 1.5,
                  "timed_out": False, "rc": 0, "checks": {}, "artifacts": {}, "work": "w", "log": "l",
                  "max_temp_c": None, "max_power_w": None}
        with tempfile.TemporaryDirectory() as tmp:
            run_bench.write_report(Path(tmp), [result], info)
            self.assertIn("Strata 0.1.50", (Path(tmp) / "report.md").read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "meta.json").exists())


class ManagerBenchListTests(unittest.TestCase):
    def test_lists_new_and_old_runs_newest_first(self):
        import json
        from unittest import mock
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import manager
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench"
            for name, meta in (("20261001-100000", None), ("20261026-100942", {"model": "m", "engine": "Strata"})):
                (bench / name).mkdir(parents=True)
                (bench / name / "results.json").write_text(json.dumps([{"agent": "claude", "task": "svg", "passed": 3,
                                                                       "total": 3, "minutes": 1, "checks": {}}]), encoding="utf-8")
                if meta:
                    (bench / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with mock.patch.object(manager.agents, "default_workspace", return_value=Path(tmp)):
                reports = manager.bench_reports()["reports"]
        self.assertEqual([r["id"] for r in reports], ["20261026-100942", "20261001-100000"])
        self.assertEqual(reports[0]["meta"]["engine"], "Strata")
        self.assertEqual(reports[1]["meta"], {})
        self.assertNotIn("checks", reports[0]["results"][0])
