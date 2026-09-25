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
