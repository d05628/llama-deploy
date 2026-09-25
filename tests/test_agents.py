import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agents

GATEWAY = "http://127.0.0.1:11434"


class AgentIsolationTests(unittest.TestCase):
    def test_every_agent_gets_its_own_home_and_never_the_official_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for agent in agents.AGENTS:
                home = agents.agent_home(base, agent)
                env = agents.build_env(agent, home, GATEWAY, "local", "k", 47104)
                agents.prepare_home(agent, home, GATEWAY, "local", 47104)
                # 所有写入都落在我们自己的 agents/<id>/ 下
                written = [p for p in base.rglob("*") if p.is_file()]
                self.assertTrue(all(str(p).startswith(str(base / "agents")) for p in written), agent)
                # 环境变量里提到的路径也都指向我们的目录
                for value in env.values():
                    if value.startswith(str(base)) or ":\\" in value or value.startswith("/"):
                        self.assertTrue(value.startswith(str(base)), (agent, value))

    def test_claude_clears_inherited_official_key_and_knows_the_real_window(self):
        env = agents.build_env("claude", Path("h"), GATEWAY, "local", "k", 47104)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], GATEWAY)
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "47104")
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "h")

    def test_codex_uses_responses_protocol_and_local_window(self):
        # Codex 0.15x 已移除 wire_api = "chat"
        config = agents.codex_config(GATEWAY, "local", 47104)
        self.assertIn('wire_api = "responses"', config)
        self.assertIn("model_context_window = 47104", config)
        self.assertIn(f'base_url = "{GATEWAY}/v1"', config)

    def test_qwen_and_gemini_skip_the_login_dialog(self):
        with tempfile.TemporaryDirectory() as tmp:
            qwen, gemini = Path(tmp) / "qwen", Path(tmp) / "gemini"
            agents.prepare_home("qwen", qwen, GATEWAY, "local", 47104)
            agents.prepare_home("gemini", gemini, GATEWAY, "local", 47104)
            q = json.loads((qwen / "settings.json").read_text(encoding="utf-8"))
            g = json.loads((gemini / ".gemini" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(q["security"]["auth"]["selectedType"], "openai")
        self.assertEqual(q["modelProviders"]["openai"][0]["generationConfig"]["contextWindowSize"], 47104)
        self.assertEqual(g["security"]["auth"]["selectedType"], "gemini-api-key")

    def test_launcher_sets_variables_only_inside_the_window(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(agents, "IS_WIN", True):
            base = Path(tmp)
            env = agents.build_env("claude", base / "agents" / "claude", GATEWAY, "local", "k", 47104)
            text = agents.write_launcher(base, "claude", env, base).read_text(encoding=agents.LAUNCHER_ENCODING)
        self.assertIn('set "ANTHROPIC_API_KEY="', text)
        # 继承来的 PWD 会让 OpenCode 写错目录
        self.assertIn(f'set "PWD={base}"', text)
        self.assertIn('set "ANTHROPIC_BASE_URL=http://127.0.0.1:11434"', text)
        # 不切 UTF-8 代码页：旧版控制台下 Qwen Code 会报 "write UNKNOWN" 闪退
        self.assertNotIn("chcp 65001", text)
        # 窗口标题带前缀，「关闭 agent」靠它识别本工具打开的窗口
        self.assertIn("title " + agents.WINDOW_TITLE_PREFIX, text)
        # 不用 setx，不写注册表：只影响这个 cmd 进程
        self.assertNotIn("setx", text.lower())
        self.assertNotIn("reg add", text.lower())

    def test_default_workspace_is_not_the_home_folder(self):
        # 默认用整个用户目录时，OpenCode 一直卡在给整个目录做快照
        self.assertNotEqual(agents.default_workspace().resolve(), Path.home().resolve())
        self.assertTrue(agents.workspace_warning(Path.home()))
        self.assertTrue(agents.workspace_warning(Path(Path.home().anchor)))
        self.assertEqual(agents.workspace_warning(agents.default_workspace()), "")

    def test_vision_is_declared_to_tools_that_would_otherwise_assume_text_only(self):
        # Qwen Code / OpenCode 对未知模型名按纯文本处理，不声明就不发图（实测回答"不支持图片"）
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for vision in (True, False):
                agents.prepare_home("qwen", base / "qwen", GATEWAY, "local", 88064, vision=vision)
                agents.prepare_home("opencode", base / "oc", GATEWAY, "local", 88064, vision=vision)
                q = json.loads((base / "qwen" / "settings.json").read_text(encoding="utf-8"))
                o = json.loads((base / "oc" / "opencode.json").read_text(encoding="utf-8"))
                gen = q["modelProviders"]["openai"][0]["generationConfig"]
                model = o["provider"]["llama-deploy"]["models"]["local"]
                self.assertIs(gen["modalities"]["image"], vision)
                self.assertIs(model["attachment"], vision)
                self.assertEqual("image" in model["modalities"]["input"], vision)

    def test_codex_skips_the_daemon_that_refuses_elevated_windows(self):
        with mock.patch.object(agents, "IS_WIN", True):
            self.assertIn("--no-daemon", agents.command_args("codex", Path("h"), "local"))

    def test_aider_confirms_edits_but_never_runs_commands_by_itself(self):
        args = agents.command_args("aider", Path("h"), "local")
        self.assertIn("--yes-always", args)
        self.assertIn("--no-suggest-shell-commands", args)

    def test_opencode_snapshots_are_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "opencode"
            agents.prepare_home("opencode", home, GATEWAY, "local", 88064)
            config = json.loads((home / "opencode.json").read_text(encoding="utf-8"))
        self.assertIs(config["snapshot"], False)

    def test_unsafe_working_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "a%PATH%b"
            bad.mkdir()
            with self.assertRaises(ValueError):
                agents.validate_cwd(str(bad))
            with self.assertRaises(ValueError):
                agents.validate_cwd(str(Path(tmp) / "missing"))
            self.assertEqual(agents.validate_cwd(tmp), Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
