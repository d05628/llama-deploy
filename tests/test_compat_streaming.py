import io
import json
import unittest
from unittest import mock

import compat


def make_handler():
    handler = object.__new__(compat.CompatHandler)
    handler.wfile = io.BytesIO()
    handler.sent = {}
    handler.send_response = lambda code, *a: handler.sent.setdefault("status", code)
    handler.send_header = lambda k, v: handler.sent.setdefault(k, v)
    handler.end_headers = lambda: None
    handler._cors = lambda: None
    handler.log_message = lambda *a: None
    return handler


def events(raw: bytes) -> list:
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            out.append(json.loads(data[0]))
    return out


class ResponsesStreamTests(unittest.TestCase):
    def test_items_are_announced_before_their_deltas_and_tool_calls_are_replayed(self):
        # Codex 报 "OutputTextDelta without active item"：此前只发 delta，工具调用也没回放
        response = {
            "id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output_text": "reading the file",
            "output": [
                {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
                 "content": [{"type": "output_text", "text": "reading the file", "annotations": []}]},
                {"id": "fc_1", "type": "function_call", "status": "completed", "name": "shell",
                 "call_id": "call_1", "arguments": "{\"cmd\": \"ls\"}"},
            ],
        }
        handler = make_handler()
        handler._stream_openai_response(response)
        seq = [e["type"] for e in events(handler.wfile.getvalue())]
        self.assertEqual(seq[0], "response.created")
        self.assertLess(seq.index("response.output_item.added"), seq.index("response.output_text.delta"))
        self.assertIn("response.function_call_arguments.done", seq)
        self.assertEqual(seq.count("response.output_item.done"), 2)
        self.assertEqual(seq[-1], "response.completed")
        numbers = [e["sequence_number"] for e in events(handler.wfile.getvalue())]
        self.assertEqual(numbers, sorted(numbers))


class GeminiToolHistoryTests(unittest.TestCase):
    def test_function_calls_in_history_become_real_tool_messages(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "make a file"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "write_file", "args": {"path": "a.txt"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "write_file", "response": {"ok": True}}}]},
            ],
            "tools": [{"functionDeclarations": [{"name": "write_file", "parameters": {"type": "object"}}]}],
        }
        messages = compat.gemini_to_openai(payload, "m")["messages"]
        assistant = next(m for m in messages if m["role"] == "assistant")
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "write_file")
        self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])
        # 不能再出现压平后的文本标记
        self.assertFalse(any("[function_call" in (m.get("content") or "") for m in messages))


class GeminiToolSchemaTests(unittest.TestCase):
    def test_both_schema_fields_reach_the_model(self):
        new = {"name": "write_file", "parametersJsonSchema": {
            "type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
        old = {"name": "read_file", "parameters": {
            "type": "OBJECT", "properties": {"path": {"type": "STRING"}}}}
        tools = compat.gemini_to_openai({"contents": [], "tools": [{"functionDeclarations": [new, old]}]}, "m")["tools"]
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["file_path"])
        self.assertEqual(tools[1]["function"]["parameters"]["properties"]["path"]["type"], "string")


PNG = "iVBORw0KGgo="


class ImagePassthroughTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(compat.VISION.update, {"enabled": compat.VISION["enabled"]})

    def test_images_from_all_protocols_reach_a_vision_upstream(self):
        compat.VISION["enabled"] = True
        blocks = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},  # Anthropic
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}},            # OpenAI
            {"type": "input_image", "image_url": "data:image/png;base64," + PNG},                   # Responses
            {"inlineData": {"mimeType": "image/png", "data": PNG}},                                 # Gemini
        ]
        msg = compat.normalize_openai_messages([{"role": "user", "content": [{"type": "text", "text": "看图"}] + blocks}])[0]
        urls = [p["image_url"]["url"] for p in msg["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(urls), 4)
        self.assertTrue(all(u.endswith(PNG) for u in urls))

    def test_images_are_replaced_by_a_hint_when_upstream_has_no_vision(self):
        compat.VISION["enabled"] = False
        msg = compat.normalize_openai_messages([{"role": "user", "content": [
            {"type": "text", "text": "看图"}, {"inlineData": {"mimeType": "image/png", "data": PNG}}]}])[0]
        self.assertIsInstance(msg["content"], str)
        self.assertIn("未开启视觉", msg["content"])
        self.assertNotIn(PNG, msg["content"])   # base64 不能被当成文本塞进提示词

    def test_tool_result_images_follow_the_tool_messages(self):
        compat.VISION["enabled"] = True
        blocks = [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "frame"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}]
        msgs = compat.normalize_openai_messages(compat.anthropic_blocks_to_openai("user", blocks))
        self.assertEqual(msgs[0]["role"], "tool")
        self.assertEqual(msgs[0]["tool_call_id"], "t1")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertTrue(any(p.get("type") == "image_url" for p in msgs[1]["content"]))


class ToolHistoryTests(unittest.TestCase):
    def test_anthropic_tool_use_becomes_real_tool_calls(self):
        msgs = compat.anthropic_blocks_to_openai("assistant", [
            {"type": "text", "text": "let me look"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}}])
        self.assertEqual(msgs[0]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(msgs[0]["content"], "let me look")

    def test_responses_function_calls_keep_their_ids(self):
        messages = compat.responses_input_to_messages({"input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]})
        assistant = next(m for m in messages if m["role"] == "assistant")
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(assistant["tool_calls"][0]["id"], "c1")
        self.assertEqual(tool["tool_call_id"], "c1")


class InvalidToolCallTests(unittest.TestCase):
    def test_unfixable_call_goes_back_to_the_client_instead_of_ending_the_turn(self):
        # 此前改写成说明文字：Claude Code 把它当成回答完毕，渲染报错后没机会修脚本就收工了
        resp = {"choices": [{"finish_reason": "tool_calls", "message": {"content": "fixing the camera", "tool_calls": [
            {"id": "c1", "function": {"name": "Edit", "arguments": json.dumps({"old_string": "a", "new_string": "b"})}}]}}]}
        schema = {"Edit": {"type": "object", "required": ["file_path", "old_string", "new_string"],
                           "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                          "new_string": {"type": "string"}}}}
        msg = compat.openai_to_anthropic(resp, {}, "m", schema)
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual([b["type"] for b in msg["content"]], ["text", "tool_use"])
        self.assertEqual(msg["content"][1]["name"], "Edit")


class HeartbeatTests(unittest.TestCase):
    def test_slow_upstream_sends_heartbeats_and_returns_result(self):
        # 长回复要生成 2-3 分钟，期间不发任何字节客户端就会断开重试（实测 Claude Code 无限重发）
        import time as _time
        handler = make_handler()

        def slow():
            _time.sleep(0.35)
            return 200, {"ok": True}

        result = handler._wait_with_heartbeat(slow, b": keepalive\n\n", interval=0.1)
        self.assertEqual(result, (200, {"ok": True}))
        self.assertGreaterEqual(handler.wfile.getvalue().count(b": keepalive"), 2)

    def test_upstream_errors_propagate(self):
        handler = make_handler()

        def boom():
            raise RuntimeError("upstream down")

        with self.assertRaisesRegex(RuntimeError, "upstream down"):
            handler._wait_with_heartbeat(boom, b": keepalive\n\n", interval=0.05)


class StalePidTests(unittest.TestCase):
    def test_reused_pid_of_another_program_is_not_the_gateway(self):
        # 重启后网关旧 PID 被 conhost.exe 复用：不能判为"在运行"，更不能被 stop 杀掉
        other = mock.Mock(returncode=0, stdout='"conhost.exe","16296","Console","1","13,100 K"')
        gateway = mock.Mock(returncode=0, stdout='"python.exe","16296","Console","1","40,000 K"')
        with mock.patch.object(compat, "IS_WIN", True):
            with mock.patch.object(compat.subprocess, "run", return_value=other):
                self.assertFalse(compat.pid_running(16296))
            with mock.patch.object(compat.subprocess, "run", return_value=gateway):
                self.assertTrue(compat.pid_running(16296))

    def test_stop_never_kills_a_reused_pid(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "gw.pid"
            pid_file.write_text("16296", encoding="utf-8")
            with mock.patch.object(compat, "PID_FILE", pid_file), \
                    mock.patch.object(compat, "pid_running", return_value=False), \
                    mock.patch.object(compat.subprocess, "run") as run:
                compat.cmd_stop()
            run.assert_not_called()
            self.assertFalse(pid_file.exists())


class ChatStreamPassthroughTests(unittest.TestCase):
    def test_upstream_sse_is_forwarded_byte_for_byte(self):
        # Qwen Code / OpenCode 发 stream:true，此前网关按 JSON 解析 SSE，返回 502
        body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

        class Upstream(io.BytesIO):
            headers = {"Content-Type": "text/event-stream"}

            def getcode(self):
                return 200

        handler = make_handler()
        with mock.patch.object(compat.urllib.request, "urlopen", return_value=Upstream(body)):
            handler._proxy_sse("http://upstream/v1/chat/completions", {"stream": True}, 5)
        self.assertEqual(handler.sent["status"], 200)
        self.assertEqual(handler.wfile.getvalue(), body)


class TextToolCallRecoveryTests(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "exec_command", "parameters": {
        "type": "object", "properties": {"cmd": {"type": "string"}, "timeout_ms": {"type": "number"}}}}}]

    def recover(self, text):
        resp = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        return compat.recover_text_tool_calls(resp, self.TOOLS)["choices"][0]

    def test_name_attribute_variant_becomes_a_real_tool_call(self):
        # Codex 实测：模型写 <parameter name="cmd">，llama-server 没解析，agent 一轮就停
        text = ('先看看。\n\n<tool_call>\n<function=exec_command>\n<parameter name="cmd">\n'
                'dir "C:\\a b"\n</parameter>\n<parameter=timeout_ms>\n5000\n</parameter>\n'
                '</function>\n</tool_call>')
        choice = self.recover(text)
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "先看看。")
        call = choice["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "exec_command")
        self.assertEqual(json.loads(call["arguments"]), {"cmd": 'dir "C:\\a b"', "timeout_ms": 5000})

    def test_hermes_json_variant(self):
        choice = self.recover('<tool_call>{"name": "exec_command", "arguments": {"cmd": "ls"}}</tool_call>')
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"cmd": "ls"})

    def test_unknown_tool_or_plain_text_is_left_alone(self):
        for text in ('<tool_call><function=rm_rf><parameter=path>/</parameter></function></tool_call>',
                     "解释一下 <tool_call> 标签的格式"):
            choice = self.recover(text)
            self.assertEqual(choice["message"]["content"], text)
            self.assertNotIn("tool_calls", choice["message"])

    def test_upstream_call_applies_recovery_only_when_tools_declared(self):
        text = '<tool_call><function=exec_command><parameter=cmd>ls</parameter></function></tool_call>'
        upstream = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        handler = make_handler()
        with mock.patch.object(compat, "http_json", return_value=(200, upstream)):
            _, with_tools = handler._call_openai_chat({"messages": [], "tools": self.TOOLS}, {})
            _, without = handler._call_openai_chat({"messages": []}, {})
        self.assertEqual(with_tools["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(without["choices"][0]["message"]["content"], text)

    def test_gpt_code_mode_request_gets_an_explanation_instead_of_silence(self):
        data = {"model": "gpt-5.6-sol", "input": [
            {"type": "additional_tools", "role": "developer", "tools": [{"type": "namespace", "name": "functions"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
        self.assertIn("llama-deploy-local", compat.responses_code_mode_notice(data))
        data["tools"] = [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]
        self.assertIsNone(compat.responses_code_mode_notice(data))

    def stream(self, pieces, tools=True):
        class Upstream(io.BytesIO):
            headers = {"Content-Type": "text/event-stream"}

            def getcode(self):
                return 200

        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}]
        chunks += [{"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in pieces]
        chunks += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        handler = make_handler()
        payload = {"stream": True, "tools": self.TOOLS} if tools else {"stream": True}
        with mock.patch.object(compat.urllib.request, "urlopen", return_value=Upstream(body)):
            handler._proxy_sse("http://upstream/v1/chat/completions", payload, 5)
        raw = handler.wfile.getvalue()
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
        return raw[:-len(b"data: [DONE]\n\n")]

    def test_stream_text_tool_call_split_across_chunks_is_recovered(self):
        out = events(self.stream(["好的<tool", "_call>\n<function=exec_command>\n<parameter name=\"cmd\">",
                                  "ls\n</parameter>\n</function>\n</tool_call>"]))
        text = "".join((e["choices"][0]["delta"].get("content") or "") for e in out if e.get("choices"))
        self.assertEqual(text, "好的")
        calls = [c for e in out if e.get("choices") for c in e["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(calls[0]["function"]["name"], "exec_command")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "ls"})
        self.assertEqual([e["choices"][0]["finish_reason"] for e in out if e.get("choices")][-1], "tool_calls")

    def test_stream_plain_text_keeps_every_character(self):
        out = events(self.stream(["a <", "b> <tool", " c"]))
        text = "".join((e["choices"][0]["delta"].get("content") or "") for e in out if e.get("choices"))
        self.assertEqual(text, "a <b> <tool c")
        self.assertEqual(out[-1]["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
