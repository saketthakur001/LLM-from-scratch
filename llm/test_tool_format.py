"""Unit tests for llm.tool_format: the parser/validator and the constrained
decoding character tracker. These are constructed-example tests -- no
trained (or even loaded) model is involved, matching the rest of this repo's
"pipeline is built, nothing is trained" stance.

Run with:

    python -m unittest llm.test_tool_format -v
"""
import unittest

from .tool_format import (
    JSONCharTracker,
    encode_tool_call,
    encode_training_example,
    parse_tool_call,
)


class TestEncode(unittest.TestCase):
    def test_encode_roundtrips_through_parser(self):
        text = encode_tool_call("get_weather", {"location": "Paris", "units": "metric"})
        result = parse_tool_call(text)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.tool_call.name, "get_weather")
        self.assertEqual(result.tool_call.arguments, {"location": "Paris", "units": "metric"})

    def test_training_example_contains_prompt_and_call(self):
        example = encode_training_example("What's the weather in Paris?", "get_weather", {"location": "Paris"})
        self.assertIn("What's the weather in Paris?", example)
        self.assertIn("<tool_call>", example)
        self.assertIn("</tool_call>", example)


class TestParse(unittest.TestCase):
    def test_valid_call(self):
        result = parse_tool_call('noise before <tool_call>{"name":"set_timer","arguments":{"seconds":300}}</tool_call> noise after')
        self.assertTrue(result.ok)
        self.assertEqual(result.tool_call.name, "set_timer")
        self.assertEqual(result.tool_call.arguments, {"seconds": 300})

    def test_missing_open_tag(self):
        result = parse_tool_call('{"name":"x","arguments":{}}</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("open tag", result.error)

    def test_missing_close_tag(self):
        result = parse_tool_call('<tool_call>{"name":"x","arguments":{}}')
        self.assertFalse(result.ok)
        self.assertIn("close tag", result.error)

    def test_invalid_json(self):
        result = parse_tool_call('<tool_call>{"name": "x", "arguments": }</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("invalid json", result.error)

    def test_missing_name(self):
        result = parse_tool_call('<tool_call>{"arguments": {}}</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("'name'", result.error)

    def test_missing_arguments(self):
        result = parse_tool_call('<tool_call>{"name": "x"}</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("'arguments'", result.error)

    def test_arguments_wrong_type(self):
        result = parse_tool_call('<tool_call>{"name": "x", "arguments": "not an object"}</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("'arguments' must be", result.error)

    def test_extra_top_level_key_rejected(self):
        result = parse_tool_call('<tool_call>{"name": "x", "arguments": {}, "id": 7}</tool_call>')
        self.assertFalse(result.ok)
        self.assertIn("unexpected keys", result.error)

    def test_nested_arguments(self):
        text = encode_tool_call("book_flight", {"from": "SFO", "to": "JFK", "options": {"nonstop": True, "seats": 2}})
        result = parse_tool_call(text)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.tool_call.arguments["options"], {"nonstop": True, "seats": 2})


class TestJSONCharTracker(unittest.TestCase):
    def _consume(self, tracker: JSONCharTracker, text: str) -> bool:
        for ch in text:
            if not tracker.step(ch):
                return False
        return True

    def test_accepts_full_valid_call(self):
        tracker = JSONCharTracker()
        text = encode_tool_call("get_weather", {"location": "Paris"})
        self.assertTrue(self._consume(tracker, text))
        self.assertEqual(tracker.state.name, "DONE")

    def test_rejects_wrong_open_tag(self):
        tracker = JSONCharTracker()
        self.assertFalse(tracker.step("["))  # must start with "<"

    def test_rejects_malformed_tag(self):
        tracker = JSONCharTracker()
        # "<tool_calx>" -- diverges from "<tool_call>" partway through
        ok = True
        for ch in "<tool_calx":
            ok = tracker.step(ch)
            if not ok:
                break
        self.assertFalse(ok)

    def test_rejects_unquoted_key(self):
        tracker = JSONCharTracker()
        ok = self._consume(tracker, "<tool_call>{name")
        self.assertFalse(ok)

    def test_rejects_trailing_comma(self):
        tracker = JSONCharTracker()
        ok = self._consume(tracker, '<tool_call>{"name":"x",}')
        self.assertFalse(ok)

    def test_accepts_nested_object_and_array(self):
        tracker = JSONCharTracker()
        text = encode_tool_call("f", {"items": [1, 2, 3], "opts": {"a": True}})
        self.assertTrue(self._consume(tracker, text))

    def test_allowed_next_at_start_is_open_bracket(self):
        tracker = JSONCharTracker()
        self.assertEqual(tracker.allowed_next(), {"<"})

    def test_incremental_prefix_rejection_matches_full_check(self):
        # A constrained decoder must reject a bad continuation *before* the
        # full text is known, not just after the fact. Check every prefix of
        # a broken example fails at the exact character it goes wrong.
        text = '<tool_call>{"name": "x" "arguments": {}}</tool_call>'  # missing comma
        tracker = JSONCharTracker()
        failed_at = None
        for i, ch in enumerate(text):
            if not tracker.step(ch):
                failed_at = i
                break
        self.assertIsNotNone(failed_at)
        self.assertEqual(text[failed_at], '"')  # fails on the stray quote after "x"


if __name__ == "__main__":
    unittest.main()
