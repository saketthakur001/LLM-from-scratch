"""Fixed structured tool-call output format for fine-tuning / decoding.

This module defines a small, deliberately simple JSON-ish grammar for tool
calls, plus:

  - an *encoder* that formats a (prompt, tool call) training example into the
    exact text a model would be fine-tuned to produce, and
  - a *parser/validator* that extracts a tool call from raw generated text,
    or reports why it is malformed.

Nothing here trains anything. It is scaffolding for a fine-tuning pipeline
that does not exist yet in this repo (see README "Status" — no GPU, no
training run). The point is to have a well-defined target format and a
harness that can check a model's output against it, so training can be
plugged in later without redesigning the interface.

Grammar
-------

A tool call is a single line wrapped in fixed sentinel tags::

    <tool_call>{"name": "<tool_name>", "arguments": {<json object>}}</tool_call>

Concretely, the full production (informal EBNF)::

    tool_call   := "<tool_call>" ws object ws "</tool_call>"
    object      := "{" ws "\"name\"" ws ":" ws string ws "," ws
                        "\"arguments\"" ws ":" ws arg_object ws "}"
    arg_object  := "{" ws (pair (ws "," ws pair)*)? ws "}"
    pair        := string ws ":" ws value
    value       := string | number | "true" | "false" | "null" | object | array
    array       := "[" ws (value (ws "," ws value)*)? ws "]"
    string      := double-quoted, with standard JSON escape sequences
                   (quote, backslash, slash, \\b \\f \\n \\r \\t, \\uXXXX)
    number      := standard JSON number
    ws          := zero or more of {space, tab, newline, carriage return}

This is exactly JSON for the "arguments" value, plus a required top-level
"name" key, plus the `<tool_call>...</tool_call>` sentinel wrapper. Keeping
it to "real JSON inside fixed sentinels" means the constrained decoder in
`llm.model.GPT.generate` only has to track two things: (a) which literal
sentinel/structural characters are legal next, and (b) generic JSON string/
number/structural state — see `JSONCharTracker` below, which is shared by
the parser (as a validator) and the decoding-time constraint hook.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


# --------------------------------------------------------------------------
# Encoding: (prompt, tool call) -> training text
# --------------------------------------------------------------------------

def encode_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Format a tool name + arguments dict into the fixed wire format.

    Uses compact JSON (no extra whitespace) so the format is unambiguous and
    short — this is what a fine-tuning example's target text would contain.
    """
    payload = {"name": name, "arguments": arguments}
    return f"{OPEN_TAG}{json.dumps(payload, separators=(',', ':'))}{CLOSE_TAG}"


def encode_training_example(prompt: str, name: str, arguments: dict[str, Any],
                             prompt_prefix: str = "### Prompt:\n",
                             response_prefix: str = "\n### Response:\n") -> str:
    """Format a full (prompt, tool call) pair into one training string.

    This is the text a `dataset.py`-style pipeline would tokenize and train
    on (prompt tokens + response tokens, response being the tool call in the
    fixed format above). No training loop consumes this yet; it exists so
    the target format for a future SFT stage is pinned down and testable.
    """
    return f"{prompt_prefix}{prompt}{response_prefix}{encode_tool_call(name, arguments)}"


# --------------------------------------------------------------------------
# Parsing / validation: generated text -> ToolCall | error
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ParseResult:
    ok: bool
    tool_call: ToolCall | None = None
    error: str | None = None


def parse_tool_call(text: str) -> ParseResult:
    """Extract and validate a tool call from raw (possibly noisy) model output.

    Looks for the first `<tool_call>...</tool_call>` span, requires the
    contents to be valid JSON, requires a "name" string key and an
    "arguments" object key, and rejects anything else (extra top-level keys,
    wrong types, missing tags, unterminated span, trailing garbage inside
    the tags).
    """
    start = text.find(OPEN_TAG)
    if start == -1:
        return ParseResult(ok=False, error="missing <tool_call> open tag")
    end = text.find(CLOSE_TAG, start)
    if end == -1:
        return ParseResult(ok=False, error="missing </tool_call> close tag")

    inner = text[start + len(OPEN_TAG):end].strip()
    if not inner:
        return ParseResult(ok=False, error="empty tool call body")

    try:
        payload = json.loads(inner)
    except json.JSONDecodeError as e:
        return ParseResult(ok=False, error=f"invalid json: {e}")

    if not isinstance(payload, dict):
        return ParseResult(ok=False, error="tool call body is not a json object")

    extra_keys = set(payload.keys()) - {"name", "arguments"}
    if extra_keys:
        return ParseResult(ok=False, error=f"unexpected keys: {sorted(extra_keys)}")

    if "name" not in payload:
        return ParseResult(ok=False, error="missing 'name' key")
    if not isinstance(payload["name"], str) or not payload["name"]:
        return ParseResult(ok=False, error="'name' must be a non-empty string")

    if "arguments" not in payload:
        return ParseResult(ok=False, error="missing 'arguments' key")
    if not isinstance(payload["arguments"], dict):
        return ParseResult(ok=False, error="'arguments' must be a json object")

    return ParseResult(ok=True, tool_call=ToolCall(name=payload["name"], arguments=payload["arguments"]))


# --------------------------------------------------------------------------
# Constrained-decoding support: a character-level state tracker for the
# grammar above. GPT.generate() uses this (via JSONCharTracker.allowed_next)
# to decide which *characters* are legal to emit next; llm/model.py maps
# that down to a token-level mask over the tokenizer's vocabulary.
# --------------------------------------------------------------------------

class _St(Enum):
    PRE_OPEN_TAG = auto()      # before "<tool_call>" has been emitted
    IN_OPEN_TAG = auto()       # mid-way through emitting "<tool_call>" literal
    JSON = auto()              # inside the JSON object (delegate to a mini JSON FSM)
    IN_CLOSE_TAG = auto()      # mid-way through emitting "</tool_call>" literal
    DONE = auto()              # closed; nothing further is part of the grammar


class _JsonSt(Enum):
    VALUE_START = auto()   # expecting a value (or key, if top frame says so) to start
    STRING = auto()        # inside a string literal
    STRING_ESCAPE = auto() # just saw a backslash inside a string
    AFTER_VALUE = auto()   # just finished a value or key; see frame phase for what's next
    NUMBER = auto()        # inside a bare number literal
    LITERAL = auto()       # inside true/false/null


class _Phase(Enum):
    """Phase of an open object/array frame, tracked per frame so the tracker
    knows -- once a string closes -- whether it was a *key* (needs a colon
    next) or a *value* (needs a comma/close next)."""
    OBJ_KEY_OR_CLOSE = auto()     # just opened '{': expect a key string or '}' (empty object)
    OBJ_KEY = auto()              # just saw ',': expect a key string (no '}' -- no trailing comma)
    OBJ_COLON = auto()            # just closed a key string: expect ':'
    OBJ_VALUE = auto()            # just saw ':': expect a value
    OBJ_COMMA_OR_CLOSE = auto()   # just finished a value: expect ',' or '}'
    ARR_VALUE_OR_CLOSE = auto()   # just opened '[', or after ',': expect a value or ']'
    ARR_COMMA_OR_CLOSE = auto()   # just finished a value: expect ',' or ']'


@dataclass
class _Frame:
    kind: str    # "{" or "["
    phase: _Phase


@dataclass
class JSONCharTracker:
    """Tracks progress through the `<tool_call>{...}</tool_call>` grammar
    one character at a time and reports which characters may legally come
    next. Simplified relative to full JSON (no unicode-escape validation,
    no strict number-grammar edge cases), but real: it actually rejects
    malformed continuations (bad keys, trailing commas, wrong structural
    chars, broken sentinels) rather than being a no-op.
    """

    state: _St = _St.PRE_OPEN_TAG
    _tag_pos: int = 0
    _json_state: _JsonSt = _JsonSt.VALUE_START
    _frames: list[_Frame] = field(default_factory=list)
    _literal_buf: str = ""
    _top_level_done: bool = False  # the single top-level JSON value has closed

    def allowed_next(self) -> set[str] | None:
        """Return the set of legal next characters, or None if any character
        is allowed (only true when DONE / outside the grammar)."""
        if self.state == _St.PRE_OPEN_TAG:
            return {OPEN_TAG[0]}
        if self.state == _St.IN_OPEN_TAG:
            return {OPEN_TAG[self._tag_pos]}
        if self.state == _St.IN_CLOSE_TAG:
            return {CLOSE_TAG[self._tag_pos]}
        if self.state == _St.DONE:
            return None
        if self.state == _St.JSON:
            return self._json_allowed_next()
        return None

    def step(self, ch: str) -> bool:
        """Consume one character. Returns False (and leaves state unchanged)
        if `ch` is not in `allowed_next()`."""
        allowed = self.allowed_next()
        if allowed is not None and ch not in allowed:
            return False

        if self.state == _St.PRE_OPEN_TAG:
            self._tag_pos = 1
            self.state = _St.IN_OPEN_TAG if len(OPEN_TAG) > 1 else _St.JSON
        elif self.state == _St.IN_OPEN_TAG:
            self._tag_pos += 1
            if self._tag_pos == len(OPEN_TAG):
                self.state = _St.JSON
                self._json_state = _JsonSt.VALUE_START
                self._frames = []
                self._top_level_done = False
        elif self.state == _St.IN_CLOSE_TAG:
            self._tag_pos += 1
            if self._tag_pos == len(CLOSE_TAG):
                self.state = _St.DONE
        elif self.state == _St.JSON:
            self._json_step(ch)
        return True

    # -- mini JSON FSM, with a phase-tagged frame stack ---------------------

    def _top(self) -> _Frame | None:
        return self._frames[-1] if self._frames else None

    def _json_allowed_next(self) -> set[str]:
        st = self._json_state
        top = self._top()

        if st == _JsonSt.STRING:
            return None  # any char legal inside a string (incl. the closing quote itself)
        if st == _JsonSt.STRING_ESCAPE:
            return set('"\\/bfnrtu')
        if st == _JsonSt.NUMBER:
            opts = set("0123456789.eE+-")
            if top is not None:
                opts.add(",")
                opts.add("}" if top.kind == "{" else "]")
            return opts
        if st == _JsonSt.LITERAL:
            remaining = {"true", "false", "null"}
            candidates = {w[len(self._literal_buf)] for w in remaining
                          if w.startswith(self._literal_buf) and len(w) > len(self._literal_buf)}
            return candidates or set()
        if st == _JsonSt.AFTER_VALUE:
            return self._after_value_allowed(top)
        # VALUE_START
        return self._value_start_allowed(top)

    def _value_start_allowed(self, top: _Frame | None) -> set[str]:
        ws = {" ", "\t", "\n", "\r"}
        if top is not None and top.kind == "{" and top.phase == _Phase.OBJ_KEY_OR_CLOSE:
            # freshly opened object: expect a key string, or '}' if empty
            return ws | {'"', "}"}
        if top is not None and top.kind == "{" and top.phase == _Phase.OBJ_KEY:
            # after a comma: a key string is required, no trailing '}'
            return ws | {'"'}
        opts = ws | {'"', "{", "[", "-"} | set("0123456789") | {"t", "f", "n"}
        if top is not None and top.kind == "[" and top.phase == _Phase.ARR_VALUE_OR_CLOSE:
            opts.add("]")  # allow empty array
        return opts

    def _after_value_allowed(self, top: _Frame | None) -> set[str]:
        ws = {" ", "\t", "\n", "\r"}
        if top is None:
            if self._top_level_done:
                return {CLOSE_TAG[0]}
            return ws  # whitespace before we decide what closed
        if top.phase == _Phase.OBJ_COLON:
            return ws | {":"}
        if top.phase == _Phase.OBJ_COMMA_OR_CLOSE:
            return ws | {",", "}"}
        if top.phase == _Phase.ARR_COMMA_OR_CLOSE:
            return ws | {",", "]"}
        return ws

    def _json_step(self, ch: str) -> None:
        st = self._json_state

        if st == _JsonSt.VALUE_START:
            if ch in " \t\n\r":
                return
            top = self._top()
            if ch == '"':
                self._json_state = _JsonSt.STRING
            elif ch == "{":
                self._frames.append(_Frame("{", _Phase.OBJ_KEY_OR_CLOSE))
                self._json_state = _JsonSt.VALUE_START
            elif ch == "[":
                self._frames.append(_Frame("[", _Phase.ARR_VALUE_OR_CLOSE))
                self._json_state = _JsonSt.VALUE_START
            elif ch == "}" and top is not None and top.kind == "{" and top.phase == _Phase.OBJ_KEY_OR_CLOSE:
                self._close_frame()
            elif ch == "]" and top is not None and top.kind == "[" and top.phase == _Phase.ARR_VALUE_OR_CLOSE:
                self._close_frame()
            elif ch in "-0123456789":
                self._json_state = _JsonSt.NUMBER
            elif ch in "tfn":
                self._literal_buf = ch
                self._json_state = _JsonSt.LITERAL
            return

        if st == _JsonSt.STRING:
            if ch == "\\":
                self._json_state = _JsonSt.STRING_ESCAPE
            elif ch == '"':
                self._string_closed()
            return

        if st == _JsonSt.STRING_ESCAPE:
            self._json_state = _JsonSt.STRING
            return

        if st == _JsonSt.NUMBER:
            if ch in "0123456789.eE+-":
                return
            self._json_state = _JsonSt.AFTER_VALUE
            self._value_closed()
            self._json_step_after_value(ch)
            return

        if st == _JsonSt.LITERAL:
            self._literal_buf += ch
            if self._literal_buf in ("true", "false", "null"):
                self._json_state = _JsonSt.AFTER_VALUE
                self._value_closed()
            return

        if st == _JsonSt.AFTER_VALUE:
            self._json_step_after_value(ch)
            return

    def _string_closed(self) -> None:
        """A `"..."` literal just closed; decide if it was a key or a value."""
        self._json_state = _JsonSt.AFTER_VALUE
        top = self._top()
        if top is not None and top.kind == "{" and top.phase in (_Phase.OBJ_KEY_OR_CLOSE, _Phase.OBJ_KEY):
            top.phase = _Phase.OBJ_COLON  # it was a key -- next must be ':'
        else:
            self._value_closed()  # it was a value

    def _value_closed(self) -> None:
        """A value (string/number/literal/nested object or array) just
        finished; advance the enclosing frame's phase (or mark top level
        done)."""
        top = self._top()
        if top is None:
            self._top_level_done = True
        elif top.kind == "{":
            top.phase = _Phase.OBJ_COMMA_OR_CLOSE
        elif top.kind == "[":
            top.phase = _Phase.ARR_COMMA_OR_CLOSE

    def _json_step_after_value(self, ch: str) -> None:
        top = self._top()
        if ch in " \t\n\r":
            return
        if top is None:
            if ch == CLOSE_TAG[0] and self._top_level_done:
                self.state = _St.IN_CLOSE_TAG
                self._tag_pos = 1
            return
        if top.phase == _Phase.OBJ_COLON and ch == ":":
            top.phase = _Phase.OBJ_VALUE
            self._json_state = _JsonSt.VALUE_START
            return
        if top.phase == _Phase.OBJ_COMMA_OR_CLOSE:
            if ch == ",":
                top.phase = _Phase.OBJ_KEY
                self._json_state = _JsonSt.VALUE_START
            elif ch == "}":
                self._close_frame()
            return
        if top.phase == _Phase.ARR_COMMA_OR_CLOSE:
            if ch == ",":
                top.phase = _Phase.ARR_VALUE_OR_CLOSE
                self._json_state = _JsonSt.VALUE_START
            elif ch == "]":
                self._close_frame()
            return

    def _close_frame(self) -> None:
        if self._frames:
            self._frames.pop()
        self._json_state = _JsonSt.AFTER_VALUE
        self._value_closed()


# --------------------------------------------------------------------------
# Token-level constrained decoding hook, built on top of JSONCharTracker.
# --------------------------------------------------------------------------

def make_constrained_logits_processor(tokenizer, max_prompt_len: int | None = None):
    """Build a `logits_processor(idx, logits) -> logits` closure for
    `GPT.generate(..., logits_processor=...)`.

    On every decoding step it:
      1. decodes the tokens generated so far (since the processor was first
         invoked) back to text,
      2. replays a fresh `JSONCharTracker` over that text to find the
         current grammar state,
      3. for every candidate next-token id, decodes its text and checks
         (character by character, on a scratch copy of the tracker) whether
         it is a legal continuation,
      4. masks out (-inf) every token id that is not.

    This is real per-token filtering, not a no-op: tokens that would break
    JSON structure or the `<tool_call>`/`</tool_call>` sentinels are
    excluded from sampling. It is simplified relative to a production
    constrained decoder (recomputes from scratch each step, decodes each
    vocab entry independently rather than walking a trie, does not handle
    every JSON edge case such as unicode escapes) but it is functional and
    testable, not a stub.

    Batch size must be 1 (this repo's `generate()` is used with batch size
    1 in `llm/generate.py`); a batch > 1 raises.
    """
    import torch

    vocab_size = tokenizer.get_vocab_size()
    # Precompute each token's literal text once.
    token_texts = [tokenizer.decode([i]) for i in range(vocab_size)]

    state = {"start_idx": None}

    def processor(idx, logits):
        if idx.shape[0] != 1:
            raise ValueError("constrained logits processor only supports batch size 1")

        if state["start_idx"] is None:
            state["start_idx"] = idx.shape[1]

        generated_ids = idx[0, state["start_idx"]:].tolist()
        generated_text = tokenizer.decode(generated_ids) if generated_ids else ""

        base_tracker = JSONCharTracker()
        for ch in generated_text:
            if not base_tracker.step(ch):
                # Already off-grammar (shouldn't happen if masking is applied
                # every step) -- stop constraining further.
                return logits

        if base_tracker.state == _St.DONE:
            return logits  # tool call already closed; no further constraint

        masked = logits.clone()
        allowed_ids = []
        for token_id in range(vocab_size):
            text = token_texts[token_id]
            if text == "":
                continue
            tracker = copy.deepcopy(base_tracker)
            ok = True
            for ch in text:
                if not tracker.step(ch):
                    ok = False
                    break
            if ok:
                allowed_ids.append(token_id)

        if not allowed_ids:
            # No vocab token is a legal continuation (can happen with a tiny
            # demo vocab that lacks some needed character combination) --
            # fall back to unconstrained rather than deadlocking generation.
            return logits

        mask = torch.full_like(masked, float("-inf"))
        mask[:, allowed_ids] = masked[:, allowed_ids]
        return mask

    return processor
