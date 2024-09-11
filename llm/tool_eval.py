"""Tiny synthetic tool-use eval harness.

This is deliberately small and deterministic: a handful of (prompt,
expected tool call) pairs, plus a scorer that checks generated text against
`llm.tool_format.parse_tool_call`. It is designed to run against *any*
checkpoint, including a freshly, randomly-initialized `GPT` — with random
weights the model will almost certainly fail every example, and that is the
point: this proves the eval harness itself is wired correctly end-to-end
(prompt -> generate -> parse -> score) before any actual training exists.

Two things are scored per example:
  - valid_format: did parsing succeed at all (right tags, valid JSON, right
    shape)?
  - exact_match: valid format AND name + arguments match the expected call
    exactly.

Run directly for a demo against a random model:

    python -m llm.tool_eval
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .tool_format import ToolCall, parse_tool_call, encode_tool_call


@dataclass
class EvalExample:
    prompt: str
    expected_name: str
    expected_arguments: dict[str, Any]


EVAL_SET: list[EvalExample] = [
    EvalExample(
        prompt="What's the weather in Paris?",
        expected_name="get_weather",
        expected_arguments={"location": "Paris"},
    ),
    EvalExample(
        prompt="Convert 10 miles to kilometers.",
        expected_name="convert_units",
        expected_arguments={"value": 10, "from_unit": "miles", "to_unit": "kilometers"},
    ),
    EvalExample(
        prompt="Set a timer for 5 minutes.",
        expected_name="set_timer",
        expected_arguments={"duration_seconds": 300},
    ),
    EvalExample(
        prompt="Look up the definition of 'ephemeral'.",
        expected_name="dictionary_lookup",
        expected_arguments={"word": "ephemeral"},
    ),
    EvalExample(
        prompt="Send an email to bob@example.com saying hi.",
        expected_name="send_email",
        expected_arguments={"to": "bob@example.com", "body": "hi"},
    ),
]


@dataclass
class ExampleResult:
    prompt: str
    generated_text: str
    valid_format: bool
    exact_match: bool
    error: str | None


@dataclass
class EvalReport:
    results: list[ExampleResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def valid_format_rate(self) -> float:
        return sum(r.valid_format for r in self.results) / self.n if self.n else 0.0

    @property
    def exact_match_rate(self) -> float:
        return sum(r.exact_match for r in self.results) / self.n if self.n else 0.0

    def summary(self) -> str:
        lines = [f"{'PROMPT':<45} {'VALID':<7} {'EXACT':<7} NOTE"]
        for r in self.results:
            note = r.error or ""
            lines.append(f"{r.prompt[:44]:<45} {str(r.valid_format):<7} {str(r.exact_match):<7} {note}")
        lines.append("")
        lines.append(f"valid_format_rate = {self.valid_format_rate:.2f}  "
                      f"exact_match_rate = {self.exact_match_rate:.2f}  (n={self.n})")
        return "\n".join(lines)


def score_example(example: EvalExample, generated_text: str) -> ExampleResult:
    result = parse_tool_call(generated_text)
    if not result.ok:
        return ExampleResult(example.prompt, generated_text, False, False, result.error)

    call = result.tool_call
    exact = (call.name == example.expected_name and call.arguments == example.expected_arguments)
    return ExampleResult(example.prompt, generated_text, True, exact, None if exact else "name/arguments mismatch")


def run_eval(generate_fn: Callable[[str], str], eval_set: list[EvalExample] | None = None) -> EvalReport:
    """`generate_fn(prompt) -> raw generated text`. Model-agnostic on purpose:
    the caller wires up tokenizer + GPT.generate() + decode; this function
    only knows about prompts, generated text, and scoring."""
    eval_set = eval_set if eval_set is not None else EVAL_SET
    results = [score_example(ex, generate_fn(ex.prompt)) for ex in eval_set]
    return EvalReport(results)


def _random_model_generate_fn() -> Callable[[str], str]:
    """Build a generate_fn backed by a real, randomly-initialized GPT +
    byte-level tokenizer, run through the constrained-decoding hook, to
    prove the eval harness works end-to-end without any trained checkpoint.
    """
    import torch
    from tokenizers import ByteLevelBPETokenizer

    from .config import GPTConfig
    from .model import GPT
    from .tool_format import make_constrained_logits_processor

    # A throwaway BPE tokenizer trained on a tiny in-memory corpus so this
    # demo needs no external files or downloads.
    corpus = [
        "What's the weather in Paris?", "Convert 10 miles to kilometers.",
        "Set a timer for 5 minutes.", "Look up the definition of 'ephemeral'.",
        "Send an email to bob@example.com saying hi.",
        encode_tool_call("get_weather", {"location": "Paris"}),
        encode_tool_call("convert_units", {"value": 10, "from_unit": "miles", "to_unit": "kilometers"}),
        "<tool_call>{}</tool_call>",
    ]
    import tempfile, os
    tokenizer = ByteLevelBPETokenizer()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "corpus.txt")
        with open(path, "w") as f:
            f.write("\n".join(corpus))
        tokenizer.train(files=[path], vocab_size=512, min_frequency=1,
                         special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(vocab_size=vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(config)
    model.eval()

    def generate_fn(prompt: str) -> str:
        ids = tokenizer.encode(prompt).ids or [0]
        idx = torch.tensor([ids], dtype=torch.long)
        processor = make_constrained_logits_processor(tokenizer)
        out = model.generate(idx, max_new_tokens=40, temperature=1.0, top_k=None,
                              logits_processor=processor)
        return tokenizer.decode(out[0, len(ids):].tolist())

    return generate_fn


def main():
    generate_fn = _random_model_generate_fn()
    report = run_eval(generate_fn)
    print(report.summary())
    print()
    print("This ran against a randomly-initialized (untrained) model, so low")
    print("scores are expected — this demonstrates the eval harness wiring")
    print("works end-to-end, not that the model can call tools.")


if __name__ == "__main__":
    main()
