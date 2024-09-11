"""Thin wrapper around a byte-level BPE tokenizer (train / load / encode / decode)."""
import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def train_tokenizer(txt_paths: list[str], vocab_size: int, save_dir: str, min_frequency: int = 2) -> None:
    if not txt_paths:
        raise ValueError("no training files given")
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=txt_paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(save_dir, "hermes")


def load_tokenizer(vocab_path: str, merges_path: str) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(vocab_path, merges_path)


def collect_txt_paths(root: str) -> list[str]:
    paths = []
    for r, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith("."):
                continue
            paths.append(os.path.join(r, f))
    return sorted(paths)


def encode_files(tokenizer, txt_paths: list[str]) -> list[int]:
    """Encode every file and concatenate ids into one flat stream, separated by </s>."""
    eos_id = tokenizer.token_to_id("</s>")
    ids: list[int] = []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        ids.extend(tokenizer.encode(text).ids)
        ids.append(eos_id)
    return ids
