"""End-to-end data prep: train a BPE tokenizer over txt/ and encode it into
tokenized_texts.json, ready for llm/train.py. Replaces the ad-hoc notebook cells."""
import argparse

from .dataset import save_token_ids
from .tokenizer import collect_txt_paths, encode_files, load_tokenizer, train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", default="txt")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--out_ids", default="tokenized_texts.json")
    args = parser.parse_args()

    paths = collect_txt_paths(args.txt_dir)
    if not paths:
        raise SystemExit(f"no files found under {args.txt_dir!r}")

    train_tokenizer(paths, args.vocab_size, args.out_dir)
    tokenizer = load_tokenizer(f"{args.out_dir}/hermes-vocab.json", f"{args.out_dir}/hermes-merges.txt")

    ids = encode_files(tokenizer, paths)
    save_token_ids(ids, args.out_ids)
    print(f"encoded {len(paths)} files into {len(ids)} tokens -> {args.out_ids}")


if __name__ == "__main__":
    main()
