"""Load a checkpoint and sample text from it."""
import argparse

import torch

from .model import GPT
from .tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--vocab", default="hermes-vocab.json")
    parser.add_argument("--merges", default="hermes-merges.txt")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = GPT(ckpt["gpt_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(args.vocab, args.merges)
    start_ids = tokenizer.encode(args.prompt).ids or [0]
    idx = torch.tensor([start_ids], dtype=torch.long, device=args.device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
