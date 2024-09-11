"""Full training loop for the GPT model. Not run in this environment (no GPU) —
kept as a ready-to-execute script for whenever compute is available."""
import math
import os

import torch

from .config import GPTConfig, TrainConfig
from .dataset import TokenBlockDataset, load_token_ids
from .model import GPT


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, eval_iters: int, device: str):
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        it = iter(loader)
        for i in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(token_ids_path: str, gpt_cfg: GPTConfig, train_cfg: TrainConfig):
    torch.manual_seed(train_cfg.seed)
    os.makedirs(train_cfg.out_dir, exist_ok=True)

    ids = load_token_ids(token_ids_path)
    split = int(0.9 * len(ids))
    train_ds = TokenBlockDataset(ids[:split], gpt_cfg.block_size)
    val_ds = TokenBlockDataset(ids[split:], gpt_cfg.block_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=True)

    model = GPT(gpt_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )

    train_iter = iter(train_loader)
    for step in range(train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(train_cfg.device), y.to(train_cfg.device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            losses = estimate_loss(
                model, {"train": train_loader, "val": val_loader}, train_cfg.eval_iters, train_cfg.device
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            torch.save(
                {"model": model.state_dict(), "gpt_config": gpt_cfg, "step": step},
                os.path.join(train_cfg.out_dir, "ckpt.pt"),
            )

    return model


if __name__ == "__main__":
    # Example wiring — adjust paths once tokenized_texts.json exists and a GPU is available.
    gpt_cfg = GPTConfig()
    train_cfg = TrainConfig()
    train("tokenized_texts.json", gpt_cfg, train_cfg)
