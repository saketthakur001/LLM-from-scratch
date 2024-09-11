from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 20000
    block_size: int = 256       # max context length (tokens per training sequence)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True           # bias in Linear/LayerNorm layers


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cpu"         # switch to "cuda" once a GPU is available
    out_dir: str = "checkpoints"
    seed: int = 1337
