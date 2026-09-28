"""TensorBoard logging + checkpointing callbacks."""
from __future__ import annotations

import os


class TrainLogger:
    def __init__(self, log_dir: str = "results/tensorboard", run_name: str = "run",
                 use_tensorboard: bool = True):
        self.use_tb = use_tensorboard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(os.path.join(log_dir, run_name))
            except Exception:
                self.writer = None

    def log(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
