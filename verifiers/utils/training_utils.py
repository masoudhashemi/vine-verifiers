import math

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup


class DummyOptim(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        return loss


class DummyScheduler:
    def __init__(self, optimizer, total_num_steps, warmup_num_steps):
        self.optimizer = optimizer
        self.total_num_steps = total_num_steps
        self.warmup_num_steps = warmup_num_steps

    def step(self):
        pass


def update_logs(logs, metrics_dict):
    for key, value in metrics_dict.items():
        if type(value).__module__.startswith("wandb"):
            logs[key] = value
        else:
            if key not in logs:
                logs[key] = []
            logs[key].append(value)
    return logs


def remove_trailing_pads(tensor: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Remove trailing padding tokens from a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len)
        pad_token_id (int): Token ID used for padding

    Returns:
        torch.Tensor: Tensor with trailing pads removed for each sequence
    """
    # Find last non-pad token for each sequence
    mask = tensor != pad_token_id
    last_nonpad = mask.long().argmax(dim=1)

    # Handle case where sequence is all padding
    all_pad = ~mask.any(dim=1)
    last_nonpad[all_pad] = 0

    # Get maximum length needed
    max_len = last_nonpad.max().item() + 1

    # Truncate to remove unnecessary padding
    return tensor[:, :max_len]


class EarlyStopping:
    """Early stopping handler class.

    Args:
        metric (str): Metric to monitor for early stopping
        mode (str): One of ['min', 'max']. Whether to look for minimum or maximum of metric
        patience (int): Number of epochs to wait for improvement before stopping
        warmup (int): Number of epochs to wait before starting to monitor for early stopping
    """

    def __init__(self, metric: str, mode: str = "min", patience: int = 3, warmup: int = 0):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch = 0

    def step(self, metrics: dict) -> tuple[bool, bool]:
        """Update early stopping state.

        Args:
            metrics (dict): Dictionary of metrics

        Returns:
            tuple[bool, bool]: (should_checkpoint, should_stop)
        """
        self.epoch += 1
        if self.epoch <= self.warmup:
            return False, False

        score = metrics.get(self.metric)
        if score is None:
            return False, False

        if self.best_score is None:
            self.best_score = score
            return True, False

        if self.mode == "min":
            improvement = self.best_score - score > 1e-5
        else:
            improvement = score - self.best_score > 1e-5

        if improvement:
            self.best_score = score
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            return False, self.counter >= self.patience


def apply_pad(tensors, padding_value=0, padding_side="right"):
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t
    return output


def get_warmup_steps(num_warmup_steps: float, max_train_steps: int) -> int:
    """Calculate the number of warmup steps.

    Args:
        num_warmup_steps (float): If < 1.0, will be treated as ratio of total steps
        max_train_steps (int): Total number of training steps

    Returns:
        int: Number of warmup steps to use
    """
    if num_warmup_steps < 1:
        return int(max_train_steps * num_warmup_steps)
    return int(num_warmup_steps)


def setup_scheduler(optimizer, num_warmup_steps, max_train_steps, accelerator=None, scheduler_type="cosine"):
    """Set up learning rate scheduler with warmup and decay.

    Args:
        optimizer: The optimizer to use
        num_warmup_steps: Number of warmup steps (or fraction of total steps)
        max_train_steps: Total number of training steps
        accelerator: Optional accelerator for preparation
        scheduler_type: Type of scheduler - "cosine", "linear", or "constant"

    Returns:
        The learning rate scheduler
    """
    # Calculate actual warmup steps
    warmup_steps = get_warmup_steps(num_warmup_steps, max_train_steps)

    # For linear scheduler, use the built-in transformers function
    if scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps
        )
        if accelerator is not None:
            lr_scheduler = accelerator.prepare(lr_scheduler)
        return lr_scheduler

    def cosine_lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay from 1.0 to 0.1
        progress = float(current_step - warmup_steps) / float(max(1, max_train_steps - warmup_steps))
        # Ensure progress is clamped between 0 and 1
        progress = min(1.0, max(0.0, progress))
        # Modified cosine schedule that decays from 1.0 to 0.1
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    def constant_lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))

        # Constant LR after warmup
        return 1.0

    # Choose the appropriate lambda function based on scheduler type
    if scheduler_type == "cosine":
        lr_lambda = cosine_lr_lambda
    elif scheduler_type == "constant":
        lr_lambda = constant_lr_lambda
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # Prepare with accelerator if provided
    if accelerator is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    return lr_scheduler
