from typing import Sized, Optional
import torch
from torch.utils.data import Sampler

class RepeatSequentialSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a sequential manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        # Use sequential indices: [0, 1, 2, 3, 4, ...]
        indexes = list(range(self.num_samples))

        # Split into batches: [[0, 1, 2], [3, 4, 5], ...]
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        # Keep only complete batches
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        # Calculate the number of complete batches
        num_complete_batches = self.num_samples // self.batch_size
        return num_complete_batches * self.batch_size * self.mini_repeat_count * self.repeat_count 