"""
https://www.deepspeed.ai/training/#simplified-data-loader
DeepSpeed abstracts away data and model parallelism when it comes to data loading. 
Users simply provide a PyTorch dataset and DeepSpeed will automatically handle 
batch creation appropriately.
"""

import torch
import numpy as np
import os
from torch.utils.data import IterableDataset

# TODO: IO & Shuffling. For distributed training we may want to precompute index list per epoch
# TODO: save RNG state for resuming from checkpoint

class MemmapIterableDataset(IterableDataset):
    """
    Yields full micro-batches (cfg.batch_size, cfg.block_size).
    pin_memory=True for faster host -> device transfer
    worker_init_fn / dataset.open_memmap() in __iter__ to open memmap per worker 
    """
    def __init__(self, cfg, root="/root/data_dir", split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.filepath = os.path.join(root, f"{split}.bin")
        self._len = None

    def _get_len(self):
        if self._len is None:
            # everything is stored in raw bytes => file_size // block_size = number of images
            file_size = os.path.getsize(self.filepath)
            self._len = file_size // self.cfg.block_size
        return self._len

    def __iter__(self):
        data = np.memmap(self.filepath, dtype=np.uint8, mode="r")
        n_positions = data.shape[0] // self.cfg.block_size
        # shard across workers deterministically
        # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info
        worker_info = torch.utils.data.get_worker_info()
        rng = None
        if worker_info is None:
            # single-process DataLoader
            rng = np.random.default_rng()
        else:
            # create a worker-local RNG
            # TODO: do we need to incorporate RANK/WORLD_SIZE into data loader to make 
            # everything deterministic?
            worker_seed = torch.initial_seed() ^ worker_info.id
            rng = np.random.default_rng(worker_seed)
        # infinite iterator (could probably wrap this like lucidrains dataloader ?)
        while True:
            if not self.cfg.overfit:
                starts = rng.integers(0, n_positions - 1, size=self.cfg.batch_size)
            else:
                # when overfitting we just sample the same image over and over again
                starts = np.zeros(self.cfg.batch_size, dtype=np.int64)

            x_list, y_list = [], []
            for s in starts:
                offset = int(s) * self.cfg.block_size
                x_seq = np.asarray(data[offset:offset + self.cfg.block_size], dtype=np.int64)
                y_seq = np.asarray(data[offset + 1:offset + 1 + 3072], dtype=np.int64)
                # patch last byte
                y_seq[-1] = x_seq[-1]
                x_list.append(torch.from_numpy(x_seq))
                y_list.append(torch.from_numpy(y_seq))
            x = torch.stack(x_list)  # (batch_size, block_size)
            y = torch.stack(y_list)
            # pipeline format: ((idx, targets, global_step), {args})
            # TODO: no nested tuples, just tuple of tensors
            yield((x, y, None), {})

