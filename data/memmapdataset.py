"""
https://www.deepspeed.ai/training/#simplified-data-loader
DeepSpeed abstracts away data and model parallelism when it comes to data loading. 
Users simply provide a PyTorch dataset and DeepSpeed will automatically handle 
batch creation appropriately.
"""

import argparse
import torch
import numpy as np
import os
import deepspeed

import torch.distributed as dist
from torch.utils.data import IterableDataset, Dataset

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
            # DeepSpeed expects tuple of tensors 
            # TODO: something fucky is going on, switching to normal dataset
            yield x, y

class CIFAR10Dataset(Dataset):
    """
    Basic PyTorch Dataset that returns CIFAR-10 Image tensor stacks
    """
    def __init__(self, cfg, root="/root/data_dir", split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.filepath = os.path.join(root, f"{split}.bin")

    def __len__(self):
        # everything is stored in raw bytes => file_size // block_size = number of images
        file_size = os.path.getsize(self.filepath)
        return file_size // self.cfg.block_size

    def __getitem__(self, idx):
        data = np.memmap(self.filepath, dtype=np.uint8, mode="r")

        if self.cfg.overfit:
            # when overfitting we just sample the same image over and over again
            start = 0
        else: 
            start = idx

        offset = int(start) * self.cfg.block_size
        x_seq = np.asarray(data[offset:offset + self.cfg.block_size], dtype=np.int64)
        y_seq = np.asarray(data[offset + 1:offset + 1 + 3072], dtype=np.int64)
        # patch last byte
        y_seq[-1] = x_seq[-1]
        x = torch.from_numpy(x_seq)
        y = torch.from_numpy(y_seq)
        return x, y


def pipeline_trainset(local_rank, cfg, split, ds=CIFAR10Dataset):
    """
    Helper for pipeline parallel training
    """
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = ds(cfg, split=split)
    if local_rank == 0:
        dist.barrier()
    return trainset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    
    # This will add all DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args