# Training loop ported for DeepSpeed
import os
import json
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader

from data.memmapdataset import CIFAR10Dataset, get_args
from modules.dense import GPTConfig
from modules.pipeline import GPTPipe

def create_dataloader(cfg, split="train"):
    """Create a simple DataLoader for DeepSpeed"""
    dataset = CIFAR10Dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return loader

def train_pipe(args, cfg):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    # Model configuration
    model_args = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        mlp_dim=cfg.mlp_dim,
        qk_dim=cfg.qk_dim,
        block_size=cfg.block_size,
        bias=cfg.bias,
        vocab_size=256,  # Set directly
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
        pipeline_parallel_stages=args.pipeline_parallel_size,  # Use CLI arg
        pp_partition_method=cfg.deepspeed.pp_partition_method,
        pp_activation_checkpoint_interval=cfg.deepspeed.pp_activation_checkpoint_interval,
    )

    gptconf = GPTConfig(**model_args)
    model = GPTPipe(gptconf)

    # Create data loader
    train_loader = create_dataloader(cfg, "train")

    # Initialize DeepSpeed - CRITICAL: Use the correct argument names
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        # args=args,
        model=model,
        training_data=train_loader,
        config = "ds_config.json"  # Pass config directly
    )

    # Training loop
    for step in range(args.steps):
        loss = model_engine.train_batch()
        if model_engine.global_rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")

if __name__ == '__main__':
    args = get_args()
    
    # Initialize distributed backend first
    deepspeed.init_distributed(dist_backend=args.backend)
    
    # Set device
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(args.local_rank)
    
    # Load config
    cfg = OmegaConf.load("./config/DS-Pipeline-16.yaml")
    
    # Ensure pipeline stages match CLI argument
    cfg.deepspeed.pipeline_parallel_stages = args.pipeline_parallel_size
    
    train_pipe(args, cfg)