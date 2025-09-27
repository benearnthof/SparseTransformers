# Training loop ported for DeepSpeed
import os
import json
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader

from data.memmapdataset import CIFAR10Dataset, get_args, pipeline_trainset
from modules.dense import GPTConfig
from modules.pipeline import GPTPipe


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
    trainset = pipeline_trainset(args.local_rank, cfg, split="train")

    # Initialize DeepSpeed - CRITICAL: Use the correct argument names
    model_engine, _, _, _ = deepspeed.initialize(
        # args=args,
        model=model,
        training_data=trainset,
        config = "./config/ds_config.json"  # Pass config directly
    )

    # Training loop
    for step in range(args.steps):
        loss = model_engine.train_batch()
        if model_engine.global_rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")

if __name__ == '__main__':
    args = get_args()
    # args.pipeline_parallel_size = 1
    # Initialize distributed backend first
    print(args)
    deepspeed.init_distributed(dist_backend=args.backend)
    print("xd")
    # Set device
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(args.local_rank)
    
    # Load config
    cfg = OmegaConf.load("./config/DS-Pipeline-16.yaml")
    print(cfg)
    
    # Ensure pipeline stages match CLI argument
    cfg.deepspeed.pipeline_parallel_stages = args.pipeline_parallel_size
    
    train_pipe(args, cfg)