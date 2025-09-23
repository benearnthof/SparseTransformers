# Training loop ported for DeepSpeed

import random
random.seed(1) # set a seed so that the results are consistent
import os
import numpy as np

import time
import math
import pickle
import yaml
import json
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import deepspeed
from torch.utils.data import DataLoader
import torch.distributed as dist
from deepspeed.runtime.dataloader import RepeatingLoader

from data.memmapdataset import CIFAR10Dataset
from modules.dense import GPTConfig
from modules.pipeline import GPTPipe
# TODO: rework these two utils to work with DeepSpeed Pipeline
from utils import (
    get_iterator,
    get_batch,
    generate_samples,
    generate_samples_pipe,
)

deepspeed.init_distributed()
# TODO: pass this as commandline argument
cfg = OmegaConf.load(r"./config/DS-Pipeline-16.yaml")

# dump deepspeed config to file
with open("ds_config.json", "w") as f:
    json.dump(OmegaConf.to_container(cfg.deepspeed, resolve=True), f, indent=2)

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

master_process = dist.get_rank() == 0
seed_offset = 0

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

# DeepSpeed manages autocast and loss-scaling internally.

os.makedirs("data_dir", exist_ok=True)

iter_num = 0
best_val_loss = 1e9
meta_vocab_size = 256

# TODO: join model args from config and commandline
model_args = dict(
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    mlp_dim=cfg.mlp_dim,
    qk_dim=cfg.qk_dim,
    block_size=cfg.block_size,
    bias=cfg.bias,
    vocab_size=None,
    attn_dropout=cfg.attn_dropout,
    resid_dropout=cfg.resid_dropout,
    progressive_layer_drop=False, #cfg.deepspeed.progressive_layer_drop.enabled,
    pld_theta=1,#cfg.deepspeed.progressive_layer_drop.theta,
    pld_gamma=0.000001, #cfg.deepspeed.progressive_layer_drop.gamma,
    pipeline_parallel_stages=cfg.deepspeed.pipeline_parallel_stages,
    pp_partition_method=cfg.deepspeed.pp_partition_method,
    pp_activation_checkpoint_interval=cfg.deepspeed.pp_activation_checkpoint_interval,
) 

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
gptconf = GPTConfig(**model_args)

print(gptconf)

# pipeline expects iterators
train_ds = CIFAR10Dataset(cfg, split="train")
# TODO: actually pass val_ds to validation functions xd
val_ds = CIFAR10Dataset(cfg, split="val")

# build pipeline model
model = GPTPipe(gptconf)

# initialize with DeepSpeed
model_engine, optimizer, training_loader, _ = deepspeed.initialize(
    model=model,
    training_data=train_ds,
    # TODO: manually set optimizer groups like we did originally, passed in as model_parameters
    config="ds_config.json"
)

if master_process:
    print_peak_memory("Max memory allocated after creating model/engine", 0)

tokens_per_iter = cfg.deepspeed.gradient_accumulation_steps * model_engine.world_size * cfg.batch_size * cfg.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if cfg.ckpt_path is not None:
    # TODO: verify this works
    print(f"Loading pretrained model from {cfg.ckpt_path}")
    load_path, client_state = model_engine.load_checkpoint(
        cfg.ckpt_path,
        tag=None,  # https://www.deepspeed.ai/getting-started/#model-checkpointing
    )
    if load_path is None:
        print(f"WARNING: No DeepSpeed checkpoint found at {cfg.ckpt_path}")
    else:
        print(f"DeepSpeed checkpoint loaded successfully from {load_path}")
else:
    print("Training from scratch.")
    checkpoint = None

# helps estimate an arbitrarily accurate loss over either split using many batches
# TODO: pass keep_prob = 1 to PLD during eval
@torch.no_grad()
def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        it = get_iterator(CIFAR10Dataset, cfg, split)
        losses = []
        for _ in range(cfg.eval_iters):
            loss = model_engine.eval_batch(it)  # iterator directly
            losses.append(loss.detach().float().cpu())
        out[split] = torch.stack(losses).mean().item()
    return out

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

# Debug memory snapshot passes
if cfg.debug_memory:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    it = get_iterator(CIFAR10Dataset, cfg, split="train")
    for _ in range(2):
        loss = model_engine.train_batch(it)
    # Dump snapshot to disk
    try: # TODO: update filenames for DeepSpeed
        fname = f"layers_{cfg.n_layer}_remat_{0}_batchsize_{cfg.batch_size}.pickle"
        if dist.get_rank() == 0:
            torch.cuda.memory._dump_snapshot(fname)
        print(f"[Memory Debug] Dumped CUDA snapshot to {fname}")
    except Exception as e:
        print(f"[Memory Debug] Failed to capture memory snapshot: {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

# logging
# TODO: move to deepspeed config
if cfg.wandb_log and master_process:
    import wandb
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name + f"{str(time.time())}", config=dict(cfg))
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file("ds_config.json")
    wandb.log_artifact(artifact)
    if cfg.debug_memory:
        artifact = wandb.Artifact("memory_snapshot", type="memory_snapshot")
        # TODO: update in case we do gradient remat/activation checkpointing
        artifact.add_file(f"layers_{cfg.n_layer}_remat_{0}_batchsize_{cfg.batch_size}.pickle")
        wandb.log_artifact(artifact)
        print(f"Logged memory snapshot to W&B.")

while True:
    # evaluation & checkpointing
    if iter_num % cfg.eval_interval == 0 and master_process:
        # Build pipeline argument (EmbeddingStage expects idx, targets, global_step)
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        img_path = f"eval_{iter_num}.jpg"
        # TODO: change split to eval for actual training runs
        # TODO: generate samples should use inference https://www.deepspeed.ai/inference/
        generate_samples_pipe(model_engine, n=cfg.eval_imgs, temperature=1.0, top_k=None, save_path=img_path, cfg=cfg, split="train")
        if cfg.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": model_engine.lr_scheduler.get_lr()[0],
                "tpi": tokens_per_iter, # TODO: calculate mfu from tokens/iter & tokens/second
                "eval_images": wandb.Image(img_path),

            })
        if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                client_state = {
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': cfg,
                }
                save_dir = os.path.join(cfg.out_dir, f'{cfg.wandb_run_name}_ds_ckpt')
                # TODO: use val_loss as checkpoint tag
                tag = f"step_{iter_num}"
                print(f"[DeepSpeed] saving checkpoint to {save_dir}, tag {tag}")
                # TODO: more descriptive checkpoint names
                model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)

    if iter_num == 0 and cfg.eval_only:
        break

    # train_batch combines forward, backward and optim steps
    loss = model_engine.train_batch()

    # timing and logging (uses 'loss' from last micro-batch)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0 and master_process:
        # TODO: Gradient Accumulation setting in DeepSpeed Config
        lossf = loss.item() * cfg.deepspeed.gradient_accumulation_steps
        # TODO: calculate mfu via tokens/second    
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.deepspeed.scheduler.params.total_num_steps:
        break
