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

from contextlib import nullcontext

import torch
import deepspeed

from modules.dense_pipeline import GPT, GPTConfig
from utils import get_batch, generate_samples 

deepspeed.init_distributed() 
# TODO: pass this as commandline argument
cfg = OmegaConf.load(r"./config/DS-Pipeline-16.yaml")

# dump deepspeed config to file
with open("ds_config.json", "w") as f:
    json.dump(OmegaConf.to_container(cfg.deepspeed, resolve=True), f, indent=2)

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

master_process = True
seed_offset = 0

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]

# DeepSpeed manages autocast and loss-scaling internally.
ctx = nullcontext()
scaler = None

os.makedirs("data_dir", exist_ok=True)

iter_num = 0
best_val_loss = 1e9
meta_vocab_size = 256

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
    progressive_layer_drop=cfg.deepspeed.progressive_layer_drop.enabled,
    pld_theta=cfg.deepspeed.progressive_layer_drop.theta,
    pld_gamma=cfg.deepspeed.progressive_layer_drop.gamma,
    pipeline_parallel_stages=cfg.pipeline_parallel_stages,
) # start with model_args from command line

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
gptconf = GPTConfig(**model_args)

print(gptconf)
# init model but move to device later (deepspeed compatibility)
model = GPT(gptconf)

model_engine, optimizer = model.configure_model_engine(cfg)

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
@torch.no_grad()
def estimate_loss(step:int=None):
    # TODO when we're layer dropping this is broken for some reason
    out = {}
    model_eval_target = raw_model  # raw_model already points to engine.module for DS
    model_eval_target.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, cfg)
            with ctx:
                logits, loss = model_eval_target(X, Y, global_step=step)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_eval_target.train()
    return out

# training loop
X, Y = get_batch('train', cfg) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model used for logging/generation/estimate_mfu:
raw_model = model_engine.module
running_mfu = -1.0

# Debug memory snapshot passes
if cfg.debug_memory:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    for _ in range(2):
        with ctx:
            logits, loss = model_engine(X, Y)
        # DS handles scaling and backward
        model_engine.backward(loss)
        # clip + step handled below as appropriate for DS; but for the micro-run we call step then zero
        model_engine.step()
        model_engine.zero_grad()
    try:
        torch.cuda.memory._dump_snapshot(
            # TODO: update in case we do gradient remat/activation checkpointing
            f"layers_{cfg.n_layer}_remat_{0}_batchsize_{cfg.batch_size}.pickle"
        )
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")
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
        losses = estimate_loss(step=iter_num)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        img_path = f"eval_{iter_num}.jpg"
        # TODO: change split to eval for actual training runs
        # TODO: update generate_samples function to do predictions without layer dropping
        generate_samples(raw_model, n=cfg.eval_imgs, temperature=1.0, top_k=None, save_path=img_path, cfg=cfg, split="train")
        theta = 0
        if hasattr(model_engine.module, "progressive_layer_drop"):
            theta = model_engine.module.progressive_layer_drop.get_theta()
        if cfg.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": model_engine.lr_scheduler.get_lr()[0],
                "mfu": running_mfu*100, # convert to percentage
                "eval_images": wandb.Image(img_path),
                "pld_theta": theta,
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

    # forward backward update, gradient accumulation is handled by DeepSpeed
    with ctx:
        logits, loss = model_engine(X, Y, global_step=iter_num)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train', cfg)
    # backward
    model_engine.backward(loss)

    # clip the gradient
    if cfg.grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), cfg.grad_clip)

    # optimizer step + zero grad
    model_engine.step()          
    model_engine.zero_grad()

    # timing and logging (uses 'loss' from last micro-batch)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0 and master_process:
        # TODO: Gradient Accumulation setting in DeepSpeed Config
        lossf = loss.item() * cfg.deepspeed.gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.deepspeed.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.deepspeed.scheduler.params.total_num_steps:
        break
