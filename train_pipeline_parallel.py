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
from torch.utils.data import DataLoader
# from deepspeed.utils import 
from deepspeed.runtime.dataloader import RepeatingLoader, DeepSpeedDataLoader

from data.memmapdataset import CIFAR10Dataset
from modules.dense_pipeline import GPT, GPTConfig, GPTPipe
# TODO: rework these two utils to work with DeepSpeed Pipeline
from utils import get_batch, generate_samples, generate_samples_pipe 

# os.environ["WORLD_SIZE"] = "2"
deepspeed.init_distributed()
# TODO: pass this as commandline argument
cfg = OmegaConf.load(r"./config/DS-Pipeline-16.yaml")

# dump deepspeed config to file
with open("ds_config.json", "w") as f:
    json.dump(OmegaConf.to_container(cfg.deepspeed, resolve=True), f, indent=2)

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

# TODO: only one master_process
master_process = True
seed_offset = 0

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
# TODO: clean up since we're using DeepSpeed there is no more need to set these manually
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]

# DeepSpeed manages autocast and loss-scaling internally.
ctx = nullcontext()

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
    progressive_layer_drop=cfg.deepspeed.progressive_layer_drop.enabled,
    pld_theta=cfg.deepspeed.progressive_layer_drop.theta,
    pld_gamma=cfg.deepspeed.progressive_layer_drop.gamma,
    pipeline_parallel_stages=cfg.deepspeed.pipeline_parallel_stages,
    pp_partition_method=cfg.deepspeed.pp_partition_method,
    pp_activation_checkpoint_interval=cfg.deepspeed.pp_activation_checkpoint_interval,
) 

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
gptconf = GPTConfig(**model_args)

print(gptconf)

# pipeline expects iterators
train_ds = CIFAR10Dataset(cfg, split="train")
train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, pin_memory=True)
train_repl = RepeatingLoader(train_loader)
train_iter = iter(train_repl)

val_ds = CIFAR10Dataset(cfg, split="val")
val_loader = DataLoader(train_ds, batch_size=None, num_workers=4, pin_memory=True)

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
step = 0
k = 0
split = "train"

ds = CIFAR10Dataset(cfg, split=split)
dl = DataLoader(ds, 4, pin_memory=True)
it = iter(RepeatingLoader(dl))

data = next(iter(dl))
data[0] = data[0].to(model_engine.device)
data[1] = data[1].to(model_engine.device)
# updated since we only forward data
data = data[0]

model_engine.module(data)

loss = model_engine.eval_batch(iter(training_loader))  # iterator directly

# Here they seup cifar10
# https://github.com/deepspeedai/DeepSpeedExamples/blob/master/training/pipeline_parallelism/train.py
# as a trainset for training
# we need to pass in the loss function at the end of the pipeline module












# TODO: we might need to use the deepspeed loss function interface for pipeline parallel
# as otherwise inputs and targets get unpacked incorrectly.
@torch.no_grad()
def estimate_loss(step: int = None):
    out = {}
    model_engine.eval()
    for split in ["train", "val"]:
        ds = CIFAR10Dataset(cfg, split=split)
        dl = DataLoader(ds, 4, pin_memory=True)
        it = iter(RepeatingLoader(dl))
        # passing the data like this works, eval_batch & train_batch perform
        # loss calculation at the end and do not pass targets around. 
        # TODO: add loss calculation to GPTPipe, remove targets from pipeline steps
        # TODO: to pass in global step we return nested tuples from dataloader ?  
        data = next(iter(dl))
        data[0] = data[0].to(model_engine.device)
        data[1] = data[1].to(model_engine.device)
        model_engine.module(data)

        losses = []

        for _ in range(cfg.eval_iters):
            loss = model_engine.eval_batch(it)  # iterator directly
            losses.append(loss.detach().float().cpu())

        out[split] = torch.stack(losses).mean().item()

    model_engine.train()
    return out


# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# TODO: raw_model is outdated, need version compatible with pipeline
# raw_model = model_engine.module
# running_mfu = -1.0

# Debug memory snapshot passes
if cfg.debug_memory:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    probe_ds = CIFAR10Dataset(cfg, split="train")
    probe_dl = torch.utils.data.DataLoader(
        probe_ds, batch_size=None, num_workers=0, pin_memory=True
    )
    data = next(iter(probe_dl))
    for _ in range(2):
        with ctx:
            # TODO: expects iterator, not data
            loss = model_engine.train_batch(data)
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
        losses = estimate_loss(step=iter_num)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # img_path = f"eval_{iter_num}.jpg"
        # TODO: change split to eval for actual training runs
        # TODO: update generate_samples function to do predictions without layer dropping
        # TODO: generate samples should use model_engine.eval_batch
        # generate_samples(raw_model.module, n=cfg.eval_imgs, temperature=1.0, top_k=None, save_path=img_path, cfg=cfg, split="train")
        theta = 0
        # TODO: port to pipeline
        # if hasattr(model_engine.module, "progressive_layer_drop"):
        #     theta = model_engine.module.progressive_layer_drop.get_theta()
        if cfg.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": model_engine.lr_scheduler.get_lr()[0],
                "mfu": running_mfu*100, # convert to percentage
                # "eval_images": wandb.Image(img_path),
                # "pld_theta": theta,
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
    with ctx:
        data = ((X, Y, iter_num), {})
        loss = model_engine.train_batch(data)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train', cfg)

    # clip the gradient
    if cfg.grad_clip != 0.0:
        # if using ZeRO stage >0: use engine.clip_grad_norm_ when available:
        try:
            model_engine.clip_grad_norm_(cfg.grad_clip)
        except AttributeError:
            # fallback: use torch clip (may be incorrect for sharded params)
            torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), cfg.grad_clip)

    # timing and logging (uses 'loss' from last micro-batch)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0 and master_process:
        # TODO: Gradient Accumulation setting in DeepSpeed Config
        lossf = loss.item() * cfg.deepspeed.gradient_accumulation_steps
        # if local_iter_num >= 5:
            # TODO: update estimate_mfu to work with pipeline
            # mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.deepspeed.gradient_accumulation_steps, dt)
            # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms") #, mfu {running_mfu*100:.2f}%

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.deepspeed.scheduler.params.total_num_steps:
        break
