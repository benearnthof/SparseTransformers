# TODO: split up DeepSpeed training into separate file
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer

from modules.vanilla import GPT, GPTConfig
from utils import get_batch, generate_samples 

# TODO: pass this as commandline argument
cfg = OmegaConf.load(r"./config/ZeRO.yaml")

# dump deepspeed config to file
with open("ds_config.json", "w") as f:
    json.dump(OmegaConf.to_container(cfg.deepspeed, resolve=True), f, indent=2)

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

# system
ddp = (int(os.environ.get('RANK', -1)) != -1) and not cfg.use_deepspeed # is this a ddp run?
# print(f"DDP: {ddp}")
# print(f"WORLD_SIZE:{int(os.environ['WORLD_SIZE'])}")
# print(f"GRAD_ACCUMULATION_STEPS:{cfg.gradient_accumulation_steps}")
if ddp:
    init_process_group(backend=cfg.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert cfg.gradient_accumulation_steps % ddp_world_size == 0
    cfg.gradient_accumulation_steps //= ddp_world_size
else:
    # non DeepSpeed base case
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]

if cfg.use_deepspeed:
    # DeepSpeed manages autocast and loss-scaling internally.
    ctx = nullcontext()
    scaler = None
else: 
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


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
    rematerialization_steps=cfg.rematerialization_steps,
    use_selective_checkpointing=cfg.use_selective_checkpointing
) # start with model_args from command line

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
gptconf = GPTConfig(**model_args)

print(gptconf)
# init model but move to device later (deepspeed compatibility)
model = GPT(gptconf)

if cfg.use_deepspeed:
    # This returns model_engine + optimizer wrapper
    model_engine, optimizer = model.configure_optimizers(cfg)
else:
    model = model.to(cfg.device)
    optimizer = model.configure_optimizers(cfg)

if master_process:
    print_peak_memory("Max memory allocated after creating model/engine", 0)


if cfg.ckpt_path is not None:
    # TODO: verify this works
    print(f"Loading pretrained model from {cfg.ckpt_path}")
    if cfg.use_deepspeed:
        # Let deepspeed handle checkpoint partitioning
        # Expects folder with mp_rank_*/ files inside
        load_path, client_state = model_engine.load_checkpoint(
            cfg.ckpt_path,
            tag=None,  # https://www.deepspeed.ai/getting-started/#model-checkpointing
        )
        if load_path is None:
            print(f"WARNING: No DeepSpeed checkpoint found at {cfg.ckpt_path}")
        else:
            print(f"DeepSpeed checkpoint loaded successfully from {load_path}")
    else:
        # Vanilla torch checkpoint
        full_checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)
        model_state_dict, optim_state_dict = full_checkpoint["model"], full_checkpoint["optimizer"]

        # DDP adds `_orig_mod.` to keys
        new_state_dict = OrderedDict(
            (k.replace("_orig_mod.", ""), v) for k, v in model_state_dict.items()
        )
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        print(f"Model + optimizer state dicts loaded successfully.")
        # free up memory
        del full_checkpoint, model_state_dict, optim_state_dict, new_state_dict
else:
    print("Training from scratch.")
    checkpoint = None

# initialize a GradScaler. If enabled=False scaler is a no-op
if not cfg.use_deepspeed:
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

# wrap model into DDP container & always compile
if ddp:
    model = DDP(torch.compile(model), device_ids=[ddp_local_rank])
if master_process:
    print_peak_memory("Max memory allocated after creating DDP", 0)

# backwards compatibility & deepspeed
def forward_model(X, Y):
    """Run forward and return (logits, loss)."""
    if cfg.use_deepspeed:
        # model_engine(...) preserves the model signature
        return model_engine(X, Y)
    else:
        return model(X, Y)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model_eval_target = raw_model  # raw_model already points to engine.module for DS
    model_eval_target.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, cfg)
            with ctx:
                logits, loss = forward_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_eval_target.train()
    return out

# learning rate decay scheduler (cosine with warmup)
# TODO: move to DeepSpeed
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# training loop
X, Y = get_batch('train', cfg) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
if cfg.use_deepspeed:
    # raw_model used for logging/generation/estimate_mfu:
    raw_model = model_engine.module
else:
    model_engine = None
    raw_model = model.module if ddp else model
running_mfu = -1.0

# Debug memory snapshot passes
if cfg.debug_memory:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    for _ in range(2):
        with ctx:
            logits, loss = forward_model(X, Y)
            loss = loss / cfg.gradient_accumulation_steps
        if cfg.use_deepspeed:
            # DS handles scaling and backward
            model_engine.backward(loss)
            # clip + step handled below as appropriate for DS; but for the micro-run we call step then zero
            model_engine.step()
            model_engine.zero_grad()
        else:
            # vanilla: use scaler if float16 else normal backward
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    try:
        torch.cuda.memory._dump_snapshot(
            f"layers_{cfg.n_layer}_remat_{cfg.rematerialization_steps}_batchsize_{cfg.batch_size}.pickle"
        )
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

# logging
if cfg.wandb_log and master_process:
    import wandb
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name + f"{str(time.time())}", config=dict(cfg))
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file("ds_config.json")
    wandb.log_artifact(artifact)
    if cfg.debug_memory:
        artifact = wandb.Artifact("memory_snapshot", type="memory_snapshot")
        artifact.add_file(f"layers_{cfg.n_layer}_remat_{cfg.rematerialization_steps}_batchsize_{cfg.batch_size}.pickle")
        wandb.log_artifact(artifact)
        print(f"Logged memory snapshot to W&B.")

while True:
    # determine and set the learning rate for this iteration
    # TODO: learning rate schedule can be set in DeepSpeed config, split this like the rest
    lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
    opt_to_update = model_engine.optimizer if cfg.use_deepspeed else optimizer
    for param_group in opt_to_update.param_groups:
        param_group['lr'] = lr

    # evaluation & checkpointing
    if iter_num % cfg.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        img_path = f"eval_{iter_num}.jpg"
        # TODO: change split to eval for actual training runs
        generate_samples(raw_model, n=cfg.eval_imgs, temperature=1.0, top_k=None, save_path=img_path, cfg=cfg, split="train")
        if cfg.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "eval_images": wandb.Image(img_path)
            })
        if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                if cfg.use_deepspeed:
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
                    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
                else:
                    # vanilla torch checkpoint
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    print(f"saving checkpoint to {cfg.out_dir}")
                    # TODO: more descriptive checkpoint names
                    torch.save(checkpoint, os.path.join(cfg.out_dir, f'{cfg.wandb_run_name}_ckpt.pt'))

    if iter_num == 0 and cfg.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(cfg.gradient_accumulation_steps):
        if ddp and not cfg.use_deepspeed:
            model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = forward_model(X, Y)
            loss = loss / cfg.gradient_accumulation_steps # scaleing for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', cfg)
        # backward
        if cfg.use_deepspeed:
            model_engine.backward(loss)
        else:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

    # clip the gradient
    if cfg.grad_clip != 0.0:
        if cfg.use_deepspeed:
            torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), cfg.grad_clip)
        else:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # optimizer step + zero grad
    if cfg.use_deepspeed:
        model_engine.step()          
        model_engine.zero_grad()
    else:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # timing and logging (uses 'loss' from last micro-batch)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0 and master_process:
        lossf = loss.item() * cfg.gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.max_iters:
        break

if ddp:
    destroy_process_group()
