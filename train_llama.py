import random
random.seed(1) # set a seed so that the results are consistent
import os
import numpy as np

import time
import math
import inspect

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from modules.llama import GPT, GPTConfig

def get_batch(split, device="cuda", overfit=True):
    if split == 'train':
        data = np.memmap(os.path.join("/workspace/data_dir", 'train.bin'), dtype=np.uint8, mode='r')
    else:
        data = np.memmap(os.path.join("/workspace/data_dir", 'val.bin'), dtype=np.uint8, mode='r')
    
    if not overfit:
        # overfitting on one image to test the architecture
        ix = torch.randint(len(data)//3072 - 3072, (4,))
    else:
        ix = torch.zeros(4, dtype=int)
    x_list, y_list = [], []
    for i in ix:
        # i = tensor(4634)
        offset = i * 3072
        x_seq = data[offset:offset+3072].astype(np.int64)
        # y is the same as y, offset by one extra pixel
        y_seq = data[offset+1:offset+1+3072].astype(np.int64)
        # patch last byte
        y_seq[-1] = x_seq[-1]  # repeat the last token instead of leaking
        x_list.append(torch.from_numpy(x_seq))
        y_list.append(torch.from_numpy(y_seq))
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

# system
ddp = (int(os.environ.get('RANK', -1)) != -1) # is this a ddp run?

if ddp:
    init_process_group(backend=cfg.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert cfg.gradient_accumulation_steps % ddp_world_size == 0
    cfg.gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = 1 * ddp_world_size * 4 * 3072
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs("/workspace/out_dir", exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["bfloat16"]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

iter_num = 0
best_val_loss = 1e9
LEARNING_RATE = 0.00035
BETAS = [0.95, 0.95]

gptconf = GPTConfig()

model = GPT(gptconf)

model = model.to("cuda")

def configure_optimizers(model):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=BETAS, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

optimizer = configure_optimizers(model)

if master_process:
    print_peak_memory("Max memory allocated after creating model/engine", 0)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(ptdtype == 'float16'))

# wrap model into DDP container & always compile
if ddp:
    model = DDP(torch.compile(model), device_ids=[ddp_local_rank])
else:
    model = torch.compile(model)

if master_process:
    print_peak_memory("Max memory allocated after creating DDP", 0)

# learning rate decay scheduler (cosine with warmup)
# TODO: move to DeepSpeed

WARMUP_ITERS = 500
LR_DECAY_ITERS = 5000
MIN_LR = LEARNING_RATE/10

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < 500:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
if ddp:
    # raw_model used for logging/generation/estimate_mfu:
    raw_model = model.module
else:
    raw_model = model

# TODO: we need to readd the loss function to the GPT llama module since the base implementation passes it to the pipeline directly
out = model(X)
out.shape # torch.Size([4, 3072, 256])

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
