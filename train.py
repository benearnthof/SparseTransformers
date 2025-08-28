import random
random.seed(1) # set a seed so that the results are consistent
import os
import numpy as np

import time
import math
import pickle
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from modules.vanilla import GPT, GPTConfig

# TODO: config
config = {}

# hyperparameters
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'sparse-transformer'
wandb_run_name = 'cifar-10' # 'run' + str(time.time())

# data
dataset = 'cifar-10'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# 6 seconds per batch for 24 batch size and 30 gradient accumulation steps
# 300 ms for 48 batch size and 1 gradient accumulation step
block_size = 3072

# model
# For testing purposes
n_layer = 128
n_head = 2
n_embd = 256
# TODO: add dropout for paper replication
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# for actual training:
# block_size: int = 3072
# vocab_size: int = 256 # 256 possible byte values
# n_layer: int = 128
# n_head: int = 2
# n_embd: int = 256
# dropout: float = 0.25
# bias: bool = True

# adamw optimizer
learning_rate = 0.0005 # max learning rate
# 120 epochs of 50k images = 6 million iterations / batch size = 500k iters
max_iters = 500000 # total number of training iterations
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
# gradient clipping as specified in paper
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 500000 # should be ~= max_iters per Chinchilla
min_lr = 0.000035 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
# TODO: implement distributed training

if ddp:
    pass
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs("data_dir", exist_ok=True)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join("data_dir", 'train.bin'), dtype=np.uint8, mode='r')
    else:
        data = np.memmap(os.path.join("data_dir", 'val.bin'), dtype=np.uint8, mode='r')
    # divide by block_size since we treat images as discrete samples
    ix = torch.randint(len(data)//block_size - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # TODO: instead of leaking next image we should repeat the preceding pixel
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9
meta_vocab_size = 256

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
gptconf = GPTConfig(**model_args)

print(gptconf)

model = GPT(gptconf)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# TODO: implement model.configure_optimizers
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# TODO: Utils
# TODO: Configs
# TODO: Memory profiling

# TODO: since we're training on images it would be nice to sample during training
# TODO: We need more powerful GPUs, the T4 in colab is very slow.
# A100 is powerful but would require about 4 days on an 8 GPU node
# TODO: There is no reason to train vanilla attention, we will do a short comparison to demonstrate the gain in throughput when using efficient GPU kernels
#     Flash attention reduces time per batch from ~31600ms to ~26000ms so give or take 5.5 seconds per batch (on T4)
# TODO: A100 or better should also be able to accommodate max_autotune_gemm mode https://discuss.pytorch.org/t/torch-compile-warning-not-enough-sms-to-use-max-autotune-gemm-mode/184405
# TODO: Adjust the positional encoding for image data
# TODO: Implement sparse kernels
# TODO: Recompute attention and feedforward blocks during backward pass to save memory
#   https://pytorch.org/blog/activation-checkpointing-techniques/
#   https://docs.pytorch.org/docs/stable/checkpoint.html
# TODO: Dropout only applied at the end of each residual addition.
# TODO: Pre-activation residual block of https://arxiv.org/pdf/1603.05027
# TODO: Weights and biases logging to benchmark memory usage
#   Done
# TODO: Automatic mixed precision/Mixed precision training
#   https://docs.pytorch.org/docs/stable/amp.html
#   already implemented with ctx context manager

# Currently this config peaks at around 11GB GPU HBM usage with torch.compile and no additional checkpointing
# TODO: Check impact of 3072 length ground truth
#   We should repeat the last pixel twice to avoid cross image contamination

# TODO: test learning rate warmup and other hyperparameters 
# TODO: DDP training
# TODO: Proper logging with image samples
# TODO: Load from checkpoint
# TODO: attention visualization for masked images
# TODO: visualize attention matrices for checkpoint
#   maybe during training? 

# current best with flawed implementation: 
# iter 610: loss 3.3404, time 8406.33ms, mfu 57.64%
# iter 611: loss 3.3030, time 8354.34ms, mfu 57.65%
# iter 612: loss 3.2933, time 8343.33ms, mfu 57.67%
# iter 613: loss 3.3958, time 8357.51ms, mfu 57.67%
# iter 614: loss 3.3052, time 8345.89ms, mfu 57.69%
# iter 615: loss 3.2869, time 8353.94ms, mfu 57.69%
# iter 616: loss 3.3050, time 8343.27ms, mfu 57.70%

# Current model seems to learn, albeit extremely slowly.
# Very nice resource
#  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
