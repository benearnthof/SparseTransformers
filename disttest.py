# dist_test.py
import os, torch, torch.distributed as dist
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
t = torch.ones(1, device=torch.cuda.current_device()) * (local_rank+1)
dist.all_reduce(t)
print("rank", dist.get_rank(), "after all_reduce:", t)
