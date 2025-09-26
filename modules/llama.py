"""
llama3 style flexible transformer implementation
Adapted from
https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py
The goal is to use distributed attention to enable sequence parallel training
https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
We should be able to drop in DistributedAttention with minimal changes:
https://www.deepspeed.ai/tutorials/ds-sequence/
"""
import torch
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttention(torch.nn.Module):
    # helper module for compact attention implementation
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # we do not apply dropout in the attention layers, only at the end of each residual block
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)




# We switch to RMSNorm: https://arxiv.org/pdf/1910.07467

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """repeat_interleave for GQA"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention with GQA/MQA flexibility. To recover vanilla MHA set n_kv_heads to `None`.

    Args:
        config: OmegaConf dict of model specification.
    
    Attributes: 
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension of each attention head.
        wq (nn.Linear): Linear layer for queries.
        wk (nn.Linear): Linear layer for keys.
        wv (nn.Linear): Linear layer for values.
        wo (nn.Linear): Linear layer for output.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads if cfg.n_kv_heads is None else cfg.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        # we ditch half-size qk_dim since we now use GQA
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
        # torch spda already applies dropout for us
        # torchtitan wraps this with device specific backend, DeepSpeed already does this for us (?)
        self.spda = ScaledDotProductAttention()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            # adjusted to be consistent with Sparse Transformers paper
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.125 / math.sqrt(linear.in_features))
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std) # for depth dependent init


    def forward(self, x: torch.Tensor):
        bs, seqlen, _ = x.shape 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # infer head size from -1 as TP may have sharded them after linear ops
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        # TODO: try RoPE instead of position embedding on images
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

