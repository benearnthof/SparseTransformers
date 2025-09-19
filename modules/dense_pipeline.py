
# DeepSpeed implementation of pipeline parallel model
# I may regret doing pipeline parallelism first but let's see
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import get_flops

import deepspeed
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop

# TODO: Gradient Remat / Activation Checkpointing via DeepSpeed
from deepspeed.pipe import PipelineModule, LayerSpec

# For automatic PP we have to split the model into stages wrap them in a list of LayerSpec objects
# PipelineModule is a wrapper that takes in a list of layers and several arguments:
# num_stages: degree of parallelism
# topology: optional process topology that maps n-dimensional cartesian coordinates to linear indices
# loss_fn optional callable loss function for loss computation
# seed_layers: bool, optional; determine if a different seed should be used for each layer
# partition method: default parameters
# activation_checkpoint_interval: int, optional: The granularity of activation checkpointing in terms of number of layers
# https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction


class LayerNorm(nn.Module):
  """LayerNorm with optional bias"""
  def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

  def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # assert config.qk_dim % config.n_head == 0
    # key, query, value projections for all heads, but in a batch
    # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.c_q = nn.Linear(config.n_embd, config.qk_dim, bias=config.bias)
    self.c_k = nn.Linear(config.n_embd, config.qk_dim, bias=config.bias)
    self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    # regularization
    self.attn_dropout = nn.Dropout(config.attn_dropout)
    self.resid_dropout = nn.Dropout(config.resid_dropout)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.qk_dim = config.qk_dim
    self.dropout = config.attn_dropout
    # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    q = self.c_q(x)  # (B, T, qk_dim)
    k = self.c_k(x)  # (B, T, qk_dim)
    v = self.c_v(x)  # (B, T, n_embd)
    k = k.view(B, T, self.n_head, self.qk_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, self.qk_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if self.flash:
      # efficient attention using Flash Attention CUDA kernels
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
      # manual implementation of attention
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.resid_dropout(self.c_proj(y))
    return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.mlp_dim * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.mlp_dim * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.config = config

    def forward(self, x, p_i=None):
        """
        p_i: scalar keep-probability in (0,1], or None to disable PLD for this block.
        Behaviour:
          - if p_i is None or model is in eval mode => standard block
          - else: sample G ~ Bernoulli(p_i) using device-safe random and apply scaling by 1/p_i
        """
        # TODO: set flag for inference here
        if p_i is None: # or not self.training:
            # Standard transformer block
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        device = x.device
        keep = (torch.rand((), device=device) < p_i).to(x.dtype) 
        # Progressive Layer Dropping: Section 4.1 in https://arxiv.org/pdf/2010.13369
        if keep.item() == 0.0:
            # skip entire sublayer, return x unchanged
            return x

        x = x + self.attn(self.ln_1(x)) / p_i
        x = x + self.mlp(self.ln_2(x)) / p_i
        return x

# Helper to init pipeline stages
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.125 / math.sqrt(module.in_features))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        d = module.embedding_dim
        n = module.num_embeddings

        # vocab embeddings are size 256, positional embeddings are size 32
        if n > 32: # vocab embedding
            std = 0.125 / math.sqrt(d)
        else: # row/col/chan embeddings
            n_emb = 3
            std = 0.125 / math.sqrt(d * n_emb)

        torch.nn.init.normal_(module.weight, mean=0.0, std=std)

# Pipeline Parallel in DeepSpeed requires us to split the transformer into stages, such 
# that we can wrap them up in a list and pass it to the scheduler
class EmbeddingStage(nn.Module):
    # To accommodate PLD we simply pass global_step from stage to stage
    def __init__(self, config, init_fn=_init_weights):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.row_emb = nn.Embedding(32, config.n_embd)
        self.col_emb = nn.Embedding(32, config.n_embd)
        self.chan_emb = nn.Embedding(3, config.n_embd)
        self.drop = nn.Dropout(config.attn_dropout)
        self.apply(init_fn)

    def forward(self, idx, targets=None, global_step=None):
        b, t = idx.size()
        device = idx.device
        H, W = 32, 32
        tok_emb = self.wte(idx)
        positions = torch.arange(t, device=device)
        chans = positions // (H * W)
        rows  = (positions % (H * W)) // W
        cols  = positions % W
        row_emb = self.row_emb(rows)[None, :, :].expand(b, -1, -1)
        col_emb = self.col_emb(cols)[None, :, :].expand(b, -1, -1)
        chan_emb = self.chan_emb(chans)[None, :, :].expand(b, -1, -1)
        x = self.drop(tok_emb + row_emb + col_emb + chan_emb)
        # forward x along with targets/global_step for later stages
        return x, targets, global_step

class TransformerStage(nn.Module):
    def __init__(self, config, layer_idx, use_pld=False, pld_theta=None, pld_gamma=None, init_fn=_init_weights):
        """
        layer_idx: zero-based index used to compute the per-layer p_i.
        use_pld: whether to enable PLD here.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.block = Block(config)
        self.use_pld = use_pld
        if use_pld:
            # Each stage can keep its own scheduler instance
            self.pld = ProgressiveLayerDrop(theta=pld_theta, gamma=pld_gamma)
        self.apply(init_fn)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, x, targets=None, global_step=None):
        # compute p_i if PLD enabled and we have a global_step
        p_i = None
        if self.use_pld and (global_step is not None):
            # update scheduler and get theta
            self.pld.update_state(int(global_step))
            theta_t = self.pld.get_theta()
            L = float(self.block.config.n_layer) if hasattr(self.block, "config") else float(self.block.config.n_layer)
            # get_layer_prob:
            L = float(self.block.config.n_layer)
            drop_fraction = (self.layer_idx + 1) / L * (1.0 - theta_t)
            p_i = float(min(max(1.0 - drop_fraction, 1e-6), 1.0))
        # if p_i is None, block will act normally
        x = self.block(x, p_i=p_i)
        # pass through tuple for next stage
        return x, targets, global_step

class FinalStage(nn.Module):
    def __init__(self, config, init_fn=_init_weights):
        super().__init__()
        self.config = config
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(init_fn)
        # lm_head is zero init
        torch.nn.init.zeros_(self.lm_head.weight)

    def forward(self, x, targets=None, global_step=None):
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss


@dataclass
class GPTConfig:
    block_size: int = 3072
    vocab_size: int = 256 # 256 possible byte values
    n_layer: int = 128
    n_head: int = 2
    n_embd: int = 256
    mlp_dim: int = 2
    qk_dim: int = 128
    attn_dropout: float = 0.1
    resid_dropout: float = 0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    progressive_layer_drop: bool = False
    pld_theta: float = 0.5
    pld_gamma: float = 0.001
    pipeline_parallel_stages: int = 1

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if config.progressive_layer_drop:
            self.progressive_layer_drop = ProgressiveLayerDrop(
                theta=config.pld_theta,
                gamma=config.pld_gamma,
            )
        if self.config.pipeline_parallel_stages == 1:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                # positional encoding for CIFAR-10 images
                row_emb = nn.Embedding(32, config.n_embd),  # 32 rows
                col_emb = nn.Embedding(32, config.n_embd),  # 32 cols
                chan_emb = nn.Embedding(3, config.n_embd),  # 3 channels (RGB)
                drop = nn.Dropout(config.attn_dropout), # the paper does not specify if this is kept or removed, we keep it
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # init all weights
            self.apply(self._init_weights)
            torch.nn.init.zeros_(self.lm_head.weight)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        else:
            self.transformer = self.build_gpt_pipeline(self.config)   
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def build_gpt_pipeline(self, config):
        layers = []
        # Stage 0: embeddings
        layers.append(LayerSpec(EmbeddingStage, config))
        # Stage 1: one LayerSpec per transformer block with layer_idx passed
        for i in range(config.n_layer):
            layers.append(
                LayerSpec(
                    TransformerStage,
                    config,
                    i,
                    config.progressive_layer_drop,
                    config.pld_theta,
                    config.pld_gamma
                )
            )
        # Final stage
        layers.append(LayerSpec(FinalStage, config))
        pipeline = PipelineModule(
            layers=layers,
            num_stages=config.pipeline_parallel_stages,
            loss_fn=None,                 # FinalStage returns loss when targets present
            partition_method="uniform",
            seed_layers=True              # optional: use per-layer seeding for deterministic init across ranks
        )
        return pipeline

    def get_layer_prob(self, layer_idx: int, global_step: int):
        """
        Per-Layer  keep probability p_i as defined in Section 4.2: Distributing along the depth dimension in https://arxiv.org/pdf/2010.13369
        Using DeepSpeed scheduler initialized during model_engine init with args from config.
        """
        assert hasattr(self, "progressive_layer_drop"), "ProgressiveLayerDrop not configured for model."
        self.progressive_layer_drop.update_state(global_step)
        theta_t = self.progressive_layer_drop.get_theta()  # theta(t)
        L = float(self.config.n_layer)
        drop_fraction = (layer_idx + 1) / L * (1.0 - theta_t)
        p_i = 1.0 - drop_fraction
        # numerical safeguards, may need to set to 1e-3 if activations spike too much
        p_i = float(min(max(p_i, 1e-6), 1.0))
        return p_i

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.transformer.row_emb.weight.numel()
            n_params -= self.transformer.col_emb.weight.numel()
            n_params -= self.transformer.chan_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        # Updated initialization (Section 6 of the paper)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.125 / math.sqrt(module.in_features))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            d = module.embedding_dim
            # token embedding
            if module is self.transformer.wte:
                std = 0.125 / math.sqrt(d)
            # positional embeddings (row/col/chan)
            else:
                n_emb = 3  # row, col, chan
                std = 0.125 / math.sqrt(d * n_emb)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None, global_step=None, **kwargs):
        # TODO clean this up, maybe with flag: inference mode or something, & see if kwargs is needed still
        if global_step is None: # for deepspeed shenanigans
            global_step = kwargs.get("global_step", None)
        # print(f"Global step in forward:{global_step}")

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        H, W, C = 32, 32, 3
        positions = torch.arange(t, device=device)
        chans = positions // (H * W)                 # 0..2
        rows  = (positions % (H * W)) // W           # 0..31
        cols  = positions % W              
        row_emb = self.transformer.row_emb(rows)[None, :, :].expand(b, -1, -1)
        col_emb = self.transformer.col_emb(cols)[None, :, :].expand(b, -1, -1)
        chan_emb = self.transformer.chan_emb(chans)[None, :, :].expand(b, -1, -1)
        # initialization has been adjusted so summing should be fine
        x = self.transformer.drop(tok_emb + row_emb + col_emb + chan_emb)
        
        for i, block in enumerate(self.transformer.h):
            p_i = None
            if hasattr(self, "progressive_layer_drop") and global_step is not None:
                p_i = self.get_layer_prob(i, global_step) # num_layers in config
            #print(f"layer={i}, p_i={p_i}, is_training_call={is_training_call}, global_step={global_step}")
            x = block(x, p_i=p_i)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def configure_model_engine(self, cfg):
        """
        Configure and return model_engine and optimizer objects for DeepSpeed.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.deepspeed.optimizer.params.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            config="ds_config.json", model=self, model_parameters=optim_groups
        )
        return model_engine, optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # TODO: Pass in FP16 flops of hardware we're training on.
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = get_flops()
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



# Then we adjust the training script to roughly accommodate the following
"""
# init distributed
deepspeed.init_distributed()

# config
gptconf = GPTConfig(...)

# build pipeline model
model = build_gpt_pipeline(gptconf)

# initialize with DeepSpeed
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters()],
    config="ds_config.json"
)"""
