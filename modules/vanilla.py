
# now let's implement our vanilla attention baseline
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import deepspeed

# gradient/activation checkpointing
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

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

    def forward_attention(self, x):
        # This will be checkpointed (recomputed)
        return self.attn(self.ln_1(x))

    def forward_mlp(self, x):
        # This will be kept in memory
        return self.mlp(self.ln_2(x))

    def forward(self, x):
        # Use checkpoint for attention, keep MLP in memory
        if self.config.use_selective_checkpointing:
            # Checkpoint the attention computation
            attn_output = torch.utils.checkpoint.checkpoint(
                self.forward_attention, 
                x, 
                use_reentrant=False
            )
            x = x + attn_output
            x = x + self.forward_mlp(x)
        else:
            # Standard forward pass
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

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
    rematerialization_steps: int = 8 # activation checkpointing to reduce GPU memory requirements of very deep nets
    use_selective_checkpointing: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
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
        # Section 6: The weight matrix for the output logits was initialized to 0.
        # The authors note that this, in combination with the smaller init requires more warmup steps
        torch.nn.init.zeros_(self.lm_head.weight)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # positional embedding for CIFAR-10 Images
        # decode flat indices (0..3071) into row, col, chan
        H, W, C = 32, 32, 3
        # we load CIFAR-10 images as 3072 bytes where the first 1024 correspond to the first 
        # channel of the image, 32 bytes of which correspond to the first row.
        positions = torch.arange(t, device=device)
        chans = positions // (H * W)                 # 0..2
        rows  = (positions % (H * W)) // W           # 0..31
        cols  = positions % W              
        row_emb = self.transformer.row_emb(rows)[None, :, :].expand(b, -1, -1)
        col_emb = self.transformer.col_emb(cols)[None, :, :].expand(b, -1, -1)
        chan_emb = self.transformer.chan_emb(chans)[None, :, :].expand(b, -1, -1)

        # initialization has been adjusted so summing should be fine
        x = self.transformer.drop(tok_emb + row_emb + col_emb + chan_emb)
        
        # TODO: option is still in config, wrap this up to make it available again
        # activation checkpointing as specified in config
        # segments = self.config.rematerialization_steps
        # x = checkpoint_sequential(self.transformer.h, segments, x, use_reentrant=False)

        for block in self.transformer.h:
            x = block(x)

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


    def configure_optimizers(self, cfg):
        """
        Configure optimizers for DDP.
        Basic implementation only requires the following keys in cfg:
            weight_decay, learning_rate, betas, device_type
        If cfg.use_deepspeed is True, everything is set up for ZeRO and CPU offloading.
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
            {'params': decay_params, 'weight_decay': cfg.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # backwards compatibility
        if not cfg.use_deepspeed:
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=cfg.betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            return optimizer
        else:

            model_engine, optimizer, _, _ = deepspeed.initialize(
                config="ds_config.json", model=self, model_parameters=optim_groups
            )
            return model_engine, optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS # 8.1 for the T4 in colab
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
