
"""
DeepSpeed PipelineModule wrapper for Llama modules. Nearly equivalent to pipeline.dense.py
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.llama import (
    Block,
    GPTConfig
)

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

def _init_embedding(module):
    # helper to initialize embeddings
    n = module.num_embeddings
    d = module.embedding_dim
    # vocab embeddings are size 256, positional embeddings are size 32
    if n > 32: # vocab embedding # TODO: set to 64 for imagenet?
        std = 0.125 / math.sqrt(d)
    else: # row/col/chan embeddings
        n_emb = 3
        std = 0.125 / math.sqrt(d * n_emb)
    nn.init.normal_(module.weight, mean=0.0, std=std)


# Helper to init pipeline stages
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.125 / math.sqrt(module.in_features))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        _init_embedding(module)


# split the transformer forward pass into three separate steps
class EmbeddingStage(nn.Module):
    """
    Pipeline Stage: token + positional/row/col/channel embeddings + optional dropout.
    """
    def __init__(self, cfg: GPTConfig, init_fn=_init_weights):
        super().__init__()
        self.cfg = cfg
        # match names from your GPT model
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.row_emb = nn.Embedding(32, cfg.n_embd)
        self.col_emb = nn.Embedding(32, cfg.n_embd)
        self.chan_emb = nn.Embedding(3, cfg.n_embd)
        # optional dropout (you can set to 0)
        attn_dropout = getattr(cfg, "attn_dropout", 0.0)
        self.drop = nn.Dropout(attn_dropout)
        self.apply(init_fn)

    def forward(self, inputs):
        # Expected inputs: token indices tensor [batch, seq_len]
        idx = inputs
        b, t = idx.size()
        device = idx.device
        H, W = 32, 32  # TODO: make this flexible for imagenet64
        tok_emb = self.tok_emb(idx)
        positions = torch.arange(t, device=device)
        chans = positions // (H * W)
        rows  = (positions % (H * W)) // W
        cols  = positions % W
        row_emb = self.row_emb(rows)[None, :, :].expand(b, -1, -1)
        col_emb = self.col_emb(cols)[None, :, :].expand(b, -1, -1)
        chan_emb = self.chan_emb(chans)[None, :, :].expand(b, -1, -1)
        x = self.drop(tok_emb + row_emb + col_emb + chan_emb)
        return x


class TransformerStage(nn.Module):
    """
    A single transformer block wrapped for pipeline lazy instantiation.

    DeepSpeed will construct one instance of this LayerSpec per LayerSpec entry.
    We accept both cfg and layer_id so we can preserve depth-dependent initialization.
    """
    def __init__(self, cfg: GPTConfig, layer_id: int, init_fn=_init_weights):
        super().__init__()
        self.block = Block(layer_id, cfg)
        self.apply(init_fn)
        for pn, p in self.named_parameters():
            # scaled init for output projection like in paper
            if pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, inputs):
        x = inputs
        return self.block(x)


class FinalStage(nn.Module):
    """Final stage: RMSNorm (or LayerNorm) + lm head. Loss is computed by the PipelineModule's loss_fn."""
    def __init__(self, cfg: GPTConfig, init_fn=_init_weights):
        super().__init__()
        self.cfg = cfg
        self.ln_f = nn.RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(init_fn)
        # zero init lm_head
        torch.nn.init.zeros_(self.lm_head.weight)

    def forward(self, inputs):
        x = inputs
        x = self.ln_f(x)
        logits = self.lm_head(x) if self.lm_head is not None else x
        return logits


class CustomCrossEntropyLoss(nn.Module):
    """Loss function for PipelineModule: receives logits and targets."""
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index)


# We split up PipelineParallelism into a separate module like in Megatron-DeepSpeed
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/8387ae17c4704f6579f88a84500b535d19d7fbbf/megatron/model/gpt_model.py


class GPTPipe(PipelineModule):
    """
    PipelineModule that splits the Llama-style GPT into stages using LayerSpec.
    One EmbeddingStage
    One LayerSpec(TransformerStage, cfg, layer_id=i) per transformer block
    FinalStage at the end. 
    Every pipeline stage only receives inputs, targets are passed to loss function by the pipeline.
    """
    def __init__(self, cfg: GPTConfig):
        # Build specs (LayerSpec will be instantiated on each pipeline rank)
        specs = []
        # Embedding stage
        specs.append(LayerSpec(EmbeddingStage, cfg))
        # One LayerSpec per transformer block (preserves layer_id)
        for i in range(cfg.n_layers):
            specs.append(LayerSpec(TransformerStage, cfg, i))
        # Final classification stage
        specs.append(LayerSpec(FinalStage, cfg))

        # Build PipelineModule
        super().__init__(
            layers=specs,
            loss_fn=CustomCrossEntropyLoss(ignore_index=-1),
            num_stages=getattr(cfg, "pipeline_parallel_stages", 1),
            partition_method=getattr(cfg, "pp_partition_method", "type:transformer"),
            seed_layers=True,
            activation_checkpoint_interval=getattr(cfg, "pp_activation_checkpoint_interval", 0),
        )