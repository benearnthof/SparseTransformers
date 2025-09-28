
# DeepSpeed implementation of pipeline parallel model
# I may regret doing pipeline parallelism first but let's see
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.dense import (
    Block,
    MLP,
    CausalSelfAttention,
    LayerNorm,
)

import deepspeed
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


# Helper to init pipeline stages
def _init_weights(module):
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
# TODO: progressive layer drop needs to be implemented like so
# https://github.com/deepspeedai/DeepSpeedExamples/blob/01f520e91d6b3235a4cabb1e7e634d9940319047/training/bing_bert/nvidia/modelingpreln_layerdrop.py#L537
# theta exists here: https://github.com/deepspeedai/DeepSpeedExamples/blob/01f520e91d6b3235a4cabb1e7e634d9940319047/training/bing_bert/nvidia/modelingpreln_layerdrop.py#L668C33-L668C38
# def forward(self, batch, **kwargs):
#     progressive_layer_drop = kwargs.get('progressive_layer_drop', False)
#     theta = kwargs.get('pld_theta', 1.0)
# can we get theta from kwargs? 
# https://github.com/deepspeedai/DeepSpeedExamples/blob/6bd444a7c62e9d7d320dd4c1e1142062f50c861d/bing_bert/nvidia/modelingpreln_layerdrop.py#L1159-L1160

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

    def forward(self, inputs):
        idx = inputs
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
        return x

class TransformerStage(nn.Module):
    def __init__(self, config, init_fn=_init_weights):
        """
        layer_idx: zero-based index used to compute the per-layer p_i.
        use_pld: whether to enable PLD here.
        """
        super().__init__()
        self.block = Block(config)
        # TODO: https://github.com/deepspeedai/DeepSpeedExamples/issues/169
        # the deepspeed engine maintains theta and gamma values already here: 
        # https://github.com/deepspeedai/DeepSpeed/blob/9bf1e9af3a3a958fc74b5d5d57e56b72559f5458/deepspeed/runtime/engine.py#L1530-L1531
        # state is also already updated after every global step
        self.apply(init_fn)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, inputs, progressive_layer_drop=False):
        x = inputs
        # TODO: get pld theta from engine kwargs https://github.com/deepspeedai/DeepSpeed/blob/9bf1e9af3a3a958fc74b5d5d57e56b72559f5458/deepspeed/runtime/engine.py#L1530-L1531
        x = self.block(x, p_i=None)
        # pass through tuple for next stage
        return x

class FinalStage(nn.Module):
    def __init__(self, config, init_fn=_init_weights):
        super().__init__()
        self.config = config
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(init_fn)
        # lm_head is zero init
        torch.nn.init.zeros_(self.lm_head.weight)

    def forward(self, inputs):
        x = inputs
        x = self.ln_f(x)
        logits = self.lm_head(x) # [4, 3072, 256]
        # loss calculation is performed by pipeline engine
        return logits

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index)


# We split up PipelineParallelism into a separate module like in Megatron-DeepSpeed
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/8387ae17c4704f6579f88a84500b535d19d7fbbf/megatron/model/gpt_model.py

class GPTPipe(PipelineModule):
    """
    DeepSpeed Pipeline for Pipeline Parallel training.
    """
    def __init__(self, config):
        self.specs = []
        # Stage 0: embeddings
        self.specs.append(LayerSpec(EmbeddingStage, config))
        # Stage 1: one LayerSpec per transformer block with layer_idx passed
        for i in range(config.n_layer):
            self.specs.append(
                LayerSpec(
                    TransformerStage,
                    config,
                )
            )
        # Final stage
        self.specs.append(LayerSpec(FinalStage, config))
        # TODO: once we integrate other forms of parallelism we want fine grained control over the topology
        # from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        # topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
        #                                     num_mp=mpu.get_tensor_model_parallel_world_size(),
        #                                     num_dp=mpu.get_data_parallel_world_size())

        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        super().__init__(
            layers=self.specs,
            loss_fn=CustomCrossEntropyLoss(ignore_index=-1), #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            num_stages=config.pipeline_parallel_stages,
            partition_method=config.pp_partition_method, # TODO: type:transformer partitioning for better balancing
            seed_layers=True,
            activation_checkpoint_interval=config.pp_activation_checkpoint_interval)
