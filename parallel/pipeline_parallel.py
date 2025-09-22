import torch
import torch.nn as nn
import torch.nn.functional as F

import process_group_manager as pgm
from comms import pipeline_communicate, bidirectional_pipeline_communicate
import comms as comm
from utils import WeightGradStore


from comms import (
    append_irecv, append_isend, commit_and_wait, set_p2p_tensor_shapes, set_p2p_tensor_dtype
)
from utils import WeightGradStore, run_backward, scatter, gather
import process_group_manager as pgm


class PipelineParallel(nn.Module):
    """
    Implements pipeline parallelism by distributing model layers across multiple GPUs.
    Each GPU processes a subset of the model's layers in a pipeline fashion.
    """
    def __init__(self, model, config):
        super().__init__()
        # Determine which layers should be assigned to this GPU
        self.layer_distribution = self.distribute_layers(config.num_hidden_layers)
        # Only first stage has embedding layer, others use Identity
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        # Assign relevant decoder layers to this GPU
        self.decoder_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in self.layer_distribution})
        # Only last stage has normalization and projection layers
        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.final_proj = model.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize or reset all model parameters for this pipeline stage."""
        if pgm.process_group_manager.pp_is_first_stage:
            self.embedding.reset_parameters()

        for layer in self.decoder_layers.values():
            layer.input_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()

        if pgm.process_group_manager.pp_is_last_stage:
            self.final_norm.reset_parameters()
            self.final_proj.reset_parameters()

    def distribute_layers(self, num_layers):
        """
        Distribute model layers across GPUs as evenly as possible.
        Returns the layer indices that should be processed by this GPU.
        """
        # Calculate layers per GPU, handling uneven distribution
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        # Calculate starting layer for this GPU
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    def forward(self, input_ids, position_ids, hidden_states):
        """
        Forward pass for this pipeline stage.
        Processes input through assigned layers and passes result to next stage.
        """
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embedding(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norm(x)
        return self.final_proj(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """
        Backward pass for this pipeline stage.
        Computes gradients for assigned layers using received gradient from next stage.
        """
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        # torch.autograd.backward will automatically accumulates gradients in the leaves (cf: https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):    
    """
    Implements 1F1B (one-forward-one-backward) pipeline parallel training.
    Interleaves forward and backward passes to improve GPU utilization.
    
    Pipeline stages:
    1. Warmup phase: Forward passes to fill pipeline
    2. Steady state: Alternating forward and backward passes
    3. Cooldown phase: Remaining backward passes
    
    Args:
        model: The pipeline parallel model
        data_loader: Iterator providing training batches
        tensor_shapes: Expected shapes of tensors for communication
        device: Device to run computations on
        dtype: Data type for tensors
    """
    # Calculate number of warmup microbatches needed
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    
    def _forward_step(input_tensor):
        """Helper function to perform a single forward step in the pipeline."""
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    # Warmup Phase: Fill the pipeline with forward passes
    for _ in range(num_warmup_microbatches):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        # Store tensors for later backward passes during cooldown phase
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # Steady State Phase: Alternate between forward and backward passes
    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    #NOTE: Explanation as to how to make DP and PP work together: https://github.com/huggingface/picotron/pull/5#issue-2629838274
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    for ith_microbatch in range(num_microbatches_remaining):  # 1F1B steady state
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        # Store current tensors for next backward pass
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # Retrieve oldest tensors for current backward pass (FIFO order)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    # Cooldown Phase: Complete remaining backward passes
    for ith_warmup_microbatches in range(num_warmup_microbatches):
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        # Process remaining stored tensors from warmup phase in FIFO order
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

### Work in progress, does not implement phase flipping (forward rank0, & rankN reversed) yet, 
# because picotron assumes a unidirectional pipeline.
# overlap of forward and backward is also not implemented, requires update to picotron layers
def train_step_pipeline_dualpipe(model, data_loader, tensor_shapes, device, dtype):
    """
    DualPipe training step.
    Implements zero-bubble, overlapped forward-backward pipeline schedule
    inspired by DeepSeek's DualPipe, but in Picotron style.

    Args:
        model: PipelineParallel model for this stage
        data_loader: Iterator yielding training batches
        tensor_shapes: Expected tensor shapes for P2P communication
        device: Local device for tensors
        dtype: Tensor dtype (e.g., torch.bfloat16)
    """
    # configure global comm buffers
    set_p2p_tensor_shapes(tensor_shapes)
    set_p2p_tensor_dtype(dtype)

    num_microbatches = data_loader.grad_acc_steps
    world_size = pgm.process_group_manager.pp_world_size
    rank = pgm.process_group_manager.pp_rank
    is_first = pgm.process_group_manager.pp_is_first_stage
    is_last = pgm.process_group_manager.pp_is_last_stage

    assert num_microbatches % 2 == 0 and num_microbatches >= 2 * world_size, \
        f"DualPipe requires even microbatches >= 2*pp_world_size, got {num_microbatches=}, {world_size=}"

    # scatter inputs into microbatches (like DeepSeek)
    batch = next(data_loader)
    inputs = (batch["input_ids"].to(device), batch["position_ids"].to(device))
    labels = batch["target_ids"].to(device)
    micro_inputs = scatter(inputs, num_microbatches, dim=0)
    micro_labels = scatter(labels, num_microbatches, dim=0)

    logging_loss = 0.0
    comm_ops = []

    # reset grad store
    WeightGradStore.clear()

    # buffers to hold intermediate states
    fwd_outputs, fwd_inputs = [], []
    bwd_output_grads = []

    # Step 1: warmup forward
    # push (world_size-1-rank) microbatches into pipeline
    warmup_forwards = (world_size - rank - 1)
    for i in range(warmup_forwards):
        mb_input = micro_inputs[i] if is_first else append_irecv(comm_ops, pgm.process_group_manager.pp_prev_rank, pgm.process_group_manager.pp_group, device)[0]
        commit_and_wait(comm_ops)
        output = model.forward(*mb_input) if is_first else model.forward(None, None, mb_input[0])
        if is_last:
            loss = F.cross_entropy(output.flatten(0, 1), micro_labels[i].flatten(), reduction="mean")
            logging_loss += loss.item() / num_microbatches
            output = loss
        fwd_inputs.append(mb_input)
        fwd_outputs.append(output)
        if not is_last:
            append_isend(comm_ops, [output], pgm.process_group_manager.pp_next_rank, pgm.process_group_manager.pp_group)
        commit_and_wait(comm_ops)

    # Step 2: overlapped steady state
    # alternate forward + backward (2F2B style)
    for i in range(warmup_forwards, num_microbatches):
        # recv forward input if not first stage
        if not is_first:
            mb_input = append_irecv(comm_ops, pgm.process_group_manager.pp_prev_rank, pgm.process_group_manager.pp_group, device)[0]
        else:
            mb_input = micro_inputs[i]

        # recv backward grad if not last stage
        if not is_last and bwd_output_grads:
            output_grad = bwd_output_grads.pop(0)
        else:
            output_grad = None

        commit_and_wait(comm_ops)

        # do forward
        output = model.forward(*mb_input) if is_first else model.forward(None, None, mb_input[0])
        if is_last:
            loss = F.cross_entropy(output.flatten(0, 1), micro_labels[i].flatten(), reduction="mean")
            logging_loss += loss.item() / num_microbatches
            output = loss
        fwd_inputs.append(mb_input)
        fwd_outputs.append(output)

        # send forward to next stage
        if not is_last:
            append_isend(comm_ops, [output], pgm.process_group_manager.pp_next_rank, pgm.process_group_manager.pp_group)

        # backward pass on oldest stored microbatch
        if len(fwd_outputs) > 1:  # overlap: keep pipeline filled
            old_input, old_output = fwd_inputs.pop(0), fwd_outputs.pop(0)
            if not is_first:
                grad_in = append_irecv(comm_ops, pgm.process_group_manager.pp_next_rank, pgm.process_group_manager.pp_group, device)[0]
            else:
                grad_in = None
            commit_and_wait(comm_ops)
            input_grad = model.backward(old_input[0], old_output, grad_in)
            if not is_first:
                append_isend(comm_ops, [input_grad], pgm.process_group_manager.pp_prev_rank, pgm.process_group_manager.pp_group)

        commit_and_wait(comm_ops)

    # Step 3: cooldown backward
    while fwd_outputs:
        old_input, old_output = fwd_inputs.pop(0), fwd_outputs.pop(0)
        grad_in = None if is_last else append_irecv(comm_ops, pgm.process_group_manager.pp_next_rank, pgm.process_group_manager.pp_group, device)[0]
        commit_and_wait(comm_ops)
        input_grad = model.backward(old_input[0], old_output, grad_in)
        if not is_first:
            append_isend(comm_ops, [input_grad], pgm.process_group_manager.pp_prev_rank, pgm.process_group_manager.pp_group)
        commit_and_wait(comm_ops)

    return logging_loss
