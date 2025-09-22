"""
Pipeline Communication
"""

import os
import torch
import torch.distributed as dist
import process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"


def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None, ops=None):
    """
    If ops is None: run immediately (blocking).
    If ops is a list: append a P2POp to it (non-blocking, DualPipe).
    """
    global STEP, VERBOSE

    is_send = operation.startswith("send")
    peer_rank, tensor_out = None, tensor

    if operation == "recv_forward":
        if pgm.process_group_manager.pp_is_first_stage:
            return None
        tensor_out = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        peer_rank = pgm.process_group_manager.pp_prev_rank

    elif operation == "send_forward":
        if pgm.process_group_manager.pp_is_last_stage:
            return
        peer_rank = pgm.process_group_manager.pp_next_rank

    elif operation == "recv_backward":
        if pgm.process_group_manager.pp_is_last_stage:
            return None
        tensor_out = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        peer_rank = pgm.process_group_manager.pp_next_rank

    elif operation == "send_backward":
        if pgm.process_group_manager.pp_is_first_stage:
            return
        peer_rank = pgm.process_group_manager.pp_prev_rank

    if peer_rank is None:
        return tensor_out if not is_send else None

    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor_out, peer_rank)

    if ops is None:
        # Immediate (blocking) mode
        if VERBOSE:
            print(
                f"{operation} | {'sending' if is_send else 'receiving'} "
                f"{operation.split('_')[1]} "
                f"{pgm.process_group_manager.pp_rank} "
                f"{'→' if is_send else '←'} {peer_rank} "
                f"| STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}",
                flush=True,
            )
        [req.wait() for req in dist.batch_isend_irecv([op])]
        torch.cuda.synchronize()
        if VERBOSE:
            STEP += 1
    else:
        # Non-blocking mode (DualPipe)
        ops.append(op)

    return tensor_out if not is_send else None


def bidirectional_pipeline_communicate(
    operation, send_tensor, recv_shapes, device, dtype, ops=None
):
    """
    If ops is None: run immediately.
    If ops is provided: append both ops to it.
    """
    global STEP, VERBOSE

    is_fwd = operation == "send_fwd_recv_bwd"
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (
        not is_fwd and pgm.process_group_manager.pp_is_first_stage
    ):
        return None

    peer_rank = (
        pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    )
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)

    send_op = dist.P2POp(dist.isend, send_tensor, peer_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, peer_rank)

    if ops is None:
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        if VERBOSE:
            print(
                f"{operation} | sending {'next' if is_fwd else 'prev'} "
                f"{pgm.process_group_manager.pp_rank} -> {peer_rank} | "
                f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pgm.process_group_manager.pp_rank} | "
                f"STEP {STEP=} | RANK:{pgm.process_group_manager.pp_rank}",
                flush=True,
            )
        [req.wait() for req in reqs]
        torch.cuda.synchronize()
        if VERBOSE:
            STEP += 1
    else:
        ops.extend([send_op, recv_op])

    return recv_tensor
### DualPipe
# requires [append, non-blocking isend, non-blocking irecv, batch wait, batch commit]
# utils: [scatter, gather] for splitting and merging batches into num_chunks
# WeightGradStore, run_backward for zero bubble & backward overlap & deferring weight updates
# split modules into two parts per rank, dual pipe expects phase0 and phase1 on each rank
# memory management & bookkeeping: queues of input / output chunks, to_free, deallocate output tensor

TENSOR_SHAPES = None
TENSOR_DTYPE = None

def set_p2p_tensor_shapes(shapes):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes

def set_p2p_tensor_dtype(dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype

def build_from_tensor_shapes(device):
    return [
        torch.empty(s, dtype=TENSOR_DTYPE, device=device, requires_grad=True)
        for s in TENSOR_SHAPES
    ]

def append_irecv(ops, src, group, device):
    """
    Append one or more irecv ops to the input ops and return allocated tensors.
    src is a logical pipeline rank, not the global rank
    """
    global_rank = dist.distributed.c10d.get_global_rank(group, src)
    tensors = build_from_tensor_shapes(device)
    for t in tensors:
        ops.append(dist.P2POp(dist.irecv, t, global_rank))
    return tensors

def append_isend(ops, tensors, dist, group):
    """
    append isend ops for a list of tensors
    """
    destination = dist.distributed_c10d.get_global_rank(group, dst)
    for t in tensors:
        ops.append(dist.P2POp(dist.isend, t, destination))
    
def commit_and_wait(ops, verbose=False):
    """
    launch and wait on all queued ops.
    """
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()
    if verbose:
        print(f"[DualPipe] committed {len(ops)} ops on rank {pgm.process_group_manager.pp_rank}", flush=True)
    ops.clear()
