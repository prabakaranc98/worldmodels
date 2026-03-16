"""
VMC Distributed Training Utilities
====================================
Single source of truth for all distributed concerns.

Supports three strategies:
    "none"  — single device (MPS / CUDA / CPU), zero overhead
    "ddp"   — DistributedDataParallel via torchrun
    "fsdp"  — FullyShardedDataParallel (optional; for future large models)

Launch patterns:
    # single GPU / MPS — no change needed
    python -m VMC.run_training --phases train_vae

    # multi-GPU DDP
    torchrun --nproc_per_node=2 -m VMC.run_training --phases train_vae --dist_strategy ddp

    # multi-GPU FSDP (for large model variants)
    torchrun --nproc_per_node=4 -m VMC.run_training --phases train_vae --dist_strategy fsdp

torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically.
No model code changes are needed — callers use wrap_model() / is_main() / all_reduce_dict().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DistConfig:
    strategy: str = "none"               # "none" | "ddp" | "fsdp"
    backend: str = "nccl"                # "nccl" for CUDA, "gloo" for CPU/MPS
    find_unused_parameters: bool = False  # DDP only; set True for partial backprop


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

def init_process_group(cfg: DistConfig) -> None:
    """
    Initialise the distributed process group when strategy != "none".

    Safe to call unconditionally — no-ops when strategy is "none".
    torchrun sets RANK / LOCAL_RANK / WORLD_SIZE in the environment.
    """
    if cfg.strategy == "none":
        return
    if dist.is_initialized():
        return

    # fall back to gloo when nccl is unavailable (CPU / MPS)
    backend = cfg.backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend)


def cleanup() -> None:
    """Destroy the process group. Safe to call when not distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Global rank of this process. 0 if not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Local rank within the current node (used for CUDA device assignment)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Total number of processes. 1 if not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main() -> bool:
    """True only on rank 0 — use to gate checkpointing, wandb, printing."""
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------

def wrap_model(
    model: nn.Module,
    cfg: DistConfig,
    device: torch.device,
) -> nn.Module:
    """
    Wrap model in DDP or FSDP according to cfg.strategy.

    Must be called *after* model.to(device).

    DDP:  each GPU holds a full model copy; gradients are averaged via
          all-reduce after each backward pass.  Ideal for VMC model sizes.

    FSDP: shards parameters, gradients, and optimizer state across GPUs.
          Use when a future model variant doesn't fit on one GPU.
    """
    if cfg.strategy == "none":
        return model

    if cfg.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [device.index] if device.type == "cuda" else None
        return DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=cfg.find_unused_parameters,
        )

    if cfg.strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools

        # auto-wrap sub-modules with >1M parameters
        wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        return FSDP(model, auto_wrap_policy=wrap_policy, device_id=device)

    raise ValueError(f"Unknown dist strategy: {cfg.strategy!r}. Choose 'none', 'ddp', or 'fsdp'.")


# ---------------------------------------------------------------------------
# Data sampler
# ---------------------------------------------------------------------------

def build_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool,
    cfg: DistConfig,
) -> torch.utils.data.Sampler:
    """
    Return a DistributedSampler when running distributed, else a standard sampler.

    Usage in DataLoader:
        sampler = build_sampler(train_ds, shuffle=True, cfg=dist_cfg)
        loader  = DataLoader(train_ds, sampler=sampler, ...)

    In the training loop, call sampler.set_epoch(epoch) each epoch so that
    each rank sees a different shuffle order per epoch.
    """
    if cfg.strategy != "none" and dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    return torch.utils.data.SequentialSampler(dataset)


# ---------------------------------------------------------------------------
# Metric synchronisation
# ---------------------------------------------------------------------------

def all_reduce_dict(
    metrics: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    """
    Average a dict of scalar metrics across all ranks.

    No-op when not distributed (returns input unchanged).

    Usage:
        raw = {"recon": 0.12, "kl": 0.04, "total": 0.16}
        synced = all_reduce_dict(raw, device)
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return metrics

    keys = sorted(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return {k: tensor[i].item() for i, k in enumerate(keys)}
