"""
VMC Training Runner
===================
Single entry point for the full VMC training pipeline.

Phases (run in order, or select individually):
    1  collect    — collect random rollouts from the gym environment
    2  train_vae  — train BetaVAE on raw frames
    3  encode     — encode all episodes to latent sequences with trained VAE
    4  train_mdn  — train MDN-RNN on encoded latent sequences
    5  train_ctrl — optimise LinearController with CMA-ES

Usage examples:
    # Run all phases end-to-end
    python -m VMC.run_training --all

    # Run individual phases
    python -m VMC.run_training --phases collect train_vae encode train_mdn train_ctrl

    # Override key hyperparams
    python -m VMC.run_training --all --n_episodes 50 --vae_epochs 10 --z_dim 32

    # Resume VAE training from checkpoint
    python -m VMC.run_training --phases train_vae --vae_resume ./VMC_checkpoints/vae_epoch_0010.pt

    # Disable wandb
    WANDB_MODE=disabled python -m VMC.run_training --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m VMC.run_training",
        description="VMC training pipeline — Vision · Memory · Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Phase selection ---
    phase_group = p.add_mutually_exclusive_group(required=True)
    phase_group.add_argument(
        "--all", action="store_true",
        help="Run all 5 phases in sequence.",
    )
    phase_group.add_argument(
        "--phases", nargs="+",
        choices=["collect", "train_vae", "encode", "train_mdn", "train_ctrl"],
        metavar="PHASE",
        help="One or more phases to run (in the order given).",
    )

    # --- Shared / paths ---
    p.add_argument("--data_dir", default="./VMC_data", help="Data directory.")
    p.add_argument("--checkpoint_dir", default="./VMC_checkpoints", help="Checkpoint directory.")
    p.add_argument("--env_id", default="CarRacing-v3", help="Gymnasium environment ID.")

    # --- Data collection ---
    p.add_argument("--n_episodes", type=int, default=10_000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=-1,
                   help="Parallel workers for collect + CMA-ES. -1 = all CPU cores.")
    p.add_argument("--seed", type=int, default=0)

    # --- Shared architecture ---
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--action_dim", type=int, default=3)

    # --- VAE ---
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--vae_epochs", type=int, default=50)
    p.add_argument("--vae_lr", type=float, default=1e-4)
    p.add_argument("--vae_batch", type=int, default=64)
    p.add_argument("--vae_resume", type=str, default=None,
                   help="Path to VAE checkpoint to resume from.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")

    # --- MDN-RNN ---
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_mixtures", type=int, default=5)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--mdn_epochs", type=int, default=30)
    p.add_argument("--mdn_lr", type=float, default=1e-3)
    p.add_argument("--mdn_batch", type=int, default=32)
    p.add_argument("--mdn_resume", type=str, default=None)

    # --- Controller (CMA-ES) ---
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--sigma0", type=float, default=0.1)
    p.add_argument("--n_rollouts", type=int, default=16,
                   help="Rollouts averaged per CMA-ES candidate.")
    p.add_argument("--n_envs", type=int, default=-1,
                   help="AsyncVectorEnv workers for CMA-ES evaluation. -1 = os.cpu_count().")
    p.add_argument("--vae_ckpt", type=str, default=None,
                   help="VAE checkpoint for controller rollouts (auto-detected if omitted).")
    p.add_argument("--mdn_ckpt", type=str, default=None,
                   help="MDN checkpoint for controller rollouts (auto-detected if omitted).")

    # --- Distributed training ---
    p.add_argument("--dist_strategy", choices=["none", "ddp", "fsdp"], default="none",
                   help="Distributed strategy. Use with torchrun for ddp/fsdp.")
    p.add_argument("--dist_backend", choices=["nccl", "gloo"], default="nccl",
                   help="Distributed backend. Use gloo for CPU/MPS, nccl for CUDA.")

    return p


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _make_data_cfg(args: argparse.Namespace):
    from .data import DataConfig
    return DataConfig(
        env_id=args.env_id,
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
        img_size=args.img_size,
        data_dir=args.data_dir,
        n_workers=args.n_workers,
        seed=args.seed,
    )


def _make_vae_cfg(args: argparse.Namespace):
    from .vision import VAEConfig
    return VAEConfig(
        img_channels=3,
        img_size=args.img_size,
        z_dim=args.z_dim,
        beta=args.beta,
    )


def _make_mdn_cfg(args: argparse.Namespace):
    from .memory import MDNRNNConfig
    return MDNRNNConfig(
        z_dim=args.z_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
    )


def _make_ctrl_cfg(args: argparse.Namespace):
    from .controller import ControllerConfig
    return ControllerConfig(
        z_dim=args.z_dim,
        h_dim=args.hidden_size,
        action_dim=args.action_dim,
        sigma0=args.sigma0,
        popsize=args.popsize,
        max_iter=args.max_iter,
        n_rollouts_per_candidate=args.n_rollouts,
        n_envs=args.n_envs,
        seed=args.seed,
        checkpoint_path=str(Path(args.checkpoint_dir) / "ctrl_best.pt"),
    )


def _make_dist_cfg(args: argparse.Namespace):
    from .distributed import DistConfig
    return DistConfig(strategy=args.dist_strategy, backend=args.dist_backend)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_collect(args: argparse.Namespace) -> None:
    from .data import collect_rollouts
    print("\n" + "=" * 60)
    print("PHASE 1 — Collect rollouts")
    print("=" * 60)
    cfg = _make_data_cfg(args)
    collect_rollouts(cfg)


def phase_train_vae(args: argparse.Namespace) -> None:
    from .trainer import VAETrainer, VAETrainerConfig
    print("\n" + "=" * 60)
    print("PHASE 2 — Train VAE")
    print("=" * 60)

    cfg = VAETrainerConfig(
        vae_cfg=_make_vae_cfg(args),
        data_cfg=_make_data_cfg(args),
        dist_cfg=_make_dist_cfg(args),
        epochs=args.vae_epochs,
        lr=args.vae_lr,
        batch_size=args.vae_batch,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=not args.no_compile,
    )

    if args.vae_resume:
        trainer = VAETrainer.from_checkpoint(args.vae_resume, cfg)
    else:
        trainer = VAETrainer(cfg)

    trainer.train()


def phase_encode(args: argparse.Namespace) -> None:
    import torch
    from .data import encode_dataset
    from .trainer import get_device
    from .vision import BetaVAE, VAEConfig

    print("\n" + "=" * 60)
    print("PHASE 3 — Encode latents")
    print("=" * 60)

    vae_ckpt = args.vae_ckpt or _find_latest(args.checkpoint_dir, "vae_final.pt", "vae_*.pt")
    if vae_ckpt is None:
        sys.exit("[encode] No VAE checkpoint found. Run train_vae first.")
    print(f"  Using VAE checkpoint: {vae_ckpt}")

    vae_cfg = _make_vae_cfg(args)
    vae = BetaVAE(vae_cfg)
    ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])

    device = get_device()
    encode_dataset(vae, data_dir=args.data_dir, device=device)


def phase_train_mdn(args: argparse.Namespace) -> None:
    from .trainer import MDNRNNTrainer, MDNRNNTrainerConfig
    print("\n" + "=" * 60)
    print("PHASE 4 — Train MDN-RNN")
    print("=" * 60)

    cfg = MDNRNNTrainerConfig(
        mdn_cfg=_make_mdn_cfg(args),
        data_cfg=_make_data_cfg(args),
        dist_cfg=_make_dist_cfg(args),
        seq_len=args.seq_len,
        epochs=args.mdn_epochs,
        lr=args.mdn_lr,
        batch_size=args.mdn_batch,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=not args.no_compile,
    )

    if args.mdn_resume:
        trainer = MDNRNNTrainer.from_checkpoint(args.mdn_resume, cfg)
    else:
        trainer = MDNRNNTrainer(cfg)

    trainer.train()


def phase_train_ctrl(args: argparse.Namespace) -> None:
    from .controller import CMAESTrainer, LinearController

    print("\n" + "=" * 60)
    print("PHASE 5 — Train Controller (CMA-ES)")
    print("=" * 60)

    vae_ckpt = args.vae_ckpt or _find_latest(args.checkpoint_dir, "vae_final.pt", "vae_*.pt")
    mdn_ckpt = args.mdn_ckpt or _find_latest(args.checkpoint_dir, "mdn_final.pt", "mdn_*.pt")

    if vae_ckpt is None:
        sys.exit("[ctrl] No VAE checkpoint found.")
    if mdn_ckpt is None:
        sys.exit("[ctrl] No MDN checkpoint found.")

    print(f"  VAE: {vae_ckpt}")
    print(f"  MDN: {mdn_ckpt}")

    ctrl_cfg = _make_ctrl_cfg(args)
    ctrl = LinearController(ctrl_cfg)

    # Build a picklable rollout function bound to the checkpoint paths
    rollout_fn = _make_rollout_fn(
        vae_ckpt=vae_ckpt,
        mdn_ckpt=mdn_ckpt,
        vae_cfg_kwargs=dict(
            img_channels=3, img_size=args.img_size, z_dim=args.z_dim, beta=args.beta
        ),
        mdn_cfg_kwargs=dict(
            z_dim=args.z_dim,
            action_dim=args.action_dim,
            hidden_size=args.hidden_size,
            num_mixtures=args.num_mixtures,
        ),
        ctrl_cfg_kwargs=ctrl_cfg.__dict__.copy(),
        env_id=args.env_id,
        max_steps=args.max_steps,
    )

    trainer = CMAESTrainer(
        ctrl_cfg, ctrl,
        vec_rollout_fn=_make_vec_rollout_fn(
            vae_ckpt=vae_ckpt,
            mdn_ckpt=mdn_ckpt,
            vae_cfg_kwargs=dict(img_channels=3, img_size=args.img_size, z_dim=args.z_dim, beta=args.beta),
            mdn_cfg_kwargs=dict(z_dim=args.z_dim, action_dim=args.action_dim,
                                hidden_size=args.hidden_size, num_mixtures=args.num_mixtures),
            ctrl_cfg_kwargs=ctrl_cfg.__dict__.copy(),
        ),
        env_id=args.env_id,
        max_steps=args.max_steps,
    )
    best_params, best_reward = trainer.train()
    print(f"\nController training done. Best reward: {best_reward:.2f}")


# ---------------------------------------------------------------------------
# Rollout function factory (module-level for pickling)
# ---------------------------------------------------------------------------

def _make_vec_rollout_fn(
    vae_ckpt: str,
    mdn_ckpt: str,
    vae_cfg_kwargs: dict,
    mdn_cfg_kwargs: dict,
    ctrl_cfg_kwargs: dict,
):
    """
    Return a vectorised rollout function for CMAESTrainer.

    Signature:
        vec_rollout_fn(params_list, seeds, env_id, n_envs, max_steps) → list[float]

    Uses gymnasium.vector.AsyncVectorEnv — persistent env subprocesses shared
    across all calls for a generation. Each call steps `n_envs` envs in parallel
    with their respective controller params until all episodes are done.

    Dill (multiprocess) serialises the closure correctly on macOS/spawn.
    """

    def vec_rollout_fn(
        params_list: list,
        seeds: list,
        env_id: str,
        n_envs: int,
        max_steps: int,
    ) -> list[float]:
        import gymnasium as gym
        import numpy as np
        import torch

        from VMC.controller import ControllerConfig, LinearController
        from VMC.memory import MDNRNN, MDNRNNConfig
        from VMC.model import WorldModel, WorldModelConfig
        from VMC.vision import BetaVAE, VAEConfig

        # Build one WorldModel per env (each carries its own LSTM hidden state)
        def make_wm(params_flat):
            vae_cfg = VAEConfig(**vae_cfg_kwargs)
            mdn_cfg = MDNRNNConfig(**mdn_cfg_kwargs)
            ctrl_cfg = ControllerConfig(**ctrl_cfg_kwargs)
            wm_cfg = WorldModelConfig(vae_cfg=vae_cfg, mdn_cfg=mdn_cfg, ctrl_cfg=ctrl_cfg)
            wm = WorldModel.from_checkpoints(
                vae_ckpt, mdn_ckpt,
                ctrl_path=ctrl_cfg_kwargs.get("checkpoint_path", ""),
                cfg=wm_cfg,
            )
            wm.ctrl.set_params(params_flat)
            wm.ctrl.eval()
            wm.reset()
            return wm

        world_models = [make_wm(p) for p in params_list]

        # Persistent vectorised envs — no spawn overhead between candidates
        envs = gym.vector.AsyncVectorEnv([
            (lambda s=seeds[i]: lambda: gym.make(env_id))()   # one factory per env
            for i in range(n_envs)
        ])

        obs_batch, _ = envs.reset(seed=seeds)  # (n_envs, H, W, C)
        totals = [0.0] * n_envs
        dones = [False] * n_envs

        for _ in range(max_steps):
            if all(dones):
                break

            actions = []
            for i, (wm, obs) in enumerate(zip(world_models, obs_batch)):
                if dones[i]:
                    actions.append(envs.action_space.sample()[:1][0])  # dummy action
                else:
                    action, _, _ = wm.step(obs)
                    actions.append(action)

            obs_batch, rewards, terminated, truncated, _ = envs.step(np.array(actions))

            for i in range(n_envs):
                if not dones[i]:
                    totals[i] += float(rewards[i])
                    if terminated[i] or truncated[i]:
                        dones[i] = True

        envs.close()
        return totals

    return vec_rollout_fn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_latest(checkpoint_dir: str, *patterns: str) -> str | None:
    """
    Look for checkpoints matching patterns (tried in order).
    For wildcard patterns, return the lexicographically last match (highest epoch).
    """
    base = Path(checkpoint_dir)
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return str(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PHASE_MAP = {
    "collect":    phase_collect,
    "train_vae":  phase_train_vae,
    "encode":     phase_encode,
    "train_mdn":  phase_train_mdn,
    "train_ctrl": phase_train_ctrl,
}

ALL_PHASES = ["collect", "train_vae", "encode", "train_mdn", "train_ctrl"]


def main(argv: list[str] | None = None) -> None:
    from .distributed import cleanup, init_process_group, is_main

    parser = build_parser()
    args = parser.parse_args(argv)

    dist_cfg = _make_dist_cfg(args)
    init_process_group(dist_cfg)   # no-op when strategy="none"

    try:
        phases = ALL_PHASES if args.all else args.phases

        if is_main():
            print(f"VMC Training — phases: {phases}  dist: {dist_cfg.strategy}")

        for phase in phases:
            PHASE_MAP[phase](args)

        if is_main():
            print("\nAll requested phases complete.")
    finally:
        cleanup()   # no-op when not distributed


if __name__ == "__main__":
    main()
