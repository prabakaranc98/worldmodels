"""
VMC Controller Module
=====================
Linear controller optimised by CMA-ES (Covariance Matrix Adaptation
Evolution Strategy), as used in Ha & Schmidhuber (2018).

Components:
    LinearController  — W * [z; h] + b, parameterised by a flat numpy vector
    CMAESTrainer      — wraps `cma.CMAEvolutionStrategy` for population-based
                        controller optimisation with parallel rollout evaluation

CMA-ES notes:
    - The `cma` library operates entirely in NumPy (float64).
    - MPS/CUDA do not support float64, so the CMA-ES loop stays on CPU/NumPy.
    - Only the controller *forward pass* inside rollout_fn runs on the target
      device (float32 tensors), after params are converted with .astype(float32).

Self-contained with respect to other VMC files.
The rollout_fn accepted by CMAESTrainer is a user-supplied module-level function:
    rollout_fn(params_flat: np.ndarray, seed: int) → float (total reward)
This design keeps controller.py free of direct gymnasium / environment deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    z_dim: int = 32
    h_dim: int = 256              # LSTM hidden size (must match MDNRNNConfig.hidden_size)
    action_dim: int = 3
    action_low: list[float] = field(default_factory=lambda: [-1.0, 0.0, 0.0])
    action_high: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    sigma0: float = 0.1           # initial CMA-ES step size
    popsize: int = 64             # CMA-ES population size
    max_iter: int = 1000          # max CMA-ES generations
    n_rollouts_per_candidate: int = 16   # rollouts averaged per candidate
    n_workers: int = 4            # parallel worker processes
    seed: int = 42
    checkpoint_path: str = "./VMC_checkpoints/ctrl_best.pt"
    wandb_project: str = "VMC-Controller"
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# LinearController
# ---------------------------------------------------------------------------

class LinearController(nn.Module):
    """
    W * [z; h] + b  →  clamp to [action_low, action_high].

    At default config: (32 + 256) × 3 + 3 = 867 parameters.

    Intentionally tiny — the heavy lifting is done by V and M;
    the controller only needs to learn a simple policy on top of
    the learned latent representation.
    """

    def __init__(self, cfg: ControllerConfig = ControllerConfig()):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.z_dim + cfg.h_dim
        self.linear = nn.Linear(input_dim, cfg.action_dim)

        self.register_buffer("low", torch.tensor(cfg.action_low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(cfg.action_high, dtype=torch.float32))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_dim) or (z_dim,)
        h: (B, h_dim) or (h_dim,)
        Returns: (B, action_dim) clamped to [low, high]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        x = torch.cat([z, h], dim=-1)          # (B, z_dim + h_dim)
        a = self.linear(x)                      # (B, action_dim)
        return torch.clamp(a, self.low, self.high)

    def get_params(self) -> np.ndarray:
        """Flatten W and b into a 1-D float64 numpy array for CMA-ES."""
        params = torch.cat([
            self.linear.weight.detach().cpu().flatten(),
            self.linear.bias.detach().cpu().flatten(),
        ])
        return params.numpy().astype(np.float64)

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat numpy parameter vector back into the linear layer."""
        w_size = self.cfg.action_dim * (self.cfg.z_dim + self.cfg.h_dim)
        w = torch.tensor(params[:w_size], dtype=torch.float32)
        b = torch.tensor(params[w_size:], dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(w.view(self.cfg.action_dim, -1))
            self.linear.bias.copy_(b)

    @property
    def n_params(self) -> int:
        """Total number of controller parameters."""
        return (self.cfg.z_dim + self.cfg.h_dim) * self.cfg.action_dim + self.cfg.action_dim


# ---------------------------------------------------------------------------
# CMAESTrainer
# ---------------------------------------------------------------------------

class CMAESTrainer:
    """
    Optimises LinearController parameters using CMA-ES.

    The fitness function wraps a user-supplied rollout_fn:
        rollout_fn(params_flat: np.ndarray, seed: int) → float

    rollout_fn must be a module-level function (picklable for multiprocess).
    It should load the V+M+C world model internally, run a gym episode,
    and return total reward.

    Each generation evaluates cfg.popsize candidates, each averaged over
    cfg.n_rollouts_per_candidate random seeds, using cfg.n_workers workers.

    CMA-ES minimises negative reward (convention: minimisation problem).
    """

    def __init__(
        self,
        cfg: ControllerConfig,
        controller: LinearController,
        rollout_fn: Callable[[np.ndarray, int], float],
    ):
        self.cfg = cfg
        self.controller = controller
        self.rollout_fn = rollout_fn

        self.wandb_run = None
        if _WANDB_AVAILABLE:
            try:
                import os
                if os.environ.get("WANDB_MODE") != "disabled":
                    self.wandb_run = _wandb.init(
                        project=cfg.wandb_project,
                        name=cfg.wandb_run_name,
                        config=cfg.__dict__,
                    )
            except Exception:
                pass

    def train(self) -> tuple[np.ndarray, float]:
        """
        Run CMA-ES optimisation.

        Returns (best_params, best_reward).
        Saves best controller to cfg.checkpoint_path.
        """
        import cma
        import multiprocess as mp
        from pathlib import Path

        x0 = self.controller.get_params()

        es = cma.CMAEvolutionStrategy(
            x0,
            self.cfg.sigma0,
            {
                "popsize": self.cfg.popsize,
                "seed": self.cfg.seed,
                "maxiter": self.cfg.max_iter,
                "verbose": -9,          # suppress cma's own prints
            },
        )

        best_params = x0.copy()
        best_reward = -np.inf

        with mp.Pool(processes=self.cfg.n_workers) as pool:
            generation = 0
            while not es.stop():
                solutions = es.ask()            # list of np.ndarray (float64)
                fitnesses = self._evaluate_population(solutions, pool)

                es.tell(solutions, fitnesses)   # CMA-ES minimises → neg reward

                rewards = [-f for f in fitnesses]
                gen_best = max(rewards)
                gen_mean = float(np.mean(rewards))

                if gen_best > best_reward:
                    best_reward = gen_best
                    best_idx = int(np.argmax(rewards))
                    best_params = solutions[best_idx].copy()
                    self._save_checkpoint(best_params, best_reward)

                print(
                    f"[CMA-ES] gen {generation+1:>4}  "
                    f"best={best_reward:.2f}  mean={gen_mean:.2f}  "
                    f"sigma={es.sigma:.4f}"
                )

                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "generation": generation,
                        "best_reward": best_reward,
                        "mean_reward": gen_mean,
                        "sigma": es.sigma,
                    })

                generation += 1

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return best_params, best_reward

    def _evaluate_population(
        self,
        population: list[np.ndarray],
        pool: object,  # multiprocess.Pool
    ) -> list[float]:
        """
        Evaluate all candidates in a generation in parallel.

        Each candidate is evaluated cfg.n_rollouts_per_candidate times
        (different random seeds) and the mean negative reward is returned.
        CMA-ES minimises, so fitness = -mean_reward.
        """
        # Build (params, seed) argument list for each (candidate, rollout) pair
        args: list[tuple[np.ndarray, int]] = []
        for candidate in population:
            base_seed = int(self.cfg.seed + hash(candidate.tobytes()) & 0xFFFF)
            for r in range(self.cfg.n_rollouts_per_candidate):
                args.append((candidate, base_seed + r))

        # Parallel evaluation
        rewards = pool.starmap(self.rollout_fn, args)

        # Average over rollouts per candidate → negate for minimisation
        n = self.cfg.n_rollouts_per_candidate
        fitnesses = []
        for i in range(len(population)):
            mean_reward = float(np.mean(rewards[i * n : (i + 1) * n]))
            fitnesses.append(-mean_reward)

        return fitnesses

    def _save_checkpoint(self, params: np.ndarray, reward: float) -> None:
        from pathlib import Path
        path = Path(self.cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": params,
                "best_reward": reward,
                "config": self.cfg.__dict__,
            },
            path,
        )

    @staticmethod
    def load_best_params(checkpoint_path: str) -> np.ndarray:
        """Load best controller params from a saved checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt["params"]
