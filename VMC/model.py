"""
VMC Model Module
================
WorldModel integrates Vision (V) + Memory (M) + Controller (C) into a single
callable object for inference, rollout evaluation, and dream rollouts.

This is NOT an nn.Module — it is an orchestrator that holds references to
the three component nn.Modules and coordinates their interaction.

Typical usage:
    wm = WorldModel.from_checkpoints(vae_path, mdn_path, ctrl_path)
    reward = wm.eval_episode(env)

Rollout loop (real env):
    1. obs → vae.encode → z_t          (deterministic: eval mode)
    2. [z_t, h_t] → controller → a_t
    3. env.step(a_t) → next_obs, reward, done
    4. [z_t, a_t] → mdn → h_{t+1}     (updates hidden state)
    5. repeat

Dream rollout (no real env):
    Same loop but step 3 samples z_{t+1} from MDN instead of env.

Imports: vision, memory, controller  (no trainer, no data — no circular deps)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .controller import ControllerConfig, LinearController
from .data import preprocess_frame
from .memory import MDNRNN, MDNRNNConfig
from .vision import BetaVAE, VAEConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    vae_cfg: VAEConfig = field(default_factory=VAEConfig)
    mdn_cfg: MDNRNNConfig = field(default_factory=MDNRNNConfig)
    ctrl_cfg: ControllerConfig = field(default_factory=ControllerConfig)
    device: str = "auto"   # "auto" | "mps" | "cuda" | "cpu"


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Assembled VMC world model for inference and evaluation.

    Attributes:
        vae   — BetaVAE (vision encoder / decoder)
        mdn   — MDNRNN  (memory / dynamics model)
        ctrl  — LinearController (action policy)
    """

    def __init__(self, cfg: WorldModelConfig = WorldModelConfig()):
        self.cfg = cfg
        self._device = self._resolve_device(cfg.device)

        self.vae = BetaVAE(cfg.vae_cfg).to(self._device).eval()
        self.mdn = MDNRNN(cfg.mdn_cfg).to(self._device).eval()
        self.ctrl = LinearController(cfg.ctrl_cfg).to(self._device).eval()

        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        vae_path: str,
        mdn_path: str,
        ctrl_path: str,
        cfg: WorldModelConfig = WorldModelConfig(),
    ) -> "WorldModel":
        """Load V, M, C from separate checkpoint files."""
        wm = cls(cfg)
        device = wm._device

        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        wm.vae.load_state_dict(vae_ckpt["model_state_dict"])

        mdn_ckpt = torch.load(mdn_path, map_location=device, weights_only=False)
        wm.mdn.load_state_dict(mdn_ckpt["model_state_dict"])

        ctrl_ckpt = torch.load(ctrl_path, map_location=device, weights_only=False)
        # controller checkpoint stores flat params (saved by CMAESTrainer)
        if "model_state_dict" in ctrl_ckpt:
            wm.ctrl.load_state_dict(ctrl_ckpt["model_state_dict"])
        elif "params" in ctrl_ckpt:
            wm.ctrl.set_params(ctrl_ckpt["params"])
        else:
            raise ValueError(f"Unrecognised controller checkpoint format in {ctrl_path}")

        return wm

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state (call at the start of each episode)."""
        self._hidden = self.mdn.init_hidden(batch_size, self._device)

    # ------------------------------------------------------------------
    # Single inference step (real env)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,    # (H, W, 3) uint8 raw frame
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Process one observation and return the controller's action.

        1. Preprocess obs → tensor
        2. VAE encode → z_t
        3. Controller → a_t  (using current hidden state h_t)
        4. MDN forward → update hidden state to h_{t+1}

        Returns: (action_np, z_t, h_t)
            action_np — (action_dim,) numpy array for env.step()
            z_t       — (1, z_dim)  latent vector
            h_t       — (1, h_dim)  LSTM hidden (first layer, squeezed)
        """
        if self._hidden is None:
            self.reset(batch_size=1)

        # preprocess
        frame = preprocess_frame(obs, self.cfg.vae_cfg.img_size).unsqueeze(0).to(self._device)

        # encode
        vae_out = self.vae(frame)
        z_t = vae_out.mu                # (1, z_dim) — deterministic in eval mode

        # extract first-layer hidden state for controller
        h_t = self._hidden[0][0]        # (1, h_dim)  layer-0 hidden

        # controller action
        a_t = self.ctrl(z_t, h_t)       # (1, action_dim)

        # advance memory with current (z, a) pair
        mdn_out = self.mdn(
            z_t.unsqueeze(1),           # (1, 1, z_dim)
            a_t.unsqueeze(1),           # (1, 1, action_dim)
            hidden=self._hidden,
        )
        self._hidden = (mdn_out.h, mdn_out.c)

        action_np = a_t.squeeze(0).cpu().numpy()
        return action_np, z_t, h_t

    # ------------------------------------------------------------------
    # Dream rollout (imagined, no real env)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dream_rollout(
        self,
        z0: torch.Tensor,               # (z_dim,) or (1, z_dim)  initial latent
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Pure imagined rollout inside the world model.

        The controller acts on the current latent + hidden state; the MDN
        samples the next latent instead of stepping the real environment.

        Returns:
            'z':         (T, z_dim)      latent trajectory
            'a':         (T, action_dim) action trajectory
            'done_prob': (T,)            predicted termination probability
        """
        self.reset(batch_size=1)

        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        z_t = z0.to(self._device)

        zs, acts, done_probs = [], [], []

        for _ in range(horizon):
            h_t = self._hidden[0][0]        # (1, h_dim)
            a_t = self.ctrl(z_t, h_t)       # (1, action_dim)

            mdn_out = self.mdn(
                z_t.unsqueeze(1),
                a_t.unsqueeze(1),
                hidden=self._hidden,
            )
            self._hidden = (mdn_out.h, mdn_out.c)

            z_next = self.mdn.sample(mdn_out, temperature=temperature)  # (1, z_dim)

            zs.append(z_t.squeeze(0).cpu())
            acts.append(a_t.squeeze(0).cpu())

            if mdn_out.done_logit is not None:
                done_p = torch.sigmoid(mdn_out.done_logit).squeeze().cpu()
                done_probs.append(done_p)

            z_t = z_next

        result: dict[str, torch.Tensor] = {
            "z": torch.stack(zs),
            "a": torch.stack(acts),
        }
        if done_probs:
            result["done_prob"] = torch.stack(done_probs)

        return result

    # ------------------------------------------------------------------
    # Full episode evaluation
    # ------------------------------------------------------------------

    def eval_episode(
        self,
        env: Any,           # gymnasium.Env
        max_steps: int = 1000,
        render: bool = False,
    ) -> float:
        """
        Run one full episode in the real environment.

        Returns total episode reward (scalar).
        """
        obs, _ = env.reset()
        self.reset(batch_size=1)

        total_reward = 0.0

        for _ in range(max_steps):
            if render:
                env.render()

            action, _, _ = self.step(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        return total_reward

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Save V, M, C model state dicts and world model config to directory.

        Creates:
            {directory}/vae.pt
            {directory}/mdn.pt
            {directory}/ctrl.pt
            {directory}/config.pt
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        torch.save({"model_state_dict": self.vae.state_dict()}, out / "vae.pt")
        torch.save({"model_state_dict": self.mdn.state_dict()}, out / "mdn.pt")
        torch.save({"model_state_dict": self.ctrl.state_dict()}, out / "ctrl.pt")
        torch.save({"config": dataclasses.asdict(self.cfg)}, out / "config.pt")
        print(f"[WorldModel] saved to {out}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device_str)

    @property
    def device(self) -> torch.device:
        return self._device
