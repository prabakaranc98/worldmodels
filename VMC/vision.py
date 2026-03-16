"""
VMC Vision Module
=================
All VAE-family architectures for learning compressed visual representations.

Self-contained — no imports from other VMC files.

Extensibility: subclass BaseVAE, override encode() and decode().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VAEConfig:
    img_channels: int = 3
    img_size: int = 64          # expects square frames (H == W)
    z_dim: int = 32
    beta: float = 4.0           # KL weight; beta=1 → vanilla VAE
    hidden_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    learning_rate: float = 1e-4


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class VAEOutput:
    recon: torch.Tensor       # (B, C, H, W)  reconstructed image
    mu: torch.Tensor          # (B, z_dim)    posterior mean
    log_var: torch.Tensor     # (B, z_dim)    posterior log-variance
    z: torch.Tensor           # (B, z_dim)    sampled latent


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVAE(nn.Module):
    """
    Abstract base class for all VAE variants.

    Subclasses must implement:
        encode(x)  → (mu, log_var)
        decode(z)  → recon

    The forward() and reparameterize() methods are shared.
    The loss() staticmethod can be overridden per subclass if needed.
    """

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var), both (B, z_dim)."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed image (B, C, H, W)."""
        raise NotImplementedError

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick.
        Returns mu deterministically in eval mode (no noise).
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return VAEOutput(recon=recon, mu=mu, log_var=log_var, z=z)

    @staticmethod
    def loss(
        output: VAEOutput,
        target: torch.Tensor,
        beta: float,
    ) -> dict[str, torch.Tensor]:
        """
        Beta-VAE loss: reconstruction (MSE) + beta * KL divergence.

        Returns dict with keys: 'recon', 'kl', 'total'.
        """
        recon_loss = F.mse_loss(output.recon, target, reduction="mean")

        # KL( q(z|x) || p(z) ) — closed form for diagonal Gaussian
        kl_loss = -0.5 * torch.mean(
            1 + output.log_var - output.mu.pow(2) - output.log_var.exp()
        )

        total = recon_loss + beta * kl_loss
        return {"recon": recon_loss, "kl": kl_loss, "total": total}


# ---------------------------------------------------------------------------
# BetaVAE — convolutional implementation
# ---------------------------------------------------------------------------

class BetaVAE(BaseVAE):
    """
    Convolutional Beta-VAE.

    Encoder: 4× Conv2d(stride=2) + BatchNorm2d + LeakyReLU(0.2)
             64 → 32 → 16 → 8 → 4 (spatial)
             → flatten → Linear → (mu, log_var)

    Decoder: Linear(z_dim → flat_dim) → reshape (C, 4, 4)
             → 4× ConvTranspose2d(stride=2) + LeakyReLU (except last)
             → Sigmoid output  (keeps values in [0, 1])

    Default config maps (B, 3, 64, 64) ↔ (B, z_dim).
    """

    def __init__(self, cfg: VAEConfig = VAEConfig()):
        super().__init__()
        self.cfg = cfg

        self._flat_dim = cfg.hidden_channels[-1] * (cfg.img_size // 2 ** len(cfg.hidden_channels)) ** 2

        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self._flat_dim, cfg.z_dim)
        self.fc_log_var = nn.Linear(self._flat_dim, cfg.z_dim)

        self.fc_decode = nn.Linear(cfg.z_dim, self._flat_dim)
        self.decoder = self._build_decoder()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_encoder(self) -> nn.Sequential:
        ch = self.cfg.hidden_channels
        layers: list[nn.Module] = []
        in_ch = self.cfg.img_channels
        for out_ch in ch:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        ch = list(reversed(self.cfg.hidden_channels))
        layers: list[nn.Module] = []
        in_ch = ch[0]
        for i, out_ch in enumerate(ch[1:]):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        # final layer → img_channels, no BN, Sigmoid activation
        layers += [
            nn.ConvTranspose2d(in_ch, self.cfg.img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)             # (B, 256, 4, 4)
        h = h.view(h.size(0), -1)       # (B, flat_dim)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ch_last = self.cfg.hidden_channels[-1]
        spatial = self.cfg.img_size // 2 ** len(self.cfg.hidden_channels)
        h = self.fc_decode(z)                           # (B, flat_dim)
        h = h.view(h.size(0), ch_last, spatial, spatial)  # (B, 256, 4, 4)
        return self.decoder(h)                          # (B, 3, 64, 64)
