"""
VMC Memory Module
=================
Factorized MDN-LSTM: models stochastic temporal dynamics in latent space.

Architecture:
    Input projection:  Linear(z_dim + action_dim → hidden_size)
    LSTM:              hidden_size, num_layers, batch_first=True
    MDN head:          Linear(hidden_size → K + K*z_dim + K*z_dim + 1)
        pi        (K)          — mixture weights
        mu        (K × z_dim)  — per-component means
        log_sigma (K × z_dim)  — per-component log-stds (diagonal covariance)
        done_logit (1)         — binary termination signal

"Factorized" = diagonal covariance per mixture component,
i.e. each latent dimension is independent within a component.

Self-contained — no imports from other VMC files.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MDNRNNConfig:
    z_dim: int = 32
    action_dim: int = 3          # CarRacing-v3: [steer, gas, brake]
    hidden_size: int = 256
    num_layers: int = 1
    num_mixtures: int = 5        # K mixture components
    predict_done: bool = True
    temperature: float = 1.0     # sampling temperature (inference only)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class MDNOutput:
    pi: torch.Tensor           # (B, T, K)          mixture weights (post softmax)
    mu: torch.Tensor           # (B, T, K, z_dim)   per-component means
    log_sigma: torch.Tensor    # (B, T, K, z_dim)   per-component log-stds
    done_logit: torch.Tensor | None  # (B, T, 1)    termination logit (optional)
    h: torch.Tensor            # (num_layers, B, hidden_size)  final hidden state
    c: torch.Tensor            # (num_layers, B, hidden_size)  final cell state


# ---------------------------------------------------------------------------
# MDNRNN
# ---------------------------------------------------------------------------

class MDNRNN(nn.Module):
    """
    Factorized MDN-LSTM.

    Takes a sequence of (z_t, a_t) pairs and predicts a mixture-of-Gaussians
    distribution over z_{t+1} at each step.

    Usage (training):
        model = MDNRNN(cfg)
        out = model(z, a)          # z: (B,T,z_dim)  a: (B,T,action_dim)
        losses = MDNRNN.mdn_loss(out, z_next, done)

    Usage (inference):
        model.eval()
        out = model(z_t.unsqueeze(1), a_t.unsqueeze(1), hidden=(h, c))
        z_next = model.sample(out)
        h, c = out.h, out.c
    """

    def __init__(self, cfg: MDNRNNConfig = MDNRNNConfig()):
        super().__init__()
        self.cfg = cfg
        K, z, h = cfg.num_mixtures, cfg.z_dim, cfg.hidden_size

        self.input_proj = nn.Linear(z + cfg.action_dim, h)

        self.lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

        # MDN head output size:
        #   K (pi) + K*z_dim (mu) + K*z_dim (log_sigma) + 1 (done)
        mdn_out_size = K + K * z + K * z + (1 if cfg.predict_done else 0)
        self.mdn_head = nn.Linear(h, mdn_out_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,                              # (B, T, z_dim)
        a: torch.Tensor,                              # (B, T, action_dim)
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> MDNOutput:
        B, T, _ = z.shape
        K = self.cfg.num_mixtures
        z_dim = self.cfg.z_dim

        x = torch.cat([z, a], dim=-1)       # (B, T, z_dim + action_dim)
        x = self.input_proj(x)              # (B, T, hidden_size)

        if hidden is None:
            hidden = self.init_hidden(B, z.device)

        lstm_out, (h_n, c_n) = self.lstm(x, hidden)  # lstm_out: (B, T, hidden_size)

        raw = self.mdn_head(lstm_out)       # (B, T, mdn_out_size)

        # --- split head output ---
        pi_raw = raw[..., :K]                                   # (B, T, K)
        mu_flat = raw[..., K : K + K * z_dim]                   # (B, T, K*z_dim)
        ls_flat = raw[..., K + K * z_dim : K + 2 * K * z_dim]  # (B, T, K*z_dim)

        pi = F.softmax(pi_raw / self.cfg.temperature, dim=-1)   # (B, T, K)
        mu = mu_flat.view(B, T, K, z_dim)                       # (B, T, K, z_dim)
        log_sigma = ls_flat.view(B, T, K, z_dim)
        log_sigma = torch.clamp(log_sigma, min=-4.0, max=4.0)   # stability clamp

        done_logit: torch.Tensor | None = None
        if self.cfg.predict_done:
            done_logit = raw[..., K + 2 * K * z_dim :]          # (B, T, 1)

        return MDNOutput(
            pi=pi,
            mu=mu,
            log_sigma=log_sigma,
            done_logit=done_logit,
            h=h_n,
            c=c_n,
        )

    # ------------------------------------------------------------------
    # Hidden state
    # ------------------------------------------------------------------

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns zero-initialised (h, c) tuple."""
        shape = (self.cfg.num_layers, batch_size, self.cfg.hidden_size)
        h = torch.zeros(shape, device=device)
        c = torch.zeros(shape, device=device)
        return h, c

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def mdn_loss(
        output: MDNOutput,
        z_next: torch.Tensor,                    # (B, T, z_dim)
        done: torch.Tensor | None = None,        # (B, T)  float {0, 1}
    ) -> dict[str, torch.Tensor]:
        """
        Negative log-likelihood of z_next under the predicted mixture.

        Uses log-sum-exp trick for numerical stability.
        Returns dict: {'nll', 'done', 'total'}.
        """
        # z_next: (B, T, z_dim) → (B, T, 1, z_dim) for broadcasting with (B, T, K, z_dim)
        z_expanded = z_next.unsqueeze(2)

        sigma = output.log_sigma.exp()
        dist = Normal(output.mu, sigma)
        log_p_z_given_k = dist.log_prob(z_expanded).sum(-1)  # (B, T, K)

        log_pi = torch.log(output.pi + 1e-8)                 # (B, T, K)
        log_mixture = torch.logsumexp(log_pi + log_p_z_given_k, dim=-1)  # (B, T)
        nll = -log_mixture.mean()

        total = nll
        done_loss = torch.tensor(0.0, device=nll.device)

        if output.done_logit is not None and done is not None:
            done_loss = F.binary_cross_entropy_with_logits(
                output.done_logit.squeeze(-1), done.float()
            )
            total = nll + done_loss

        return {"nll": nll, "done": done_loss, "total": total}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        output: MDNOutput,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample z_{t+1} from the predicted mixture distribution.

        output should have T=1 (single-step inference).
        Returns (B, z_dim).
        """
        # output.pi: (B, 1, K) → squeeze T dim
        pi = output.pi.squeeze(1)           # (B, K)
        mu = output.mu.squeeze(1)           # (B, K, z_dim)
        sigma = output.log_sigma.exp().squeeze(1) * temperature  # (B, K, z_dim)

        # sample component indices
        idx = torch.multinomial(pi, num_samples=1)  # (B, 1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, 1, mu.size(-1))  # (B, 1, z_dim)

        mu_k = mu.gather(1, idx_expanded).squeeze(1)      # (B, z_dim)
        sigma_k = sigma.gather(1, idx_expanded).squeeze(1)  # (B, z_dim)

        return mu_k + sigma_k * torch.randn_like(sigma_k)
