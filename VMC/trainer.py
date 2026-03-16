"""
VMC Trainer Module
==================
Training recipes for VAE and MDN-RNN components.

    VAETrainer    — trains BetaVAE on FrameDataset
    MDNRNNTrainer — trains MDNRNN on SequenceDataset (needs pre-encoded latents)

Both trainers support:
    - MPS / CUDA / CPU auto-selection
    - torch.compile (with graceful fallback)
    - wandb logging (optional — degrades gracefully if not configured)
    - Checkpointing with full config stored inside checkpoint
    - from_checkpoint() classmethod for resuming

Imports: vision, memory, data  (no reverse deps, no circular imports)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .data import DataConfig, FrameDataset, SequenceDataset
from .memory import MDNRNN, MDNRNNConfig
from .vision import BetaVAE, VAEConfig

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-select MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _try_compile(model: nn.Module) -> nn.Module:
    """
    Wrap model with torch.compile.
    Falls back silently on MPS (uses aot_eager backend) or if compile fails.
    """
    try:
        device_type = next(model.parameters()).device.type if len(list(model.parameters())) > 0 else "cpu"
        backend = "aot_eager" if device_type == "mps" else "inductor"
        return torch.compile(model, backend=backend)
    except Exception:
        return model


# ---------------------------------------------------------------------------
# VAETrainer
# ---------------------------------------------------------------------------

@dataclass
class VAETrainerConfig:
    vae_cfg: VAEConfig = field(default_factory=VAEConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    val_split: float = 0.1
    checkpoint_dir: str = "./VMC_checkpoints"
    checkpoint_every: int = 5         # save checkpoint every N epochs
    use_compile: bool = True
    wandb_project: str = "VMC-VAE"
    wandb_run_name: str | None = None
    log_recon_every: int = 10         # log reconstruction images every N epochs


class VAETrainer:
    """
    Trains BetaVAE on raw frames collected by collect_rollouts().

    Usage:
        cfg = VAETrainerConfig()
        trainer = VAETrainer(cfg)
        vae = trainer.train()
    """

    def __init__(self, cfg: VAETrainerConfig):
        self.cfg = cfg
        self.device = get_device()
        print(f"[VAETrainer] device: {self.device}")

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.model = BetaVAE(cfg.vae_cfg).to(self.device)
        if cfg.use_compile:
            self.model = _try_compile(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        dataset = FrameDataset(cfg.data_cfg.data_dir)
        n_val = max(1, int(len(dataset) * cfg.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        self.wandb_run = None
        if _WANDB_AVAILABLE and os.environ.get("WANDB_MODE") != "disabled":
            try:
                self.wandb_run = _wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config=dataclasses.asdict(cfg),
                )
            except Exception:
                pass

        self._start_epoch = 0

    def train(self) -> BetaVAE:
        """Full training loop. Returns trained model (on CPU)."""
        epoch_bar = tqdm(
            range(self._start_epoch, self.cfg.epochs),
            desc="VAE", unit="epoch", dynamic_ncols=True,
        )
        for epoch in epoch_bar:
            self.model.train()
            train_metrics = self._run_epoch(self.train_loader, train=True)

            self.model.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, train=False)

            metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            metrics["epoch"] = epoch

            epoch_bar.set_postfix(
                train=f"{train_metrics['total']:.4f}",
                val=f"{val_metrics['total']:.4f}",
            )

            if self.wandb_run is not None:
                self.wandb_run.log(metrics)
                if (epoch + 1) % self.cfg.log_recon_every == 0:
                    self._log_reconstructions(epoch)

            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(epoch, val_metrics)

        self._save_checkpoint(self.cfg.epochs - 1, val_metrics, final=True)
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return self.model.cpu()

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> dict[str, float]:
        totals: dict[str, float] = {"recon": 0.0, "kl": 0.0, "total": 0.0}
        n = 0
        desc = "train" if train else "val"
        for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            x = batch.to(self.device)

            if train:
                self.optimizer.zero_grad()

            out = self.model(x)
            losses = BetaVAE.loss(out, x, self.cfg.vae_cfg.beta)

            if train:
                losses["total"].backward()
                self.optimizer.step()

            for k, v in losses.items():
                totals[k] += v.item() * x.size(0)
            n += x.size(0)

        return {k: v / n for k, v in totals.items()}

    def _log_reconstructions(self, epoch: int) -> None:
        """Log a side-by-side grid of originals and reconstructions to wandb."""
        if self.wandb_run is None:
            return
        try:
            import torchvision.utils as vutils
            batch = next(iter(self.val_loader))
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            x = batch[:8].to(self.device)
            with torch.no_grad():
                recon = self.model(x).recon
            grid = vutils.make_grid(
                torch.cat([x.cpu(), recon.cpu()], dim=0), nrow=8, normalize=False
            )
            self.wandb_run.log({"reconstructions": _wandb.Image(grid), "epoch": epoch})
        except Exception:
            pass

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], final: bool = False
    ) -> None:
        tag = "final" if final else f"epoch_{epoch+1:04d}"
        path = Path(self.cfg.checkpoint_dir) / f"vae_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": dataclasses.asdict(self.cfg),
                "metrics": metrics,
            },
            path,
        )
        print(f"  [VAE] checkpoint → {path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, cfg: VAETrainerConfig) -> "VAETrainer":
        """Resume training from a saved checkpoint."""
        trainer = cls(cfg)
        ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer._start_epoch = ckpt["epoch"] + 1
        print(f"[VAETrainer] resumed from epoch {ckpt['epoch']+1}")
        return trainer


# ---------------------------------------------------------------------------
# MDNRNNTrainer
# ---------------------------------------------------------------------------

@dataclass
class MDNRNNTrainerConfig:
    mdn_cfg: MDNRNNConfig = field(default_factory=MDNRNNConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    seq_len: int = 32
    seq_stride: int | None = None     # None → seq_len // 2
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    val_split: float = 0.1
    checkpoint_dir: str = "./VMC_checkpoints"
    checkpoint_every: int = 5
    use_compile: bool = True
    wandb_project: str = "VMC-MDN"
    wandb_run_name: str | None = None
    grad_clip: float = 1.0            # gradient clipping (important for RNNs)


class MDNRNNTrainer:
    """
    Trains MDNRNN on pre-encoded latent sequences.

    Requires encode_dataset() to have been run first.

    Usage:
        cfg = MDNRNNTrainerConfig()
        trainer = MDNRNNTrainer(cfg)
        mdn = trainer.train()
    """

    def __init__(self, cfg: MDNRNNTrainerConfig):
        self.cfg = cfg
        self.device = get_device()
        print(f"[MDNRNNTrainer] device: {self.device}")

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.model = MDNRNN(cfg.mdn_cfg).to(self.device)
        if cfg.use_compile:
            self.model = _try_compile(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        dataset = SequenceDataset(
            cfg.data_cfg.data_dir,
            seq_len=cfg.seq_len,
            stride=cfg.seq_stride,
        )
        n_val = max(1, int(len(dataset) * cfg.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        self.wandb_run = None
        if _WANDB_AVAILABLE and os.environ.get("WANDB_MODE") != "disabled":
            try:
                self.wandb_run = _wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config=dataclasses.asdict(cfg),
                )
            except Exception:
                pass

        self._start_epoch = 0

    def train(self) -> MDNRNN:
        """Full training loop. Returns trained model (on CPU)."""
        epoch_bar = tqdm(
            range(self._start_epoch, self.cfg.epochs),
            desc="MDN", unit="epoch", dynamic_ncols=True,
        )
        for epoch in epoch_bar:
            self.model.train()
            train_metrics = self._run_epoch(self.train_loader, train=True)

            self.model.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, train=False)

            metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            metrics["epoch"] = epoch

            epoch_bar.set_postfix(
                train_nll=f"{train_metrics['nll']:.4f}",
                val_nll=f"{val_metrics['nll']:.4f}",
            )

            if self.wandb_run is not None:
                self.wandb_run.log(metrics)

            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(epoch, val_metrics)

        self._save_checkpoint(self.cfg.epochs - 1, val_metrics, final=True)
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return self.model.cpu()

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> dict[str, float]:
        totals: dict[str, float] = {"nll": 0.0, "done": 0.0, "total": 0.0}
        n = 0
        desc = "train" if train else "val"
        for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            z = batch["z"].to(self.device)
            a = batch["a"].to(self.device)
            z_next = batch["z_next"].to(self.device)
            done = batch["done"].to(self.device)

            if train:
                self.optimizer.zero_grad()

            # hidden state is reset per batch (no TBPTT)
            out = self.model(z, a)
            losses = MDNRNN.mdn_loss(out, z_next, done)

            if train:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            bs = z.size(0)
            for k, v in losses.items():
                totals[k] += v.item() * bs
            n += bs

        return {k: v / n for k, v in totals.items()}

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], final: bool = False
    ) -> None:
        tag = "final" if final else f"epoch_{epoch+1:04d}"
        path = Path(self.cfg.checkpoint_dir) / f"mdn_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": dataclasses.asdict(self.cfg),
                "metrics": metrics,
            },
            path,
        )
        print(f"  [MDN] checkpoint → {path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, cfg: MDNRNNTrainerConfig) -> "MDNRNNTrainer":
        """Resume training from a saved checkpoint."""
        trainer = cls(cfg)
        ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer._start_epoch = ckpt["epoch"] + 1
        print(f"[MDNRNNTrainer] resumed from epoch {ckpt['epoch']+1}")
        return trainer
