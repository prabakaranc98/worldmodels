"""
VMC Data Module
===============
Rollout collection, frame preprocessing, and dataset classes.

Pipeline:
    1. collect_rollouts(cfg)         — parallel episode collection → episodes.h5
    2. VAETrainer uses FrameDataset  — individual (3,64,64) frames
    3. encode_dataset(vae, ...)      — encode all episodes → encoded.h5
    4. MDNRNNTrainer uses SequenceDataset — (z, a, z_next, done) sequences

Storage layout
--------------
VMC_data/
    episodes.h5          — raw rollouts
        /ep_00000/
            frames   (T, 3, 64, 64)  float32  [chunked + lzf compressed]
            actions  (T, action_dim) float32
            rewards  (T,)            float32
            dones    (T,)            bool
    encoded.h5           — VAE latents (written once after VAE training)
        /ep_00000/
            z        (T, z_dim)      float32
            actions  (T, action_dim) float32
            dones    (T,)            bool

macOS note: Python 3.13 defaults to 'spawn' for multiprocessing.
All worker functions must be module-level (no closures / lambdas).
HDF5 note: h5py file handles are not fork-safe. Workers return tensors to the
main process, which owns the single writer handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# DataLoader worker init — h5py fork-safety fix
# ---------------------------------------------------------------------------

def h5_worker_init_fn(worker_id: int) -> None:
    """
    Reopen h5py file handle inside each DataLoader worker after fork.

    h5py file handles are not fork-safe — forked workers inherit the parent's
    handle which becomes invalid.  Pass this as worker_init_fn to DataLoader
    when using num_workers > 0 with FrameDataset.

    Handles torch.utils.data.Subset transparently (val split wraps the dataset).
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return  # called in main process — no-op
    ds = worker_info.dataset
    # unwrap Subset (created by random_split)
    if hasattr(ds, "dataset"):
        ds = ds.dataset
    if hasattr(ds, "_h5_path"):
        ds._h5 = h5py.File(ds._h5_path, "r")

if TYPE_CHECKING:
    from .vision import BaseVAE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    env_id: str = "CarRacing-v3"
    n_episodes: int = 10_000        # Ha & Schmidhuber used 10,000 rollouts
    max_steps_per_episode: int = 1000
    img_size: int = 64
    data_dir: str = "./VMC_data"
    n_workers: int = -1             # -1 = use all logical CPU cores
    seed: int = 0
    render_mode: str | None = None  # None = headless (faster)


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, img_size: int = 64) -> torch.Tensor:
    """
    np.ndarray (H, W, 3) uint8  →  torch.Tensor (3, img_size, img_size) float32 ∈ [0, 1]
    """
    from PIL import Image

    img = Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


# ---------------------------------------------------------------------------
# Single-worker episode collection  (module-level — required for spawn/pickle)
# ---------------------------------------------------------------------------

def collect_episode(
    env_id: str,
    max_steps: int,
    img_size: int,
    seed: int,
    render_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run one episode with random actions.

    Returns:
        frames:  (T, 3, img_size, img_size) float32
        actions: (T, action_dim) float32
        rewards: (T,) float32
        dones:   (T,) bool
    """
    import gymnasium as gym

    env = gym.make(env_id, render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    frames, actions, rewards, dones = [], [], [], []

    for _ in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        frames.append(preprocess_frame(obs, img_size))
        actions.append(torch.tensor(action, dtype=torch.float32))
        rewards.append(float(reward))
        dones.append(done)

        obs = next_obs
        if done:
            break

    env.close()

    return {
        "frames":  torch.stack(frames),
        "actions": torch.stack(actions),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones":   torch.tensor(dones,   dtype=torch.bool),
    }


def _collect_episode_star(args: tuple) -> dict[str, torch.Tensor]:
    """Unpack tuple args for imap_unordered (which passes a single argument)."""
    return collect_episode(*args)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _write_episode_to_h5(
    f: h5py.File,
    ep_key: str,
    ep: dict[str, torch.Tensor],
) -> None:
    """
    Write one episode dict into an open HDF5 file under group `ep_key`.

    Frames are chunked per-frame and lzf-compressed (fast + decent ratio).
    Scalars (actions, rewards, dones) are stored uncompressed — they're tiny.
    """
    grp = f.create_group(ep_key)
    frames_np = ep["frames"].numpy()   # (T, 3, H, W)
    T = frames_np.shape[0]
    grp.create_dataset(
        "frames", data=frames_np,
        chunks=(1, *frames_np.shape[1:]),   # one frame per chunk → random access
        compression="lzf",                  # fast, no external deps
    )
    grp.create_dataset("actions", data=ep["actions"].numpy())
    grp.create_dataset("rewards", data=ep["rewards"].numpy())
    grp.create_dataset("dones",   data=ep["dones"].numpy())
    grp.attrs["T"] = T


# ---------------------------------------------------------------------------
# Parallel rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(cfg: DataConfig) -> None:
    """
    Collect cfg.n_episodes episodes in parallel and store in a single HDF5 file.

    Output: {cfg.data_dir}/episodes.h5

    Speed-up notes:
    - n_workers=-1 auto-selects os.cpu_count() (all logical cores).
    - imap_unordered: main process writes each episode as workers finish —
      lower peak RAM, faster perceived progress.
    - HDF5: single file, lzf-compressed frames, ~3-5× smaller than raw .pt files.
      FrameDataset reads individual frames via chunked slicing (no full load).
    """
    import os
    import multiprocess as mp

    n_workers = cfg.n_workers if cfg.n_workers > 0 else os.cpu_count()
    print(f"  workers: {n_workers}  episodes: {cfg.n_episodes}")

    out_path = Path(cfg.data_dir) / "episodes.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        (cfg.env_id, cfg.max_steps_per_episode, cfg.img_size, cfg.seed + i, cfg.render_mode)
        for i in range(cfg.n_episodes)
    ]

    saved = 0
    with h5py.File(out_path, "w") as hf:
        with mp.Pool(processes=n_workers) as pool:
            with tqdm(total=cfg.n_episodes, desc="collect", unit="ep", dynamic_ncols=True) as pbar:
                for result in pool.imap_unordered(_collect_episode_star, args):
                    _write_episode_to_h5(hf, f"ep_{saved:05d}", result)
                    saved += 1
                    pbar.update(1)

    print(f"Done. {saved} episodes → {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Dataset: individual frames for VAE training
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """
    Lazy frame dataset for VAE training backed by episodes.h5.

    Each __getitem__ returns: torch.Tensor (3, img_size, img_size) float32

    The HDF5 file is opened once and kept open for the lifetime of the dataset.
    Frames are read via chunked slicing — only the requested frame is loaded
    from disk, not the full episode.

    Note: keep DataLoader num_workers=0 (default) — h5py handles are not
    fork-safe. For multi-worker loading, pass worker_init_fn that reopens the file.
    """

    def __init__(self, data_dir: str):
        h5_path = Path(data_dir) / "episodes.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"episodes.h5 not found at {h5_path}. Run collect_rollouts first.")

        self._h5_path = h5_path
        self._h5: h5py.File = h5py.File(h5_path, "r")

        # index: list of (ep_key, frame_idx_within_episode)
        self._index: list[tuple[str, int]] = []
        ep_keys = sorted(self._h5.keys())
        for ep_key in tqdm(ep_keys, desc="indexing frames", unit="ep", dynamic_ncols=True):
            T = self._h5[ep_key].attrs["T"]
            self._index.extend((ep_key, t) for t in range(T))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep_key, ti = self._index[idx]
        frame = self._h5[ep_key]["frames"][ti]   # (3, H, W) float32
        return torch.from_numpy(frame)

    def __del__(self) -> None:
        if hasattr(self, "_h5") and self._h5.id.valid:
            self._h5.close()


# ---------------------------------------------------------------------------
# Encode raw episodes → latent sequences (runs once after VAE training)
# ---------------------------------------------------------------------------

def encode_dataset(
    vae: "BaseVAE",
    data_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> None:
    """
    Encode all saved episodes with the trained VAE encoder.

    Reads:  {data_dir}/episodes.h5
    Writes: {data_dir}/encoded.h5
        /ep_XXXXX/z (T, z_dim), actions (T, action_dim), dones (T,)

    Uses deterministic encoding (vae.eval() → reparameterize returns mu).
    Call once after VAE training, before MDN-RNN training.
    """
    from torch.utils.data import DataLoader, TensorDataset

    ep_path  = Path(data_dir) / "episodes.h5"
    enc_path = Path(data_dir) / "encoded.h5"

    if not ep_path.exists():
        raise FileNotFoundError(f"episodes.h5 not found at {ep_path}")

    vae = vae.to(device).eval()

    with h5py.File(ep_path, "r") as src, h5py.File(enc_path, "w") as dst:
        ep_keys = sorted(src.keys())
        for ep_key in tqdm(ep_keys, desc="encoding", unit="ep", dynamic_ncols=True):
            frames  = torch.from_numpy(src[ep_key]["frames"][:])    # (T, 3, H, W)
            actions = torch.from_numpy(src[ep_key]["actions"][:])
            dones   = torch.from_numpy(src[ep_key]["dones"][:])

            zs = []
            loader = DataLoader(TensorDataset(frames), batch_size=batch_size)
            with torch.no_grad():
                for (batch,) in loader:
                    out = vae(batch.to(device))
                    zs.append(out.mu.cpu())   # deterministic: eval mode returns mu

            z = torch.cat(zs, dim=0).numpy()
            grp = dst.create_group(ep_key)
            grp.create_dataset("z",       data=z)
            grp.create_dataset("actions", data=actions.numpy())
            grp.create_dataset("dones",   data=dones.numpy())
            grp.attrs["T"] = z.shape[0]

    print(f"Done. Encoded episodes → {enc_path}  ({enc_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Dataset: latent sequences for MDN-RNN training
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Sequence dataset for MDN-RNN training backed by encoded.h5.

    Loads all encoded episodes into RAM at construction (encoded data is small:
    10k eps × 1000 steps × 32-dim float32 ≈ 1.2 GB — fits comfortably).

    Each __getitem__ returns a dict:
        'z':      (seq_len, z_dim)      current latent
        'a':      (seq_len, action_dim)
        'z_next': (seq_len, z_dim)      next latent (MDN-RNN target)
        'done':   (seq_len,) float32    termination flags
    """

    def __init__(self, data_dir: str, seq_len: int = 32, stride: int | None = None):
        enc_path = Path(data_dir) / "encoded.h5"
        if not enc_path.exists():
            raise FileNotFoundError(
                f"encoded.h5 not found at {enc_path}. Run encode_dataset() first."
            )

        self._seq_len = seq_len
        stride = stride if stride is not None else seq_len // 2

        # keep encoded episodes in memory — small (T × z_dim floats)
        self._episodes: list[dict[str, torch.Tensor]] = []
        self._index: list[tuple[int, int]] = []   # (ep_idx, start_t)

        with h5py.File(enc_path, "r") as f:
            ep_keys = sorted(f.keys())
            for ep_key in tqdm(ep_keys, desc="loading sequences", unit="ep", dynamic_ncols=True):
                T = f[ep_key].attrs["T"]
                if T < seq_len + 1:
                    continue
                ep = {
                    "z":       torch.from_numpy(f[ep_key]["z"][:]),
                    "actions": torch.from_numpy(f[ep_key]["actions"][:]),
                    "dones":   torch.from_numpy(f[ep_key]["dones"][:]),
                }
                ep_idx = len(self._episodes)
                self._episodes.append(ep)
                for start in range(0, T - seq_len, stride):
                    self._index.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self._index[idx]
        ep = self._episodes[ep_idx]
        s, e = start, start + self._seq_len
        return {
            "z":      ep["z"][s:e],
            "a":      ep["actions"][s:e],
            "z_next": ep["z"][s + 1 : e + 1],
            "done":   ep["dones"][s:e].float(),
        }
