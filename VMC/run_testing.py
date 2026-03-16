"""
VMC Testing / Evaluation Runner
================================
Evaluate a trained VMC world model and inspect its components.

Modes:
    eval         — run N episodes in the real environment, report reward stats
    record       — record a real episode to MP4 with the policy overlaid
    dream        — generate imagined rollouts (latents only)
    dream_video  — dream rollout decoded through VAE → GIF / MP4 you can watch
    encode       — encode a single frame and print latent stats
    reconstruct  — encode + decode a frame and save reconstruction image

Usage examples:
    # Evaluate 10 episodes (auto-detects latest checkpoints)
    python -m VMC.run_testing eval --n_episodes 10

    # Record a single episode to MP4
    python -m VMC.run_testing record --out_video episode.mp4

    # Dream rollout decoded to a watchable GIF
    python -m VMC.run_testing dream_video --horizon 100 --out_gif dream.gif

    # Encode a frame and print latent stats
    python -m VMC.run_testing encode --ep_idx 0 --frame_idx 0

    # Reconstruct a specific frame and save the image
    python -m VMC.run_testing reconstruct --ep_idx 0 --frame_idx 42 --out reconstruction.png
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
        prog="python -m VMC.run_testing",
        description="VMC evaluation — Vision · Memory · Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # --- Shared checkpoint args (added to each subcommand) ---
    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--checkpoint_dir", default="./VMC_checkpoints")
        sp.add_argument("--data_dir", default="./VMC_data")
        sp.add_argument("--vae_ckpt", default=None)
        sp.add_argument("--mdn_ckpt", default=None)
        sp.add_argument("--ctrl_ckpt", default=None)
        # Architecture (must match training)
        sp.add_argument("--img_size", type=int, default=64)
        sp.add_argument("--z_dim", type=int, default=32)
        sp.add_argument("--action_dim", type=int, default=3)
        sp.add_argument("--hidden_size", type=int, default=256)
        sp.add_argument("--num_mixtures", type=int, default=5)
        sp.add_argument("--env_id", default="CarRacing-v3")

    # --- eval ---
    eval_p = sub.add_parser("eval", help="Run episodes in the real environment.")
    add_shared(eval_p)
    eval_p.add_argument("--n_episodes", type=int, default=10)
    eval_p.add_argument("--max_steps", type=int, default=1000)
    eval_p.add_argument("--render", action="store_true", help="Render episodes (requires display).")
    eval_p.add_argument("--seed", type=int, default=0)

    # --- dream ---
    dream_p = sub.add_parser("dream", help="Generate imagined rollouts inside the world model.")
    add_shared(dream_p)
    dream_p.add_argument("--horizon", type=int, default=50)
    dream_p.add_argument("--temperature", type=float, default=1.0)
    dream_p.add_argument("--n_dreams", type=int, default=3)
    dream_p.add_argument("--seed", type=int, default=0)
    dream_p.add_argument("--out_dir", default="./VMC_dreams",
                         help="Directory to save dream latent trajectories.")

    # --- encode ---
    encode_p = sub.add_parser("encode", help="Encode a frame from a saved episode and print stats.")
    add_shared(encode_p)
    encode_p.add_argument("--ep_idx", type=int, default=0, help="Episode index.")
    encode_p.add_argument("--frame_idx", type=int, default=0, help="Frame index within episode.")

    # --- record ---
    record_p = sub.add_parser(
        "record",
        help="Run one episode with the trained policy and save it as an MP4.",
    )
    add_shared(record_p)
    record_p.add_argument("--max_steps", type=int, default=1000)
    record_p.add_argument("--seed", type=int, default=0)
    record_p.add_argument("--out_video", default="./VMC_videos/episode.mp4",
                          help="Output MP4 path.")
    record_p.add_argument("--fps", type=int, default=30)

    # --- dream_video ---
    dv_p = sub.add_parser(
        "dream_video",
        help="Dream rollout decoded through VAE decoder → watchable GIF or MP4.",
    )
    add_shared(dv_p)
    dv_p.add_argument("--horizon", type=int, default=100)
    dv_p.add_argument("--temperature", type=float, default=1.0)
    dv_p.add_argument("--seed", type=int, default=0)
    dv_p.add_argument("--out_gif", default="./VMC_videos/dream.gif",
                      help="Output GIF path (use .mp4 extension for MP4 output).")
    dv_p.add_argument("--fps", type=int, default=15)
    dv_p.add_argument("--n_dreams", type=int, default=1,
                      help="Number of dream sequences to render side by side.")

    # --- reconstruct ---
    recon_p = sub.add_parser("reconstruct", help="Encode + decode a frame and save the image.")
    add_shared(recon_p)
    recon_p.add_argument("--ep_idx", type=int, default=0)
    recon_p.add_argument("--frame_idx", type=int, default=0)
    recon_p.add_argument("--out", default="reconstruction.png",
                         help="Output image path (PNG).")

    return p


# ---------------------------------------------------------------------------
# World model loader
# ---------------------------------------------------------------------------

def load_world_model(args: argparse.Namespace):
    """Build WorldModelConfig and load from checkpoints."""
    from .controller import ControllerConfig
    from .memory import MDNRNNConfig
    from .model import WorldModel, WorldModelConfig
    from .vision import VAEConfig

    vae_ckpt = args.vae_ckpt or _find_latest(args.checkpoint_dir, "vae_final.pt", "vae_*.pt")
    mdn_ckpt = args.mdn_ckpt or _find_latest(args.checkpoint_dir, "mdn_final.pt", "mdn_*.pt")
    ctrl_ckpt = args.ctrl_ckpt or _find_latest(args.checkpoint_dir, "ctrl_best.pt")

    missing = [n for n, p in [("VAE", vae_ckpt), ("MDN", mdn_ckpt), ("Ctrl", ctrl_ckpt)] if p is None]
    if missing:
        sys.exit(f"Missing checkpoints: {missing}. Run run_training.py first.")

    print(f"  VAE  : {vae_ckpt}")
    print(f"  MDN  : {mdn_ckpt}")
    print(f"  Ctrl : {ctrl_ckpt}")

    cfg = WorldModelConfig(
        vae_cfg=VAEConfig(img_size=args.img_size, z_dim=args.z_dim),
        mdn_cfg=MDNRNNConfig(
            z_dim=args.z_dim,
            action_dim=args.action_dim,
            hidden_size=args.hidden_size,
            num_mixtures=args.num_mixtures,
        ),
        ctrl_cfg=ControllerConfig(
            z_dim=args.z_dim,
            h_dim=args.hidden_size,
            action_dim=args.action_dim,
        ),
    )

    wm = WorldModel.from_checkpoints(vae_ckpt, mdn_ckpt, ctrl_ckpt, cfg=cfg)
    return wm


# ---------------------------------------------------------------------------
# Mode: eval
# ---------------------------------------------------------------------------

def mode_eval(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import numpy as np

    print("\n" + "=" * 60)
    print("MODE: eval")
    print("=" * 60)

    wm = load_world_model(args)
    render_mode = "human" if args.render else None

    rewards = []
    for ep in range(args.n_episodes):
        env = gym.make(args.env_id, render_mode=render_mode)
        env.reset(seed=args.seed + ep)
        reward = wm.eval_episode(env, max_steps=args.max_steps, render=args.render)
        env.close()
        rewards.append(reward)
        print(f"  episode {ep+1:>3}/{args.n_episodes}  reward = {reward:.2f}")

    print(f"\nResults over {args.n_episodes} episodes:")
    print(f"  mean  = {np.mean(rewards):.2f}")
    print(f"  std   = {np.std(rewards):.2f}")
    print(f"  min   = {np.min(rewards):.2f}")
    print(f"  max   = {np.max(rewards):.2f}")


# ---------------------------------------------------------------------------
# Mode: dream
# ---------------------------------------------------------------------------

def mode_dream(args: argparse.Namespace) -> None:
    import torch
    import numpy as np

    print("\n" + "=" * 60)
    print("MODE: dream")
    print("=" * 60)

    wm = load_world_model(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed)

    for i in range(args.n_dreams):
        # sample a random starting latent from unit Gaussian
        z0 = torch.randn(args.z_dim, generator=rng)

        trajectory = wm.dream_rollout(
            z0=z0,
            horizon=args.horizon,
            temperature=args.temperature,
        )

        out_path = out_dir / f"dream_{i:03d}.pt"
        torch.save(trajectory, out_path)

        z_traj = trajectory["z"].numpy()       # (T, z_dim)
        a_traj = trajectory["a"].numpy()       # (T, action_dim)

        print(f"  dream {i+1}/{args.n_dreams}")
        print(f"    z  mean={z_traj.mean():.3f}  std={z_traj.std():.3f}  "
              f"min={z_traj.min():.3f}  max={z_traj.max():.3f}")
        print(f"    a  mean={a_traj.mean():.3f}  std={a_traj.std():.3f}")
        if "done_prob" in trajectory:
            done_probs = trajectory["done_prob"].numpy()
            print(f"    done_prob  mean={done_probs.mean():.3f}  max={done_probs.max():.3f}")
        print(f"    saved → {out_path}")


# ---------------------------------------------------------------------------
# Mode: encode
# ---------------------------------------------------------------------------

def mode_encode(args: argparse.Namespace) -> None:
    import torch

    print("\n" + "=" * 60)
    print("MODE: encode")
    print("=" * 60)

    vae_ckpt = args.vae_ckpt or _find_latest(args.checkpoint_dir, "vae_final.pt", "vae_*.pt")
    if vae_ckpt is None:
        sys.exit("No VAE checkpoint found.")

    from .trainer import get_device
    from .vision import BetaVAE, VAEConfig

    device = get_device()
    vae_cfg = VAEConfig(img_size=args.img_size, z_dim=args.z_dim)
    vae = BetaVAE(vae_cfg).to(device).eval()

    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])

    # load frame
    ep_path = sorted(Path(args.data_dir, "episodes").glob("ep_*.pt"))[args.ep_idx]
    ep = torch.load(ep_path, weights_only=True)
    frame = ep["frames"][args.frame_idx].unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        out = vae(frame)

    z = out.z.squeeze(0).cpu().numpy()
    mu = out.mu.squeeze(0).cpu().numpy()
    log_var = out.log_var.squeeze(0).cpu().numpy()

    print(f"  Episode : {ep_path.name}  frame {args.frame_idx}")
    print(f"  z       : mean={z.mean():.4f}  std={z.std():.4f}  "
          f"min={z.min():.4f}  max={z.max():.4f}")
    print(f"  mu      : mean={mu.mean():.4f}  std={mu.std():.4f}")
    print(f"  log_var : mean={log_var.mean():.4f}  std={log_var.std():.4f}")
    print(f"  z values: {z.tolist()}")


# ---------------------------------------------------------------------------
# Mode: reconstruct
# ---------------------------------------------------------------------------

def mode_reconstruct(args: argparse.Namespace) -> None:
    import torch

    print("\n" + "=" * 60)
    print("MODE: reconstruct")
    print("=" * 60)

    vae_ckpt = args.vae_ckpt or _find_latest(args.checkpoint_dir, "vae_final.pt", "vae_*.pt")
    if vae_ckpt is None:
        sys.exit("No VAE checkpoint found.")

    from .trainer import get_device
    from .vision import BetaVAE, VAEConfig

    device = get_device()
    vae_cfg = VAEConfig(img_size=args.img_size, z_dim=args.z_dim)
    vae = BetaVAE(vae_cfg).to(device).eval()

    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])

    ep_path = sorted(Path(args.data_dir, "episodes").glob("ep_*.pt"))[args.ep_idx]
    ep = torch.load(ep_path, weights_only=True)
    frame = ep["frames"][args.frame_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        out = vae(frame)

    original = frame.squeeze(0).cpu()
    recon = out.recon.squeeze(0).cpu()

    # compute reconstruction error
    mse = (original - recon).pow(2).mean().item()
    print(f"  MSE (original vs recon): {mse:.6f}")

    # save side-by-side image
    try:
        import torchvision.utils as vutils
        from PIL import Image
        import numpy as np

        grid = vutils.make_grid([original, recon], nrow=2, padding=4, normalize=False)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(grid_np)
        img.save(args.out)
        print(f"  Saved: {args.out}  (left=original, right=reconstruction)")
    except ImportError:
        print("  PIL not available — skipping image save.")
        # fallback: save as tensor
        torch.save({"original": original, "recon": recon}, args.out.replace(".png", ".pt"))
        print(f"  Saved tensors to {args.out.replace('.png', '.pt')}")


# ---------------------------------------------------------------------------
# Mode: record  (real episode → MP4)
# ---------------------------------------------------------------------------

def mode_record(args: argparse.Namespace) -> None:
    """
    Run one episode with the trained VMC policy and record it to MP4.

    Uses gymnasium's RecordVideo wrapper which writes frames via ffmpeg.
    The output video shows the raw environment pixels (96×96 CarRacing),
    not the 64×64 VAE input — so it looks exactly as the env renders it.
    """
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    print("\n" + "=" * 60)
    print("MODE: record")
    print("=" * 60)

    wm = load_world_model(args)

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RecordVideo needs render_mode="rgb_array"
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(out_path.parent),
        name_prefix=out_path.stem,
        episode_trigger=lambda ep: True,   # record every episode
    )

    obs, _ = env.reset(seed=args.seed)
    wm.reset()

    total_reward = 0.0
    steps = 0
    for _ in range(args.max_steps):
        action, _, _ = wm.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    env.close()

    # RecordVideo saves to <folder>/<prefix>-episode-0.mp4
    saved = sorted(out_path.parent.glob(f"{out_path.stem}*.mp4"))
    print(f"  Episode reward : {total_reward:.2f}  ({steps} steps)")
    print(f"  Video saved    : {saved[-1] if saved else '(not found)'}")


# ---------------------------------------------------------------------------
# Mode: dream_video  (dream latents decoded through VAE → GIF / MP4)
# ---------------------------------------------------------------------------

def mode_dream_video(args: argparse.Namespace) -> None:
    """
    Generate N imagined rollouts, decode each latent z_t back through the VAE
    decoder, and stitch the frames into a watchable GIF or MP4.

    This lets you see what the world model *imagines* the environment looks like —
    the key visual sanity check for whether V and M are jointly working.

    Output layout (n_dreams > 1): frames tiled horizontally, one column per dream.
    """
    import torch
    import numpy as np

    print("\n" + "=" * 60)
    print("MODE: dream_video")
    print("=" * 60)

    wm = load_world_model(args)
    out_path = Path(args.out_gif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed)

    # --- collect decoded frames for each dream ---
    all_dream_frames: list[list[np.ndarray]] = []   # [dream_idx][t] → (H, W, 3) uint8

    for d in range(args.n_dreams):
        z0 = torch.randn(args.z_dim, generator=rng)
        trajectory = wm.dream_rollout(z0=z0, horizon=args.horizon, temperature=args.temperature)

        z_seq = trajectory["z"].to(wm.device)   # (T, z_dim)
        frames: list[np.ndarray] = []

        with torch.no_grad():
            for t in range(z_seq.shape[0]):
                recon = wm.vae.decode(z_seq[t].unsqueeze(0))   # (1, 3, H, W)
                frame = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame)

        all_dream_frames.append(frames)
        print(f"  dream {d+1}/{args.n_dreams} — {len(frames)} frames decoded")

    # --- tile dreams side-by-side per timestep ---
    T = min(len(f) for f in all_dream_frames)
    tiled: list[np.ndarray] = []
    for t in range(T):
        row = np.concatenate([all_dream_frames[d][t] for d in range(args.n_dreams)], axis=1)
        tiled.append(row)

    # --- save ---
    ext = out_path.suffix.lower()
    if ext == ".gif":
        _save_gif(tiled, out_path, fps=args.fps)
        print(f"  Saved GIF → {out_path}  ({T} frames, {args.fps} fps)")
    elif ext == ".mp4":
        _save_mp4(tiled, out_path, fps=args.fps)
        print(f"  Saved MP4 → {out_path}  ({T} frames, {args.fps} fps)")
    else:
        # fallback: save GIF regardless
        gif_path = out_path.with_suffix(".gif")
        _save_gif(tiled, gif_path, fps=args.fps)
        print(f"  Unknown extension '{ext}', saved GIF → {gif_path}")


def _save_gif(frames: list, path: Path, fps: int) -> None:
    import numpy as np
    from PIL import Image

    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(np.asarray(f)) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


def _save_mp4(frames: list, path: Path, fps: int) -> None:
    """Write frames to MP4 using imageio (falls back to GIF if unavailable)."""
    try:
        import imageio.v3 as iio  # type: ignore[import-untyped]
        iio.imwrite(str(path), frames, fps=fps, codec="libx264")
    except (ImportError, Exception) as e:
        print(f"  imageio MP4 write failed ({e}), falling back to GIF.")
        gif_path = path.with_suffix(".gif")
        _save_gif(frames, gif_path, fps=fps)
        print(f"  Saved GIF → {gif_path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_latest(checkpoint_dir: str, *patterns: str) -> str | None:
    base = Path(checkpoint_dir)
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return str(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

MODE_MAP = {
    "eval":         mode_eval,
    "record":       mode_record,
    "dream":        mode_dream,
    "dream_video":  mode_dream_video,
    "encode":       mode_encode,
    "reconstruct":  mode_reconstruct,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    MODE_MAP[args.mode](args)


if __name__ == "__main__":
    main()
