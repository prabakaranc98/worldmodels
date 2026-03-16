# VMC — Vision · Memory · Controller

Modular implementation of the Ha & Schmidhuber (2018) World Models architecture on CarRacing-v3.

| Component | File | What it does |
|-----------|------|-------------|
| V | `vision.py` | Beta-VAE — compresses raw frames to latent vector `z` |
| M | `memory.py` | Factorized MDN-LSTM — models `p(z_{t+1} \| z_t, a_t)` |
| C | `controller.py` | Linear controller optimised by CMA-ES |
| — | `data.py` | Rollout collection (multiprocessing) + dataset classes |
| — | `trainer.py` | Training recipes for VAE and MDN-RNN |
| — | `model.py` | `WorldModel` — assembles V+M+C for inference |
| — | `run_training.py` | CLI training pipeline |
| — | `run_testing.py` | CLI evaluation + visual inspection |

---

## Training pipeline

Run all phases end-to-end:

```bash
python -m VMC.run_training --all
```

Or run phases individually in order:

**1. Collect rollouts** (random policy — uses all CPU cores by default)
```bash
python -m VMC.run_training --phases collect
```
Defaults: 10,000 episodes, all logical CPU cores (`--n_workers -1`).
Scale down for a quick test: `--n_episodes 500`.

**2. Train VAE**
```bash
python -m VMC.run_training --phases train_vae --vae_epochs 50
```

**3. Encode latents** (one-time forward pass with frozen VAE)
```bash
python -m VMC.run_training --phases encode
```

**4. Train MDN-RNN**
```bash
python -m VMC.run_training --phases train_mdn --mdn_epochs 30
```

**5. Train controller** (CMA-ES evolutionary search)
```bash
python -m VMC.run_training --phases train_ctrl --popsize 64 --max_iter 200
```

Quick smoke test (verifies the full pipeline runs without errors):
```bash
python -m VMC.run_training --all \
  --n_episodes 10 --vae_epochs 2 --mdn_epochs 2 --max_iter 5 --popsize 8 --n_rollouts 2
```

---

## Visual evaluation

**Check VAE quality** — original vs reconstruction side by side:
```bash
python -m VMC.run_testing reconstruct --ep_idx 0 --frame_idx 100 --out recon.png
```

**Check MDN-RNN** — decode dream rollout to GIF (should look like a shaky track):
```bash
python -m VMC.run_testing dream_video --horizon 60 --out_gif dream.gif
```

**Evaluate policy** — record a full episode to MP4:
```bash
python -m VMC.run_testing record --out_video policy.mp4
```

**Run N episodes and print reward stats:**
```bash
python -m VMC.run_testing eval --n_episodes 10
```

---

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n_episodes` | 10,000 | Episodes to collect (Ha & Schmidhuber used 10k) |
| `--n_workers` | -1 | Parallel workers; -1 = all CPU cores |
| `--z_dim` | 32 | Latent dimension (must match across all phases) |
| `--hidden_size` | 256 | LSTM hidden size |
| `--beta` | 4.0 | KL weight in Beta-VAE |
| `--no_compile` | off | Disable `torch.compile` |
| `--vae_resume` | — | Path to VAE checkpoint to resume from |
| `--mdn_resume` | — | Path to MDN checkpoint to resume from |

Set `WANDB_MODE=disabled` to suppress wandb, or `wandb login` to enable live loss curves.
