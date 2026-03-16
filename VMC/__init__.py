"""
VMC — Vision · Memory · Controller
====================================
Modular world model framework based on Ha & Schmidhuber (2018).

    V  BetaVAE          — compress raw observations to a latent vector z
    M  MDNRNN           — factorized MDN + LSTM, models p(z_{t+1} | z_t, a_t)
    C  LinearController — W·[z; h] + b, optimised by CMA-ES

Quick start
-----------
    from VMC import WorldModel, WorldModelConfig
    wm = WorldModel.from_checkpoints("vae.pt", "mdn.pt", "ctrl.pt")
    reward = wm.eval_episode(env)

Training pipeline
-----------------
    from VMC.data import DataConfig, collect_rollouts, encode_dataset
    from VMC.trainer import VAETrainer, VAETrainerConfig
    from VMC.trainer import MDNRNNTrainer, MDNRNNTrainerConfig
    from VMC.controller import ControllerConfig, LinearController, CMAESTrainer

    # 1. collect data
    collect_rollouts(DataConfig())

    # 2. train VAE
    vae = VAETrainer(VAETrainerConfig()).train()

    # 3. encode latents
    encode_dataset(vae, data_dir="./VMC_data", device=...)

    # 4. train MDN-RNN
    mdn = MDNRNNTrainer(MDNRNNTrainerConfig()).train()

    # 5. train controller (supply your own rollout_fn)
    ctrl = LinearController(ControllerConfig())
    CMAESTrainer(ControllerConfig(), ctrl, rollout_fn=my_rollout_fn).train()
"""

from .controller import CMAESTrainer, ControllerConfig, LinearController
from .memory import MDNRNN, MDNOutput, MDNRNNConfig
from .model import WorldModel, WorldModelConfig
from .vision import BaseVAE, BetaVAE, VAEConfig, VAEOutput

__all__ = [
    # Vision
    "BaseVAE",
    "BetaVAE",
    "VAEConfig",
    "VAEOutput",
    # Memory
    "MDNRNN",
    "MDNOutput",
    "MDNRNNConfig",
    # Controller
    "LinearController",
    "CMAESTrainer",
    "ControllerConfig",
    # World Model (assembled)
    "WorldModel",
    "WorldModelConfig",
]
