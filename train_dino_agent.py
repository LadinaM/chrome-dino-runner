import os
import logging
from typing import Callable, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from chrome_dino_env import ChromeDinoEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("stable_baselines3").setLevel(logging.INFO)

SEED = 42
N_ENVS = 8  # increase if CPU allows (e.g., 8)
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5

def make_env(rank: int) -> Callable[[], Monitor]:
    """
    Utility to create a thunk that builds a single Monitor-wrapped env.
    """
    def _init():
        env = ChromeDinoEnv(render_mode=None, seed=SEED + rank)
        env = Monitor(env)
        return env
    return _init

def linear_lr(initial_lr: float):
    """
    Linear LR schedule: lr = initial_lr * (1 - progress_remaining)
    """
    def _lr(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return _lr

class SyncVecNormalizeCallback(BaseCallback):
    """
    Ensures eval VecNormalize uses the latest obs_rms from the training VecNormalize
    right before each evaluation happens.
    Set sync_freq equal to EVAL_FREQ to sync just-in-time for EvalCallback.
    """
    def __init__(self, train_vecnorm: VecNormalize, eval_vecnorm: VecNormalize, sync_freq: int):
        super().__init__()
        self.train_vecnorm = train_vecnorm
        self.eval_vecnorm = eval_vecnorm
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.sync_freq == 0:
            self.eval_vecnorm.obs_rms = self.train_vecnorm.obs_rms
        return True


def main():
    set_random_seed(SEED)

    # ----- Training env(s)
    if N_ENVS > 1:
        train_vec = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        train_vec = DummyVecEnv([make_env(0)])


    # Wrap with VecNormalize for obs/reward normalization
    train_env = VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    train_env.seed(SEED)

    # ----- Evaluation env (MUST match wrapper type)
    eval_raw = DummyVecEnv([make_env(10_000)])  # separate seed stream
    eval_env = VecNormalize(
        eval_raw,
        training=False,          # <- important: eval mode
        norm_obs=True,
        norm_reward=False,       # don't normalize rewards for reporting
        clip_obs=10.0,
    )
    # ---- Callbacks
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="dino_model",
        save_replay_buffer=False,
        save_vecnormalize=True  # save VecNormalize with checkpoints
    )
    sync_callback = SyncVecNormalizeCallback(
        train_vecnorm=train_env,
        eval_vecnorm=eval_env,
        sync_freq=EVAL_FREQ
    )
    callbacks = CallbackList([sync_callback, eval_callback, checkpoint_callback])

    # ----- PPO model
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="auto",
        learning_rate=linear_lr(3e-4),
        n_steps=2048,  # per env; effective batch = n_steps * n_envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        target_kl=0.03,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=policy_kwargs,
        seed=SEED,
    )

    # ---- Train
    logger.info("Starting training...")
    total_ts = 1_000_000  # consider 1–3M for reliable mastery
    model.learn(total_timesteps=total_ts, callback=callbacks)

    # Save model and VecNormalize statistics
    model.save("dino_final_model")
    train_env.save("vecnorm_stats.pkl")

if __name__ == "__main__":
    main()
