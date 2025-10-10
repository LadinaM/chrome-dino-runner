# test_dino_agent.py
import os
import argparse
import logging
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chrome_dino_env import ChromeDinoEnv

# ---------------- Logging ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("stable_baselines3").setLevel(logging.INFO)


def build_eval_env(seed: int, render_mode: str, vecnorm_path: str | None):
    """
    Build an evaluation env.
    If vecnorm_path exists, wrap env with VecNormalize.load(...), set to eval mode.
    """
    if render_mode not in ("human", None):
        raise ValueError(f"Unsupported render_mode '{render_mode}'. "
                         f"Use 'human' for display or None for headless mode.")
    make_raw = lambda: Monitor(ChromeDinoEnv(render_mode=render_mode, seed=seed))
    raw_env = DummyVecEnv([make_raw])

    if vecnorm_path and os.path.isfile(vecnorm_path):
        logger.info(f"Loading VecNormalize stats from: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, raw_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env
    if vecnorm_path:
        logger.warning(f"VecNormalize stats not found at '{vecnorm_path}'. "
                           f"Continuing WITHOUT normalization. "
                           f"(If you trained with VecNormalize, expect a performance drop.)")
        user_input = input("Do you want to continue? [y/N] ")
        if user_input.lower() == "y":
            return raw_env
        raise RuntimeError("VecNormalize stats not found at '{vecnorm_path}' and user doesn't want to continue.")


def run_eval(model_path: str,
             vecnorm_path: str | None = "vecnorm_stats.pkl",
             seed: int = 42,
             episodes: int = 5,
             deterministic: bool = True,
             render_human: bool = True):
    """
    Load model and evaluate for N episodes.
    """
    # Load the SB3 PPO model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device="auto")

    render_mode = "human" if render_human else None
    env = build_eval_env(seed=seed + 123, render_mode=render_mode, vecnorm_path=vecnorm_path)

    logger.info("Starting evaluation...")
    total_reward = 0.0
    finished = 0

    obs = env.reset()

    while finished < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)

        # rewards is a vector of shape (n_envs,). We have n_envs=1 here.
        total_reward += float(np.mean(rewards))

        if dones.any():
            # infos is a list of dicts for VecEnv
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            ep_score = info0.get("score", None)
            if ep_score is not None:
                logger.info(f"Episode {finished + 1} finished with score: {ep_score}")
            else:
                logger.info(f"Episode {finished + 1} finished.")
            finished += 1
            # reset handled automatically by VecEnv when dones.any() is True

    env.close()
    logger.info(f"Finished {episodes} episodes. "
                f"Total reward (mean across vec steps): {total_reward:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Dino PPO agent.")
    p.add_argument("--model_path", type=str, default="dino_final_model.zip",
                   help="Path to the saved model (.zip).")
    p.add_argument("--vecnorm_path", type=str, default="vecnorm_stats.pkl",
                   help="Path to VecNormalize stats file. Leave as default if you trained with VecNormalize.")
    p.add_argument("--seed", type=int, default=42, help="Base seed for eval env.")
    p.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic actions (default: deterministic).")
    p.add_argument("--no_render", action="store_true",
                   help="Do not render (headless eval).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path if args.vecnorm_path else None,
        seed=args.seed,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render_human=not args.no_render,
    )
