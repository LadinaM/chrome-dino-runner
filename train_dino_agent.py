import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from chrome_dino_env import ChromeDinoEnv

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("stable_baselines3").setLevel(logging.INFO)

def make_env():
    """Create a single environment"""
    env = ChromeDinoEnv(render_mode=None)  # No rendering during training for speed
    env = Monitor(env)
    return env

def main():
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(1)])  # Single environment for now
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(ChromeDinoEnv(render_mode=None))])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./checkpoints/",
        name_prefix="dino_model"
    )
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,  # 10
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )
    

    logger.info("Starting training...")
    model.learn(
        total_timesteps=150_000,
        callback=[eval_callback, checkpoint_callback]
    )

    model.save("dino_final_model")

    logger.info("Testing trained model...")
    test_env = ChromeDinoEnv(render_mode="human")
    
    obs, info = test_env.reset()
    total_reward = 0
    episode_count = 0
    
    for _ in range(5):  # Test 5 episodes
        obs, info = test_env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            
            if terminated:
                logger.info(f"Episode {episode_count + 1} finished with score: {info['score']}")
                episode_count += 1
                break
    
    test_env.close()
    logger.info("Training and testing completed!")

if __name__ == "__main__":
    main()
