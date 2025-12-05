import logging

from chrome_dino_env import ChromeDinoEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_environment():
    """Smoke-test the Chrome Dino environment with random, nondeterministic actions."""
    logger.info("Testing Chrome Dino Environment...")
    env = ChromeDinoEnv(render_mode="human")
    
    # Test reset
    logger.info("Testing reset...")
    obs, info = env.reset()
    logger.info(f"Initial observation: {obs}")
    logger.info(f"Initial info: {info}")
    
    # Test a few random actions
    logger.info("Testing random actions...")
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"Step {i+1}: Action={action}, Reward={reward}, Terminated={terminated}, Score={info['score']}")
        
        if terminated:
            logger.info("Episode ended!")
            break
    
    env.close()
    logger.info("Environment test completed!")

if __name__ == "__main__":
    test_environment()
