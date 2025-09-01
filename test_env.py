import numpy as np
from chrome_dino_env import ChromeDinoEnv

def test_environment():
    """Test the Chrome Dino environment"""
    print("Testing Chrome Dino Environment...")
    
    # Create environment
    env = ChromeDinoEnv(render_mode="human")
    
    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few random actions
    print("Testing random actions...")
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Action={action}, Reward={reward}, Terminated={terminated}, Score={info['score']}")
        
        if terminated:
            print("Episode ended!")
            break
    
    env.close()
    print("Environment test completed!")

if __name__ == "__main__":
    test_environment()
