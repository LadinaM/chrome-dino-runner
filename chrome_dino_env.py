import gymnasium as gym
import numpy as np
import pygame
import random
import datetime
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# Import game figures
from figures.dinosaur import Dinosaur
from figures.cloud import Cloud
from figures.obstacles import SmallCactus, LargeCactus, Bird
from figures.configurations import load_game_assets
from helpers import get_high_score

# Import shared game settings
from game import GameSettings, GameState, GameObjects, GameRenderer, GameInitializer


class ChromeDinoEnv(gym.Env):
    """
    Chrome Dino Runner environment for reinforcement learning
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Initialize pygame
        pygame.init()
        
        # Use shared game settings
        self.SCREEN_HEIGHT = GameSettings.SCREEN_HEIGHT
        self.SCREEN_WIDTH = GameSettings.SCREEN_WIDTH
        self.SCREEN = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Chrome Dino Runner - RL Environment")
        
        # Load game assets
        self.ASSETS = load_game_assets()
        pygame.display.set_icon(self.ASSETS['ICON'])
        
        # Initialize game state and objects
        self.game_state = GameState()
        self.game_objects = GameObjects(self.ASSETS)
        self.game_objects.initialize(self.game_state.game_speed)
        
        # Game state variables
        self.clock = pygame.time.Clock()
        
        # Define action space: 0 = do nothing, 1 = jump, 2 = duck
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # We'll use a simplified state representation:
        # [dino_y_pos, dino_velocity, nearest_obstacle_x, nearest_obstacle_y, nearest_obstacle_type, game_speed]
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0, 0, 0]), 
            high=np.array([400, 20, 1200, 400, 3, 100]), 
            dtype=np.float32
        )
        
        # Render mode
        self.render_mode = render_mode
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # Get dino position and velocity
        dino_y = self.game_objects.player.dino_rect.y
        dino_vel = -self.game_objects.player.jump_vel if self.game_objects.player.dino_jump else 0
        
        # Get nearest obstacle info
        nearest_obstacle_x = 1200  # Default to screen width if no obstacles
        nearest_obstacle_y = 0
        nearest_obstacle_type = 0
        
        if len(self.game_state.obstacles) > 0:
            nearest_obstacle = self.game_state.obstacles[0]
            nearest_obstacle_x = nearest_obstacle.rect.x
            nearest_obstacle_y = nearest_obstacle.rect.y
            
            # Determine obstacle type: 0 = small cactus, 1 = large cactus, 2 = bird
            if isinstance(nearest_obstacle, SmallCactus):
                nearest_obstacle_type = 0
            elif isinstance(nearest_obstacle, LargeCactus):
                nearest_obstacle_type = 1
            elif isinstance(nearest_obstacle, Bird):
                nearest_obstacle_type = 2
        
        return np.array([
            dino_y, dino_vel, nearest_obstacle_x, nearest_obstacle_y, 
            nearest_obstacle_type, self.game_state.game_speed
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'score': self.game_state.points,
            'game_speed': self.game_state.game_speed,
            'obstacles_count': len(self.game_state.obstacles)
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset game state
        self.game_state.reset()
        
        # Reset game objects
        self.game_objects.initialize(self.game_state.game_speed)
        
        # Clear screen
        current_time = datetime.datetime.now().hour
        if 7 < current_time < 19:
            self.SCREEN.fill((255, 255, 255))
        else:
            self.SCREEN.fill((0, 0, 0))
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Convert action to pygame key state dictionary
        userInput = pygame.key.get_pressed()
        
        # Create a custom key state for our actions
        # We'll simulate key presses by creating a custom input handler
        if action == 1:  # Jump
            # Simulate UP or SPACE key press
            userInput = type('MockKeys', (), {
                '__getitem__': lambda self, key: True if key in [pygame.K_UP, pygame.K_SPACE] else False
            })()
        elif action == 2:  # Duck
            # Simulate DOWN key press
            userInput = type('MockKeys', (), {
                '__getitem__': lambda self, key: True if key == pygame.K_DOWN else False
            })()
        else:  # Do nothing
            userInput = type('MockKeys', (), {
                '__getitem__': lambda self, key: False
            })()
        
        # Update player
        self.game_objects.player.update(userInput)
        
        # Spawn obstacles
        self.game_objects.spawn_obstacle(self.game_state.obstacles)
        
        # Update obstacles
        for obstacle in self.game_state.obstacles:
            obstacle.update(self.game_state.game_speed, self.game_state.obstacles)
        
        # Check collision
        terminated = False
        reward = 0
        
        for obstacle in self.game_state.obstacles:
            if self.game_objects.player.dino_rect.colliderect(obstacle.rect):
                terminated = True
                reward = -100  # Large negative reward for collision
                break
        
        # Update score and game speed
        self.game_state.update_score()
        if self.game_state.points % 100 == 0:
            reward += 10  # Bonus for surviving longer
        
        # Small positive reward for surviving
        if not terminated:
            reward += 1
        
        # Update background
        GameRenderer.update_background(self.SCREEN, self.ASSETS, self.game_state)
        
        # Update cloud
        self.game_objects.cloud.update(self.game_state.game_speed)
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _render_frame(self):
        """Render the current frame"""
        current_time = datetime.datetime.now().hour
        if 7 < current_time < 19:
            self.SCREEN.fill((255, 255, 255))
        else:
            self.SCREEN.fill((0, 0, 0))
        
        # Draw game objects
        GameRenderer.render_game_objects(self.SCREEN, self.game_objects, self.game_state.obstacles)
        
        # Draw score
        GameRenderer.render_score(self.SCREEN, self.game_objects.font, self.game_state.points)
        
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.SCREEN)
    
    def close(self):
        """Close the environment"""
        pygame.quit()
