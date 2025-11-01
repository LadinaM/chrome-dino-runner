import pygame
import datetime
import random
from figures.dinosaur import Dinosaur
from figures.cloud import Cloud
from figures.obstacles import SmallCactus, LargeCactus, Bird
from figures.configurations import load_game_assets
from utilities.constants import MovementType, AssetClass


class GameSettings:
    """Shared game settings and constants"""
    SCREEN_WIDTH = 1100
    SCREEN_HEIGHT = 600
    INITIAL_GAME_SPEED = 20
    Y_POS_BG = 380
    SPEED_INCREASE_INTERVAL = 100
    FONT_COLOR = (0, 0, 0)
    FONT_SIZE = 20
    
    @classmethod
    def get_background_color(cls) -> tuple[int, int, int]:
        """Get background color based on time of day"""
        current_time = datetime.datetime.now().hour
        if 7 < current_time < 19:
            return 255, 255, 255  # White for day
        else:
            return 0, 0, 0  # Black for night

class GameState:
    """Shared game state management"""
    
    def __init__(self):
        self.game_speed = GameSettings.INITIAL_GAME_SPEED
        self.x_pos_bg = 0
        self.y_pos_bg = GameSettings.Y_POS_BG
        self.points = 0
        self.obstacles = []
        
    def reset(self):
        """Reset game state to initial values"""
        self.game_speed = GameSettings.INITIAL_GAME_SPEED
        self.x_pos_bg = 0
        self.y_pos_bg = GameSettings.Y_POS_BG
        self.points = 0
        self.obstacles = []
    
    def update_score(self):
        """Update score and game speed"""
        self.points += 1
        if self.points % GameSettings.SPEED_INCREASE_INTERVAL == 0:
            self.game_speed += 1

class GameObjects:
    """Shared game objects management"""
    
    def __init__(self, assets):
        self.assets = assets
        self.player = None
        self.cloud = None
        self.font = None
        
    def initialize(self, game_speed):
        """Initialize game objects"""
        self.player = Dinosaur(self.assets[MovementType.RUNNING], self.assets[MovementType.JUMPING], self.assets[MovementType.DUCKING])
        self.cloud = Cloud(self.assets[AssetClass.CLOUD], GameSettings.SCREEN_WIDTH, game_speed)
        self.font = pygame.font.Font("freesansbold.ttf", GameSettings.FONT_SIZE)
    
    def spawn_obstacle(self, obstacles):
        """Spawn a random obstacle"""
        if len(obstacles) == 0:
            obstacle_type = random.randint(0, 2)
            match obstacle_type:
                case 0:
                    obstacles.append(SmallCactus(self.assets[AssetClass.SMALL_CACTUS]))
                case 1:
                    obstacles.append(LargeCactus(self.assets[AssetClass.LARGE_CACTUS]))
                case 2:
                    obstacles.append(Bird(self.assets[AssetClass.BIRD]))

class GameRenderer:
    """Shared game rendering functions"""
    
    @staticmethod
    def update_background(screen, assets, game_state):
        """Update background position and draw"""
        image_width = assets['BG'].get_width()
        screen.blit(assets['BG'], (game_state.x_pos_bg, game_state.y_pos_bg))
        screen.blit(assets['BG'], (image_width + game_state.x_pos_bg, game_state.y_pos_bg))
        if game_state.x_pos_bg <= -image_width:
            screen.blit(assets['BG'], (image_width + game_state.x_pos_bg, game_state.y_pos_bg))
            game_state.x_pos_bg = 0
        game_state.x_pos_bg -= game_state.game_speed
    
    @staticmethod
    def render_score(screen, font, points, high_score=None):
        """Render score text"""
        if high_score is None:
            text = font.render(f"Points: {points}", True, GameSettings.FONT_COLOR)
        else:
            text = font.render(f"High Score: {high_score}  Points: {points}", True, GameSettings.FONT_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (900, 40)
        screen.blit(text, text_rect)
    
    @staticmethod
    def render_game_objects(screen, game_objects, obstacles):
        """Render all game objects"""
        # Draw player
        game_objects.player.draw(screen)
        
        # Draw obstacles
        for obstacle in obstacles:
            obstacle.draw(screen)
        
        # Draw cloud
        game_objects.cloud.draw(screen)

class GameInitializer:
    """Shared game initialization"""
    
    @staticmethod
    def initialize_pygame():
        """Initialize pygame and create screen"""
        pygame.init()
        screen = pygame.display.set_mode((GameSettings.SCREEN_WIDTH, GameSettings.SCREEN_HEIGHT))
        return screen
    
    @staticmethod
    def load_assets():
        """Load game assets"""
        assets = load_game_assets()
        return assets
    
    @staticmethod
    def setup_display(screen, assets):
        """Setup display with caption and icon"""
        pygame.display.set_caption("Chrome Dino Runner")
        pygame.display.set_icon(assets['ICON'])
