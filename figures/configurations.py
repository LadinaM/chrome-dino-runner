import os
import pygame


def load_game_assets():
    """Load all game assets and return them as a dictionary"""
    
    # Dinosaur sprites
    RUNNING = [
        pygame.image.load(os.path.join("assets/Dino", "DinoRun1.png")),
        pygame.image.load(os.path.join("assets/Dino", "DinoRun2.png")),
    ]
    JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))
    DUCKING = [
        pygame.image.load(os.path.join("assets/Dino", "DinoDuck1.png")),
        pygame.image.load(os.path.join("assets/Dino", "DinoDuck2.png")),
    ]

    # Cactus obstacles
    SMALL_CACTUS = [
        pygame.image.load(os.path.join("assets/Cactus", "SmallCactus1.png")),
        pygame.image.load(os.path.join("assets/Cactus", "SmallCactus2.png")),
        pygame.image.load(os.path.join("assets/Cactus", "SmallCactus3.png")),
    ]
    LARGE_CACTUS = [
        pygame.image.load(os.path.join("assets/Cactus", "LargeCactus1.png")),
        pygame.image.load(os.path.join("assets/Cactus", "LargeCactus2.png")),
        pygame.image.load(os.path.join("assets/Cactus", "LargeCactus3.png")),
    ]

    # Bird obstacles
    BIRD = [
        pygame.image.load(os.path.join("assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("assets/Bird", "Bird2.png")),
    ]

    # Background elements
    CLOUD = pygame.image.load(os.path.join("assets/Other", "Cloud.png"))
    BG = pygame.image.load(os.path.join("assets/Other", "Track.png"))
    
    # Icon
    ICON = pygame.image.load("assets/DinoWallpaper.png")
    
    return {
        'RUNNING': RUNNING,
        'JUMPING': JUMPING,
        'DUCKING': DUCKING,
        'SMALL_CACTUS': SMALL_CACTUS,
        'LARGE_CACTUS': LARGE_CACTUS,
        'BIRD': BIRD,
        'CLOUD': CLOUD,
        'BG': BG,
        'ICON': ICON
    }
