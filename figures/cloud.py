import pygame
import random

class Cloud:
    def __init__(self, cloud_img, screen_width, game_speed):
        self.x = screen_width + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = cloud_img
        self.width = self.image.get_width()
        self.screen_width = screen_width
        self.game_speed = game_speed

    def update(self, game_speed=None):
        if game_speed is not None:
            self.game_speed = game_speed
        self.x -= self.game_speed
        if self.x < -self.width:
            self.x = self.screen_width + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))
