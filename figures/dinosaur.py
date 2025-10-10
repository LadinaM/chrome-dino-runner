import pygame

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self, run_img, jump_img, duck_img):
        self.duck_img = duck_img
        self.run_img = run_img
        self.jump_img = jump_img

        # State flags
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        # Jump physics
        self.jump_vel = self.JUMP_VEL

        # Animation state
        self.step_index = 0
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    # Controls
    def jump(self):
        if not self.dino_jump:
            self.dino_jump = True
            self.dino_duck = False
            self.dino_run = False
            self.jump_vel = self.JUMP_VEL

    def duck(self):
        if not self.dino_jump: # Jump only when on ground
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
    def release_duck(self):
        if not self.dino_jump:
            self.dino_duck = False
            self.dino_run = True

    # Frame update
    def update(self, user_input=None):
        if self.dino_duck:
            self.duck_anim()
        if self.dino_run:
            self.run_anim()
        if self.dino_jump:
            self.jump_anim()

        if self.step_index >= 10:
            self.step_index = 0

        # Keyboard controls for human play
        if (user_input[pygame.K_UP] or user_input[pygame.K_SPACE]) and not self.dino_jump:
            self.jump()
        elif user_input[pygame.K_DOWN] and not self.dino_jump:
           self.duck()
        elif not (self.dino_jump or user_input[pygame.K_DOWN]):
            self.release_duck()

    def duck_anim(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run_anim(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump_anim(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            # Landed
            self.dino_duck = True
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
