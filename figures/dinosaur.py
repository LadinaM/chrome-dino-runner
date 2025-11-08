import pygame

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5    
    GRAVITY  = 0.8  # retained for reference; classic jump uses fixed decrement

    def __init__(self, run_img, jump_img, duck_img):
        # Sprites / assets
        self.run_img  = run_img          # sequence of frames
        self.jump_img = jump_img         # single surface
        self.duck_img = duck_img         # sequence of frames

        # State flags
        self.dino_duck = False
        self.dino_run  = True
        self.dino_jump = False

        # Animation state
        self.step_index = 0
        self.image = self.run_img[0]

        # Collision / position rect (must exist immediately)
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

        # Jump physics (start at rest)
        self.jump_vel = 0.0

    # Utility
    def on_ground(self) -> bool:
        # Consider tiny drift; on ground when at/under Y_POS and not flagged jumping
        return self.dino_rect.y >= self.Y_POS and not self.dino_jump

    # Controls
    def jump(self):
        if self.on_ground():
            self.dino_jump = True
            self.dino_duck = False
            self.dino_run  = False
            self.jump_vel  = self.JUMP_VEL

    def duck(self):
        if self.on_ground():
            self.dino_duck = True
            self.dino_run  = False
            self.dino_jump = False

    def release_duck(self):
        if self.on_ground():
            self.dino_duck = False
            self.dino_run  = True

    # Frame update
    def update(self, user_input=None):
        # Safe user_input handling (env may pass None or a proxy)
        up = down = space = False
        if user_input is not None:
            try:
                up    = bool(user_input[pygame.K_UP])
                down  = bool(user_input[pygame.K_DOWN])
                space = bool(user_input[pygame.K_SPACE])
            except Exception:
                pass

        # Keyboard-like controls for human play
        if (up or space) and self.on_ground():
            self.jump()
        elif down and self.on_ground():
            self.duck()
        elif not self.dino_jump and not down:
            self.release_duck()

        # Animate state
        if self.dino_duck:
            self.duck_anim()
        elif self.dino_run:
            self.run_anim()
        if self.dino_jump:
            self.jump_anim()

        if self.step_index >= 10:
            self.step_index = 0

    def duck_anim(self):
        self.image = self.duck_img[self.step_index // 5]
        rect = self.image.get_rect()
        rect.x = self.X_POS
        rect.y = self.Y_POS_DUCK
        self.dino_rect = rect
        self.step_index += 1

    def run_anim(self):
        self.image = self.run_img[self.step_index // 5]
        rect = self.image.get_rect()
        rect.x = self.X_POS
        rect.y = self.Y_POS
        self.dino_rect = rect
        self.step_index += 1

    def jump_anim(self):
        self.image = self.jump_img
        if self.dino_jump:
            # Classic, snappy Chrome-Dino jump: fast ascent, smooth descent
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            # Land
            self.dino_rect.y = self.Y_POS
            self.dino_run  = True
            self.dino_duck = False
            self.dino_jump = False
            self.jump_vel  = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
