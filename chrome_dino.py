import datetime
import threading

import pygame

from utilities.helpers import get_high_score
from utilities.constants import MovementType
from game import GameSettings, GameState, GameObjects, GameRenderer, GameInitializer

# Initialize pygame and create screen
SCREEN = GameInitializer.initialize_pygame()

# Load game assets
ASSETS = GameInitializer.load_assets()

# Setup display
GameInitializer.setup_display(SCREEN, ASSETS)

FONT_COLOR = GameSettings.FONT_COLOR

DELAY_IN_MS: int = 2000

def main():
    """Run the interactive Chrome Dino game loop."""
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    
    # Initialize game state and objects
    game_state = GameState()
    game_objects = GameObjects(ASSETS)
    game_objects.initialize(game_state.game_speed)
    
    # Local variables for backward compatibility
    game_speed = game_state.game_speed
    x_pos_bg = game_state.x_pos_bg
    y_pos_bg = game_state.y_pos_bg
    points = game_state.points
    obstacles = game_state.obstacles
    
    death_count = 0
    pause = False

    def score():
        """Update score, adjust speed, and render HUD."""
        global points, game_speed
        game_state.update_score()
        points = game_state.points
        game_speed = game_state.game_speed
        
        current_time = datetime.datetime.now().hour
        highscore = get_high_score()
        if points > highscore:
            highscore = points 
        GameRenderer.render_score(SCREEN, game_objects.font, points, highscore)

    def background():
        """Scroll and render the background."""
        global x_pos_bg, y_pos_bg
        GameRenderer.update_background(SCREEN, ASSETS, game_state)
        x_pos_bg = game_state.x_pos_bg
        y_pos_bg = game_state.y_pos_bg

    def unpause():
        """Resume the game loop after a pause."""
        nonlocal pause, run
        pause = False
        run = True

    def paused():
        """Pause loop that waits for the unpause key."""
        nonlocal pause
        pause = True
        font = pygame.font.Font("freesansbold.ttf", 30)
        text = font.render("Game Paused, Press 'u' to Unpause", True, FONT_COLOR)
        textRect = text.get_rect()
        textRect.center = (GameSettings.SCREEN_WIDTH // 2, GameSettings.SCREEN_HEIGHT  // 3)
        SCREEN.blit(text, textRect)
        pygame.display.update()

        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                    unpause()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                run = False
                paused()

        # Fill screen with background color
        SCREEN.fill(GameSettings.get_background_color())
        
        user_input = pygame.key.get_pressed()

        game_objects.player.draw(SCREEN)
        game_objects.player.update(user_input)

        # Spawn obstacles
        game_objects.spawn_obstacle(game_state.obstacles)
        obstacles = game_state.obstacles  # Update local reference

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update(game_speed, obstacles)
            if game_objects.player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(DELAY_IN_MS)
                death_count += 1
                menu(death_count)

        background()

        game_objects.cloud.draw(SCREEN)
        game_objects.cloud.update(game_speed)

        score()

        clock.tick(30)
        pygame.display.update()


def menu(death_count):
    """Menu screen for start/restart flow."""
    global points
    global FONT_COLOR
    run: bool = True
    while run:
        FONT_COLOR = GameSettings.FONT_COLOR
        SCREEN.fill(GameSettings.get_background_color())
        font = pygame.font.Font("freesansbold.ttf", 30)

        if death_count == 0:
            text = font.render("Press any Key to Start", True, FONT_COLOR)
        elif death_count > 0:
            text = font.render("Press any Key to Restart", True, FONT_COLOR)
            score = font.render("Your Score: " + str(points), True, FONT_COLOR)
            score_rect = score.get_rect()
            score_rect.center = (GameSettings.SCREEN_WIDTH // 2, GameSettings.SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, score_rect)
            f = open("score.txt", "a")
            f.write(str(points) + "\n")
            f.close()
            highscore = get_high_score()
            hs_score_text = font.render(
                "High Score : " + str(highscore), True, FONT_COLOR
            )
            hs_score_rect = hs_score_text.get_rect()
            hs_score_rect.center = (GameSettings.SCREEN_WIDTH // 2, GameSettings.SCREEN_HEIGHT // 2 + 100)
            SCREEN.blit(hs_score_text, hs_score_rect)
        text_rect = text.get_rect()
        text_rect.center = (GameSettings.SCREEN_WIDTH // 2, GameSettings.SCREEN_HEIGHT // 2)
        SCREEN.blit(text, text_rect)
        SCREEN.blit(ASSETS[MovementType.RUNNING][0], (GameSettings.SCREEN_WIDTH // 2 - 20, GameSettings.SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                main()


t1 = threading.Thread(target=menu(death_count=0), daemon=True)
t1.start()
