import gymnasium as gym
import numpy as np
import pygame
import random
import datetime
import logging
from gymnasium.utils import seeding
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# Game bits
from figures.dinosaur import Dinosaur
from figures.obstacles import SmallCactus, LargeCactus, Bird
from figures.configurations import load_game_assets

# Shared game settings
from game import GameSettings, GameState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("pygame").setLevel(logging.INFO)


class ChromeDinoEnv(gym.Env):
    """
    Chrome Dino Runner environment for reinforcement learning
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            max_episode_steps: int = 3000,
            frame_skip: int = 2,
            alive_reward: float = 0.1,
            milestone_points: int = 0,
            milestone_bonus: float = 0.0,  # small bonus
            death_penalty: float = -1.0,
            avoid_reward: float = 1.0,  # Reward for successfully avoiding an obstacle
            speed_increases: bool = False,
            seed: Optional[int] = None
    ):
        super().__init__()

        self.render_mode = render_mode

        # Initialize pygame
        pygame.init()
        self._w, self._h = GameSettings.SCREEN_WIDTH, GameSettings.SCREEN_HEIGHT

        if self.render_mode == "human":
            self.SCREEN = pygame.display.set_mode((self._w, self._h))
            pygame.display.set_caption("Chrome Dino Runner - RL Environment")
        else:
            # Offscreen surface (width, height) — not (height, width)
            self.SCREEN = pygame.Surface((self._w, self._h))

        # Load game assets
        self.ASSETS = load_game_assets()
        if self.render_mode == "human" and "ICON" in self.ASSETS:
            try:
                pygame.display.set_icon(self.ASSETS['ICON'])
            except Exception as e:
                logger.warning(f"Could not set window icon: {e}")

        # Initialize game state and player
        self.game_state = GameState(speed_increases=speed_increases)
        self.player = Dinosaur(self.ASSETS["RUNNING"], self.ASSETS["JUMPING"], self.ASSETS["DUCKING"])

        # Clock
        self.clock = pygame.time.Clock()

        # ----- RL spaces -----
        self.action_space = spaces.Discrete(3)  # 0 noop, 1 jump, 2 duck
        # Observation shape: [y, vy, rel_x, rel_y, onehot(3), speed, ttc, airborne]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Rewards & rollout control
        self._alive_reward = float(alive_reward)
        self._death_penalty = float(death_penalty)
        self._avoid_reward = float(avoid_reward)
        self._milestone_points = int(milestone_points)
        self._milestone_bonus = float(milestone_bonus)
        self._max_episode_steps = int(max_episode_steps)
        self._frame_skip = int(frame_skip)

        # Bookkeeping
        self._steps = 0
        self._prev_dino_y = float(self.player.dino_rect.y)  # for vy finite diff
        self._prev_obstacle_ids = set()  # Track obstacles from previous step for avoidance rewards
        self._avoided_this_episode = 0  # Count obstacles the dino successfully passed

        # Seeding
        self.np_random, _ = seeding.np_random(seed)
        if seed is not None:
            random.seed(seed)

    # ---------- Observations & info ----------
    def _get_obs(self) -> np.ndarray:
        """
        Build a normalized 10-D observation vector:
        [ y_n, vy_n, rel_x_n, rel_y_n, onehot_s, onehot_l, onehot_b, speed_n, ttc_n, airborne ]

        where:
        y_n      : dino vertical position / screen height  (0..1)
        vy_n     : vertical velocity (positive = moving up) / screen height
        rel_x_n  : horizontal distance to nearest obstacle / screen width (0..1)
        rel_y_n  : vertical offset to nearest obstacle / screen height (-1..1)
        onehot_* : obstacle type one-hot (small cactus, large cactus, bird)
        speed_n  : game speed normalized by a cap (default 100)
        ttc_n    : time-to-collision (frames) normalized by a cap (default 300)
        airborne : 1.0 if dino is airborne else 0.0
        """
        dino_rect = self.player.dino_rect
        y = float(dino_rect.y)

        # vy: positive when moving UP
        vy = self._prev_dino_y - y
        self._prev_dino_y = y

        # pick nearest obstacle ahead (x >= dino_x)
        dino_x = float(dino_rect.x)
        ahead = [o for o in self.game_state.obstacles if o.rect.x >= dino_x]
        nearest = min(ahead, key=lambda o: o.rect.x, default=None)

        if nearest is not None:
            rel_x = float(nearest.rect.x - dino_x)
            rel_y = float(nearest.rect.y - dino_rect.y)
            # type: 0 small cactus, 1 large cactus, 2 bird
            if isinstance(nearest, SmallCactus):
                t = 0
            elif isinstance(nearest, LargeCactus):
                t = 1
            else:
                t = 2
        else:
            # no obstacle: treat as far away and level with dino
            rel_x, rel_y, t = float(self._w), 0.0, 0

        # one-hot for obstacle type
        onehot = np.zeros(3, dtype=np.float32)
        onehot[t] = 1.0

        # configurable caps (fallback to sensible defaults if not set on self)
        speed_cap = getattr(self, "_obs_speed_cap", 100.0)   # denominator for speed_n
        ttc_cap   = getattr(self, "_obs_ttc_cap",   300.0)   # denominator for ttc_n

        # normalization by screen size (keeps values ~[-1,1])
        y_n     = y  / float(self._h)
        vy_n    = vy / float(self._h)
        rel_x_n = float(np.clip(rel_x / float(self._w), 0.0, 1.0))
        rel_y_n = float(np.clip(rel_y / float(self._h), -1.0, 1.0))

        # normalized speed
        speed = float(self.game_state.game_speed)
        speed_n = float(np.clip(speed / speed_cap, 0.0, 1.0))

        # time-to-collision (frames). If no obstacle, use max cap.
        # add small eps to avoid division by zero at very low speeds
        eps = 1e-6
        if nearest is not None and speed > eps:
            ttc = rel_x / max(speed, eps)
        else:
            ttc = ttc_cap
        ttc_n = float(np.clip(ttc / ttc_cap, 0.0, 1.0))

        airborne = 1.0 if self._is_airborne() else 0.0

        obs = np.array(
            [y_n, vy_n, rel_x_n, rel_y_n, onehot[0], onehot[1], onehot[2], speed_n, ttc_n, airborne],
            dtype=np.float32
        )
        return obs



    def _get_info(self) -> Dict[str, Any]:
        return {
            'score': int(self.game_state.points),
            'game_speed': int(self.game_state.game_speed),
            'obstacles_count': int(len(self.game_state.obstacles))
        }

    # ---------- Gym API ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            random.seed(seed)

        # Reset game state & player
        self.game_state.reset()
        self.player = Dinosaur(self.ASSETS["RUNNING"], self.ASSETS["JUMPING"], self.ASSETS["DUCKING"])
        self._steps = 0
        self._prev_dino_y = float(self.player.dino_rect.y)
        self._prev_obstacle_ids = set()
        self._avoided_this_episode = 0

        # Optional: clear screen for human mode
        self._fill_bg()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = False

        # apply action intent once, then update multiple frames (frame skip)
        self._apply_action(action)

        # Track obstacles at start of step (before any updates)
        dino_x = float(self.player.dino_rect.x)
        obstacles_ahead_at_start = {id(o): o.rect.x for o in self.game_state.obstacles if o.rect.x >= dino_x}
        
        for _ in range(self._frame_skip):
            # update player
            self._update_player()

            # spawn/update obstacles
            self._spawn_and_update_obstacles()

            # collision
            if self._check_collision():
                terminated = True
                reward += self._death_penalty
                break

            # Reward for successfully avoiding obstacles (passed behind dino)
            # Use a per-obstacle flag to ensure we count each obstacle once.
            for ob in self.game_state.obstacles:
                passed = (ob.rect.x + ob.rect.width) < dino_x
                already_counted = getattr(ob, "_passed_counted", False)
                if passed and not already_counted:
                    setattr(ob, "_passed_counted", True)
                    reward += self._avoid_reward
                    self._avoided_this_episode += 1

            # score and speed
            prev_points = self.game_state.points
            self.game_state.update_score()
            if self._milestone_points > 0 and self.game_state.points // self._milestone_points > prev_points // self._milestone_points:
                reward += self._milestone_bonus

            reward += self._alive_reward

            if self.render_mode == "human":
                self._render_frame()
                self._pump_events()
        
        # Update obstacle tracking for next step (after all frame skips)
        dino_x_final = float(self.player.dino_rect.x)
        self._prev_obstacle_ids = {id(o) for o in self.game_state.obstacles if o.rect.x >= dino_x_final}

        # horizon truncation
        self._steps += 1
        if not terminated and self._steps >= self._max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["episode_length"] = self._steps
        if terminated or truncated:
            info["avoided_count"] = int(self._avoided_this_episode)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------- Game ----------------
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
            self._pump_events()
        elif self.render_mode == "rgb_array":
            # Draw a fresh frame before capturing to avoid visual artifacts
            self._render_frame()
            arr = pygame.surfarray.array3d(self.SCREEN)  # (W, H, 3)
            return np.transpose(arr, (1, 0, 2))  # (H, W, 3)

    def close(self):
        pygame.quit()

    # ---------- Internals ----------
    def _apply_action(self, action: int) -> None:
        grounded = not self._is_airborne()

        # Prefer direct hooks if present
        has_jump = hasattr(self.player, "jump") and callable(self.player.jump)
        has_duck = hasattr(self.player, "duck") and callable(self.player.duck)
        has_rel  = hasattr(self.player, "release_duck") and callable(getattr(self.player, "release_duck", None))

        if has_jump and has_duck and has_rel:
            if action == 1 and grounded:
                self.player.jump()                # jump only from ground
                self.player.release_duck()        # ensure not ducking while initiating jump
            elif action == 2 and grounded:
                self.player.duck()                # duck only on ground
            else:
                self.player.release_duck()        # default: stand / keep jumping physics to update
            return

        # Fallback: emulate key state for Dinosaur.update(userInput)
        class _KeyProxy:
            def __init__(self, up: bool, down: bool, space: bool):
                self._u, self._d, self._s = up, down, space
            def __getitem__(self, key: int) -> bool:
                if key == pygame.K_UP:    return self._u
                if key == pygame.K_DOWN:  return self._d
                if key == pygame.K_SPACE: return self._s
                return False

        up    = (action == 1) and grounded
        down  = (action == 2) and grounded
        space = up  # typical mapping: jump = up/space

        self.player.update(_KeyProxy(up, down, space))


    def _spawn_and_update_obstacles(self) -> None:
        # spawn if none
        if len(self.game_state.obstacles) == 0:
            typ = self.np_random.integers(0, 3)
            if typ == 0:
                self.game_state.obstacles.append(SmallCactus(self.ASSETS["SMALL_CACTUS"]))
            elif typ == 1:
                self.game_state.obstacles.append(LargeCactus(self.ASSETS["LARGE_CACTUS"]))
            else:
                self.game_state.obstacles.append(Bird(self.ASSETS["BIRD"]))

        # update obstacles (and remove off-screen safely)
        for ob in list(self.game_state.obstacles):
            ob.update(self.game_state.game_speed, self.game_state.obstacles)
            if ob.rect.x < -ob.rect.width and ob in self.game_state.obstacles:
                self.game_state.obstacles.remove(ob)

        # background scroll
        self._update_background()

    def _update_player(self) -> None:
        try:
            self.player.update(None)  # allow None if you modified Dinosaur.update
        except TypeError:
            class _NullKeys:
                def __getitem__(self, key: int) -> bool:
                    return False

            self.player.update(_NullKeys())

    def _update_background(self) -> None:
        bg = self.ASSETS["BG"]
        w = bg.get_width()
        # draw twice for wrap-around
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg, GameSettings.Y_POS_BG))
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg + w, GameSettings.Y_POS_BG))
        # wrap
        if self.game_state.x_pos_bg <= -w:
            self.game_state.x_pos_bg = 0
        self.game_state.x_pos_bg -= self.game_state.game_speed

    def _fill_bg(self) -> None:
        current_hour = datetime.datetime.now().hour
        color = (255, 255, 255) if 7 < current_hour < 19 else (0, 0, 0)
        self.SCREEN.fill(color)

    def _render_frame(self):
        self._fill_bg()
        # Draw the ground background
        bg = self.ASSETS["BG"]
        w = bg.get_width()
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg, GameSettings.Y_POS_BG))
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg + w, GameSettings.Y_POS_BG))
        # Draw player and obstacles
        self.player.draw(self.SCREEN)
        for obstacle in self.game_state.obstacles:
            obstacle.draw(self.SCREEN)
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _check_collision(self):
        dino = self.player.dino_rect
        for ob in self.game_state.obstacles:
            if dino.colliderect(ob.rect):
                return True
        return False

    def _is_airborne(self):
        # grounded if y is at either Y_POS or Y_POS_DUCK
        return self.player.dino_rect.y < Dinosaur.Y_POS

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
