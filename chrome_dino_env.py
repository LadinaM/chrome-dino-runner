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

        # Base rewards
        alive_reward: float = 0.1,
        death_penalty: float = -1.0,
        avoid_reward: float = 0.0,   # (start at 0.0; you can raise later)

        # Milestones (optional)
        milestone_points: int = 0,
        milestone_bonus: float = 0.0,

        # Game dynamics
        speed_increases: bool = False,

        # Duck learning aids
        spawn_probs: tuple = (0.3, 0.2, 0.5),  # small, large, bird
        bird_only_phase: bool = False,         # curriculum: spawn only birds
        duck_window_ttc: tuple = (6, 24),      # frames window near bird
        duck_bonus: float = 0.3,               # +r if duck within window
        wrong_jump_penalty: float = 0.2,       # -r if jump within window
        idle_duck_penalty: float = 0.01,       # -r per frame ducking w/out need
        airtime_penalty: float = 0.005,        # -r per frame airborne w/out need

        # Obs normalization caps (can be overridden by trainer)
        obs_speed_cap: float = 100.0,
        obs_ttc_cap: float = 300.0,

        seed: Optional[int] = None,
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
            # Offscreen surface (width, height)
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
        # Observation shape: [y_n, vy_n, rel_x_n, rel_y_n, onehot(3), speed_n, ttc_n, airborne]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Rewards & rollout control
        self._alive_reward = float(alive_reward)
        self._death_penalty = float(death_penalty)
        self._avoid_reward = float(avoid_reward)
        self._milestone_points = int(milestone_points)
        self._milestone_bonus = float(milestone_bonus)
        self._max_episode_steps = int(max_episode_steps)
        self._frame_skip = int(frame_skip)

        # Duck curriculum + shaping
        self._spawn_probs = np.array(spawn_probs, dtype=np.float64)
        self._spawn_probs = self._spawn_probs / self._spawn_probs.sum()
        self._bird_only_phase = bool(bird_only_phase)
        self._duck_ttc_lo, self._duck_ttc_hi = duck_window_ttc
        self._duck_bonus = float(duck_bonus)
        self._wrong_jump_penalty = float(wrong_jump_penalty)
        self._idle_duck_penalty = float(idle_duck_penalty)
        self._airtime_penalty = float(airtime_penalty)

        # Observation caps
        self._obs_speed_cap = float(obs_speed_cap)
        self._obs_ttc_cap = float(obs_ttc_cap)

        # Bookkeeping
        self._steps = 0
        self._prev_dino_y = float(self.player.dino_rect.y)  # for vy finite diff
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

        # normalization by screen size (keeps values ~[-1,1])
        y_n     = y  / float(self._h)
        vy_n    = vy / float(self._h)
        rel_x_n = float(np.clip(rel_x / float(self._w), 0.0, 1.0))
        rel_y_n = float(np.clip(rel_y / float(self._h), -1.0, 1.0))

        # normalized speed
        speed = float(self.game_state.game_speed)
        speed_n = float(np.clip(speed / self._obs_speed_cap, 0.0, 1.0))

        # time-to-collision (frames). If no obstacle, use max cap.
        eps = 1e-6
        if nearest is not None and speed > eps:
            ttc = rel_x / max(speed, eps)
        else:
            ttc = self._obs_ttc_cap
        ttc_n = float(np.clip(ttc / self._obs_ttc_cap, 0.0, 1.0))

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
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            random.seed(seed)

        # Reset game state & player
        self.game_state.reset()
        self.player = Dinosaur(self.ASSETS["RUNNING"], self.ASSETS["JUMPING"], self.ASSETS["DUCKING"])
        self._steps = 0
        self._prev_dino_y = float(self.player.dino_rect.y)
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
            for ob in self.game_state.obstacles:
                passed = (ob.rect.x + ob.rect.width) < float(self.player.dino_rect.x)
                already_counted = getattr(ob, "_passed_counted", False)
                if passed and not already_counted:
                    setattr(ob, "_passed_counted", True)
                    reward += self._avoid_reward
                    self._avoided_this_episode += 1

            # ---- contextual duck shaping ----
            rel_x_b, rel_y_b, ttc_b = self._nearest_bird_info()
            if ttc_b is not None:
                if self._duck_ttc_lo <= ttc_b <= self._duck_ttc_hi:
                    if action == 2 and not self._is_airborne():
                        reward += self._duck_bonus
                    elif action == 1:
                        reward -= self._wrong_jump_penalty

            # gentle regularizers to avoid degenerate policies
            need_duck = (ttc_b is not None) and (self._duck_ttc_lo <= ttc_b <= self._duck_ttc_hi)
            if self._is_airborne() and not need_duck:
                reward -= self._airtime_penalty
            if (getattr(self.player, "dino_duck", False)) and not need_duck:
                reward -= self._idle_duck_penalty

            # score and speed
            prev_points = self.game_state.points
            self.game_state.update_score()
            if self._milestone_points > 0 and self.game_state.points // self._milestone_points > prev_points // self._milestone_points:
                reward += self._milestone_bonus

            reward += self._alive_reward

            if self.render_mode == "human":
                self._render_frame()
                self._pump_events()

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
            # draw fresh frame before capture
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
            if self._bird_only_phase:
                typ = 2
            else:
                typ = int(self.np_random.choice([0, 1, 2], p=self._spawn_probs))

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

    def _nearest_bird_info(self):
        """Return (rel_x, rel_y, ttc_frames) for the nearest bird ahead or (None, None, None)."""
        dino_x = float(self.player.dino_rect.x)
        ahead_birds = [o for o in self.game_state.obstacles
                       if isinstance(o, Bird) and o.rect.x >= dino_x]
        if not ahead_birds:
            return None, None, None
        nearest = min(ahead_birds, key=lambda o: o.rect.x)
        rel_x = float(nearest.rect.x - dino_x)
        rel_y = float(nearest.rect.y - self.player.dino_rect.y)
        speed = max(float(self.game_state.game_speed), 1e-6)
        ttc = rel_x / speed
        return rel_x, rel_y, ttc

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
