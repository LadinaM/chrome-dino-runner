import datetime
import logging
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

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
    Chrome Dino Runner environment for reinforcement learning, with:
      - skill-shaped rewards to learn DUCK on low birds,
      - spawn probability control per obstacle type,
      - optional curriculum knobs passed at construction time,
      - normalized 10-D observation.

    Observation (float32, shape (10,)):
      [ y_n, vy_n, rel_x_n, rel_y_n, onehot_small, onehot_large, onehot_bird,
        speed_n, ttc_n, airborne ]
    Action (Discrete(3)): 0 noop, 1 jump, 2 duck
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        # rollout control
        max_episode_steps: int = 3000,
        frame_skip: int = 1,
        # environment dynamics
        speed_increases: bool = False,
        spawn_probs: Tuple[float, float, float] = (0.3, 0.2, 0.5),  # small, large, bird
        # reward shaping (base)
        alive_reward: float = 0.05,
        death_penalty: float = -1.0,
        avoid_reward: float = 0.0,
        milestone_points: int = 0,
        milestone_bonus: float = 0.0,
        # DUCK curriculum shaping
        duck_window_ttc: Tuple[int, int] = (6, 24),
        duck_bonus: float = 0.0,            # +reward if duck in window vs bird
        wrong_jump_penalty: float = 0.0,    # -reward if jump in window vs bird
        idle_duck_penalty: float = 0.0,     # -reward if duck w/o bird-in-window
        airtime_penalty: float = 0.0,       # -reward per frame airborne (discourages spam jumps)
        # obs normalization caps
        obs_speed_cap: float = 100.0,
        obs_ttc_cap: float = 300.0,
        seed: Optional[int] = None
    ):
        super().__init__()

        # -------------- Display & assets --------------
        self.render_mode = render_mode
        pygame.init()
        self._w, self._h = GameSettings.SCREEN_WIDTH, GameSettings.SCREEN_HEIGHT
        if self.render_mode == "human":
            self.SCREEN = pygame.display.set_mode((self._w, self._h))
            pygame.display.set_caption("Chrome Dino Runner - RL Environment")
        else:
            self.SCREEN = pygame.Surface((self._w, self._h))  # offscreen surface

        self.ASSETS = load_game_assets()
        if self.render_mode == "human" and "ICON" in self.ASSETS:
            try:
                pygame.display.set_icon(self.ASSETS['ICON'])
            except Exception as e:
                logger.warning(f"Could not set window icon: {e}")

        # -------------- Game state --------------
        self.game_state = GameState(speed_increases=speed_increases)
        self.player = Dinosaur(self.ASSETS["RUNNING"], self.ASSETS["JUMPING"], self.ASSETS["DUCKING"])
        self.clock = pygame.time.Clock()

        # -------------- RL spaces --------------
        self.action_space = spaces.Discrete(3)  # 0 noop, 1 jump, 2 duck
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # -------------- Config / rewards --------------
        self._max_episode_steps = int(max_episode_steps)
        self._frame_skip = int(frame_skip)

        self._alive_reward = float(alive_reward)
        self._death_penalty = float(death_penalty)
        self._avoid_reward = float(avoid_reward)
        self._milestone_points = int(milestone_points)
        self._milestone_bonus = float(milestone_bonus)

        # DUCK curriculum shaping
        self._duck_window = (int(duck_window_ttc[0]), int(duck_window_ttc[1]))  # inclusive window in frames
        self._duck_bonus = float(duck_bonus)
        self._wrong_jump_penalty = float(wrong_jump_penalty)
        self._idle_duck_penalty = float(idle_duck_penalty)
        self._airtime_penalty = float(airtime_penalty)

        # spawn probabilities
        sp = np.array(spawn_probs, dtype=np.float64)
        sp = np.clip(sp, 0.0, None)
        self._spawn_probs = (sp / sp.sum()).tolist() if sp.sum() > 0 else [1.0, 0.0, 0.0]

        # obs caps
        self._obs_speed_cap = float(obs_speed_cap)
        self._obs_ttc_cap = float(obs_ttc_cap)

        # -------------- Bookkeeping --------------
        self._steps = 0
        self._prev_dino_y = float(self.player.dino_rect.y)  # for vy
        self._avoided_this_episode = 0
        self._avoided_bird = 0
        self._avoided_other = 0
        self._avoid_reward_accum = 0.0
        self._duck_skill = {"window": 0, "hits": 0}  # for skill-gated curriculum

        # -------------- RNG --------------
        self.np_random, _ = seeding.np_random(seed)
        if seed is not None:
            random.seed(seed)

    # ---------- Observations & info ----------
    def _nearest_obstacle(self, dino_x: float):
        """Return the closest obstacle ahead of the dino, if any."""
        ahead = [o for o in self.game_state.obstacles if o.rect.x >= dino_x]
        return min(ahead, key=lambda o: o.rect.x, default=None)

    def _get_obs(self) -> np.ndarray:
        """
        10-D normalized observation vector.
        """
        dino_rect = self.player.dino_rect
        y = float(dino_rect.y)

        # vy: positive when moving up
        vy = self._prev_dino_y - y
        self._prev_dino_y = y

        # nearest obstacle
        dino_x = float(dino_rect.x)
        nearest = self._nearest_obstacle(dino_x)

        if nearest is not None:
            rel_x = float(nearest.rect.x - dino_x)
            rel_y = float(nearest.rect.y - dino_rect.y)
            if isinstance(nearest, SmallCactus):
                t = 0
            elif isinstance(nearest, LargeCactus):
                t = 1
            else:
                t = 2
        else:
            rel_x, rel_y, t = float(self._w), 0.0, 0

        onehot = np.zeros(3, dtype=np.float32)
        onehot[t] = 1.0

        # normalize
        y_n = y / float(self._h)
        vy_n = vy / float(self._h)
        rel_x_n = float(np.clip(rel_x / float(self._w), 0.0, 1.0))
        rel_y_n = float(np.clip(rel_y / float(self._h), -1.0, 1.0))

        speed = float(self.game_state.game_speed)
        speed_n = float(np.clip(speed / self._obs_speed_cap, 0.0, 1.0))

        eps = 1e-6
        ttc = (rel_x / max(speed, eps)) if nearest is not None and speed > eps else self._obs_ttc_cap
        ttc_n = float(np.clip(ttc / self._obs_ttc_cap, 0.0, 1.0))

        airborne = 1.0 if self._is_airborne() else 0.0

        return np.array(
            [y_n, vy_n, rel_x_n, rel_y_n, onehot[0], onehot[1], onehot[2], speed_n, ttc_n, airborne],
            dtype=np.float32
        )

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info about the current game state."""
        return {
            'score': int(self.game_state.points),
            'game_speed': int(self.game_state.game_speed),
            'obstacles_count': int(len(self.game_state.obstacles))
        }

    # ---------- Gym API ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
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
        self._avoided_bird = 0
        self._avoided_other = 0
        self._avoid_reward_accum = 0.0
        self._duck_skill = {"window": 0, "hits": 0}

        self._fill_bg()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step with the given action."""
        reward = 0.0
        terminated = False
        truncated = False

        # cache nearest before frameskip (for shaping window)
        dino_x = float(self.player.dino_rect.x)
        nearest = self._nearest_obstacle(dino_x)
        is_bird = isinstance(nearest, Bird) if nearest is not None else False

        # compute TTC (frames) for shaping window
        speed = max(float(self.game_state.game_speed), 1e-6)
        rel_x = float(nearest.rect.x - dino_x) if nearest is not None else self._w
        ttc_frames = rel_x / speed
        t_low, t_high = self._duck_window
        in_duck_window = is_bird and (t_low <= ttc_frames <= t_high)

        grounded = not self._is_airborne()

        # --- DUCK skill accounting (per-step window counters) ---
        if in_duck_window:
            self._duck_skill["window"] += 1
            if action == 2 and grounded:
                self._duck_skill["hits"] += 1

        # apply action once
        self._apply_action(action)

        # frame skip loop
        for _ in range(self._frame_skip):
            self._update_player()
            self._spawn_and_update_obstacles()

            # collision
            if self._check_collision():
                terminated = True
                reward += self._death_penalty
                break

            # avoidance reward (count once per obstacle)
            for ob in self.game_state.obstacles:
                passed = (ob.rect.x + ob.rect.width) < dino_x
                already_counted = getattr(ob, "_passed_counted", False)
                if passed and not already_counted:
                    setattr(ob, "_passed_counted", True)
                    # base avoid reward
                    reward += self._avoid_reward
                    self._avoid_reward_accum += self._avoid_reward
                    self._avoided_this_episode += 1
                    # bird vs non-bird stats
                    if isinstance(ob, Bird):
                        self._avoided_bird += 1
                    else:
                        self._avoided_other += 1

            # time-based rewards
            prev_points = self.game_state.points
            self.game_state.update_score()

            if self._milestone_points > 0 and self.game_state.points // self._milestone_points > prev_points // self._milestone_points:
                reward += self._milestone_bonus

            reward += self._alive_reward

            # Shaping rewards: encourage ducking in the bird window, discourage idle ducks, and penalize airtime.
            # 1) in duck window vs bird
            if in_duck_window:
                if action == 2 and grounded:
                    reward += self._duck_bonus
                elif action == 1:  # jumped instead of duck
                    reward -= self._wrong_jump_penalty

            # 2) idle-duck penalty (ducking when not required)
            if action == 2 and not in_duck_window and grounded:
                reward -= self._idle_duck_penalty

            # 3) airtime penalty to discourage jump spam
            if self._is_airborne():
                reward -= self._airtime_penalty

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
            info["avoided_bird"] = int(self._avoided_bird)
            info["avoided_other"] = int(self._avoided_other)
            info["avoid_reward_total"] = float(self._avoid_reward_accum)
            info["bird_window_total"] = int(self._duck_skill["window"])
            info["bird_duck_hits"] = int(self._duck_skill["hits"])
            # reset for next episode (in case env is reused)
            self._duck_skill = {"window": 0, "hits": 0}
            self._avoided_this_episode = 0
            self._avoided_bird = 0
            self._avoided_other = 0
            self._avoid_reward_accum = 0.0

        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------- Game helpers ----------------
    def render(self):
        """Render the current frame and return an array if rgb_array mode."""
        if self.render_mode == "human":
            self._render_frame()
            self._pump_events()
        elif self.render_mode == "rgb_array":
            self._render_frame()
            arr = pygame.surfarray.array3d(self.SCREEN)  # (W, H, 3)
            return np.transpose(arr, (1, 0, 2))  # (H, W, 3)

    def close(self):
        """Close pygame resources."""
        pygame.quit()

    # ---------- Internals ----------
    def _apply_action(self, action: int) -> None:
        """Map an integer action to player controls."""
        grounded = not self._is_airborne()
        if action == 1 and grounded:
            self.player.jump()
            self.player.release_duck()
        elif action == 2 and grounded:
            self.player.duck()
        else:
            self.player.release_duck()

    def _spawn_and_update_obstacles(self) -> None:
        """Spawn obstacles as needed and advance all active obstacles."""
        # spawn if none, according to probabilities
        if len(self.game_state.obstacles) == 0:
            r = self.np_random.random()
            cumulative = np.cumsum(self._spawn_probs)
            if r < cumulative[0]:
                self.game_state.obstacles.append(SmallCactus(self.ASSETS["SMALL_CACTUS"]))
            elif r < cumulative[1]:
                self.game_state.obstacles.append(LargeCactus(self.ASSETS["LARGE_CACTUS"]))
            else:
                self.game_state.obstacles.append(Bird(self.ASSETS["BIRD"]))

        # update obstacles
        for ob in list(self.game_state.obstacles):
            ob.update(self.game_state.game_speed, self.game_state.obstacles)
            if ob.rect.x < -ob.rect.width and ob in self.game_state.obstacles:
                self.game_state.obstacles.remove(ob)

        # background scroll
        self._update_background()

    def _update_player(self) -> None:
        """Advance the player animation and physics without keyboard input."""
        try:
            self.player.update(None)
        except TypeError:
            class _NullKeys:
                def __getitem__(self, key: int) -> bool:
                    return False
            self.player.update(_NullKeys())

    def _update_background(self) -> None:
        """Scroll the background and reset when it wraps."""
        bg = self.ASSETS["BG"]
        w = bg.get_width()
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg, GameSettings.Y_POS_BG))
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg + w, GameSettings.Y_POS_BG))
        if self.game_state.x_pos_bg <= -w:
            self.game_state.x_pos_bg = 0
        self.game_state.x_pos_bg -= self.game_state.game_speed

    def _fill_bg(self) -> None:
        """Fill the background based on day/night for render output."""
        current_hour = datetime.datetime.now().hour
        color = (255, 255, 255) if 7 < current_hour < 19 else (0, 0, 0)
        self.SCREEN.fill(color)

    def _render_frame(self):
        """Draw all visible elements to the screen surface."""
        self._fill_bg()
        bg = self.ASSETS["BG"]
        w = bg.get_width()
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg, GameSettings.Y_POS_BG))
        self.SCREEN.blit(bg, (self.game_state.x_pos_bg + w, GameSettings.Y_POS_BG))
        self.player.draw(self.SCREEN)
        for obstacle in self.game_state.obstacles:
            obstacle.draw(self.SCREEN)
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _check_collision(self):
        """Return True if the player collides with any obstacle."""
        dino = self.player.dino_rect
        for ob in self.game_state.obstacles:
            if dino.colliderect(ob.rect):
                return True
        return False

    def _is_airborne(self):
        """Return True if the player is not on the ground."""
        return self.player.dino_rect.y < Dinosaur.Y_POS

    def _pump_events(self):
        """Process pending pygame events and exit cleanly on quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
