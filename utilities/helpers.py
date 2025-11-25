from typing import Callable

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from chrome_dino_env import ChromeDinoEnv


def get_high_score():
    """Read the high score from score.txt file (human game mode use)."""
    try:
        with open("score.txt", "r") as f:
            content = f.read().strip()
            if content:
                score_ints = [int(x) for x in content.split()]
                return max(score_ints)
            else:
                return 0
    except (FileNotFoundError, ValueError):
        return 0


def make_env(rank: int, seed: int, **kwargs) -> Callable[[], gym.Env]:
    """
    Factory for vectorized environments. All curriculum/phase kwargs are passed through.
    Expected kwargs (all optional, with sensible defaults in ChromeDinoEnv):
      - frame_skip: int
      - speed_increases: bool
      - spawn_probs: tuple[float, float, float]
      - alive_reward, death_penalty, avoid_reward, milestone_points, milestone_bonus: floats/ints
      - duck_window_ttc: tuple[int, int]
      - duck_bonus, wrong_jump_penalty, idle_duck_penalty, airtime_penalty: floats
      - obs_speed_cap, obs_ttc_cap: floats
    """
    def thunk():
        env = ChromeDinoEnv(
            render_mode=None,
            frame_skip=kwargs.get("frame_skip", 1),
            speed_increases=kwargs.get("speed_increases", False),
            spawn_probs=kwargs.get("spawn_probs", (0.3, 0.2, 0.5)),
            alive_reward=kwargs.get("alive_reward", 0.05),
            death_penalty=kwargs.get("death_penalty", -1.0),
            avoid_reward=kwargs.get("avoid_reward", 0.0),
            milestone_points=kwargs.get("milestone_points", 0),
            milestone_bonus=kwargs.get("milestone_bonus", 0.0),
            duck_window_ttc=kwargs.get("duck_window_ttc", (6, 24)),
            duck_bonus=kwargs.get("duck_bonus", 0.0),
            wrong_jump_penalty=kwargs.get("wrong_jump_penalty", 0.0),
            idle_duck_penalty=kwargs.get("idle_duck_penalty", 0.0),
            airtime_penalty=kwargs.get("airtime_penalty", 0.0),
            obs_speed_cap=kwargs.get("obs_speed_cap", 100.0),
            obs_ttc_cap=kwargs.get("obs_ttc_cap", 300.0),
            seed=seed + rank,
        )
        env = RecordEpisodeStatistics(env, deque_size=1000)
        return env
    return thunk
