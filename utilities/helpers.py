import gymnasium as gym

from typing import Callable

from gymnasium.wrappers import RecordEpisodeStatistics

from chrome_dino_env import ChromeDinoEnv


def get_high_score():
    """Read the high score from score.txt file"""
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

def make_env(rank, seed, **kwargs):
    def thunk():
        env = ChromeDinoEnv(
            render_mode=None,
            frame_skip=1,                  # <— important for learning
            alive_reward=kwargs.get("alive_reward", 0.05),
            avoid_reward=kwargs.get("avoid_reward", 1.0),
            death_penalty=kwargs.get("death_penalty", -1.0),
            milestone_points=kwargs.get("milestone_points", 0),
            milestone_bonus=kwargs.get("milestone_bonus", 0.0),
            speed_increases=kwargs.get("speed_increases", True),
            seed=seed + rank,
        )
        return env
    return thunk

