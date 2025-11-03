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

def make_env(
    rank: int, 
    seed: int, 
    render_mode=None, 
    speed_increases: bool = True,
    alive_reward: float = 0.1,
    death_penalty: float = -1.0,
    avoid_reward: float = 1.0,
    milestone_points: int = 10,
    milestone_bonus: float = 2.0
) -> Callable[[], gym.Env]:
    def _thunk():
        env = ChromeDinoEnv(
            render_mode=render_mode, 
            seed=seed + rank, 
            speed_increases=speed_increases,
            alive_reward=alive_reward,
            death_penalty=death_penalty,
            avoid_reward=avoid_reward,
            milestone_points=milestone_points,
            milestone_bonus=milestone_bonus
        )
        env = RecordEpisodeStatistics(env, deque_size=1000)
        return env
    return _thunk
