# utilities/helpers.py
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
    """
    Factory that builds a ChromeDinoEnv with curriculum/shaping knobs.

    Known kwargs (all optional; sensible defaults set here):
      frame_skip:int=1
      speed_increases:bool=True
      alive_reward:float=0.05
      avoid_reward:float=0.0
      death_penalty:float=-1.0
      milestone_points:int=0
      milestone_bonus:float=0.0

      # Curriculum / spawn
      spawn_probs:tuple=(0.3,0.2,0.5)   # small, large, bird
      bird_only_phase:bool=False

      # Duck shaping
      duck_window_ttc:tuple=(6,24)
      duck_bonus:float=0.3
      wrong_jump_penalty:float=0.2
      idle_duck_penalty:float=0.01
      airtime_penalty:float=0.005

      # Observation normalization caps
      obs_speed_cap:float=100.0
      obs_ttc_cap:float=300.0
    """
    def thunk():
        env = ChromeDinoEnv(
            render_mode=None,
            frame_skip=kwargs.get("frame_skip", 1),
            speed_increases=kwargs.get("speed_increases", True),

            alive_reward=kwargs.get("alive_reward", 0.05),
            avoid_reward=kwargs.get("avoid_reward", 0.0),
            death_penalty=kwargs.get("death_penalty", -1.0),
            milestone_points=kwargs.get("milestone_points", 0),
            milestone_bonus=kwargs.get("milestone_bonus", 0.0),

            # curriculum & shaping
            spawn_probs=kwargs.get("spawn_probs", (0.3, 0.2, 0.5)),
            bird_only_phase=kwargs.get("bird_only_phase", False),
            duck_window_ttc=kwargs.get("duck_window_ttc", (6, 24)),
            duck_bonus=kwargs.get("duck_bonus", 0.3),
            wrong_jump_penalty=kwargs.get("wrong_jump_penalty", 0.2),
            idle_duck_penalty=kwargs.get("idle_duck_penalty", 0.01),
            airtime_penalty=kwargs.get("airtime_penalty", 0.005),

            # obs caps
            obs_speed_cap=kwargs.get("obs_speed_cap", 100.0),
            obs_ttc_cap=kwargs.get("obs_ttc_cap", 300.0),

            seed=seed + rank,
        )
        env = RecordEpisodeStatistics(env, deque_size=1000)
        return env
    return thunk

