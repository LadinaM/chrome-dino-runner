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

class AssetClass:
    CLOUD: str = 'CLOUD'
    SMALL_CACTUS: str = 'SMALL_CACTUS'
    LARGE_CACTUS: str = 'LARGE_CACTUS'
    BIRD: str = 'BIRD'


class MovementType:
    RUNNING: str = 'RUNNING'
    JUMPING: str = 'JUMPING'
    DUCKING: str = 'DUCKING'