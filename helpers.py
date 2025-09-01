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
