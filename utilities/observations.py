import random

import numpy as np
import torch

import torch.optim as optim
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ObsNorm:
    """
    Simple observation normalizer (running mean/var).
    """
    def __init__(self, shape, eps: float = 1e-8, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.clip = clip

    def update(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        tot = self.count + count
        new_mean = self.mean + delta * count / tot
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + delta**2 * self.count * count / tot
        new_var = M2 / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(x, -self.clip, self.clip)


