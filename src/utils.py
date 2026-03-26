import random
import numpy as np
import torch


def setSeed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def truncLogUni(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...],
    add: float = 0.0,
    round: bool = False,
) -> np.ndarray:
    """sample from log uniform in [low, high) + {add} and optionally floor to integer"""
    assert 0 < low, 'lower bound must be positive'
    assert low <= high, 'lower bound smaller than upper bound'
    log_low = np.log(low)
    log_high = np.log(high)
    out = rng.uniform(log_low, log_high, size)
    out = np.exp(out) + add
    if round:
        out = np.floor(out).astype(int)
    return out
