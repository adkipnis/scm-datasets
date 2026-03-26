"""Run deterministic and stochastic post-hoc transformations.

This script applies each post-hoc layer to random data and prints correlation summaries.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posthoc import (
    Categorical,
    Geometric,
    MultiThreshold,
    NegativeBinomial,
    Poisson,
    QuantileBins,
    Rank,
    Threshold,
)
from src.utils import setSeed


def test(model, x, statistic: str = 'pearson') -> None:
    y = torch.cat([model(x), model(x)], dim=-1)
    corr_sum = 0
    for i in range(len(y)):
        if statistic == 'pearson':
            corr = np.corrcoef(y[i], rowvar=False)
        elif statistic == 'spearman':
            corr = spearmanr(y[i], axis=0)[0]
        else:
            raise ValueError('unknown statistic')
        corr_sum += corr
    corr_mean = corr_sum / len(y)
    unique = corr_mean[np.triu_indices_from(corr_mean, k=1)]
    print(unique)


if __name__ == '__main__':
    setSeed(1)

    batch, n, d = 64, 100, 8
    x = torch.randn((batch, n, d)) * 3

    _ = Threshold(d, 2)(x)
    _ = MultiThreshold(d, 1, levels=3)(x)
    _ = QuantileBins(d, 2, levels=2)(x)
    _ = Rank(d, 5)(x)

    test(Categorical(d, 1), x)
    test(Categorical(d, 2), x)
    test(Poisson(d, 1), x)
    test(Geometric(d, 1), x)
    test(NegativeBinomial(d, 1), x)
