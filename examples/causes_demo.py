"""Visualize cause samples across supported distributions."""

import matplotlib.pyplot as plt
import torch

from scamd.causes import CauseSampler
from scamd.utils import setSeed


def gridplot(
    x: torch.Tensor,
    nrow: int,
    ncol: int,
    figsize: tuple[int, int],
    title: str = '',
) -> None:
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = list(getattr(axes, 'flat', [axes]))
    n_causes = x.shape[-1]
    for i in range(n_causes):
        axes[i].hist(x[:, i], bins=24)
        axes[i].set_title(f'Cause {i}', size=10)
    for ax in axes[n_causes:]:
        ax.set_visible(False)
    plt.suptitle(title, size=18)
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(42)

    config = {
        'n_samples': 1000,
        'n_causes': 8,
        'dist': 'normal',
        'fixed': True,
    }

    # --- Normal distribution with fixed parameters
    x = CauseSampler(**config).sample()
    gridplot(
        x, nrow=3, ncol=3, figsize=(10, 8), title='Causes: Normal (Fixed)'
    )

    # --- Normal distribution with random parameters
    config.update({'fixed': False})
    x = CauseSampler(**config).sample()
    gridplot(
        x, nrow=3, ncol=3, figsize=(10, 8), title='Causes: Normal (Random)'
    )

    # --- Uniform distribution with random parameters
    config.update({'dist': 'uniform'})
    x = CauseSampler(**config).sample()
    gridplot(
        x, nrow=3, ncol=3, figsize=(10, 8), title='Causes: Uniform (Random)'
    )

    # --- Mixed distributions (normal, uniform, multinomial, zipf) with random parameters
    config.update({'dist': 'mixed', 'fixed': False})
    x = CauseSampler(**config).sample()
    gridplot(
        x, nrow=3, ncol=3, figsize=(10, 8), title='Causes: Mixed (Random)'
    )
