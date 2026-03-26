# scm-datasets

Generate realistic synthetic tabular datasets with a structural causal model (SCM).

`scm-datasets` builds latent causes, transforms them through deep nonlinear mechanisms, and applies optional post-hoc feature transformations (categorical/count/rank/binning) to create mixed-type, high-dependency feature matrices.

## Design at a glance

1. **Cause sampling** (`src/causes.py`)
   Sample root variables from configurable distributions.

2. **Structural mechanism** (`src/scm.py`)
   Pass causes through a deep noisy MLP with stochastic activations/weights.

3. **Post-hoc feature transforms** (`src/posthoc.py`)
   Inject realistic tabular effects (thresholding, quantiles, categorical/count outputs).

## Install

```bash
uv sync
```

Python `>=3.12` is required.

## Quickstart

```python
from torch import nn

from src.scm import SCM, Posthoc
from src.utils import setSeed

setSeed(42)

scm = SCM(
    n_samples=1000,
    n_features=20,
    n_causes=12,
    cause_dist="mixed",   # "normal" | "uniform" | "mixed"
    n_layers=8,
    n_hidden=64,
    activation=nn.Tanh,
    blockwise=True,
)

x_latent = scm.sample()      # torch.Tensor, shape: (1000, 20)
x = Posthoc(n_features=20)(x_latent)  # np.ndarray, standardized, shape: (1000, 20)
```

