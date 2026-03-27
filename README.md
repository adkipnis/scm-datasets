# Structural Causal Model Datasets (scamd)

Generate realistic synthetic tabular datasets using structural causal models.

`scamd` builds latent causes, transforms them through deep nonlinear mechanisms, and applies optional post-hoc feature transformations (categorical/count/rank/binning) to create mixed-type, high-dependency feature matrices.

## Design at a glance

1. **Cause sampling** (`scamd/causes.py`)
   Sample root variables from configurable distributions.

2. **Structural mechanism** (`scamd/scm.py`)
   Pass causes through a deep noisy MLP with stochastic activations/weights.

3. **Post-hoc feature transforms** (`scamd/posthoc.py`)
   Inject realistic tabular effects (thresholding, quantiles, categorical/count outputs).

## Install

```bash
uv sync
uv pip install . -e
```

Python `>=3.12` is required.

Plotting examples use `pandas` and `seaborn`.

## Quickstart

```python
from torch import nn
import pandas as pd
from matplotlib import pyplot as plt

from scamd import generate_dataset, plot_dataset
from scamd.utils import setSeed

setSeed(42)

# Fast start: use a preset shared by demos.
x = generate_dataset(
    n_samples=1000,
    n_features=20,
    n_causes=12,
    n_layers=8,
    n_hidden=64,
    blockwise=True,
    preset="balanced_realistic",
)
print("preset shape:", x.shape)

# Optional: custom configuration.
x = generate_dataset(
    n_samples=1000,
    n_features=20,
    n_causes=12,
    cause_dist="mixed",   # "normal" | "uniform" | "mixed"
    n_layers=8,
    n_hidden=64,
    activation=nn.Tanh,
    blockwise=True,
)

print("custom shape:", x.shape)

# Optional: pair-grid visualization on a feature subset.
df = pd.DataFrame(x[:, :6], columns=[f"x{i+1}" for i in range(6)])
plot_dataset(df, color="teal", title="scamd quickstart sample", kde=True)
plt.show()

```

Presets only control behavior defaults (`pool_preset`, `p_posthoc`, `cause_dist`,
`fixed`). You should always set explicit SCM size and sampling knobs:
`n_samples`, `n_features`, `n_causes`, `n_layers`, `n_hidden`, and `blockwise`.

See `examples/quickstart.py` for a runnable script.

## Demos

- `python examples/pool_demo.py`: activation pool presets and sampled curves.
- `python examples/scm_demo.py`: dependence-spectrum presets + pair-grid examples.
- `python examples/posthoc_demo.py`: behavior of each post-hoc transform.
