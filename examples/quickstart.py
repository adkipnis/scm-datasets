from torch import nn

from scamd import generate_dataset
from scamd.utils import setSeed

setSeed(42)

x = generate_dataset(
    n_samples=1000,
    n_features=20,
    n_causes=12,
    cause_dist='mixed',
    n_layers=8,
    n_hidden=64,
    activation=nn.Tanh,
    blockwise=True,
)

print(x.shape)
