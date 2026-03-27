import pandas as pd
from matplotlib import pyplot as plt

from scamd import generate_dataset, plot_dataset
from scamd.utils import setSeed

# setSeed(42)

# Fast start: use a named preset shared by demos.
x = generate_dataset(
    n_samples=300,
    n_features=8,
    n_causes=12,
    n_layers=8,
    n_hidden=16,
    blockwise=False,
    preset='balanced_realistic',
)

# # Optional: custom configuration.
# x = generate_dataset(
#     n_samples=300,
#     n_features=20,
#     n_causes=12,
#     cause_dist='mixed',
#     n_layers=8,
#     n_hidden=64,
#     activation=nn.Tanh,
#     blockwise=True,
# )
# print('custom shape:', x.shape)

# Optional: visualize the first few features.
df = pd.DataFrame(x[:, :4], columns=[f'x{i + 1}' for i in range(4)])
plot_dataset(df, color='teal', title='scamd quickstart sample', kde=False)
plt.show()


# from tqdm import tqdm
# for _ in tqdm(range(10_000)):
#     x = generate_dataset(
#         n_samples=300,
#         n_features=5,
#         n_causes=6,
#         n_layers=8,
#         n_hidden=16,
#         blockwise=False,
#         preset='balanced_realistic',
#     )