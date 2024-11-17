import pyro

from namgcv.basemodels.bnam import BayesianNN
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import torch
import numpy as np

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange

import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(
    x_true: np.ndarray or torch.Tensor,
    y_true: np.ndarray or torch.Tensor,
    x_obs: np.ndarray,
    y_obs: np.ndarray or torch.Tensor,
    x_test: np.ndarray or torch.Tensor,
    y_pred: np.ndarray or torch.Tensor,
    y_std: np.ndarray or torch.Tensor,
    xlims: list = [-0.5, 1.5],
    ylims: list = [-1.5, 2.5]
):
    GREEN_RGB_COLORS = [
        '#004c00',  # '#004e00', '#005000', '#005100', '#005300',
        # '#005500', # '#005700', '#005900', '#005a00', '#005c00',
        '#005e00',  # '#006000', '#006200', '#006300', '#006500',
        # '#006700', # '#006900', '#006b00', '#006c00', '#006e00',
        '#007000',  # '#007200', '#007400', '#007500', '#007700',
        # '#007900', # '#007b00', '#007d00', '#007e00', '#008000',
        '#008200',  # '#008400', '#008600', '#008800', '#008900',
        # '#008b00', # '#008d00', '#008f00', '#009100', '#009200',
        '#009400',  # '#009600', '#009800', '#009a00', '#009b00',
        # '#009d00', # '#009f00', '#00a100', '#00a300', '#00a400',
        '#00a600',  # '#00a800', '#00aa00', '#00ac00', '#00ad00',
        # '#00af00', # '#00b100', '#00b300', '#00b500', '#00b600',
        '#00b800',  # '#00ba00', '#00bc00', '#00be00', '#00bf00',
        # '#00c100', # '#00c300', '#00c500', '#00c700', '#00c800',
        '#00ca00',  # '#00cc00', '#00ce00', '#00d000', '#00d100',
        # '#00d300', # '#00d500', '#00d700', '#00d900', '#00da00',
        '#00dc00',  # '#00de00', '#00e000', '#00e200', '#00e300',
        # '#00e500', # '#00e700', '#00e900', '#00eb00', '#00ec00',
        '#00ee00',  # '#00f000', '#00f200', '#00f400', '#00f500',
        # '#00f700', # '#00f900', '#00fb00', '#00fd00', '#00ff00'
    ]

    sns.set_style("whitegrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    i = 0
    if x_true is not None and y_true is not None:
        sns.lineplot(
            x=x_true, y=y_true,
            color=GREEN_RGB_COLORS[i],
            label="True DGP",
            linewidth=1,
            ax=ax
        )
        i += 1

    if x_obs is not None and y_obs is not None:
        sns.scatterplot(
            x=x_obs, y=y_obs,
            color=GREEN_RGB_COLORS[i],
            label="Observed Data",
            linewidth=0.5,
            ax=ax
        )
        i += 1

    if x_test is not None and y_pred is not None and y_std is not None:
        sns.lineplot(
            x=x_test, y=y_pred,
            color=GREEN_RGB_COLORS[i],
            label="Predictive Posterior",
            linewidth=1,
            ax=ax
        )
        i += 1
        ax.fill_between(
            x_test,
            y_pred - 2 * y_std,
            y_pred + 2 * y_std,
            alpha=0.2,
            color=GREEN_RGB_COLORS[-1],
            zorder=5,
            label="95% Prediction Interval"
        )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc=4, fontsize=12, frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    pyro.set_rng_seed(42)

    # Generate some synthetic data.
    x_obs = np.hstack(
        [
            np.linspace(start=-0.2, stop=0.2, num=100),
            np.linspace(start=0.6, stop=1, num=100)
        ]
    )
    noise = 0.02 * np.random.randn(x_obs.shape[0])
    y_obs = (
        x_obs + 0.5 * np.sin(2 * np.pi * (x_obs + noise))
        +
        0.5 * np.cos(4 * np.pi * (x_obs + noise))
        +
        noise
    )
    x_true = np.linspace(-0.5, 1.5, 1000)
    y_true = (
        x_true + 0.5 * np.sin(2 * np.pi * x_true)
        +
        0.5 * np.cos(4 * np.pi * x_true)
    )

    xlims = [-0.5, 1.5]
    ylims = [-1.5, 2.5]

    plot_data(
        x_true=x_true, y_true=y_true,
        x_obs=x_obs, y_obs=y_obs,
        x_test=None, y_pred=None, y_std=None,
        xlims=xlims, ylims=ylims
    )

    model = BayesianNN(
        in_dim=1,
        out_dim=1,
        config=DefaultBayesianNNConfig()
    )

    x_train = torch.from_numpy(x_obs).float()
    y_train = torch.from_numpy(y_obs).float()
    x_test = torch.linspace(xlims[0], xlims[1], 3000)

    inference_method = 'mcmc'
    y_pred, y_std = model.infer(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        num_samples=50,
        inference_method=inference_method
    )

    plot_data(
        x_true=x_true, y_true=y_true,
        x_obs=x_obs, y_obs=y_obs,
        x_test=x_test, y_pred=y_pred, y_std=y_std,
        xlims=xlims, ylims=ylims
    )
    pass