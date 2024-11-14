import pyro

from namgcv.basemodels.bnam import BayesianNN
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import torch
import numpy as np

from pyro.infer import MCMC, NUTS, Predictive

import matplotlib.pyplot as plt
import seaborn as sns

GREEN_RGB_COLORS = [
    '#004c00', # '#004e00', '#005000', '#005100', '#005300',
    # '#005500', # '#005700', '#005900', '#005a00', '#005c00',
    '#005e00', # '#006000', '#006200', '#006300', '#006500',
    # '#006700', # '#006900', '#006b00', '#006c00', '#006e00',
    '#007000', # '#007200', '#007400', '#007500', '#007700',
    # '#007900', # '#007b00', '#007d00', '#007e00', '#008000',
    '#008200', # '#008400', '#008600', '#008800', '#008900',
    # '#008b00', # '#008d00', '#008f00', '#009100', '#009200',
    '#009400', # '#009600', '#009800', '#009a00', '#009b00',
    # '#009d00', # '#009f00', '#00a100', '#00a300', '#00a400',
    '#00a600', # '#00a800', '#00aa00', '#00ac00', '#00ad00',
    # '#00af00', # '#00b100', '#00b300', '#00b500', '#00b600',
    '#00b800', # '#00ba00', '#00bc00', '#00be00', '#00bf00',
    # '#00c100', # '#00c300', '#00c500', '#00c700', '#00c800',
    '#00ca00', # '#00cc00', '#00ce00', '#00d000', '#00d100',
    # '#00d300', # '#00d500', '#00d700', '#00d900', '#00da00',
    '#00dc00', # '#00de00', '#00e000', '#00e200', '#00e300',
    # '#00e500', # '#00e700', '#00e900', '#00eb00', '#00ec00',
    '#00ee00', # '#00f000', '#00f200', '#00f400', '#00f500',
    # '#00f700', # '#00f900', '#00fb00', '#00fd00', '#00ff00'
]



if __name__ == "__main__":
    np.random.seed(42)
    pyro.set_rng_seed(42)

    # Generate some synthetic data.
    x_obs = np.hstack(
        [
            np.linspace(start=-0.2, stop=0.2, num=1000),
            np.linspace(start=0.6, stop=1, num=1000)
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
    sns.set()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    sns.lineplot(
        x=x_true, y=y_true,
        color=GREEN_RGB_COLORS[0],
        label="True DGP",
        linewidth=1,
        ax=ax
    )
    sns.scatterplot(
        x=x_obs, y=y_obs,
        color=GREEN_RGB_COLORS[1],
        label="Observed Data",
        linewidth=0.5,
        ax=ax
    )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc=4, fontsize=12, frameon=False)
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()

    model = BayesianNN(
        in_dim=1,
        out_dim=1,
        config=DefaultBayesianNNConfig()
    )

    x_train = torch.from_numpy(x_obs).float()
    y_train = torch.from_numpy(y_obs).float()

    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(nuts_kernel, num_samples=100)
    mcmc.run(x_train, y_train)

    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    x_test = torch.linspace(xlims[0], xlims[1], 3000)
    preds = predictive(x_test)

    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    xlims = [-0.5, 1.5]
    ylims = [-1.5, 2.5]
    sns.lineplot(
        x=x_true, y=y_true,
        color=GREEN_RGB_COLORS[0],
        label="True DGP",
        linewidth=1,
        ax=ax
    )
    sns.scatterplot(
        x=x_obs, y=y_obs,
        color=GREEN_RGB_COLORS[1],
        label="Observed Data",
        linewidth=0.5,
        ax=ax
    )
    sns.lineplot(
        x=x_test, y=y_pred,
        color=GREEN_RGB_COLORS[2],
        label="Predictive Posterior",
        linewidth=1,
        ax=ax
    )
    ax.fill_between(
        x_test, y_pred - 2 * y_std,
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
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()

    pass