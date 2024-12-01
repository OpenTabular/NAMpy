import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import jax.numpy as jnp
import jax.random as random

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from namgcv.basemodels.bnam import BayesianNAM
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import pickle

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


def plot_feature_contributions(
        num_features: dict,
        cat_features: dict,
        interaction_features: dict,
        submodel_contributions: dict,
        feature_names_mapping=None
):
    """
    Plots feature contributions for numerical, categorical, and interaction features.

    Args:
        num_features (Dict[str, jnp.ndarray]):
            Dictionary of numerical features.
        cat_features (Dict[str, jnp.ndarray]):
            Dictionary of categorical features.
        interaction_features (Dict[str, jnp.ndarray]):
            Dictionary of interaction features.
        submodel_contributions (Dict[str, np.ndarray]):
            Dictionary of feature contributions with keys as feature names and values
            as numpy arrays of shape [num_samples, batch_size].
        feature_names_mapping (Dict[str, List[str]]):
            Optional. Mapping from feature names to category names for categorical features.
    """
    sns.set_style("whitegrid", {"axes.facecolor": ".9"})

    # Plot numerical features
    if num_features:
        num_plots = len(num_features)
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=1,
            figsize=(12, 6 * num_plots),
            squeeze=False  # Ensure ax is always a 2D array
        )
        ax = ax.flatten()
        for i, (feature_name, feature_array) in enumerate(num_features.items()):
            # Get feature values and contributions.
            feature_values = np.array(feature_array).flatten()  # Convert JAX array to NumPy
            contributions = submodel_contributions[feature_name]  # Shape: [num_samples, batch_size]
            mean_contrib = contributions.mean(axis=0)  # Shape: [batch_size]
            lower = np.percentile(contributions, 5.0, axis=0)
            upper = np.percentile(contributions, 95.0, axis=0)

            # Sort the data for plotting.
            sorted_idx = np.argsort(feature_values)
            feature_values_sorted = feature_values[sorted_idx]
            mean_contrib_sorted = mean_contrib[sorted_idx]
            lower_sorted = lower[sorted_idx]
            upper_sorted = upper[sorted_idx]

            sns.lineplot(
                x=feature_values_sorted,
                y=mean_contrib_sorted,
                color=GREEN_RGB_COLORS[0],
                label="Mean Contribution",
                ax=ax[i]
            )
            ax[i].fill_between(
                feature_values_sorted, lower_sorted, upper_sorted,
                alpha=0.2, color=GREEN_RGB_COLORS[-1],
                label="95% Credible Interval"
            )
            ax[i].set_xlabel(f"{feature_name}", fontsize=12)
            ax[i].set_ylabel("Feature Contribution", fontsize=12)
            ax[i].set_title(f"Feature Contribution for {feature_name}", fontsize=12)
            ax[i].legend(loc='best', fontsize=12, frameon=False)
            ax[i].grid(True)

        plt.tight_layout()
        plt.savefig('num_feature_contributions.png')
        plt.show()

    # Plot interaction features
    if interaction_features:
        num_interactions = len(interaction_features)
        # Decide on grid size for subplots
        ncols = 2  # Adjust number of columns as needed
        nrows = (num_interactions + ncols - 1) // ncols  # Ceiling division to get the number of rows
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(12 * ncols, 6 * nrows),
            squeeze=False
        )
        axes = axes.flatten()

        for idx, (interaction_name, feature_arrays) in enumerate(interaction_features.items()):
            # feature_arrays is a JAX array of shape [batch_size, num_interaction_features]
            # Get the features involved in the interaction
            feature_names = interaction_name.split(":")
            num_interaction_features = len(feature_names)
            if num_interaction_features != 2:
                print(f"Skipping interaction {interaction_name}: only supports pairwise interactions.")
                continue

            feature1_name, feature2_name = feature_names
            feature1_values = np.array(feature_arrays[:, 0])
            feature2_values = np.array(feature_arrays[:, 1])

            contributions = submodel_contributions[interaction_name]  # Shape: [num_samples, batch_size]
            mean_contrib = contributions.mean(axis=0)  # Shape: [batch_size]

            # Create grid for plotting
            num_points = 100  # Adjust for finer grid
            x = np.linspace(feature1_values.min(), feature1_values.max(), num_points)
            y = np.linspace(feature2_values.min(), feature2_values.max(), num_points)
            xx, yy = np.meshgrid(x, y)

            # Interpolate the mean contributions onto the grid
            from scipy.interpolate import griddata
            points = np.stack((feature1_values, feature2_values), axis=-1)
            grid_z = griddata(points, mean_contrib, (xx, yy), method='linear')

            # Plot in the subplot
            ax = axes[idx]
            cp = ax.contourf(xx, yy, grid_z, levels=20, cmap='Greens')
            fig.colorbar(cp, ax=ax, label='Mean Contribution')
            ax.set_xlabel(feature1_name, fontsize=12)
            ax.set_ylabel(feature2_name, fontsize=12)
            ax.set_title(f'Interaction: {feature1_name} & {feature2_name}', fontsize=14)
            ax.grid(True)

        # Hide any unused subplots
        total_subplots = nrows * ncols
        if num_interactions < total_subplots:
            for idx in range(num_interactions, total_subplots):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig('interaction_feature_contributions.png')
        plt.show()


if __name__ == "__main__":
    # -----------------
    # Data preparation.
    # -----------------
    data = fetch_california_housing()

    X = pd.DataFrame(
        data=data.data,
        columns=data.feature_names
    )
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

    input_dim = X_train_scaled.shape[1]

    # ---------------
    # Model training.
    # ---------------
    inference_method = 'svi'
    model = BayesianNAM(
        cat_feature_info={},
        num_feature_info={
            feature_name: {
                "input_dim": 1,
                "output_dim": 1
            } for feature_name in X_train.columns
        },
        num_classes=1,
        config=DefaultBayesianNAMConfig(),
        subnetwork_config=DefaultBayesianNNConfig()
    )
    try:
        model_dir = f"bnam_numpyro_{inference_method}.pkl"
        model.load_model(filepath=model_dir)
    except FileNotFoundError:
        model.train_model(
            num_features={
                feature_name: jnp.array(
                    X_train_scaled[:, col_idx]
                ) for col_idx, feature_name in enumerate(X_train.columns)
            },
            cat_features={},
            target=jnp.array(y_train),
            inference_method=inference_method
        )
        model.save_model(filepath=model_dir)

    num_features = {
        feature_name: jnp.array(
            X_test_scaled[:, col_idx]
        ) for col_idx, feature_name in enumerate(X_test.columns)
    }
    cat_features = {}
    y_pred, y_std, submodel_contributions = model.predict(
        num_features=num_features,
        cat_features=cat_features,
    )

    # -----------------
    # Results analysis.
    # -----------------
    interaction_feature_information = {}
    all_features = {**num_features, **cat_features}
    for interaction_name in submodel_contributions.keys():
        if ":" not in interaction_name:
            continue

        feature_names = interaction_name.split(":")
        interaction_feature_information[interaction_name] = jnp.concatenate(
            [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
            axis=-1
        )

    plot_feature_contributions(
        num_features=num_features,
        cat_features=cat_features,
        interaction_features=interaction_feature_information,
        submodel_contributions=submodel_contributions
    )

    sns.set_style("whitegrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    sns.scatterplot(
        x=y_test,
        y=y_pred,
        color=GREEN_RGB_COLORS[0],
        label="Predictions",
        ax=ax
    )
    ax.set_xlabel("Actuals", fontsize=12)
    ax.set_ylabel("Predictions", fontsize=12)
    ax.set_title(
        f"Predictions vs. Actuals | MSE: {np.mean((y_test - y_pred) ** 2):.4f}",
        fontsize=12
    )
    ax.legend(loc='best', fontsize=12, frameon=False)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    pass