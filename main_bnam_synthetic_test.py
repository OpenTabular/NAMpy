import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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
        feature_names_mapping=None,
        y=None
):
    """
    Plots feature contributions for numerical, categorical, and interaction features.
    Now includes a separate heatmap panel for the uncertainty of interaction features.

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
            squeeze=False
        )
        ax = ax.flatten()
        for i, (feature_name, feature_array) in enumerate(num_features.items()):
            feature_values = np.array(feature_array).flatten()  # Convert JAX array to NumPy

            contributions = submodel_contributions[feature_name]  # Shape: [num_samples, batch_size]
            mean_contrib = contributions.mean(axis=0)  # [batch_size]
            lower = np.percentile(contributions, 5.0, axis=0)
            upper = np.percentile(contributions, 95.0, axis=0)

            # Sort the data for a cleaner plot
            sorted_idx = np.argsort(feature_values)
            feature_values_sorted = feature_values[sorted_idx]
            mean_contrib_sorted = mean_contrib[sorted_idx]
            lower_sorted = lower[sorted_idx]
            upper_sorted = upper[sorted_idx]

            if y is not None:
                sns.scatterplot(
                    x=feature_values,
                    y=y,
                    color=GREEN_RGB_COLORS[5],
                    label="Data Points",
                    ax=ax[i]
                )

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

    if interaction_features:
        num_interactions = len(interaction_features)
        # For each interaction, we create a row with two plots:
        # Left: Mean contribution contourf with black dashed contour lines.
        # Right: Uncertainty (upper - lower) as a heatmap.
        ncols = 2
        nrows = num_interactions
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(12 * ncols, 6 * nrows),
            squeeze=False
        )

        for idx, (interaction_name, feature_arrays) in enumerate(interaction_features.items()):
            feature_names = interaction_name.split(":")
            if len(feature_names) != 2:
                print(
                    f"Skipping interaction {interaction_name}: only supports pairwise interactions.")
                continue

            feature1_name, feature2_name = feature_names
            feature1_values = np.array(feature_arrays[:, 0])
            feature2_values = np.array(feature_arrays[:, 1])

            contributions = submodel_contributions[interaction_name]  # [num_samples, batch_size]
            mean_contrib = contributions.mean(axis=0)  # [batch_size]
            lower = np.percentile(contributions, 5.0, axis=0)
            upper = np.percentile(contributions, 95.0, axis=0)
            uncertainty = upper - lower  # Width of the credible interval

            # Create a grid for contour plot
            num_points = 100
            x = np.linspace(feature1_values.min(), feature1_values.max(), num_points)
            y = np.linspace(feature2_values.min(), feature2_values.max(), num_points)
            xx, yy = np.meshgrid(x, y)

            from scipy.interpolate import griddata
            points = np.stack((feature1_values, feature2_values), axis=-1)
            grid_z_mean = griddata(points, mean_contrib, (xx, yy), method='linear')
            grid_z_unc = griddata(points, uncertainty, (xx, yy), method='linear')

            # --------------------
            # Plot posterior mean.
            # --------------------
            ax_mean = axes[idx, 0]
            cp = ax_mean.contourf(xx, yy, grid_z_mean, levels=20, cmap='Greens', alpha=0.8)
            fig.colorbar(cp, ax=ax_mean, label='Mean Contribution')

            cl = ax_mean.contour(
                xx, yy, grid_z_mean,
                levels=10,
                colors='black',
                linestyles='dashed'
            )  # Add black dashed contour lines on top.
            ax_mean.clabel(cl, inline=True, fontsize=10)
            ax_mean.set_xlabel(feature1_name, fontsize=12)
            ax_mean.set_ylabel(feature2_name, fontsize=12)
            ax_mean.set_title(f'Mean Interaction: {feature1_name} & {feature2_name}', fontsize=14)
            ax_mean.grid(True)

            # ------------------------------
            # Plot uncertainty as a heatmap.
            # ------------------------------
            ax_unc = axes[idx, 1]
            img = ax_unc.imshow(
                grid_z_unc,
                origin='lower',
                aspect='auto',
                extent=[x.min(), x.max(), y.min(), y.max()],
                cmap='RdPu'
            )
            fig.colorbar(img, ax=ax_unc, label='Uncertainty (Width of Credible Interval)')
            ax_unc.set_xlabel(feature1_name, fontsize=12)
            ax_unc.set_ylabel(feature2_name, fontsize=12)
            ax_unc.set_title(f'Uncertainty: {feature1_name} & {feature2_name}', fontsize=14)
            ax_unc.grid(False)  # No grid for heatmaps.

        plt.tight_layout()
        plt.savefig('interaction_feature_contributions_uncertainty.png')
        plt.show()

def get_synthetic_data() -> pd.DataFrame:
    """
    Function to generate synthetic data for testing the BayesianNAM model.
    Returns a pandas DataFrame containing the synthetic data.
    """
    # ------------------
    # Generate features.
    # ------------------
    np.random.seed(42)
    n_samples = 50

    numerical_1 = np.random.uniform(low=0, high=10, size=n_samples)
    numerical_2 = np.random.uniform(low=-5, high=5, size=n_samples)
    categorical_1 = np.random.choice(a=['A', 'B', 'C'], size=n_samples)
    categorical_2 = np.random.choice(a=['X', 'Y', 'Z'], size=n_samples)

    encoder = OneHotEncoder()
    cat_1_encoded = pd.DataFrame(
        encoder.fit_transform(categorical_1.reshape(-1, 1)).toarray(),
        columns=[f"Cat1_{cat}" for cat in encoder.categories_[0]]
    )
    cat_2_encoded = pd.DataFrame(
        encoder.fit_transform(categorical_2.reshape(-1, 1)).toarray(),
        columns=[f"Cat2_{cat}" for cat in encoder.categories_[0]]
    )

    true_effects = {
        "numerical_1": {
            "response": np.sin(numerical_1),
            "feature": numerical_1
        },
        "numerical_2": {
            "response": numerical_2 ** 3,
            "feature": numerical_2
        },
        "numerical_1:numerical_2": {
            "response": numerical_1 * numerical_2,
            "feature": np.stack([numerical_1, numerical_2], axis=-1)
        }
    }
    noise_parameters = {
        "numerical_1": {"loc": 0, "scale": 0.5, "size": n_samples},
        "numerical_2": {"loc": 0, "scale": 10, "size": n_samples},
        # "categorical_1": {"a": [-1, 0, 1], "p": [0.1, 0.8, 0.1], "size": n_samples},
        # "categorical_2": {"a": [-1, 0, 1], "p": [0.1, 0.8, 0.1], "size": n_samples},
    }
    noise_parameters["numerical_1:numerical_2"] = {
            "loc": (
                    noise_parameters["numerical_1"]["loc"] *
                    noise_parameters["numerical_2"]["loc"]
            ),
            "scale": np.sqrt(
                (
                    noise_parameters["numerical_1"]["loc"]**2 *
                    noise_parameters["numerical_2"]["loc"]**2
                ) + (
                    noise_parameters["numerical_1"]["loc"]**2 *
                    noise_parameters["numerical_2"]["scale"]**2
                ) + (
                    noise_parameters["numerical_2"]["loc"]**2 *
                    noise_parameters["numerical_1"]["scale"] ** 2
                ) + (
                    noise_parameters["numerical_1"]["scale"]**2 *
                    noise_parameters["numerical_2"]["scale"]**2
                )
            ),
            "size": n_samples
        }
    noise = {
        "numerical_1": np.random.normal(**noise_parameters["numerical_1"]),
        "numerical_2": np.random.normal(**noise_parameters["numerical_2"]),
        # "categorical_1": np.random.choice(**noise_parameters["categorical_1"]),
        # "categorical_2": np.random.choice(**noise_parameters["categorical_2"])
        "numerical_1:numerical_2": np.random.normal(**noise_parameters["numerical_1:numerical_2"])
    }
    true_effects["numerical_1"]["noisy_response"] = (
            true_effects["numerical_1"]["response"] +
            noise["numerical_1"]
    )
    true_effects["numerical_2"]["noisy_response"] = (
            true_effects["numerical_2"]["response"] +
            noise["numerical_2"]
    )
    true_effects["numerical_1:numerical_2"]["noisy_response"] = (
            true_effects["numerical_1:numerical_2"]["response"] +
            noise["numerical_1:numerical_2"]
    )

    response = (
            true_effects["numerical_1"]["response"] +
            true_effects["numerical_2"]["response"] +
            true_effects["numerical_1:numerical_2"]["response"]
    )

    # --------------
    # Plot the data.
    # --------------
    sns.set_style("whitegrid", {"axes.facecolor": ".9"})
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6*2))
    ci_multiplier = 1.96

    for i, feature_name in enumerate(["numerical_1", "numerical_2"]):
        x_vals = true_effects[feature_name]["feature"]
        y_mean = true_effects[feature_name]["response"]

        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_sorted = y_mean[sort_idx]

        y_lower = y_sorted - ci_multiplier * noise_parameters[feature_name]["scale"]
        y_upper = y_sorted + ci_multiplier * noise_parameters[feature_name]["scale"]
        axes[i].fill_between(x_sorted, y_lower, y_upper, color=GREEN_RGB_COLORS[-1], alpha=0.2)

        sns.scatterplot(
            x=true_effects[feature_name]["feature"],
            y=true_effects[feature_name]["noisy_response"],
            color=GREEN_RGB_COLORS[5],
            label="Noisy Response",
            ax=axes[i]
        )
        sns.lineplot(
            x=true_effects[feature_name]["feature"],
            y=true_effects[feature_name]["response"],
            color=GREEN_RGB_COLORS[0],
            label="True Marginal Effect",
            ax=axes[i],
        )
        axes[i].set_title(f"{feature_name} Effect", fontsize=14)
        axes[i].set_xlabel(f"{feature_name}", fontsize=12)
        axes[i].set_ylabel("Effect", fontsize=12)
        axes[i].legend()
        axes[i].grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot interaction: numerical_1:numerical_2 ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    feature_name = "numerical_1:numerical_2"
    x1_vals = true_effects[feature_name]["feature"][:, 0]
    x2_vals = true_effects[feature_name]["feature"][:, 1]
    y_mean = true_effects[feature_name]["response"]
    y_noisy = true_effects[feature_name]["noisy_response"]

    x1_i = np.linspace(x1_vals.min(), x1_vals.max(), 200)
    x2_i = np.linspace(x2_vals.min(), x2_vals.max(), 200)
    X1_i, X2_i = np.meshgrid(x1_i, x2_i)
    Y_i = X1_i * X2_i  # True interaction function.

    Y_i_lower = Y_i - ci_multiplier * noise_parameters[feature_name]["scale"]
    Y_i_upper = Y_i + ci_multiplier * noise_parameters[feature_name]["scale"]
    Y_i_uncertainty = Y_i_upper - Y_i_lower

    # --------------------
    # Plot posterior mean.
    # --------------------
    ax_mean = axes[0]
    cp = ax_mean.contourf(X1_i, X2_i, Y_i, levels=20, cmap='Greens', alpha=0.8)
    fig.colorbar(cp, ax=ax_mean, label='Mean Interaction Effect')
    cl = ax_mean.contour(X1_i, X2_i, Y_i, levels=10, colors='black', linestyles='dashed')
    ax_mean.clabel(cl, inline=True, fontsize=10)
    ax_mean.set_title("Interaction Mean Effect", fontsize=14)
    ax_mean.set_xlabel("numerical_1", fontsize=12)
    ax_mean.set_ylabel("numerical_2", fontsize=12)
    ax_mean.grid(True)

    # ------------------------------
    # Plot uncertainty as a heatmap.
    # ------------------------------
    ax_unc = axes[1]
    img = ax_unc.imshow(
        Y_i_uncertainty,
        origin='lower',
        aspect='auto',
        # extent=[x1_i.min(), x1_i.max(), x2_i.min(), x2_i.max()],
        cmap='RdPu'
    )
    fig.colorbar(img, ax=ax_unc, label='Uncertainty (Width of 95% CI)', orientation='vertical')
    ax_unc.set_title("Interaction Uncertainty", fontsize=14)
    ax_unc.set_xlabel("numerical_1", fontsize=12)
    ax_unc.set_ylabel("numerical_2", fontsize=12)
    ax_unc.grid(False)

    plt.tight_layout()
    plt.show()

    # -------
    # Return.
    # -------
    return pd.concat(
        [
            pd.DataFrame(
                data={
                    'numerical_1': numerical_1,
                    'numerical_2': numerical_2,
                }
            ),
            # cat_1_encoded,
            # cat_2_encoded,
            pd.DataFrame(data={'Response': response})
        ], axis=1
    )



if __name__ == "__main__":
    # -----------------
    # Data preparation.
    # -----------------
    data = get_synthetic_data()

    X = data.drop(columns=['Response'])
    y = data['Response']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    input_dim = X_train_scaled.shape[1]

    # ---------------
    # Model training.
    # ---------------
    inference_method = 'mcmc'
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
    model_dir = f"bnam_numpyro_{inference_method}.pkl"
    try:
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
        # model.save_model(filepath=model_dir)

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
        submodel_contributions=submodel_contributions,
        y=y_test
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