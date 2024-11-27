import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from namgcv.basemodels.bnam import BayesianNN, BayesianNAM
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import pickle

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_scaled.shape[1]

    # ---------------
    # Model training.
    # ---------------
    inference_method = 'mcmc'
    try:
        model_dir = f'{inference_method}_joint_bnam.pickle'
        with open(model_dir, 'rb') as handle:
            model = pickle.load(handle)
    except FileNotFoundError:
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

        model.train_model(
            num_features={
                feature_name: torch.from_numpy(
                    X_train_scaled[:, col_idx]
                ).float().to(device) for col_idx, feature_name in enumerate(X_train.columns)
            },
            cat_features={},
            target=torch.from_numpy(y_train).float().to(device),
            num_samples=50,
            inference_method=inference_method
        )

        with open(model_dir, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    y_pred, y_std, submodel_contributions = model.predict(
        num_features={
            feature_name: torch.from_numpy(
                X_test_scaled[:, col_idx]
            ).float().to(device) for col_idx, feature_name in enumerate(X_test.columns)
        },
        cat_features={}
    )


    # -----------------
    # Results analysis.
    # -----------------
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
            submodel_contributions: dict,
            feature_names_mapping=None
    ):
        """
        Plots feature contributions for numerical and categorical features.

        Args:
            num_features (Dict[str, torch.Tensor]):
                Dictionary of numerical features.
            cat_features (Dict[str, torch.Tensor]):
                Dictionary of categorical features.
            submodel_contributions (Dict[str, np.ndarray]):
                Dictionary of feature contributions with keys as feature names and values
                as numpy arrays of shape [num_samples, batch_size].
            feature_names_mapping (Dict[str, List[str]]):
                Optional. Mapping from feature names to category names for categorical features.
        """

        sns.set_style("whitegrid", {"axes.facecolor": ".9"})

        if num_features:
            fig, ax = plt.subplots(nrows=len(num_features), ncols=1, figsize=(12, 6*len(num_features)))
            for i, (feature_name, feature_tensor) in enumerate(num_features.items()):
                # Get feature values and contributions.
                feature_values = feature_tensor.detach().numpy().flatten()  # Shape: [batch_size]
                contributions = submodel_contributions[feature_name]  # Shape: [num_samples, batch_size]
                mean_contrib = contributions.mean(axis=0)  # Shape: [batch_size]
                lower = np.percentile(contributions, 2.5, axis=0)
                upper = np.percentile(contributions, 97.5, axis=0)

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
                ax[i].legend(loc=4, fontsize=12, frameon=False)
                ax[i].grid(True)

            plt.tight_layout()
            plt.savefig('num_feature_contributions.png')
            plt.show()

        if cat_features:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6*len(cat_features)))
            for feature_name, feature_tensor in cat_features.items():
                # Get the contributions.
                contributions = submodel_contributions[feature_name]  # Shape: [num_samples, batch_size]
                mean_contrib = contributions.mean(axis=0)  # Shape: [batch_size]
                lower = np.percentile(contributions, 2.5, axis=0)
                upper = np.percentile(contributions, 97.5, axis=0)

                # Get category indices.
                feature_values = feature_tensor.detach().numpy()  # Shape: [batch_size, num_categories]
                categories = np.argmax(feature_values, axis=1)  # Shape: [batch_size]

                # Map category indices to names if provided.
                if feature_names_mapping and feature_name in feature_names_mapping:
                    category_names = feature_names_mapping[feature_name]
                else:  # Default to indices as category names.
                    category_names = [str(i) for i in range(feature_values.shape[1])]

                unique_categories = np.unique(categories)
                category_contribs = []
                category_lowers = []
                category_uppers = []
                category_labels = []

                for category in unique_categories:
                    idx = categories == category
                    category_contrib = mean_contrib[idx].mean()
                    category_lower = lower[idx].mean()
                    category_upper = upper[idx].mean()
                    category_contribs.append(category_contrib)
                    category_lowers.append(category_lower)
                    category_uppers.append(category_upper)
                    category_labels.append(category_names[category])

                y_err_lower = np.array(category_contribs) - np.array(category_lowers)
                y_err_upper = np.array(category_uppers) - np.array(category_contribs)
                y_err = [y_err_lower, y_err_upper]
                sns.barplot(
                    x=category_labels,
                    y=category_contribs,
                    yerr=y_err, capsize=5, color=GREEN_RGB_COLORS[0],
                    edgecolor='black', ax=ax
                )
                ax.set_xlabel(f"{feature_name}", fontsize=12)
                ax.set_ylabel("Feature Contribution", fontsize=12)
                ax.set_title(f"Feature Contribution for {feature_name}", fontsize=12)
                ax.grid(True)

            plt.tight_layout()
            plt.savefig('cat_feature_contributions.png')
            plt.show()

    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = np.mean((y_pred_unscaled - y_test_unscaled) ** 2)
    sns.set_style(
        "whitegrid",
        {"axes.facecolor": ".9"}
    )
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(12, 6)
    )
    sns.scatterplot(
        x=y_test_unscaled,
        y=y_pred_unscaled,
        color=GREEN_RGB_COLORS[0],
        ax=ax
    )
    ax.set_xlabel("True Target", fontsize=12)
    ax.set_ylabel("Predicted Target", fontsize=12)
    ax.set_title(f"Predicted vs True Target (MSE: {mse:.2f})", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    plot_feature_contributions(
        num_features={
            feature_name: torch.from_numpy(
                # X_test.iloc[:, col_idx].values
                X_test_scaled[:, col_idx]
            ).float() for col_idx, feature_name in enumerate(X_test.columns)
        },
        cat_features={},
        submodel_contributions=submodel_contributions
    )

