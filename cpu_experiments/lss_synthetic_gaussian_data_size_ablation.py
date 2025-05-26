import os

num_chains = 10
n_devices = min(os.cpu_count(), num_chains)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'

import pandas as pd
import numpy as np
import jax.numpy as jnp

import numpyro
numpyro.set_platform('cpu')

import matplotlib.pyplot as plt
import seaborn as sns

import jax
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
print(f"Default backend for JAX: {jax.default_backend()}")
print(
    f"Number of devices available on default backend: "
    f"{jax.local_device_count(backend=jax.default_backend())}"
)

from namgcv.basemodels.bnam import BayesianNAM
from namgcv.configs.experimental.small_synthetic_nam import DefaultBayesianNAMConfig
from namgcv.configs.experimental.small_synthetic_nn import DefaultBayesianNNConfig

import random

import properscoring
import numpyro.distributions as dist



def get_gaussian_synthetic_data(n_samples: int=3000, seed=42, plot=True):
    lower_bound, upper_bound = -1, 1
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = X[:, 0], X[:, 1]

    def theta1_func(x1, x2):
        return (
            # 2 + np.sin(1.2 * x1) + 0.5 * np.log1p(x2**2)

            x1**2 + np.sin(4*x2)

            # 0.5 * np.sin(np.pi * x2) + 0.25 * x1
        )

    def theta2_func(x1, x2):
        # z = 0.3 * (x1 - 1.5)**2 + np.exp(-0.5 * x2)
        z = 0.01 + np.exp(x1)/10 + 0.1 + x2**2

        # z = np.sqrt(
        #     np.exp(
        #         np.log(0.05 + 0.2*x1**2)
        #         +
        #         np.log1p(np.exp(-1.3 + 0.8*x2**2))
        #     )
        # )
        # return np.log1p(np.exp(z))
        return z

    theta_functions = [theta1_func, theta2_func]

    theta1 = theta1_func(x1, x2)
    theta2 = theta2_func(x1, x2)

    # Sample from Gaussian
    y = np.random.normal(loc=theta1, scale=theta2)

    # Plot distributions
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('θ1')
        axs[0].set_xlabel('θ2')
        axs[0].grid(True)

        sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
        axs[1].set_title('θ2')
        axs[1].set_xlabel('θ2')
        axs[1].grid(True)

        sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
        axs[2].set_title('Y ~ Normal(θ1, θ2)')
        axs[2].set_xlabel('Y')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Partial dependence plots
        x_grid = np.linspace(lower_bound, upper_bound, 100)
        theta_labels = ["θ1", "θ2"]
        feature_labels = ["x1", "x2"]

        sns.set_style("white", {"axes.grid": True})
        sns.set_palette("Reds")
        fig, axes = plt.subplots(
            n_features,
            len(theta_functions),
            figsize=(6 * len(theta_functions), 6 * n_features)
        )

        for i in range(n_features):
            for j, theta_func in enumerate(theta_functions):
                pdp_vals = []
                for val in x_grid:
                    X_temp = X.copy()
                    X_temp[:, i] = val
                    y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                    pdp_vals.append(np.mean(y_vals))

                feature_data = X[:, i]
                theta_data = theta_func(X[:, 0], X[:, 1])

                ax = axes[i, j]
                sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
                ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

                ax.set_title(
                    f'Marginal Effect of {feature_labels[i]} '
                    f'on {theta_labels[j]}', fontsize=12
                )
                ax.set_ylabel(f'{theta_labels[j]}', fontsize=12)
                ax.set_xlabel(feature_labels[i])
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'theta1': theta1,
        'theta2': theta2,
        'response': y,
    })


def plot_feature_contributions(
        num_features: dict,
        cat_features: dict,
        interaction_features: dict,
        submodel_contributions: dict,
        num_outputs: int
):
    sns.set_style("white")
    sns.set_palette("Greens")

    for feature_dict in [num_features, cat_features, interaction_features]:
        if not feature_dict:
            continue

        num_plots = len(feature_dict)
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=num_outputs,
            figsize=(6 * (num_outputs + 1), 6 * (num_plots + 1)),
            squeeze=False
        )
        for i, (feature_name, feature_array) in enumerate(feature_dict.items()):
            feature_values = np.array(feature_array).flatten()  # Convert JAX array to NumPy

            # Shape: [num_mcmc_samples, batch_size, network_output_dim]
            contributions = submodel_contributions[feature_name]

            # [batch_size, network_output_dim]
            mean_contribution_all_params = contributions.mean(axis=0)
            for j in range(num_outputs):
                mean_param_contribution = mean_contribution_all_params[:, j] \
                    if num_outputs > 1 else mean_contribution_all_params

                sorted_idx = np.argsort(feature_values)
                feature_values_sorted = feature_values[sorted_idx]
                mean_param_contribution_sorted = mean_param_contribution[sorted_idx]

                ax_to_plot = ax[i, j]
                uncertainty = np.std(
                    submodel_contributions[feature_name][:, :, j],
                    axis=0
                )[sorted_idx] \
                    if num_outputs > 1 \
                    else np.std(
                    submodel_contributions[feature_name],
                    axis=0
                )[sorted_idx]
                for ci_multiplier, alpha in zip([3, 2, 1], [0.3, 0.6, 0.9]):
                    ax_to_plot.fill_between(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty,
                        alpha=alpha,
                        label=f"Aleatoric Uncertainty - {ci_multiplier} Std. Deviations"
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )

                # num_bins = 30
                # counts, bin_edges = np.histogram(feature_values, bins=num_bins)
                # norm_counts = counts / counts.max()
                # fixed_height = ax_to_plot.get_ylim()[1] - ax_to_plot.get_ylim()[0]
                # for k in range(num_bins):
                #     ax_to_plot.bar(
                #         bin_edges[k],
                #         height=fixed_height,
                #         bottom=ax_to_plot.get_ylim()[0],
                #         width=bin_edges[k + 1] - bin_edges[k],
                #         color=plt.cm.Blues(norm_counts[k]),
                #         alpha=0.3
                #     )
                #
                sns.lineplot(
                    x=feature_values_sorted,
                    y=mean_param_contribution_sorted,
                    label="Mean Output Parameter Contribution",
                    ax=ax_to_plot,
                    color="black",
                    drawstyle="steps" if np.all(
                        np.isin(feature_values, [0, 1])
                    ) else None
                )

                ax_to_plot.set_xlabel(f"{feature_name}", fontsize=16)
                ax_to_plot.set_ylabel("Feature Contribution", fontsize=16)
                ax_to_plot.set_title(f"Feature Contribution of {feature_name} on theta{j}", fontsize=16)
                ax_to_plot.legend(
                    loc='upper left',
                    fontsize=12,
                    frameon=False,
                    bbox_to_anchor=(0.1, 1),
                    # borderaxespad=0.
                )
                ax_to_plot.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import pickle

    num_runs_per_sample_size = 1
    data_sizes = [
        100,
        200,
        500,
        1000,
        1500,
        # 2000,
        # 4000,
        # 8000,
        # 16000
    ]
    results = {
        data_size: None for data_size in data_sizes
    }
    LOAD = False
    if LOAD:
        for data_size in data_sizes:
            loaded_res_list = []
            for run_num in range(1, num_runs_per_sample_size):
                loaded_res_list.append(
                    pickle.load(
                        open(f"res//my_new_results{run_num}_{data_size}.pkl", "rb")
                    )
                )
            # combine the two lists in loaded_res_list into a single list
            loaded_res_list_final = []
            for i in range(len(loaded_res_list)):
                loaded_res_list_final += loaded_res_list[i]

            results[data_size] = loaded_res_list_final

    else:
        results = {
            n_samples: [] for n_samples in data_sizes
        }
        for experiment_idx in range(num_runs_per_sample_size):
            for n_samples in data_sizes:
                data = get_gaussian_synthetic_data(n_samples=n_samples, plot=False)

                X = data.drop(columns=['response'] + [col for col in data.columns if "theta" in col])
                y = data['response']

                # One hot encode the categorical features.
                try:
                    cat_features = pd.get_dummies(
                        data=X[[col for col in X.columns if X[col].dtype == "category"]]
                    ).astype(int)
                except ValueError:  # No categorical features.
                    cat_features = pd.DataFrame()

                num_features = X[[col for col in X.columns if X[col].dtype != "category"]]
                print(f"Categorical features: {cat_features.columns.tolist()}")
                print(f"Numeric features: {num_features.columns.tolist()}")

                # --- Model Setup ---
                num_outputs = 2
                cat_feature_info, cat_feature_inputs = {}, {}
                for cat_col in [col for col in X.columns if X[col].dtype == "category"]:
                    all_cols_for_feature = [
                        col for col in cat_features.columns if col.startswith(cat_col)
                    ]
                    cat_feature_info[cat_col] = {
                        "input_dim": len(all_cols_for_feature),
                        "output_dim": num_outputs
                    }
                    cat_feature_inputs[cat_col] = jnp.array(
                        cat_features[all_cols_for_feature]
                    )

                num_feature_info = {
                    feature_name: {
                        "input_dim": 1,
                        "output_dim": num_outputs
                    } for feature_name in num_features.columns
                }
                num_feature_inputs = {
                    feature_name: jnp.array(
                        num_features.loc[:, feature_name]
                    ) for feature_name in num_feature_info.keys()
                }

                warmstart_path = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "namgcv",
                    "bnam_de_warmstart_checkpoints",
                    "warmstart"
                )
                if os.path.exists(warmstart_path):
                    for file in os.listdir(warmstart_path):
                        if file.endswith(".npz"):
                            os.remove(os.path.join(warmstart_path, file))
                    if os.path.exists(os.path.join(warmstart_path, "..", "tree")):
                        os.remove(os.path.join(warmstart_path, "..", "tree"))

                numpyro.set_host_device_count(
                    DefaultBayesianNAMConfig().num_chains
                )
                model = BayesianNAM(
                    cat_feature_info=cat_feature_info,
                    num_feature_info=num_feature_info,
                    config=DefaultBayesianNAMConfig(),
                    subnetwork_config=DefaultBayesianNNConfig(),
                    rng_key=jax.random.PRNGKey(
                        random.randint(1, 10000)
                    ),
                )

                # --- Training ---
                CV = False
                if CV:
                    cv_results = model.cross_validation(
                        num_features=num_feature_inputs,
                        cat_features=cat_feature_inputs,
                        target=jnp.array(y),
                    )
                else:
                    data_loader = model.train_model(
                        num_features={
                            feature_name: jnp.array(
                                num_features.iloc[:, col_idx]
                            ) for col_idx, feature_name in enumerate(
                                num_features.columns
                            )
                        },
                        cat_features={
                            feature_name: jnp.array(
                                cat_features.iloc[:, col_idx]
                            ) for col_idx, feature_name in enumerate(
                                cat_features.columns
                            )
                        },
                        target=jnp.array(y),
                    )

                final_params, submodel_contributions = model.predict(
                    data_loader=data_loader
                )
                batch_iter = data_loader.iter(
                    split="test",
                    batch_size=None
                )
                data_dict = next(batch_iter)  # First and only batch.
                num_features = data_dict["feature"]["numerical"]
                cat_features = data_dict["feature"]["categorical"]
                target = data_dict["target"]

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

                PLOT_EFFECTS = False
                if PLOT_EFFECTS:
                    plot_feature_contributions(
                        num_features=num_features,
                        cat_features=cat_features,
                        interaction_features=interaction_feature_information,
                        submodel_contributions=submodel_contributions,
                        num_outputs=num_outputs
                    )

                import numpyro.distributions as dist

                log_probs = dist.Normal(
                    final_params[..., 0].mean(axis=0),
                    final_params[..., 1].mean(axis=0)
                ).log_prob(target)  # shape: (num_samples, num_data)
                nll = -log_probs
                print("NLL:", jnp.mean(nll))
                from properscoring import crps_gaussian
                crps = crps_gaussian(
                    x=target,
                    mu=final_params[..., 0].mean(axis=0),
                    sig=final_params[..., 1].mean(axis=0)
                )
                print("CRPS:", jnp.mean(crps))
                lppd = [
                    dist.Normal(
                        final_params[..., 0][i, ...],
                        final_params[..., 1][i, ...]
                    ).log_prob(target)
                    for i in range(final_params.shape[0])
                ]
                lppd = jnp.mean(jnp.array(lppd), axis=0)
                print("LPPD:", jnp.mean(lppd))

                results[n_samples].append({
                    "nll": nll,
                    "crps": crps,
                    "lppd": lppd,
                    "final_params": final_params,
                    "submodel_contributions": submodel_contributions,
                    "data_loader": data_loader,
                    "samples": model._get_posterior_param_samples(
                        group_by_chain=True
                    )
                })

                with open(f"new_new_{n_samples}.pkl", "wb") as f:
                    pickle.dump(results[n_samples], f)

                PLOT_POSTERIOR = False
                if PLOT_POSTERIOR:
                    # Plot the Weight distributions
                    posterior_param_samples_dict = model._get_posterior_param_samples()
                    sns.set_style(style="white")
                    for posterior_params_name, posterior_param_samples in (
                            posterior_param_samples_dict.items()
                    ):
                        if not posterior_param_samples:
                            continue  # Intercept may be None.

                        if "weight" not in posterior_params_name:
                            continue  # Only care about weights.

                        feature_names = list({
                            key.split("/")[0] for key in posterior_param_samples.keys()
                        })
                        for feature_name in feature_names:
                            param_names = [
                                key for key in posterior_param_samples.keys()
                                if feature_name in key
                            ]  # Parameters for this feature.
                            for i, param_name in enumerate(param_names):
                                layer_num = 4
                                if f"dense_{layer_num}" not in param_name:
                                    continue  # Let's only look at the last layer.
                                if "scale" in param_name:
                                    continue  # Not interested in the hierarchical prior.

                                # Convert the posterior samples into a DataFrame for easier plotting.
                                num_samples, num_layer_weights, num_outputs = (
                                    posterior_param_samples[param_name].shape
                                )
                                samples_reshaped = posterior_param_samples[param_name].reshape(
                                    num_samples,
                                    num_layer_weights * num_outputs
                                )
                                column_names = [
                                    f"Dense Layer {layer_num} - Neuron {i} (Output {j})"
                                    for j in range(num_outputs)
                                    for i in range(num_layer_weights)
                                ]
                                samples_df = pd.DataFrame(samples_reshaped, columns=column_names)

                                # Create the plot of bivariate marginals for the last layer weights.
                                for j in range(num_outputs):
                                    # Plot all combinations of weights for the last layer.
                                    for k in range(num_layer_weights):
                                        for l in range(k + 1, num_layer_weights):
                                            # Extract x and y for joint distribution
                                            x_vals = samples_df.iloc[:, k + j * num_layer_weights]
                                            y_vals = samples_df.iloc[:, l + j * num_layer_weights]

                                            # Set up joint grid with marginals
                                            g = sns.JointGrid(
                                                data=samples_df,
                                                x=x_vals.name,
                                                y=y_vals.name,
                                                height=12
                                            )
                                            # Main bivariate KDE
                                            g.plot_joint(
                                                sns.kdeplot,
                                                cmap="Greens",
                                                fill=True,
                                                levels=10,
                                                alpha=0.8
                                            )
                                            g.plot_joint(
                                                sns.kdeplot,
                                                fill=False,
                                                color="green",
                                                linewidths=1.0,
                                                levels=10
                                            )
                                            # Marginal KDEs
                                            g.plot_marginals(sns.kdeplot, color="green", fill=True)

                                            g.ax_joint.set_xlabel(
                                                f"Dense Layer {layer_num} - Kernel Weight {k} (Output {j})")
                                            g.ax_joint.set_ylabel(
                                                f"Dense Layer {layer_num} - Kernel Weight  {l} (Output {j})")
                                            g.ax_joint.grid(True)
                                            plt.suptitle(
                                                f"{feature_name} BNN Bivariate Marginal Kernel Posterior",
                                            )
                                            plt.tight_layout()
                                            plt.show()

        with open(f"new_new.pkl", "wb") as f:
            pickle.dump(results, f)





nll_dict, crps_dict, lppd_dict = (
    {experiment_idx: [] for experiment_idx in range(len(results[list(results.keys())[0]]))},
    {experiment_idx: [] for experiment_idx in range(len(results[list(results.keys())[0]]))},
    {experiment_idx: [] for experiment_idx in range(len(results[list(results.keys())[0]]))},
)

for data_size in results.keys():
    for experiment_idx, experiment in enumerate(results[data_size]):
        crps_dict[experiment_idx].append(np.mean(experiment["crps"]))
        nll_dict[experiment_idx].append(np.mean(experiment["nll"]))
        lppd_dict[experiment_idx].append(np.mean(experiment["lppd"]))

nll_mean, crps_mean, lppd_mean = [], [], []
nll_std, crps_std, lppd_std = [], [], []
for experiment_idx in range(len(results[list(results.keys())[0]])):
    crps_mean.append(crps_dict[experiment_idx])
    nll_mean.append(nll_dict[experiment_idx])
    lppd_mean.append(lppd_dict[experiment_idx])

nll_std = pd.DataFrame(nll_mean, dtype=float).std(axis=0)#.sort_values(ascending=False)
crps_std = pd.DataFrame(crps_mean, dtype=float).std(axis=0)#.sort_values(ascending=False)
lppd_std = pd.DataFrame(lppd_mean, dtype=float).std(axis=0)#.sort_values(ascending=False)

nll_mean = pd.DataFrame(nll_mean, dtype=float).mean(axis=0)#.sort_values(ascending=False)
crps_mean = pd.DataFrame(crps_mean, dtype=float).mean(axis=0)#.sort_values(ascending=False)
lppd_mean = pd.DataFrame(lppd_mean, dtype=float).mean(axis=0)#.sort_values(ascending=True)

nll_std = nll_std[nll_mean.index]
crps_std = crps_std[crps_mean.index]
lppd_std = lppd_std[lppd_mean.index]

fig, ax = plt.subplots(1, 3, figsize=(6 * 3, 6))
# Plot CRPS on left y-axis
ax1 = ax[0]
ax1.plot(data_sizes, crps_mean, 'o-', color='forestgreen', label='Mean CRPS')
# Plot +- 1 std deviation
ax1.fill_between(
    data_sizes,
    crps_mean + crps_std,
    crps_mean - crps_std,
    alpha=0.3,
    color="limegreen",
    label=f"Std. Deviation"
)

ax1.set_xlabel("Number of Samples")
ax1.set_ylabel("CRPS", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title("CRPS")
ax1.legend()
ax1.grid(True)

# Create second y-axis for NLL
# ax2 = ax1.twinx()
ax2 = ax[1]
ax2.plot(data_sizes, nll_mean, 'o-', color='forestgreen', label='Mean NLL')
ax2.fill_between(
    data_sizes,
    nll_mean + nll_std,
    nll_mean - nll_std,
    alpha=0.3,
    color="limegreen",
    label=f"Std. Deviation"
)
ax2.set_xlabel("Number of Samples")
ax2.set_ylabel("NLL", color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_title("NLL")
ax2.legend()
ax2.grid(True)

ax3 = ax[2]
ax3.plot(
    data_sizes, lppd_mean, 'o-', color='forestgreen', label='Mean LPPD'
)
ax3.fill_between(
    data_sizes,
    lppd_mean + lppd_std,
    lppd_mean - lppd_std,
    alpha=0.3,
    color="limegreen",
    label=f"Std. Deviation"
)
ax3.set_xlabel("Number of Samples")
ax3.set_ylabel("LPPD", color='black')
ax3.tick_params(axis='y', labelcolor='black')
ax3.set_title("LPPD")
ax3.legend()
ax3.grid(True)

plt.suptitle("Metrics vs Number of Samples")
plt.tight_layout()
plt.grid(True)
plt.show()


