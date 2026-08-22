"""Generic additive-term extraction for GAM/GA2M-style models.

Module-level functions operating on plain data: ``vals``/``counts`` dicts
keyed by feature index (int), interaction index pair (tuple), or -1 for the
intercept, each mapping to per-unique-value pandas Series/DataFrames. Any
additive model that can produce per-term outputs can compose them; the
NODE-GAM blocks do so via ``GAMAdditiveMixin`` in
``additive_trees.py``.

Purification is adapted from https://github.com/zzzace2000/nodegam.
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def convert_onehot_vector_to_integers(terms):
    """Make onehot or multi-hot vectors into a list of integers or tuple.

    Args:
        terms (Pytorch tensor): a one-hot matrix with each column has only one entry as 1.
            Shape: [in_features, uniq_GAM_terms].

    Returns:
        tuple_terms (list): A list of integers or tuples of all the GAM terms.
    """
    r_idx, c_idx = torch.nonzero(terms, as_tuple=True)
    tuple_terms = []
    for c in range(terms.shape[1]):
        n_interaction = (c_idx == c).sum()

        if n_interaction > 2:
            print(
                f"WARNING: it is not a GA2M with a {n_interaction}-way term. "
                f"Ignore this term."
            )
            continue
        if n_interaction == 1:
            tuple_terms.append(int(r_idx[c_idx == c].item()))
        elif n_interaction == 2:
            tuple_terms.append(tuple(r_idx[c_idx == c][:2].cpu().numpy()))
    return tuple_terms


def terms_from_feature_selectors(feature_selectors, return_inverse=False):
    """Derive the learned additive terms from concatenated feature selectors.

    Args:
        feature_selectors: Nonnegative selector weights of shape
            [in_features, num_selectors, depth] (e.g. entmax outputs), where any
            positive weight means the selector uses that feature.
        return_inverse (bool): If True, also return the map from each selector
            column back to the index of its term.

    Returns:
        tuple_terms (list): A list of integers or tuples representing all the
            additive terms. E.g. [2, 4, (2, 3), (1, 4)].
    """
    fs = feature_selectors.sum(dim=-1)
    fs[fs > 0.0] = 1.0
    # ^-- [in_features, num_selectors] binary features

    result = torch.unique(fs, dim=1, sorted=True, return_inverse=return_inverse)
    # ^-- ([in_features, uniq_terms], [num_selectors])

    terms = result
    if isinstance(result, tuple):  # return_inverse=True
        terms = result[0]

    # To make additive terms human-readable, it transforms the one-hot vector into an integer,
    # and a 2-hot vector (interaction) into a tuple of integer.
    tuple_terms = convert_onehot_vector_to_integers(terms)

    if isinstance(result, tuple):
        return tuple_terms, result[1]
    return tuple_terms


def aggregate_term_values(results, X, terms):
    """Group per-sample term outputs by unique input value.

    Args:
        results: The per-term model outputs, a numpy tensor of shape
            [num_data, num_unique_terms, output_dim] aligned with `terms`.
        X: The inputs of the data (pandas DataFrame, unnormalized).
        terms (list): All the main and interaction terms. E.g. [1, 2, (2, 3)].

    Returns:
        vals (dict): A dict that has keys as feature index and value as a pandas
            Series/DataFrame that maps the unique value of input X to the output of the
            model. For example, if a model learns 2 main effects for features 1 and 2,
            and an interaction term between features 1 and 2, we could have::
            {1: {0: -0.2, 1: 0.3, 2: 1},
             2: {1: 0.3, 2: -0.5},
             (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
        counts (dict): Same format as `vals` but the values are the counts in the data.
    """
    vals, counts = {}, {}
    for idx, t in enumerate(tqdm(terms)):
        if not isinstance(t, tuple):  # main effect term
            index = X.iloc[:, t]
            scores = pd.Series(results[:, idx, 0], index=index)

            tmp = scores.groupby(level=0).agg(["count", "first"])
            vals[t] = tmp["first"]
            counts[t] = tmp["count"]
        else:
            tmp = pd.Series(
                results[:, idx, 0],
                index=pd.MultiIndex.from_frame(X.iloc[:, list(t)]),
            )

            # One groupby to return both vals and counts
            tmp2 = tmp.groupby(level=[0, 1]).agg(["count", "first"])

            the_vals = tmp2["first"]
            the_counts = tmp2["count"]

            vals[t] = the_vals.unstack(level=-1).fillna(0.0)
            counts[t] = the_counts.unstack(level=-1).fillna(0).astype(int)

    # For each interaction tuple (i, j), initialize the main effect term i and j since they
    # will have some values during the purification.
    for t in terms:
        if not isinstance(t, tuple):
            continue

        for i in t:
            if i in vals:
                continue
            a = X.iloc[:, i]
            the_counts = a.groupby(a).agg(["count"])
            counts[i] = the_counts["count"]
            vals[i] = the_counts["count"].copy()
            vals[i][:] = 0.0

    return vals, counts


def purify_interactions(vals, counts, tol=1e-3):
    """Purify the interaction terms to move mass from interactions to main effects.

    See the Supp. D of the NODE-GAM paper for details. It modifies `vals` in-place.

    Args:
        vals (dict): Per-term unique-value outputs as returned by
            :func:`aggregate_term_values`.
        counts (dict): Per-term unique-value counts, same keys as `vals`.
        tol: Stop once the largest purified average value is below this tolerance.
    """
    for t in vals:
        # If it's not an interaction term, continue.
        if not isinstance(t, tuple):
            continue

        # Continue purify the interactions until the purified average value is smaller than tol.
        biggest_epsilon = np.inf
        while biggest_epsilon > tol:
            biggest_epsilon = -np.inf

            avg = (vals[t] * counts[t]).sum(axis=1).values / counts[t].sum(
                axis=1
            ).values
            if np.max(np.abs(avg)) > biggest_epsilon:
                biggest_epsilon = np.max(np.abs(avg))

            vals[t] -= avg.reshape(-1, 1)
            vals[t[0]] += avg

            avg = (vals[t] * counts[t]).sum(axis=0).values / counts[t].sum(
                axis=0
            ).values
            if np.max(np.abs(avg)) > biggest_epsilon:
                biggest_epsilon = np.max(np.abs(avg))

            vals[t] -= avg.reshape(1, -1)
            vals[t[1]] += avg


def center_main_effects(vals, counts, bias):
    """Center each main effect to weighted mean zero, folding the shifts into the intercept.

    Modifies `vals` in-place: `vals[-1]` accumulates `bias` plus each main effect's
    count-weighted average.

    Args:
        vals (dict): Per-term unique-value outputs; entry -1 is the intercept.
        counts (dict): Per-term unique-value counts.
        bias: The model's global bias to add to the intercept term.
    """
    vals[-1] += bias
    for t in vals:
        # If it's an interaction term or the bias term, continue.
        if isinstance(t, tuple) or t == -1:
            continue

        weights = counts[t].values
        avg = np.average(vals[t].values, weights=weights)

        vals[-1] += avg
        vals[t] -= avg


def build_terms_frame(vals, counts, feature_names):
    """Organize purified per-term values into the result table.

    Args:
        vals (dict): Per-term unique-value outputs; entry -1 is the intercept.
        counts (dict): Per-term unique-value counts.
        feature_names: Sequence mapping feature index to name (e.g. `X.columns`).

    Returns:
        A pandas table with one row per term. The columns include::
        feat_name: The feature name. E.g. "Hour".
        feat_idx: The feature index. E.g. 2.
        x: The unique values of the feature. E.g. [0.5, 3, 4.7].
        y: The values of the output. E.g. [-0.2, 0.3, 0.5].
        importance: The feature importance. It's calculated as the weighted average of
            the absolute value of y weighted by the counts of each unique value.
        counts: The counts of each unique value in the data. E.g. [20, 10, 3].
    """
    # Initialize with the bias term.
    results = [
        {
            "feat_name": "offset",
            "feat_idx": -1,
            "x": None,
            "y": np.full(1, vals[-1]),
            "importance": -1,
            "counts": None,
        }
    ]

    for t in tqdm(vals):
        if t == -1:
            continue

        if not isinstance(t, tuple):
            x = vals[t].index.values
            y = vals[t].values
            count = counts[t].values
            tmp = np.argsort(x)
            x, y, count = x[tmp], y[tmp], count[tmp]
        else:
            # Make 2d back to 1d
            tmp = vals[t].stack()
            tmp_count = counts[t].values.reshape(-1)
            selected_entry = (tmp.values != 0) | (tmp_count > 0)
            tmp = tmp[selected_entry]
            x = tmp.index.values
            y = tmp.values
            count = tmp_count[selected_entry]

        imp = np.average(np.abs(np.array(y)), weights=np.array(count))
        results.append(
            {
                "feat_name": (
                    feature_names[t]
                    if not isinstance(t, tuple)
                    else f"{feature_names[t[0]]}_{feature_names[t[1]]}"
                ),
                "feat_idx": t,
                "x": x.tolist(),
                "y": y.tolist(),
                "importance": imp,
                "counts": count.tolist(),
            }
        )

    df = pd.DataFrame(results)
    df["tmp"] = df.feat_idx.apply(
        lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x)
    )
    df = df.sort_values("tmp").drop("tmp", axis=1)
    df = df.reset_index(drop=True)
    return df
