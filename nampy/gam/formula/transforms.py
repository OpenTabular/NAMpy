"""Fitted RHS transformations corresponding to R base/stats model frames."""

from __future__ import annotations

import ast
import itertools

import numpy as np
import pandas as pd

TRANSFORMS = {"factor", "ordered", "interaction", "cut", "poly"}


def polynomial_matrix(x, degree, state=None):
    """Mirror stats::poly's QR normalization and prediction recurrence."""
    x = np.asarray(x, dtype=float)
    if state is None:
        if degree < 1 or degree >= len(np.unique(x)):
            raise ValueError("poly degree must be positive and less than unique points")
        mean = np.mean(x)
        centered = x - mean
        q, r = np.linalg.qr(centered[:, None] ** np.arange(degree + 1))
        z = q * np.diag(r)
        norms = np.sum(z * z, axis=0)
        alpha = (np.sum(centered[:, None] * z * z, axis=0) / norms + mean)[:degree]
        state = {"alpha": alpha.tolist(), "norm2": np.r_[1.0, norms].tolist()}
    else:
        alpha, norms = np.asarray(state["alpha"]), np.asarray(state["norm2"])
        z = np.ones((len(x), degree + 1))
        z[:, 1] = x - alpha[0]
        for i in range(1, degree):
            z[:, i + 1] = (x - alpha[i]) * z[:, i] - norms[i + 1] / norms[i] * z[
                :, i - 1
            ]
        norms = norms[1:]
    return (z / np.sqrt(norms))[:, 1:], state


def _factor_label(value):
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")
    return str(value)


def _format_cut_number(value, digits):
    # base::cut.default uses formatC(format="g"), i.e. C significant digits.
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    return format(float(value) + 0.0, f".{max(1, digits)}g")


def ordered_contrasts(n):
    if n < 2 or n > 95:
        raise ValueError("ordered contrasts require between 2 and 95 levels")
    return polynomial_matrix(np.arange(1, n + 1, dtype=float), n - 1)[0]


def evaluate_transform(expr, data, *, state=None):
    """Return values and serializable fitted state; reject unimplemented options."""
    from ..specs.build import _evaluate_formula_numeric_expr_node

    fitted = {} if state is None else state
    training = state is None

    def evaluate(node, path="root"):
        if isinstance(node, ast.Name):
            return data[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cbind"
        ):
            if node.keywords or not node.args:
                raise NotImplementedError(
                    "cbind transform inputs require positional columns"
                )
            return np.column_stack(
                [evaluate(arg, path + f".{i}") for i, arg in enumerate(node.args)]
            )
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id not in TRANSFORMS
        ):
            return _evaluate_formula_numeric_expr_node(node, data, expr=expr)
        name = node.func.id
        args = node.args
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise NotImplementedError(
                    "Expanded transform keyword arguments are unsupported"
                )
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        allowed = {
            "factor": {"levels", "labels", "ordered"},
            "ordered": {"levels", "labels"},
            "interaction": {"drop", "sep", "lex_order"},
            "cut": {
                "breaks",
                "labels",
                "right",
                "include_lowest",
                "ordered_result",
                "dig_lab",
            },
            "poly": {"degree", "raw"},
        }[name]
        if set(kwargs) - allowed:
            raise NotImplementedError(
                f"Unsupported {name} options: {sorted(set(kwargs) - allowed)}"
            )
        if not args:
            raise ValueError(f"{name} requires a data argument")
        values = evaluate(args[0], path + ".0")
        if name in {"factor", "ordered"}:
            if len(args) > 1:
                if len(args) > 2 or "levels" in kwargs:
                    raise NotImplementedError(
                        "factor positional syntax supports x and levels only"
                    )
                kwargs["levels"] = ast.literal_eval(args[1])
            levels = kwargs.get("levels")
            relabel_prediction = not training and "labels" in kwargs and levels is None
            if training or relabel_prediction:
                if levels is None:
                    levels = list(pd.Categorical(values).categories)
                labels = kwargs.get("labels", levels)
                if isinstance(labels, (str, int, float)):
                    labels = [labels]
                labels = [_factor_label(value) for value in labels]
                if len(labels) == 1 and len(levels) != 1:
                    labels = [f"{labels[0]}{i + 1}" for i in range(len(levels))]
                if len(labels) != len(levels):
                    raise ValueError(
                        "factor labels must have length one or the number of levels"
                    )
                entry = {"levels": list(levels), "labels": list(labels)}
                if training:
                    fitted[path] = entry
            else:
                entry = fitted[path]
            if (
                pd.isna(values).any()
                or not pd.Series(values).isin(entry["levels"]).all()
            ):
                raise ValueError(f"{name} contains missing or unseen levels")
            cat = pd.Categorical(values, categories=entry["levels"])
            if np.any(cat.codes < 0):
                raise ValueError(f"{name} contains missing or unseen levels")
            unique_labels = list(pd.unique(np.asarray(entry["labels"], dtype=object)))
            label_codes = pd.Categorical(
                entry["labels"], categories=unique_labels
            ).codes
            return pd.Categorical.from_codes(
                label_codes[cat.codes],
                unique_labels,
                ordered=name == "ordered"
                or kwargs.get(
                    "ordered",
                    bool(getattr(getattr(values, "dtype", None), "ordered", False)),
                ),
            )
        if name == "poly":
            rest = [evaluate(arg, path + f".{i}") for i, arg in enumerate(args[1:], 1)]
            degree = kwargs.get("degree", 1)
            if len(rest) == 1 and np.asarray(rest[0]).size == 1:
                degree = np.asarray(rest.pop()).reshape(-1)[0]
            if (
                not np.isscalar(degree)
                or isinstance(degree, bool)
                or int(degree) != degree
                or degree < 1
            ):
                raise ValueError("poly degree must be a positive integer")
            degree = int(degree)
            arrays = [
                np.asarray(list(v) if isinstance(v, pd.Series) else v, dtype=float)
                for v in [values, *rest]
            ]
            if any(a.ndim == 0 or len(a) != len(data) for a in arrays):
                raise ValueError("poly arguments must have the same length")
            matrix = np.column_stack(arrays)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError("poly requires finite numeric columns")
            raw = kwargs.get("raw", False)
            marginal = []
            saved = []
            for i in range(matrix.shape[1]):
                if raw:
                    basis = matrix[:, i, None] ** np.arange(1, degree + 1)
                else:
                    previous = (
                        None
                        if training
                        else (
                            fitted[path]
                            if matrix.shape[1] == 1
                            else fitted[path]["marginals"][i]
                        )
                    )
                    if previous is not None and "marginals" in previous:
                        raise ValueError(
                            "multivariate poly prediction needs matching columns; use cbind for single-row prediction"
                        )
                    basis, coefficients = polynomial_matrix(
                        matrix[:, i], degree, previous
                    )
                    saved.append(coefficients)
                marginal.append(np.column_stack([np.ones(len(data)), basis]))
            if not raw:
                fitted[path] = saved[0] if len(saved) == 1 else {"marginals": saved}
            # stats::polym expand.grid varies the first exponent fastest.
            powers = [
                v[::-1]
                for v in itertools.product(range(degree + 1), repeat=matrix.shape[1])
            ]
            powers = [v for v in powers if 0 < sum(v) <= degree]
            return np.column_stack(
                [
                    np.prod(
                        np.column_stack(
                            [marginal[i][:, power] for i, power in enumerate(v)]
                        ),
                        axis=1,
                    )
                    for v in powers
                ]
            )
        if name == "interaction":
            cats = [
                pd.Categorical(evaluate(arg, path + f".{i}"))
                for i, arg in enumerate(args)
            ]
            if any(np.any(cat.codes < 0) for cat in cats):
                raise ValueError("interaction contains missing values")
            lex = kwargs.get("lex_order", False)
            combos = list(
                itertools.product(
                    *(cat.categories for cat in (cats if lex else cats[::-1]))
                )
            )
            if not lex:
                combos = [combo[::-1] for combo in combos]
            sep = kwargs.get("sep", ".")
            levels = [sep.join(map(_factor_label, combo)) for combo in combos]
            observed = [
                sep.join(map(_factor_label, combo)) for combo in zip(*cats, strict=True)
            ]
            if training:
                fitted[path] = {
                    "levels": [
                        v
                        for v in levels
                        if not kwargs.get("drop", False) or v in observed
                    ]
                }
            if not pd.Series(observed).isin(fitted[path]["levels"]).all():
                raise ValueError("interaction contains unseen levels")
            return pd.Categorical(observed, categories=fitted[path]["levels"])
        if len(args) > 2:
            raise NotImplementedError(
                "cut positional syntax supports x and breaks only"
            )
        breaks = kwargs.get(
            "breaks", ast.literal_eval(args[1]) if len(args) > 1 else None
        )
        if breaks is None:
            raise ValueError("cut requires explicit breaks")
        if np.isscalar(breaks):
            if not np.isfinite(breaks) or breaks < 2:
                raise ValueError("invalid number of cut intervals")
            count = int(breaks + 1)
            low, high = np.min(values), np.max(values)
            width = high - low
            if width == 0:
                width = abs(low) if low != 0 else 1.0
                breaks = np.linspace(low - width / 1000, high + width / 1000, count)
            else:
                breaks = np.linspace(low, high, count)
                breaks[0], breaks[-1] = low - width / 1000, high + width / 1000
        else:
            breaks = np.sort(np.asarray(breaks, dtype=float))
        if len(breaks) < 2 or np.any(np.diff(breaks) <= 0):
            raise ValueError("cut breaks must be unique and increasing")
        labels = kwargs.get("labels")
        if labels is None:
            digits = kwargs.get("dig_lab", 3)
            if not isinstance(digits, int) or digits < 0:
                raise ValueError("cut dig_lab must be a nonnegative integer")
            for precision in range(digits, max(12, digits) + 1):
                formatted = [_format_cut_number(v, precision) for v in breaks]
                if all(
                    a != b for a, b in zip(formatted[:-1], formatted[1:], strict=True)
                ):
                    break
            else:
                formatted = None
            right = kwargs.get("right", True)
            if formatted is None:
                labels = [f"Range_{i + 1}" for i in range(len(breaks) - 1)]
            else:
                labels = [
                    ("(" if right else "[") + a + "," + b + ("]" if right else ")")
                    for a, b in zip(formatted[:-1], formatted[1:], strict=True)
                ]
                if kwargs.get("include_lowest", False):
                    if right:
                        labels[0] = "[" + labels[0][1:]
                    else:
                        labels[-1] = labels[-1][:-1] + "]"
        if labels is not False and (
            not isinstance(labels, (list, tuple)) or len(labels) != len(breaks) - 1
        ):
            raise ValueError("cut labels must have one label per interval")
        cut = pd.cut(
            values,
            bins=sorted(breaks),
            labels=False,
            right=kwargs.get("right", True),
            include_lowest=kwargs.get("include_lowest", False),
        )
        codes = np.asarray(cut, dtype=float).copy()
        if not kwargs.get("right", True) and kwargs.get("include_lowest", False):
            codes[np.asarray(values) == max(breaks)] = len(breaks) - 2
        if np.isnan(codes).any():
            raise ValueError("cut produced missing values outside the breaks")
        if labels is False:
            return codes + 1
        labels = [_factor_label(value) for value in labels]
        unique_labels = list(pd.unique(np.asarray(labels, dtype=object)))
        label_codes = pd.Categorical(labels, categories=unique_labels).codes
        return pd.Categorical.from_codes(
            label_codes[codes.astype(int)],
            unique_labels,
            ordered=kwargs.get("ordered_result", False),
        )

    return evaluate(ast.parse(expr, mode="eval").body), fitted


def is_fitted_transform(expr):
    node = ast.parse(expr, mode="eval").body
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in TRANSFORMS
    )
