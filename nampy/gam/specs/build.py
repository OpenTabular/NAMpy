"""Formula build stage: extracted formula intent -> canonical predictor specs."""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..formula.extract import (
    ExtractedParametricTerm,
    ExtractedPredictor,
    ExtractedSmoothTerm,
)
from .base import TermSpec
from .predictors import LinearPredictorSpec
from .smooth import (
    CubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    FactorSmoothInteractionSpec,
    GPSmoothSpec,
    MarkovRandomFieldSmoothSpec,
    PSplineSmoothSpec,
    RandomEffectSmoothSpec,
    SmoothSpec,
    SumToZeroFactorSmoothSpec,
    TensorANOVASmoothSpec,
    TensorInteractionSmoothSpec,
    TensorProductSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    ThinPlateSmoothSpec,
    replace_smooth_spec,
)

BuiltPredictorSpec = LinearPredictorSpec
BuiltTermSpec = TermSpec

_SMOOTH_SPEC_DEFAULTS: dict[str, object] = {
    "special": "s",
    "bs": "cr",
    "k": -1,
    "fx": False,
    "select": False,
    "m": None,
    "xt": None,
    "sp": None,
    "pc": None,
    "knots": None,
    "constraint_mode": "auto",
    "shared_basis_setup": None,
    "mc": None,
    "full": False,
    "ord_": None,
}

_FORMULA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORMULA_NUMERIC_FUNCTIONS: dict[str, object] = {
    "I": lambda x: x,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}


@dataclass(frozen=True)
class FormulaBuildResult:
    predictor_specs: list[BuiltPredictorSpec]
    working_data: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    used_columns: list[str]
    response: np.ndarray
    offsets: np.ndarray | list[np.ndarray | None] | None
    preprocess_state: dict
    response_name: str | None


def _is_bare_formula_name(expr: str | None) -> bool:
    if expr is None:
        return False
    return _FORMULA_NAME_RE.fullmatch(str(expr)) is not None


def _evaluate_formula_numeric_expr_node(node, data: pd.DataFrame, *, expr: str):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        name = str(node.id)
        if name not in data.columns:
            raise KeyError(
                f"Formula expression {expr!r} references variable {name!r}, "
                "but it is not in `data`."
            )
        return numeric_1d_values(data[name], name=name)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_formula_numeric_expr_node(node.operand, data, expr=expr)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _evaluate_formula_numeric_expr_node(node.operand, data, expr=expr)

    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        left = _evaluate_formula_numeric_expr_node(node.left, data, expr=expr)
        right = _evaluate_formula_numeric_expr_node(node.right, data, expr=expr)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError(
                f"Unsupported formula expression function in {expr!r}."
            )
        if any(kw.arg is None for kw in node.keywords) or node.keywords:
            raise NotImplementedError(
                f"Unsupported keyword arguments in formula expression {expr!r}."
            )
        func_name = str(node.func.id)
        func = _FORMULA_NUMERIC_FUNCTIONS.get(func_name)
        if func is None:
            raise NotImplementedError(
                f"Unsupported formula expression function {func_name!r} in {expr!r}."
            )
        args = [
            _evaluate_formula_numeric_expr_node(arg, data, expr=expr)
            for arg in node.args
        ]
        return func(*args)

    raise NotImplementedError(f"Unsupported formula expression {expr!r}.")


def _evaluate_formula_numeric_expression(expr: str, data: pd.DataFrame):
    from ..formula.parse import all_vars1

    parsed = ast.parse(str(expr), mode="eval")
    values = _evaluate_formula_numeric_expr_node(parsed.body, data, expr=str(expr))
    out = np.asarray(values, dtype=np.float64)
    if out.ndim == 0:
        out = np.full(len(data), float(out), dtype=np.float64)
    else:
        out = np.ravel(out)
    if out.shape[0] != len(data):
        raise ValueError(
            f"Formula expression {expr!r} produced length {out.shape[0]}, "
            f"expected {len(data)}."
        )
    if not np.isfinite(out).all():
        raise ValueError(f"Formula expression {expr!r} produced NaN or Inf.")
    return out, [str(name) for name in all_vars1(str(expr))]


def _resolve_formula_offset_column(
    data_work: pd.DataFrame, offset_name: str
) -> np.ndarray:
    if offset_name not in data_work.columns:
        raise KeyError(f"Offset column {offset_name!r} not found in `data`.")
    if not pd.api.types.is_numeric_dtype(data_work[offset_name]):
        raise NotImplementedError(
            "Current formula-based GAM fitting supports numeric offsets only. "
            f"Offset column {offset_name!r} is non-numeric."
        )
    return np.asarray(data_work[offset_name], dtype=np.float64).ravel()


def _materialize_formula_expression_column(
    expr: str,
    data_work: pd.DataFrame,
    *,
    pred_name: str,
    hidden_counter: int,
    state: dict,
):
    expr_key = str(expr)
    cache = state.setdefault("_formula_expression_cache", {})
    cached = cache.get(expr_key)
    if cached is not None:
        return (
            str(cached["hidden_name"]),
            hidden_counter,
            list(cached["source_variables"]),
        )

    values, src_vars = _evaluate_formula_numeric_expression(expr_key, data_work)
    hidden_col = f"__gam_formula__{pred_name}__{hidden_counter}"
    hidden_counter += 1
    data_work[hidden_col] = values

    record = {
        "hidden_name": hidden_col,
        "expr": expr_key,
        "source_variables": list(src_vars),
    }
    cache[expr_key] = dict(record)
    state["formula_expression_columns"].append(dict(record))
    return hidden_col, hidden_counter, list(src_vars)


def _split_formula_text(formula: str) -> tuple[str | None, str]:
    text = str(formula).strip()
    if "~" not in text:
        return None, text
    lhs, rhs = text.split("~", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    return (lhs if lhs else None), rhs


def _contains_standalone_dot(rhs: str) -> bool:
    return re.search(r"(^|[~+\-(),\s])\.([~+\-(),\s]|$)", rhs) is not None


def _build_formula_text(response_name: str | None, rhs: str) -> str:
    rhs_txt = str(rhs).strip()
    if response_name is None:
        return f"~{rhs_txt}" if rhs_txt else "~"
    return f"{response_name} ~ {rhs_txt}" if rhs_txt else f"{response_name} ~"


def _expand_dot_shorthand_formula(raw_formula: str, data: pd.DataFrame) -> str:
    response_name, rhs = _split_formula_text(raw_formula)
    if not _contains_standalone_dot(rhs):
        return raw_formula

    dot_terms = [str(col) for col in data.columns if str(col) != response_name]
    dot_rhs = " + ".join(dot_terms) if dot_terms else "1"
    expanded_rhs = re.sub(
        r"(^|[~+\-(),\s])\.([~+\-(),\s]|$)",
        rf"\1{dot_rhs}\2",
        rhs,
    )
    return _build_formula_text(response_name, expanded_rhs)


def _expand_extracted_predictors_dot_shorthand(
    extracted_predictors: list[ExtractedPredictor], data: pd.DataFrame
) -> list[ExtractedPredictor]:
    def _predictor_has_dot_term(pred: ExtractedPredictor) -> bool:
        if _contains_standalone_dot(_split_formula_text(pred.raw_formula)[1]):
            return True

        for term in pred.terms:
            if not isinstance(term, ExtractedParametricTerm):
                continue
            if term.raw_label == ".":
                return True
            if any(label == "." for label in getattr(term, "factor_labels", ())):
                return True
            if any(var == "." for var in getattr(term, "variables", ())):
                return True
        return False

    dot_predictors = [
        pred for pred in extracted_predictors if _predictor_has_dot_term(pred)
    ]
    if not dot_predictors:
        return extracted_predictors

    if len(extracted_predictors) != 1:
        raise NotImplementedError(
            "Data-aware '.' shorthand is unsupported for formula-list / "
            "multi-predictor models. Upstream mgcv::gam(list(...), data=...) "
            "rejects this surface too, so NAMpy keeps it closed for parity."
        )

    from ..formula import extract_formula_terms, parse_gam_formula

    expanded_formula = _expand_dot_shorthand_formula(
        dot_predictors[0].raw_formula, data
    )
    expanded_predictor = extract_formula_terms(parse_gam_formula(expanded_formula))[0]
    original = extracted_predictors[0]
    return [
        replace(
            expanded_predictor,
            predictor_index=original.predictor_index,
            predictor_name=original.predictor_name,
            raw_formula=original.raw_formula,
        )
    ]


def _build_s_cr(opts) -> CubicRegressionSmoothSpec:
    return CubicRegressionSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_cs(opts) -> CubicShrinkageSmoothSpec:
    return CubicShrinkageSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_cc(opts) -> CyclicCubicRegressionSmoothSpec:
    return CyclicCubicRegressionSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_ps(opts) -> PSplineSmoothSpec:
    return PSplineSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_tp(opts) -> ThinPlateSmoothSpec:
    return ThinPlateSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        xt=opts["xt"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_ts(opts) -> ThinPlateShrinkageSmoothSpec:
    return ThinPlateShrinkageSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        xt=opts["xt"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_gp(opts) -> GPSmoothSpec:
    return GPSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        xt=opts["xt"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_mrf(opts) -> MarkovRandomFieldSmoothSpec:
    return MarkovRandomFieldSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_re(opts) -> RandomEffectSmoothSpec:
    return RandomEffectSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_fs(opts) -> FactorSmoothInteractionSpec:
    return FactorSmoothInteractionSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_sz(opts) -> SumToZeroFactorSmoothSpec:
    return SumToZeroFactorSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


_S_BASIS_SPEC_BUILDERS: dict[str, object] = {
    "cr": _build_s_cr,
    "cs": _build_s_cs,
    "cc": _build_s_cc,
    "ps": _build_s_ps,
    "tp": _build_s_tp,
    "ts": _build_s_ts,
    "gp": _build_s_gp,
    "mrf": _build_s_mrf,
    "re": _build_s_re,
    "fs": _build_s_fs,
    "sz": _build_s_sz,
}


def _build_te(opts) -> TensorProductSmoothSpec:
    return TensorProductSmoothSpec(
        special="te",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        m=opts["m"],
        xt=opts["xt"],
        sp=opts["sp"],
        knots=opts["knots"],
    )


def _build_ti(opts) -> TensorInteractionSmoothSpec:
    return TensorInteractionSmoothSpec(
        special="ti",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        m=opts["m"],
        xt=opts["xt"],
        sp=opts["sp"],
        knots=opts["knots"],
        mc=opts["mc"],
    )


def _build_t2(opts) -> TensorANOVASmoothSpec:
    return TensorANOVASmoothSpec(
        special="t2",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        m=opts["m"],
        xt=opts["xt"],
        sp=opts["sp"],
        knots=opts["knots"],
        full=opts["full"],
        ord=opts["ord_"],
    )


_SPECIAL_SMOOTH_BUILDERS: dict[str, object] = {
    "te": _build_te,
    "ti": _build_ti,
    "t2": _build_t2,
}


def _dispatch_smooth_spec_from_options(opts) -> SmoothSpec:
    merged = {**_SMOOTH_SPEC_DEFAULTS, **dict(opts)}
    special_key = str(merged["special"]).lower()
    if special_key == "s":
        bs_key = str(merged["bs"]).lower()
        builder = _S_BASIS_SPEC_BUILDERS.get(bs_key)
        if builder is None:
            raise NotImplementedError(f"Unsupported s() basis {merged['bs']!r}.")
        return builder(merged)
    builder = _SPECIAL_SMOOTH_BUILDERS.get(special_key)
    if builder is None:
        raise NotImplementedError(f"Unsupported smooth special {merged['special']!r}.")
    return builder(merged)


def build_smooth_spec(
    *,
    special: str,
    bs,
    k=-1,
    fx=False,
    select: bool = False,
    m=None,
    xt=None,
    sp=None,
    pc=None,
    knots=None,
    constraint_mode: str = "auto",
    shared_basis_setup=None,
    mc=None,
    full: bool = False,
    ord_=None,
) -> SmoothSpec:
    return _dispatch_smooth_spec_from_options(locals())


def smooth_spec_from_basis_options(basis_options) -> SmoothSpec:
    raw = dict(basis_options or {})
    if "ord_" not in raw and "ord" in raw:
        raw = {**raw, "ord_": raw["ord"]}
    merged = {**_SMOOTH_SPEC_DEFAULTS, **raw}
    special_key = str(merged.get("special", "s")).lower()
    fx_raw = merged.get("fx", False)
    if special_key in {"te", "ti"} and isinstance(fx_raw, (list, tuple, np.ndarray)):
        merged["fx"] = [bool(v) for v in np.asarray(fx_raw, dtype=object).ravel()]
    else:
        merged["fx"] = bool(fx_raw)
    merged["select"] = bool(merged.get("select", False))
    merged["full"] = bool(merged.get("full", False))
    merged["constraint_mode"] = str(merged.get("constraint_mode", "auto"))
    return _dispatch_smooth_spec_from_options(merged)


def _coerce_fx(fx, *, kind, n_features):
    kind_key = str(kind).lower()
    if kind_key in {"te", "ti"} and isinstance(fx, (list, tuple, np.ndarray)):
        vals = [bool(v) for v in np.asarray(fx, dtype=object).ravel()]
        if len(vals) == 1:
            return vals * int(n_features)
        if len(vals) != int(n_features):
            import warnings

            warnings.warn("dimension of fx is wrong", stacklevel=3)
            return [False] * int(n_features)
        return vals
    if isinstance(fx, (list, tuple, np.ndarray)):
        raise NotImplementedError(
            "Vector-valued fx is currently supported only for te() and ti() terms."
        )
    return bool(fx)


def _default_k_for_basis(basis, default_k):
    basis_key = str(basis).lower()
    if basis_key in {"mrf", "gp"}:
        return -1
    return default_k


def _factor_smooth_base_basis_from_xt(xt):
    if xt is None:
        return "tp"
    if isinstance(xt, str):
        return str(xt).lower()
    if isinstance(xt, dict):
        return str(xt.get("bs", "tp")).lower()
    return None


def _default_k_for_smooth(kind, basis, features, default_k):
    kind_key = str(kind).lower()
    if kind_key in {"te", "ti", "t2"}:
        # mgcv::te()/ti()/t2() default k to 5^d per marginal. The current
        # Python tensor surface supports one feature per marginal, so d = 1.
        return [5] * len(features)
    return _default_k_for_basis(basis, default_k)


def _default_basis_for_kind(kind, default_basis):
    kind_key = str(kind).lower()
    if kind_key == "s":
        return "tp" if default_basis is None else default_basis
    if default_basis is not None and isinstance(default_basis, (list, tuple)):
        return default_basis
    return "cr"


def _knots_for_features(knots, features):
    if knots is None:
        return None
    if isinstance(knots, dict):
        vals = [knots.get(str(f), None) for f in features]
        return (
            None
            if all(v is None for v in vals)
            else vals[0] if len(features) == 1 else vals
        )
    return knots


def is_factor_like_series(s: pd.Series) -> bool:
    dtype = s.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def factor_info(s: pd.Series):
    cat = pd.Categorical(s)
    levels = list(cat.categories)
    ordered = bool(getattr(cat.dtype, "ordered", False))
    return cat, levels, ordered


def safe_token(x) -> str:
    txt = str(x)
    txt = re.sub(r"\s+", "_", txt)
    txt = re.sub(r"[^0-9A-Za-z_]+", "", txt)
    return txt or "level"


def numeric_1d_values(s: pd.Series, *, name: str):
    vals = np.asarray(s, dtype=np.float64).ravel()
    if vals.ndim != 1:
        raise ValueError(f"Column {name!r} did not coerce to a 1D numeric array.")
    if not np.isfinite(vals).all():
        raise ValueError(f"Column {name!r} contains NaN or Inf.")
    return vals


def _fs_by_without_factor_feature_base_spec(smooth_spec: FactorSmoothInteractionSpec):
    """Mirror mgcv::smooth.construct.fs.smooth.spec when no factor term is present."""
    xt = smooth_spec.xt
    if xt is None:
        base_bs = "tp"
        xt_rest = None
    elif isinstance(xt, str):
        base_bs = str(xt).lower()
        xt_rest = None
    elif isinstance(xt, dict):
        base_bs = str(xt.get("bs", "tp")).lower()
        xt_rest = {k: v for k, v in xt.items() if k != "bs"} or None
    else:
        raise NotImplementedError(
            'For `bs="fs"`, xt must be None, a basis string, or a dict '
            "containing optional key `bs`."
        )

    kwargs = {
        "special": str(smooth_spec.special),
        "bs": base_bs,
        "k": smooth_spec.k,
        "fx": smooth_spec.fx,
        "select": smooth_spec.select,
        "sp": smooth_spec.sp,
        "knots": smooth_spec.knots,
        "constraint_mode": str(getattr(smooth_spec, "constraint_mode", "auto")),
    }

    if base_bs == "ps":
        kwargs["m"] = None if xt_rest is None else xt_rest.get("m", None)
    elif base_bs in {"tp", "ts", "gp"}:
        kwargs["xt"] = xt_rest
    elif xt_rest:
        raise NotImplementedError(
            f'`bs="fs"` factor-by fallback does not support extra xt options for '
            f"base bs={base_bs!r}."
        )

    return build_smooth_spec(**kwargs)


def _expand_parametric_term(
    term,
    data_work,
    *,
    pred_name,
    include_intercept,
    hidden_counter,
    state,
):
    factor_labels = list(
        term.metadata.get(
            "factor_labels",
            [term.label],
        )
    )
    src_vars = list(
        term.metadata.get(
            "source_variables",
            list(factor_labels) if len(factor_labels) > 0 else [term.label],
        )
    )
    if len(factor_labels) == 0:
        raise ValueError(
            f"Parametric term {term.label!r} does not define any factor labels."
        )

    n = len(data_work)
    comp_lists = []
    needs_expansion = len(factor_labels) > 1 or any(
        not _is_bare_formula_name(label) for label in factor_labels
    )

    for label in factor_labels:
        if not _is_bare_formula_name(label):
            hidden_expr_name, hidden_counter, expr_src_vars = (
                _materialize_formula_expression_column(
                    str(label),
                    data_work,
                    pred_name=pred_name,
                    hidden_counter=hidden_counter,
                    state=state,
                )
            )
            vals = numeric_1d_values(data_work[hidden_expr_name], name=hidden_expr_name)
            comp_lists.append(
                [
                    {
                        "label": str(label),
                        "values": vals,
                        "recipe": {
                            "var": hidden_expr_name,
                            "type": "numeric",
                            "expr": str(label),
                            "source_variables": list(expr_src_vars),
                        },
                    }
                ]
            )
            continue

        if label not in data_work.columns:
            raise KeyError(
                f"Formula references parametric variable {label!r}, but it is not in `data`."
            )

        s = data_work[label]

        if is_factor_like_series(s):
            needs_expansion = True
            cat, levels, ordered = factor_info(s)
            active_levels = levels if not include_intercept else levels[1:]

            comps = []
            for lev in active_levels:
                comps.append(
                    {
                        "label": f"{label}[{lev}]",
                        "values": np.asarray(
                            (cat == lev).astype(float), dtype=np.float64
                        ),
                        "recipe": {
                            "var": label,
                            "type": "factor",
                            "level": lev,
                            "levels": list(levels),
                            "ordered": ordered,
                            "include_intercept": bool(include_intercept),
                        },
                    }
                )
            comp_lists.append(comps)
        else:
            vals = numeric_1d_values(s, name=label)
            comp_lists.append(
                [
                    {
                        "label": label,
                        "values": vals,
                        "recipe": {
                            "var": label,
                            "type": "numeric",
                        },
                    }
                ]
            )

    if not needs_expansion:
        return [term], hidden_counter

    if any(len(comps) == 0 for comps in comp_lists):
        return [], hidden_counter

    out_terms = []
    for combo in itertools.product(*comp_lists):
        values = np.ones(n, dtype=np.float64)
        labels = []
        recipe = []

        for c in combo:
            values = values * np.asarray(c["values"], dtype=np.float64)
            labels.append(c["label"])
            recipe.append(dict(c["recipe"]))

        hidden_col = f"__gam_param__{pred_name}__{hidden_counter}"
        hidden_counter += 1
        data_work[hidden_col] = values

        label = ":".join(labels)
        meta = dict(term.metadata)
        meta["parametric_expansion"] = {
            "source_variables": list(src_vars),
            "label": label,
            "hidden_name": hidden_col,
            "recipe": recipe,
        }

        out_terms.append(
            TermSpec(
                kind="parametric",
                features=(hidden_col,),
                by_variable=None,
                basis_options={},
                smoothing_id=None,
                label=label,
                metadata=meta,
            )
        )

        state["parametric_expansions"].append(
            {
                "hidden_name": hidden_col,
                "label": label,
                "source_variables": list(src_vars),
                "recipe": recipe,
            }
        )

    return out_terms, hidden_counter


def _expand_factor_by_term(
    term,
    data_work,
    *,
    pred_name,
    hidden_counter,
    state,
):
    if not isinstance(term, TermSpec) or term.kind != "smooth":
        return [term], hidden_counter

    by_name = term.by_variable
    if by_name is None:
        return [term], hidden_counter

    if by_name not in data_work.columns:
        raise KeyError(
            f"Formula references by-variable {by_name!r}, but it is not in `data`."
        )

    by_series = data_work[by_name]

    if not is_factor_like_series(by_series):
        return [term], hidden_counter

    smooth_spec = term.smooth_spec
    if smooth_spec is None:
        raise ValueError(f"Smooth term {term.label!r} is missing smooth_spec.")

    if str(smooth_spec.special) != "s":
        raise NotImplementedError(
            f"Factor `by` expansion is implemented for s(...) only in this step, "
            f"not for {smooth_spec.special}(...)."
        )

    if isinstance(smooth_spec, FactorSmoothInteractionSpec):
        has_factor_feature = any(
            feature in data_work.columns and is_factor_like_series(data_work[feature])
            for feature in term.features
        )
        if not has_factor_feature:
            smooth_spec = _fs_by_without_factor_feature_base_spec(smooth_spec)

    cat, levels, ordered = factor_info(by_series)
    if len(levels) == 0:
        return [], hidden_counter

    active_levels = levels[1:] if ordered else levels
    original_label = term.label
    out_terms = []

    for lev in active_levels:
        token = safe_token(lev)
        hidden_col = f"__gam_by__{pred_name}__{by_name}__{token}__{hidden_counter}"
        hidden_counter += 1

        data_work[hidden_col] = np.asarray((cat == lev).astype(float), dtype=np.float64)

        new_meta = dict(term.metadata)
        new_meta["factor_by"] = {
            "source_by": by_name,
            "level": lev,
            "ordered": ordered,
            "hidden_by": hidden_col,
            "all_levels": list(levels),
        }

        new_label = f"{original_label}:{by_name}={lev}"
        out_terms.append(
            replace(
                term,
                by_variable=hidden_col,
                smooth_spec=replace_smooth_spec(
                    smooth_spec, constraint_mode="factor_by"
                ),
                label=new_label,
                metadata=new_meta,
            )
        )

        state["factor_by_expansions"].append(
            {
                "predictor_name": pred_name,
                "source_by": by_name,
                "level": lev,
                "ordered": ordered,
                "all_levels": list(levels),
                "hidden_by": hidden_col,
                "original_label": original_label,
                "expanded_label": new_label,
            }
        )

    return out_terms, hidden_counter


def _expand_transformed_smooth_term(
    term,
    data_work,
    *,
    pred_name,
    hidden_counter,
    state,
):
    if not isinstance(term, TermSpec) or term.kind != "smooth":
        return [term], hidden_counter

    if all(_is_bare_formula_name(feature) for feature in term.features):
        return [term], hidden_counter

    # Mirror mgcv::gam.setup model.frame-style handling by materializing
    # transformed smooth covariates once, then reusing stable column names.
    new_features = []
    transforms = []

    for feature in term.features:
        feature_name = str(feature)
        if _is_bare_formula_name(feature_name):
            new_features.append(feature_name)
            continue

        hidden_col, hidden_counter, src_vars = _materialize_formula_expression_column(
            feature_name,
            data_work,
            pred_name=pred_name,
            hidden_counter=hidden_counter,
            state=state,
        )
        transform = {
            "hidden_name": hidden_col,
            "expr": feature_name,
            "source_variables": list(src_vars),
        }
        transforms.append(transform)
        new_features.append(hidden_col)

    new_meta = dict(term.metadata)
    new_meta["formula_feature_expansions"] = [dict(item) for item in transforms]

    return [
        replace(term, features=tuple(new_features), metadata=new_meta)
    ], hidden_counter


def _collect_used_columns_from_predictor_specs(predictor_specs):
    used = set()

    for pred in predictor_specs:
        for term in pred.terms:
            if isinstance(term, TermSpec) and term.kind == "parametric":
                meta = term.metadata or {}
                hidden = meta.get("parametric_expansion", {}).get("hidden_name")
                if hidden is not None:
                    used.add(str(hidden))
                else:
                    src = meta.get("source_variables", None)
                    if src is not None:
                        used.update(str(v) for v in src)
                    else:
                        used.add(term.features[0])
                continue

            if isinstance(term, TermSpec) and term.kind == "smooth":
                used.update(term.features)
                if isinstance(term.by_variable, str):
                    used.add(term.by_variable)
                continue

    return used


def _dataframe_to_feature_matrix(X_df: pd.DataFrame):
    non_numeric = [
        c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
    ]

    if len(non_numeric) == 0:
        X_np = X_df.to_numpy(dtype=np.float64)
        if not np.isfinite(X_np).all():
            raise ValueError("Referenced numeric columns contain NaN or Inf.")
        return X_np

    for c in X_df.columns:
        s = X_df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = np.asarray(s, dtype=np.float64)
            if not np.isfinite(vals).all():
                raise ValueError(
                    f"Referenced numeric column {c!r} contains NaN or Inf."
                )
        else:
            if s.isna().any():
                raise ValueError(
                    f"Referenced non-numeric column {c!r} contains missing values, "
                    "which are not currently supported in fitting."
                )

    return X_df.to_numpy(dtype=object)


def _build_predictor_spec(
    extracted_predictor: ExtractedPredictor,
    *,
    default_k,
    default_basis,
    default_select,
    knots,
    available_columns,
):
    terms = []
    available_column_names = None
    if available_columns is not None:
        available_column_names = {str(col) for col in available_columns}

    for term in extracted_predictor.terms:
        if isinstance(term, ExtractedParametricTerm):
            factor_labels = (
                list(term.factor_labels)
                if getattr(term, "factor_labels", None)
                else [term.raw_label]
            )
            terms.append(
                TermSpec(
                    kind="parametric",
                    features=(term.raw_label,),
                    by_variable=None,
                    basis_options={},
                    smoothing_id=None,
                    label=term.raw_label,
                    metadata={
                        "formula_term": term.raw_label,
                        "factor_labels": list(factor_labels),
                        "source_variables": list(term.variables),
                        "parametric_interaction": len(term.variables) > 1,
                    },
                )
            )
            continue

        if not isinstance(term, ExtractedSmoothTerm):
            raise TypeError(f"Unknown extracted term type: {type(term)}")

        kind = term.kind
        features = list(term.features)
        kw = dict(term.kwargs)

        basis = kw.pop(
            "bs",
            kw.pop("basis", _default_basis_for_kind(kind, default_basis)),
        )
        if "k" in kw:
            k = kw.pop("k")
        else:
            k_basis = basis
            if str(kind).lower() == "s" and str(basis).lower() in {"fs", "sz"}:
                k_basis = _factor_smooth_base_basis_from_xt(kw.get("xt", None)) or basis
            k = _default_k_for_smooth(kind, k_basis, features, default_k)
        by = kw.pop("by", None)
        if by is not None:
            by = str(by)
            if not _is_bare_formula_name(by):
                raise NotImplementedError(
                    "Transformed smooth `by` expressions are parsed exactly, but "
                    "downstream formula building does not yet support them."
                )
            if available_column_names is not None and by not in available_column_names:
                raise KeyError(f"by column {by!r} not found in available data columns.")
        smoothing_id = kw.pop("id", kw.pop("smoothing_id", None))
        fixed = _coerce_fx(kw.pop("fx", False), kind=kind, n_features=len(features))
        select = bool(kw.pop("select", default_select))

        m = kw.pop("m", None)
        xt = kw.pop("xt", None)
        sp = kw.pop("sp", None)
        pc = kw.pop("pc", None)

        mc = kw.pop("mc", None)
        full = kw.pop("full", False)
        ord_ = kw.pop("ord", None)

        if kw:
            raise NotImplementedError(
                f"Unsupported smooth arguments in {term.raw_label!r}: {sorted(kw)}"
            )

        term_knots = _knots_for_features(knots, features)
        terms.append(
            TermSpec(
                kind="smooth",
                features=tuple(str(f) for f in features),
                by_variable=by,
                smooth_spec=build_smooth_spec(
                    special=kind,
                    bs=basis,
                    k=k,
                    fx=fixed,
                    select=select,
                    m=m,
                    xt=xt,
                    sp=sp,
                    pc=pc,
                    knots=term_knots,
                    constraint_mode="auto",
                    shared_basis_setup=None,
                    mc=mc,
                    full=full,
                    ord_=ord_,
                ),
                smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
                label=term.raw_label,
                metadata={"formula_term": term.raw_label},
            )
        )

    return LinearPredictorSpec(
        name=extracted_predictor.predictor_name,
        terms=terms,
        has_intercept=bool(extracted_predictor.intercept),
        offset_name=extracted_predictor.offset_name,
        metadata={
            "raw_formula": extracted_predictor.raw_formula,
            "offset_name": extracted_predictor.offset_name,
            "response_name": extracted_predictor.response_name,
            "intercept": bool(extracted_predictor.intercept),
        },
    )


def _preprocess_predictor_specs(extracted_predictors, predictor_specs, data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula preprocessing requires `data` to be a pandas DataFrame."
        )

    if len(extracted_predictors) != len(predictor_specs):
        raise ValueError(
            f"Extracted predictor count ({len(extracted_predictors)}) does not match "
            f"predictor spec count ({len(predictor_specs)})."
        )

    data_work = data.copy()
    state = {
        "factor_by_expansions": [],
        "formula_expression_columns": [],
        "parametric_expansions": [],
        "_formula_expression_cache": {},
    }

    out_specs = []
    hidden_counter = 0

    for extracted_pred, pred in zip(extracted_predictors, predictor_specs):
        out_terms = []
        include_intercept = bool(extracted_pred.intercept)

        for term in pred.terms:
            if isinstance(term, TermSpec) and term.kind == "parametric":
                expanded, hidden_counter = _expand_parametric_term(
                    term,
                    data_work,
                    pred_name=pred.name,
                    include_intercept=include_intercept,
                    hidden_counter=hidden_counter,
                    state=state,
                )
                out_terms.extend(expanded)
                continue

            if not (isinstance(term, TermSpec) and term.kind == "smooth"):
                out_terms.append(term)
                continue

            transformed_terms, hidden_counter = _expand_transformed_smooth_term(
                term,
                data_work,
                pred_name=pred.name,
                hidden_counter=hidden_counter,
                state=state,
            )
            for transformed_term in transformed_terms:
                expanded, hidden_counter = _expand_factor_by_term(
                    transformed_term,
                    data_work,
                    pred_name=pred.name,
                    hidden_counter=hidden_counter,
                    state=state,
                )
                out_terms.extend(expanded)

        offset_name = pred.offset_name
        if offset_name is not None and not _is_bare_formula_name(offset_name):
            offset_name, hidden_counter, _src_vars = (
                _materialize_formula_expression_column(
                    str(offset_name),
                    data_work,
                    pred_name=pred.name,
                    hidden_counter=hidden_counter,
                    state=state,
                )
            )

        out_specs.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=out_terms,
                has_intercept=bool(getattr(pred, "has_intercept", False)),
                parameter_name=pred.parameter_name,
                offset_name=offset_name,
                metadata=dict(pred.metadata),
            )
        )

    state.pop("_formula_expression_cache", None)
    return out_specs, data_work, state


def build_formula_model(
    extracted_predictors: list[ExtractedPredictor],
    data,
    *,
    y=None,
    knots=None,
    default_k=10,
    default_basis=None,
    default_select=False,
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula-based fitting currently requires `data` to be a pandas DataFrame."
        )

    extracted_predictors = _expand_extracted_predictors_dot_shorthand(
        extracted_predictors, data
    )

    predictor_specs = [
        _build_predictor_spec(
            pred,
            default_k=default_k,
            default_basis=default_basis,
            default_select=default_select,
            knots=knots,
            available_columns=data.columns,
        )
        for pred in extracted_predictors
    ]

    predictor_specs, data_work, preprocess_state = _preprocess_predictor_specs(
        extracted_predictors, predictor_specs, data
    )

    used = _collect_used_columns_from_predictor_specs(predictor_specs)
    used_cols = [c for c in data_work.columns if c in used]
    missing = sorted(used.difference(set(data_work.columns)))
    if missing:
        raise KeyError(f"Formula references columns not present in `data`: {missing}")

    X_df = data_work[used_cols]
    X_np = _dataframe_to_feature_matrix(X_df)
    feature_names = list(X_df.columns)

    response_name = None
    for pred in extracted_predictors:
        if pred.response_name is not None:
            response_name = pred.response_name
            break

    if y is None:
        if response_name is None:
            raise ValueError(
                "Formula does not specify a response, so `y` must be supplied explicitly."
            )
        if _is_bare_formula_name(response_name):
            if response_name not in data_work.columns:
                raise KeyError(
                    f"Response column {response_name!r} not found in `data`."
                )
            y_out = np.asarray(data_work[response_name]).ravel()
        else:
            y_out, _src_vars = _evaluate_formula_numeric_expression(
                response_name, data_work
            )
    else:
        y_out = np.asarray(y).ravel()

    offset_names = tuple(pred.offset_name for pred in predictor_specs)
    offset_out: np.ndarray | list[np.ndarray | None] | None = None
    if len(predictor_specs) <= 1:
        offset_name = offset_names[0] if offset_names else None
        if offset_name is not None:
            offset_out = _resolve_formula_offset_column(data_work, offset_name)
    else:
        offset_list = []
        has_offset = False
        for offset_name in offset_names:
            if offset_name is None:
                offset_list.append(None)
                continue
            offset_list.append(_resolve_formula_offset_column(data_work, offset_name))
            has_offset = True
        if has_offset:
            offset_out = offset_list

    preprocess_state = dict(preprocess_state)
    preprocess_state["used_columns"] = list(used_cols)
    preprocess_state["offset_names"] = offset_names
    preprocess_state["offset_name"] = (
        offset_names[0] if len(offset_names) == 1 else None
    )

    return FormulaBuildResult(
        predictor_specs=predictor_specs,
        working_data=data_work,
        X=X_np,
        feature_names=feature_names,
        used_columns=list(used_cols),
        response=y_out,
        offsets=offset_out,
        preprocess_state=preprocess_state,
        response_name=response_name,
    )


def apply_formula_preprocess_to_new_data(data, preprocess_state):
    if preprocess_state is None:
        return data

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula preprocessing for prediction requires a pandas DataFrame."
        )

    out = data.copy()

    for item in preprocess_state.get("formula_expression_columns", []):
        values, _src_vars = _evaluate_formula_numeric_expression(item["expr"], out)
        out[item["hidden_name"]] = values

    for item in preprocess_state.get("parametric_expansions", []):
        vals = np.ones(len(out), dtype=np.float64)
        for comp in item["recipe"]:
            src = comp["var"]
            if src not in out.columns:
                raise KeyError(
                    f"Prediction data is missing parametric source column {src!r} "
                    f"needed to rebuild {item['hidden_name']!r}."
                )

            if comp["type"] == "numeric":
                vals = vals * numeric_1d_values(out[src], name=src)
            elif comp["type"] == "factor":
                vals = vals * np.asarray(
                    (out[src] == comp["level"]).astype(float), dtype=np.float64
                )
            else:
                raise ValueError(f"Unknown parametric recipe type {comp['type']!r}.")

        out[item["hidden_name"]] = vals

    for item in preprocess_state.get("factor_by_expansions", []):
        src = item["source_by"]
        if src not in out.columns:
            raise KeyError(
                f"Prediction data is missing factor by-variable {src!r} "
                f"needed to rebuild formula columns."
            )

        out[item["hidden_by"]] = np.asarray(
            (out[src] == item["level"]).astype(float), dtype=np.float64
        )

    return out


__all__ = [
    "BuiltPredictorSpec",
    "BuiltTermSpec",
    "FormulaBuildResult",
    "apply_formula_preprocess_to_new_data",
    "build_smooth_spec",
    "build_formula_model",
    "factor_info",
    "is_factor_like_series",
    "numeric_1d_values",
    "safe_token",
    "smooth_spec_from_basis_options",
]
