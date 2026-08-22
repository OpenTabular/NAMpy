from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.compiler.compile_predictors import compile_predictors
from nampy.gam.constraints.identifiability import (
    _penalty_root,
    apply_global_side_conditions,
)
from nampy.gam.specs.modeling import prepare_formula_inputs
from tests.mgcv_invariant_policy import (
    gam_setup_compares_dominant_penalty_spectrum,
    gam_side_uses_invariant_transform,
    penalty_spectrum,
    stable_column_space_projector,
)
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _run_mgcv_gam_setup_assembly,
    _run_mgcv_predict_on_newdata,
)
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    GENERAL_PREOPT_CASES,
)
from tests.optimization.test_mgcv_preoptimization_blocks_parity import (
    PREOPT_CASES,
    _make_factor_by_data,
    _make_numeric_by_data,
)

_UNSTABLE_DEL_INDEX_CASES = {
    # mgcv can report del.index in a constructor-only parameterization that is
    # not preserved as a raw column identity surface in NAMpy after tensor
    # reparameterization/prediction bookkeeping. Compare exact side effects via
    # final blocks, penalties, ranks, and deletion counts instead.
    "nested_te",
}


def test_gam_side_penalty_root_is_checked_through_its_identified_gram_matrix():
    """The augment.smX square-root orientation is not an mgcv contract."""
    penalty = np.asarray(
        [
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    root = _penalty_root(penalty, tol=np.finfo(np.float64).eps**0.5)

    # mgcv::mroot() promises B B' = S. Any orthogonal rotation B Q has the
    # same observable augmentation, so compare the identified Gram matrix.
    rotation = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotated_root = root @ rotation

    np.testing.assert_allclose(root @ root.T, penalty, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        rotated_root @ rotated_root.T,
        penalty,
        atol=2e-14,
        rtol=2e-14,
    )


def _coerce_int_array_1d(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr, dtype=np.int64).ravel()


def _as_tuple_or_none(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _coerce_optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text == "NA" else text


def _canonical_expected_side_smooth(value) -> dict:
    return {
        "class_name": str(value["class_name"]),
        "special": str(value["special"]),
        "label": _coerce_optional_str(value.get("label", None)),
        "term": _as_tuple_or_none(value.get("term", None)) or (),
        "by_name": _coerce_optional_str(value.get("by_name", None)),
        "by_level": _coerce_optional_str(value.get("by_level", None)),
        "id": _coerce_optional_str(value.get("id", None)),
        "dim": None if value.get("dim", None) is None else int(value["dim"]),
        "df": None if value.get("df", None) is None else int(value["df"]),
        "del_index": tuple(
            int(v) - 1 for v in _coerce_int_array_1d(value.get("del_index", None))
        ),
        "first_para": (
            None if value.get("first_para", None) is None else int(value["first_para"])
        ),
        "last_para": (
            None if value.get("last_para", None) is None else int(value["last_para"])
        ),
        "first_sp": (
            None if value.get("first_sp", None) is None else int(value["first_sp"])
        ),
        "last_sp": (
            None if value.get("last_sp", None) is None else int(value["last_sp"])
        ),
        "rank": tuple(_coerce_int_array_1d(value.get("rank", None)).tolist()),
        "null_space_dim": (
            None
            if value.get("null_space_dim", None) is None
            else int(value["null_space_dim"])
        ),
        "n_penalties": int(value.get("n_penalties", 0)),
    }


def _python_smooth_class_name(tb) -> str:
    term_spec = dict((tb.metadata or {}).get("term_spec", {}) or {})
    basis_options = dict(term_spec.get("basis_options", {}) or {})
    special = str(basis_options.get("special", "s")).lower()
    if special in {"te", "ti"}:
        return "tensor.smooth"

    basis = basis_options.get("bs", getattr(tb, "basis_name", None))
    basis_key = str(basis).lower()
    mapping = {
        "cr": "cr.smooth",
        "cs": "cs.smooth",
        "cc": "cyclic.smooth",
        "ps": "pspline.smooth",
        "tp": "tprs.smooth",
        "ts": "ts.smooth",
        "re": "random.effect",
        "fs": "fs.interaction",
        "sz": "sz.interaction",
    }
    return mapping.get(basis_key, f"{basis_key}.smooth")


def _compile_side_state(data: pd.DataFrame, formula, *, select: bool):
    model_like = SimpleNamespace(k=10, basis="tp", select=bool(select))
    (
        _parsed,
        predictor_specs,
        X,
        feature_names,
        _response,
        _used_columns,
        _offsets,
        _preprocess_state,
    ) = prepare_formula_inputs(
        model_like,
        data=data,
        formula=formula,
        y=np.zeros(len(data), dtype=np.float64),
    )
    raw_predictors = compile_predictors(
        X=X,
        feature_names=feature_names,
        predictor_specs=predictor_specs,
    )
    adjusted_predictors = []
    reports = []
    for predictor in raw_predictors:
        adjusted, report = apply_global_side_conditions(
            predictor,
            fit_intercept=bool(predictor.has_intercept),
            tol=float(np.finfo(np.float64).eps**0.5),
            warn=False,
        )
        adjusted_predictors.append(adjusted)
        reports.append(report)
    return raw_predictors, adjusted_predictors, reports


def _fit_side_state(data: pd.DataFrame, formula, family, method, *, select: bool):
    model = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method=method,
        select=select,
    )
    model.fit(data=data)
    return list(model.gam_result_.compiled_model.predictors), list(model.gam_result_.compiled_model.side_condition_reports)


def _predictor_full_matrix(predictor) -> np.ndarray:
    n_obs = predictor.design_matrix.shape[0]
    blocks: list[np.ndarray] = []
    if bool(predictor.has_intercept):
        blocks.append(np.ones((n_obs, 1), dtype=np.float64))
    for tb in predictor.compiled_terms:
        blocks.append(np.asarray(tb.basis_train, dtype=np.float64))
    return np.column_stack(blocks) if blocks else np.empty((n_obs, 0), dtype=np.float64)


def _build_actual_side_surface(raw_predictors, adjusted_predictors, reports):
    if raw_predictors is None:
        raw_predictors = adjusted_predictors

    full_blocks = [_predictor_full_matrix(pred) for pred in adjusted_predictors]
    full_X = (
        np.column_stack(full_blocks)
        if full_blocks
        else np.empty((0, 0), dtype=np.float64)
    )

    all_penalties = []
    for predictor in adjusted_predictors:
        all_penalties.extend(list(predictor.compiled_penalties))
    penalty_order = {id(pb): i + 1 for i, pb in enumerate(all_penalties)}

    smooths = []
    full_col = 1
    for raw_predictor, predictor, report in zip(
        raw_predictors, adjusted_predictors, reports, strict=True
    ):
        if bool(predictor.has_intercept):
            full_col += 1

        raw_by_term_id = {str(tb.term_id): tb for tb in raw_predictor.compiled_terms}
        report_by_term_id = {
            str(tb.term_id): term_report
            for tb, term_report in zip(
                raw_predictor.compiled_terms, report["term_reports"], strict=True
            )
        }

        for term_index, tb in enumerate(predictor.compiled_terms):
            width = int(tb.basis_train.shape[1])
            start = full_col
            stop = full_col + width - 1
            full_col += width

            if str(tb.term_type) == "parametric":
                continue

            term_penalties = [
                pb
                for pb in predictor.compiled_penalties
                if int(pb.term_index) == term_index
            ]
            rank = tuple(
                int(
                    pb.rank
                    if pb.rank is not None
                    else np.linalg.matrix_rank(np.asarray(pb.matrix, dtype=np.float64))
                )
                for pb in term_penalties
            )
            penalty_sum = np.zeros((width, width), dtype=np.float64)
            for pb in term_penalties:
                penalty_sum += np.asarray(pb.matrix, dtype=np.float64)
            null_space_dim = (
                width - int(np.linalg.matrix_rank(penalty_sum))
                if term_penalties
                else width
            )

            metadata = dict(tb.metadata or {})
            factor_by = dict(metadata.get("factor_by", {}) or {})
            by_name = factor_by.get("source_by", None)
            by_level = factor_by.get("level", None)
            if (
                by_name is None
                and getattr(tb.by_variable_info, "name", None) is not None
            ):
                raw_by = str(tb.by_variable_info.name)
                if not raw_by.startswith("__gam_by__"):
                    by_name = raw_by

            term_spec = dict(metadata.get("term_spec", {}) or {})
            basis_options = dict(term_spec.get("basis_options", {}) or {})
            report_entry = report_by_term_id[str(tb.term_id)]

            smooths.append(
                {
                    "class_name": _python_smooth_class_name(tb),
                    "special": str(basis_options.get("special", "s")),
                    "label": str(tb.label),
                    "term": tuple(str(v) for v in tb.feature_info.feature_names),
                    "by_name": None if by_name is None else str(by_name),
                    "by_level": None if by_level is None else str(by_level),
                    "id": (
                        None
                        if getattr(tb, "smoothing_group_id", None) is None
                        else str(tb.smoothing_group_id)
                    ),
                    "dim": int(len(tb.feature_info.feature_names)),
                    "df": int(width),
                    "del_index": tuple(
                        int(v)
                        for v in (
                            []
                            if tb.deleted_columns is None
                            else np.asarray(tb.deleted_columns, dtype=np.int64).tolist()
                        )
                    ),
                    "first_para": start,
                    "last_para": stop,
                    "first_sp": (
                        None
                        if not term_penalties
                        else min(int(penalty_order[id(pb)]) for pb in term_penalties)
                    ),
                    "last_sp": (
                        None
                        if not term_penalties
                        else max(int(penalty_order[id(pb)]) for pb in term_penalties)
                    ),
                    "rank": rank,
                    "null_space_dim": int(null_space_dim),
                    "n_penalties": int(len(term_penalties)),
                    "penalties": [
                        np.asarray(pb.matrix, dtype=np.float64) for pb in term_penalties
                    ],
                    "raw_basis": np.asarray(
                        raw_by_term_id[str(tb.term_id)].basis_train,
                        dtype=np.float64,
                    ),
                    "basis": np.asarray(tb.basis_train, dtype=np.float64),
                    "absorbed_centering": bool(report_entry["absorbed_centering"]),
                }
            )

    return {"X": full_X, "smooths": smooths}


def _matrix_atol_for_class(class_name: str) -> float:
    return 1e-8 if class_name == "fs.interaction" else 1e-10


def _penalty_atol_for_class(class_name: str) -> float:
    if class_name == "fs.interaction":
        return 1e-8
    if class_name == "sz.interaction":
        # SZ marginal eigenvectors are identified only up to sign.  After the
        # explicit sign congruence above, the remaining difference is the
        # accumulated LAPACK eigensolver roundoff in the penalty entries.
        return 1e-11
    if class_name in {"tprs.smooth", "ts.smooth"}:
        return 3e-10
    return 1e-12


def _solve_basis_change(actual_block, expected_block, *, atol: float) -> np.ndarray:
    actual_block = np.asarray(actual_block, dtype=np.float64)
    expected_block = np.asarray(expected_block, dtype=np.float64)
    if actual_block.shape[1] == 0:
        return np.eye(0, dtype=np.float64)
    transform, *_ = np.linalg.lstsq(actual_block, expected_block, rcond=None)
    np.testing.assert_allclose(
        actual_block @ transform,
        expected_block,
        rtol=0.0,
        atol=atol,
    )
    return np.asarray(transform, dtype=np.float64)


def _solve_column_sign_change(actual_block, expected_block, *, atol: float) -> np.ndarray:
    """Align an identified basis whose eigenvectors differ only by column sign."""
    actual_block = np.asarray(actual_block, dtype=np.float64)
    expected_block = np.asarray(expected_block, dtype=np.float64)
    dots = np.sum(actual_block * expected_block, axis=0)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    transform = np.diag(signs)
    np.testing.assert_allclose(
        actual_block @ transform,
        expected_block,
        rtol=0.0,
        atol=atol,
    )
    return transform


def _constant_projection_residual(block: np.ndarray) -> float:
    block = np.asarray(block, dtype=np.float64)
    if block.shape[1] == 0:
        return float(np.sqrt(block.shape[0]))
    ones = np.ones(block.shape[0], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(block, ones, rcond=None)
    resid = ones - block @ coef
    return float(np.linalg.norm(resid))


def _make_repeat_cr_ps_data(seed=510, n=180):
    data = _make_gaussian_data(seed=seed, n=n)[["y", "x0"]].copy()
    return data.rename(columns={"x0": "x"})


def _make_re_plus_nested_tp_data(seed=511, n_levels=7, n_rep=18):
    rng = np.random.default_rng(seed)
    levels = [f"g{i}" for i in range(n_levels)]
    f = pd.Categorical(np.repeat(levels, n_rep))
    n = len(f)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    re = {level: rng.normal(scale=0.35) for level in levels}
    y = np.array([re[str(level)] for level in f], dtype=np.float64)
    y += np.sin(0.8 * x0) + 0.25 * x0 * x1
    y += rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "f": f, "x0": x0, "x1": x1})


def _make_three_way_nested_data(seed=515, n=180):
    data = _make_gaussian_data(seed=seed, n=n)
    rng = np.random.default_rng(seed)
    data["x2"] = rng.uniform(-1.25, 1.25, size=n)
    return data


def _make_near_rank_boundary_data(seed=516, n=180, epsilon=1e-11):
    data = _make_gaussian_data(seed=seed, n=n)[["y", "x0"]].copy()
    rng = np.random.default_rng(seed)
    data["x_near"] = data["x0"].to_numpy() + float(epsilon) * rng.normal(size=n)
    return data


def _make_ordered_factor_by_data(seed=518, n=210):
    data = _make_factor_by_data(seed=seed, n=n)
    data["f"] = pd.Categorical(
        data["f"],
        categories=["a", "b", "c"],
        ordered=True,
    )
    return data


SIDE_CASES = [
    (
        "repeat_cr_ps",
        _make_repeat_cr_ps_data,
        'y ~ s(x, bs="cr", k=8) + s(x, bs="ps", k=10)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "nested_tp",
        lambda: _make_gaussian_data(seed=512, n=180),
        'y ~ s(x0, bs="cr", k=8) + s(x0, x1, bs="tp", k=15)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "nested_tp_reversed_formula_order",
        lambda: _make_gaussian_data(seed=512, n=180),
        'y ~ s(x0, x1, bs="tp", k=15) + s(x0, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "nested_ts",
        lambda: _make_gaussian_data(seed=513, n=180),
        'y ~ s(x0, bs="cr", k=8) + s(x0, x1, bs="ts", k=15)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "nested_te",
        lambda: _make_gaussian_data(seed=514, n=180),
        'y ~ s(x0, bs="cr", k=8) + te(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "main_ti",
        lambda: _make_gaussian_data(seed=517, n=180),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8) + ti(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "re_plus_nested_tp",
        _make_re_plus_nested_tp_data,
        'y ~ s(f, bs="re") + s(x0, bs="cr", k=8) + s(x0, x1, bs="tp", k=15)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "near_rank_boundary_cr",
        _make_near_rank_boundary_data,
        'y ~ s(x0, bs="cr", k=8) + s(x_near, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "zero_width_duplicate_cr",
        _make_repeat_cr_ps_data,
        'y ~ s(x, bs="cr", k=8) + s(x, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_two_cr",
        lambda: _make_gaussian_data(seed=601, n=180),
        'y ~ 0 + s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_repeat_cr_ps",
        lambda: _make_gaussian_data(seed=602, n=180)[["y", "x0"]].rename(
            columns={"x0": "x"}
        ),
        'y ~ 0 + s(x, bs="cr", k=8) + s(x, bs="ps", k=10)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_nested_tp",
        lambda: _make_gaussian_data(seed=603, n=180),
        'y ~ 0 + s(x0, bs="cr", k=8) + s(x0, x1, bs="tp", k=15)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_main_ti",
        lambda: _make_gaussian_data(seed=605, n=180),
        'y ~ 0 + s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8) + ti(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_numeric_by_cr",
        lambda: _make_numeric_by_data(seed=606, n=180),
        'y ~ 0 + s(x, by=z, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_no_intercept_factor_by_cr",
        lambda: _make_factor_by_data(seed=607, n=240),
        'y ~ 0 + f + s(x, by=f, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        True,
    ),
]


def _assert_gam_side_case(
    case_id,
    data,
    formula,
    family,
    method,
    *,
    select,
    use_model_fit: bool,
):
    if use_model_fit:
        adjusted_predictors, reports = _fit_side_state(
            data,
            formula,
            family,
            method,
            select=select,
        )
        raw_predictors = None
    else:
        raw_predictors, adjusted_predictors, reports = _compile_side_state(
            data,
            formula,
            select=select,
        )
    actual = _build_actual_side_surface(raw_predictors, adjusted_predictors, reports)
    expected = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family=family,
        method=method,
        select=select,
    )

    expected_X = np.asarray(expected["X"], dtype=np.float64)
    expected_S = [
        np.asarray(item, dtype=np.float64) for item in (expected.get("S", []) or [])
    ]
    expected_smooths = [
        _canonical_expected_side_smooth(item)
        for item in (expected.get("smooth", []) or [])
    ]

    if case_id in _UNSTABLE_DEL_INDEX_CASES:
        np.testing.assert_allclose(
            stable_column_space_projector(actual["X"]),
            stable_column_space_projector(expected_X),
            rtol=0.0,
            atol=2e-10,
            err_msg=(
                "side-condition deletion sets: "
                f"actual={[sm['del_index'] for sm in actual['smooths']]}, "
                f"expected={[sm['del_index'] for sm in expected_smooths]}"
            ),
        )

    assert len(actual["smooths"]) == len(expected_smooths)
    assert sum(len(sm["del_index"]) for sm in actual["smooths"]) == sum(
        len(sm["del_index"]) for sm in expected_smooths
    )
    assert sum(int(report["n_deleted_total"]) for report in reports) == sum(
        len(sm["del_index"]) for sm in expected_smooths
    )

    for actual_sm, expected_sm in zip(actual["smooths"], expected_smooths, strict=True):
        assert actual_sm["class_name"] == expected_sm["class_name"]
        assert actual_sm["special"] == expected_sm["special"]
        assert actual_sm["term"] == expected_sm["term"]
        assert actual_sm["by_name"] == expected_sm["by_name"]
        assert actual_sm["by_level"] == expected_sm["by_level"]
        assert actual_sm["id"] == expected_sm["id"]
        if expected_sm["class_name"] != "fs.interaction":
            assert actual_sm["dim"] == expected_sm["dim"]
        if expected_sm["class_name"] not in {"fs.interaction", "sz.interaction"}:
            assert actual_sm["df"] == expected_sm["df"]
        assert actual_sm["first_para"] == expected_sm["first_para"]
        assert actual_sm["last_para"] == expected_sm["last_para"]
        assert actual_sm["first_sp"] == expected_sm["first_sp"]
        assert actual_sm["last_sp"] == expected_sm["last_sp"]
        assert actual_sm["rank"] == expected_sm["rank"]
        assert actual_sm["null_space_dim"] == expected_sm["null_space_dim"]
        assert actual_sm["n_penalties"] == expected_sm["n_penalties"]

        if (
            case_id not in _UNSTABLE_DEL_INDEX_CASES
            or len(expected_sm["del_index"]) == 0
        ):
            assert actual_sm["del_index"] == expected_sm["del_index"]
        else:
            assert len(actual_sm["del_index"]) == len(expected_sm["del_index"])

        start = int(expected_sm["first_para"]) - 1
        stop = int(expected_sm["last_para"])
        actual_block = np.asarray(actual["X"][:, start:stop], dtype=np.float64)
        expected_block = np.asarray(expected_X[:, start:stop], dtype=np.float64)

        matrix_atol = _matrix_atol_for_class(expected_sm["class_name"])
        deletion_basis_is_nonunique = (
            case_id in _UNSTABLE_DEL_INDEX_CASES
            and len(expected_sm["del_index"]) > 0
        )
        if deletion_basis_is_nonunique:
            actual_prior = np.asarray(actual["X"][:, :start], dtype=np.float64)
            expected_prior = np.asarray(expected_X[:, :start], dtype=np.float64)
            actual_unique = actual_block - actual_prior @ np.linalg.lstsq(
                actual_prior, actual_block, rcond=None
            )[0]
            expected_unique = expected_block - expected_prior @ np.linalg.lstsq(
                expected_prior, expected_block, rcond=None
            )[0]
            T = _solve_basis_change(actual_unique, expected_unique, atol=2e-9)
        elif expected_sm["class_name"] == "sz.interaction":
            T = _solve_column_sign_change(
                actual_block, expected_block, atol=matrix_atol
            )
        elif gam_side_uses_invariant_transform(expected_sm["class_name"]):
            T = _solve_basis_change(actual_block, expected_block, atol=matrix_atol)
        else:
            T = np.eye(actual_block.shape[1], dtype=np.float64)
        if not deletion_basis_is_nonunique:
            np.testing.assert_allclose(
                actual_block @ T,
                expected_block,
                rtol=0.0,
                atol=matrix_atol,
            )

        if not deletion_basis_is_nonunique:
            np.testing.assert_allclose(
                _constant_projection_residual(actual_block),
                _constant_projection_residual(expected_block),
                rtol=0.0,
                atol=1e-8,
            )

        first_sp = expected_sm["first_sp"]
        last_sp = expected_sm["last_sp"]
        if first_sp is None or last_sp is None:
            assert actual_sm["penalties"] == []
            continue

        expected_term_penalties = expected_S[int(first_sp) - 1 : int(last_sp)]
        assert len(actual_sm["penalties"]) == len(expected_term_penalties)
        penalty_atol = _penalty_atol_for_class(expected_sm["class_name"])
        transformed_penalties = [
            T.T @ np.asarray(penalty, dtype=np.float64) @ T
            for penalty in actual_sm["penalties"]
        ]
        if expected_sm["class_name"] == "fs.interaction":
            # mgcv/R/smooth.r::smooth.construct.fs.smooth.spec forms one
            # penalty for each natural-parameter null direction. Those
            # eigendirections have no unique orientation, so individual null
            # penalties can rotate while their spectra and combined penalty
            # remain fixed.
            for actual_penalty, expected_penalty in zip(
                transformed_penalties, expected_term_penalties, strict=True
            ):
                np.testing.assert_allclose(
                    penalty_spectrum(actual_penalty),
                    penalty_spectrum(expected_penalty),
                    rtol=0.0,
                    atol=penalty_atol,
                )
            np.testing.assert_allclose(
                np.sum(transformed_penalties, axis=0),
                np.sum(expected_term_penalties, axis=0),
                rtol=0.0,
                atol=penalty_atol,
            )
            continue
        for actual_penalty_t, expected_penalty in zip(
            transformed_penalties, expected_term_penalties, strict=True
        ):
            if gam_setup_compares_dominant_penalty_spectrum(case_id):
                actual_spectrum = penalty_spectrum(actual_penalty_t)
                expected_spectrum = penalty_spectrum(expected_penalty)
                assert actual_spectrum[0] > 0.0
                assert expected_spectrum[0] > 0.0
                assert actual_spectrum[0] < 0.1 * actual_spectrum[1]
                assert expected_spectrum[0] < 0.1 * expected_spectrum[1]
                np.testing.assert_allclose(
                    actual_spectrum[1:],
                    expected_spectrum[1:],
                    rtol=0.0,
                    atol=max(penalty_atol, 2e-4),
                )
            else:
                np.testing.assert_allclose(
                    actual_penalty_t,
                    expected_penalty,
                    rtol=0.0,
                    atol=penalty_atol,
                )


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, select, _compare_design_space_only",
    PREOPT_CASES,
    ids=[case[0] for case in PREOPT_CASES],
)
def test_gam_side_matches_mgcv_current_non_general_case_matrix(
    case_id,
    data_factory,
    formula,
    family,
    method,
    select,
    _compare_design_space_only,
):
    """Verify that gam side matches mgcv current non general case matrix."""
    del _compare_design_space_only
    _assert_gam_side_case(
        case_id,
        data_factory(),
        formula,
        family,
        method,
        select=select,
        use_model_fit=True,
    )


@pytest.mark.parametrize(
    "case_id, family, formula, data_factory, method, select, _compare_design_space_only",
    GENERAL_PREOPT_CASES,
    ids=[case[0] for case in GENERAL_PREOPT_CASES],
)
def test_gam_side_matches_mgcv_current_general_case_matrix(
    case_id,
    family,
    formula,
    data_factory,
    method,
    select,
    _compare_design_space_only,
):
    """Verify that gam side matches mgcv current general case matrix."""
    del _compare_design_space_only
    _assert_gam_side_case(
        case_id,
        data_factory(),
        formula,
        family,
        method,
        select=select,
        use_model_fit=True,
    )


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, select, use_model_fit",
    SIDE_CASES,
    ids=[case[0] for case in SIDE_CASES],
)
def test_gam_side_matches_mgcv_nested_side_condition_cases(
    case_id,
    data_factory,
    formula,
    family,
    method,
    select,
    use_model_fit,
):
    """Verify that gam side matches mgcv nested side condition cases."""
    _assert_gam_side_case(
        case_id,
        data_factory(),
        formula,
        family,
        method,
        select=select,
        use_model_fit=use_model_fit,
    )


def test_three_way_nested_side_conditions_match_mgcv_behavior_invariantly():
    """Three-way nesting compares deletion rank and fitted behavior, not pivot columns."""
    data = _make_three_way_nested_data()
    formula = (
        'y ~ s(x0, bs="cr", k=7, sp=0.8)'
        ' + s(x0, x1, bs="tp", k=12, sp=1.1)'
        ' + s(x0, x1, x2, bs="tp", k=20, sp=1.3)'
    )
    raw, adjusted, reports = _compile_side_state(data, formula, select=False)
    actual_side = _build_actual_side_surface(raw, adjusted, reports)
    expected_side = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family="gaussian",
        method="fixed",
        select=False,
    )
    expected_smooths = [
        _canonical_expected_side_smooth(item) for item in expected_side["smooth"]
    ]

    assert [len(sm["del_index"]) for sm in actual_side["smooths"]] == [0, 1, 6]
    assert [len(sm["del_index"]) for sm in expected_smooths] == [0, 1, 6]
    assert [sm["df"] for sm in actual_side["smooths"]] == [
        sm["df"] for sm in expected_smooths
    ]
    assert [sm["rank"] for sm in actual_side["smooths"]] == [
        sm["rank"] for sm in expected_smooths
    ]
    for predictor in adjusted:
        for term in predictor.compiled_terms:
            np.testing.assert_allclose(
                term.predict_matrix(data[["x0", "x1", "x2"]].to_numpy()),
                term.basis_train,
                rtol=0.0,
                atol=2e-10,
            )

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[::9].drop(columns=["y"]).copy()
    actual_fit, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
    )
    # The final six-column deletion is selected inside a numerically repeated
    # augmented QR space. Exact pivot columns and the augment.smX root are not
    # identified across LAPACK implementations; keep the downstream difference
    # below 0.3% of the response scale without selecting one platform's root.
    np.testing.assert_allclose(
        actual_fit, np.asarray(expected["pred"]).ravel(), rtol=0.0, atol=3e-3
    )
    np.testing.assert_allclose(
        actual_se, np.asarray(expected["se"]).ravel(), rtol=0.0, atol=3e-3
    )


@pytest.mark.parametrize(
    "epsilon",
    [1e-13, 1e-7],
    ids=["effectively_singular", "identified_two_dimensional"],
)
def test_gam_side_rank_threshold_neighborhood_matches_mgcv(epsilon):
    """Near-degenerate nested coordinates preserve identified side effects."""
    data = _make_near_rank_boundary_data(epsilon=epsilon)
    formula = (
        'y ~ s(x0, bs="cr", k=8, sp=0.8)'
        ' + s(x0, x_near, bs="tp", k=12, sp=1.1)'
    )
    raw, adjusted, reports = _compile_side_state(data, formula, select=False)
    actual = _build_actual_side_surface(raw, adjusted, reports)
    expected = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family="gaussian",
        method="fixed",
        select=False,
    )
    expected_smooths = [
        _canonical_expected_side_smooth(item) for item in expected["smooth"]
    ]

    assert [len(item["del_index"]) for item in actual["smooths"]] == [
        len(item["del_index"]) for item in expected_smooths
    ]
    assert [item["df"] for item in actual["smooths"]] == [
        item["df"] for item in expected_smooths
    ]
    assert [item["rank"] for item in actual["smooths"]] == [
        item["rank"] for item in expected_smooths
    ]
    assert [item["null_space_dim"] for item in actual["smooths"]] == [
        item["null_space_dim"] for item in expected_smooths
    ]

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[2::17].drop(columns=["y"]).copy()
    actual_prediction, actual_se = gam.predict(
        newdata, type="response", return_se=True
    )
    expected_prediction = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual_prediction,
        np.asarray(expected_prediction["pred"]).ravel(),
        atol=2e-3,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected_prediction["se"]).ravel(),
        atol=2e-3,
        rtol=2e-3,
    )


def test_gam_side_multi_predictor_blocks_match_mgcv_independently():
    """Each distributional predictor owns its own nested side-condition pass."""
    data = _make_gaussian_data(seed=604, n=180)
    formula = [
        'y ~ s(x0, bs="cr", k=7) + s(x0, x1, bs="tp", k=12)',
        '~ s(x1, bs="cr", k=7) + s(x0, x1, bs="tp", k=12)',
    ]
    _assert_gam_side_case(
        "gaulss_two_predictor_nested",
        data,
        formula,
        "gaulss",
        "ML",
        select=False,
        use_model_fit=False,
    )


def test_ordered_factor_by_keeps_nonreference_levels_with_mgcv_blocks():
    """Ordered factor-by smooths omit the first level and retain mgcv's other blocks."""
    data = _make_ordered_factor_by_data()
    formula = 'y ~ s(x, by=f, bs="cr", k=8)'
    raw, adjusted, reports = _compile_side_state(data, formula, select=False)
    actual = _build_actual_side_surface(raw, adjusted, reports)

    # CSV does not preserve pandas' ordered categorical metadata, so the live R
    # setup contains all three unordered levels.  The b/c blocks themselves are
    # the exact mgcv reference for the ordered model's non-reference levels.
    expected = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family="gaussian",
        method="REML",
        select=False,
    )
    expected_smooths = [
        _canonical_expected_side_smooth(item) for item in expected["smooth"]
    ]

    assert [sm["by_level"] for sm in actual["smooths"]] == ["b", "c"]
    assert [sm["by_level"] for sm in expected_smooths] == ["a", "b", "c"]
    expected_X = np.asarray(expected["X"], dtype=np.float64)
    expected_bc = expected_X[:, 8:22]
    np.testing.assert_allclose(actual["X"][:, 1:], expected_bc, rtol=0.0, atol=1e-12)
    for actual_sm, expected_penalty in zip(
        actual["smooths"], expected["S"][1:], strict=True
    ):
        np.testing.assert_allclose(
            actual_sm["penalties"][0],
            np.asarray(expected_penalty, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
