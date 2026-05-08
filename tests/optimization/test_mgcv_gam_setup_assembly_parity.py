from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.state import _prediction_parameterization_map
from nampy.gam.formula import extract_formula_terms
from nampy.gam.formula.extract import ExtractedParametricTerm
from nampy.gam.smoothing_selection.reparam import (
    _full_coef_indices,
    build_estimate_gam_setup_state,
)
from tests.mgcv_invariant_policy import (
    gam_setup_compares_dominant_penalty_spectrum,
    gam_setup_uses_invariant_transform,
    penalty_spectrum,
)
from tests.mgcv_parity_utils import _run_mgcv_gam_setup_assembly
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    GENERAL_PREOPT_CASES,
)
from tests.optimization.test_mgcv_preoptimization_blocks_parity import PREOPT_CASES


def _coerce_float_array_1d(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr, dtype=np.float64).ravel()


def _coerce_int_array_1d(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr, dtype=np.int64).ravel()


def _coerce_optional_matrix(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty(tuple(int(v) for v in arr.shape), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


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


def _predictor_xlevels(
    predictor_extracted, data: pd.DataFrame
) -> dict[str, tuple[str, ...]]:
    seen: list[str] = []
    for term in predictor_extracted.terms:
        if not isinstance(term, ExtractedParametricTerm):
            continue
        for var in term.variables:
            if str(var) not in seen:
                seen.append(str(var))

    out: dict[str, tuple[str, ...]] = {}
    for var in seen:
        if var not in data.columns:
            continue
        series = data[var]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            out[var] = tuple(str(v) for v in pd.Categorical(series).categories.tolist())
    return out


def _python_xlevels(model: GAM, data: pd.DataFrame):
    extracted = list(extract_formula_terms(model.formula_))
    if len(extracted) == 1:
        return _predictor_xlevels(extracted[0], data)
    return [_predictor_xlevels(predictor, data) for predictor in extracted]


def _predictor_assign(predictor_extracted, compiled_predictor) -> np.ndarray:
    term_order = [
        str(term.raw_label)
        for term in predictor_extracted.terms
        if isinstance(term, ExtractedParametricTerm)
    ]
    order_map = {label: i + 1 for i, label in enumerate(term_order)}

    assign = [0] if bool(compiled_predictor.has_intercept) else []
    for tb in compiled_predictor.compiled_terms:
        if str(tb.term_type) != "parametric":
            continue
        formula_term = str((tb.metadata or {}).get("formula_term", tb.label))
        assign.extend(
            [int(order_map[formula_term])]
            * int(tb.coef_slice.stop - tb.coef_slice.start)
        )
    return np.asarray(assign, dtype=np.int64)


def _python_assign(model: GAM):
    extracted = list(extract_formula_terms(model.formula_))
    predictors = list(model.compiled_model_.predictors)
    assert len(extracted) == len(predictors)
    if len(extracted) == 1:
        return _predictor_assign(extracted[0], predictors[0])
    return [
        _predictor_assign(predictor_extracted, compiled_predictor)
        for predictor_extracted, compiled_predictor in zip(
            extracted, predictors, strict=True
        )
    ]


def _canonical_assign(value):
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value, dtype=np.int64).ravel().tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ()
        first = value[0]
        if isinstance(first, (list, tuple, np.ndarray)) or first is None:
            out: list[int] = []
            for item in value:
                if item is not None:
                    out.extend(_coerce_int_array_1d(item).tolist())
            return tuple(out)
        return tuple(_coerce_int_array_1d(value).tolist())
    return tuple(_coerce_int_array_1d(value).tolist())


def _canonical_xlevels(value):
    if value is None or value == {}:
        return ()
    if isinstance(value, dict):
        return (
            tuple(
                sorted(
                    (str(key), tuple(str(v) for v in vals))
                    for key, vals in dict(value).items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if item is None or item == {}:
                continue
            if isinstance(item, dict):
                entries = tuple(
                    sorted(
                        (str(key), tuple(str(v) for v in vals))
                        for key, vals in dict(item).items()
                    )
                )
                if entries:
                    out.append(entries)
        return tuple(out)
    raise TypeError(f"Unsupported xlevels payload: {type(value).__name__}")


def _canonical_offset(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ()
        first = value[0]
        if first is not None and not isinstance(first, (list, tuple, np.ndarray)):
            return tuple(_coerce_float_array_1d(value).tolist())
        out = []
        for item in value:
            if item is None:
                out.append(None)
            else:
                out.append(tuple(_coerce_float_array_1d(item).tolist()))
        while out and out[-1] is None:
            out.pop()
        return tuple(out)
    return tuple(_coerce_float_array_1d(value).tolist())


def _is_nested_offset_list(value) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return True
    first = value[0]
    return first is None or isinstance(first, (list, tuple, np.ndarray))


def _setup_sp_vector(model: GAM) -> np.ndarray:
    n_sp = int(model.compiled_model_.n_smoothing_params)
    if n_sp == 0:
        return np.empty((0,), dtype=np.float64)
    sp_all = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    out = np.full(n_sp, -1.0, dtype=np.float64)
    out[fixed_mask] = sp_all[fixed_mask]
    return out


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


def _python_smooth_summaries(model: GAM) -> list[dict]:
    compiled = model.compiled_model_
    sp_setup = _setup_sp_vector(model)
    out = []
    penalty_order = {id(pb): i + 1 for i, pb in enumerate(compiled.compiled_penalties)}
    for term_index, tb in enumerate(compiled.compiled_terms):
        if str(tb.term_type) == "parametric":
            continue

        penalties = [
            pb for pb in compiled.compiled_penalties if int(pb.term_index) == term_index
        ]
        full_idx = np.asarray(
            _full_coef_indices(model, tb.coef_slice), dtype=np.int64
        )
        start = int(full_idx[0]) + 1
        stop = int(full_idx[-1]) + 1
        width = int(stop - start + 1)
        rank = tuple(
            int(
                pb.rank
                if pb.rank is not None
                else np.linalg.matrix_rank(np.asarray(pb.matrix, dtype=np.float64))
            )
            for pb in penalties
        )
        penalty_sum = np.zeros((width, width), dtype=np.float64)
        for pb in penalties:
            penalty_sum += np.asarray(pb.matrix, dtype=np.float64)
        null_space_dim = (
            width - int(np.linalg.matrix_rank(penalty_sum)) if penalties else width
        )

        unique_sp_idx: list[int] = []
        for idx in getattr(tb, "smoothing_indices", []):
            idx_int = int(idx)
            if idx_int not in unique_sp_idx:
                unique_sp_idx.append(idx_int)
        penalty_positions = [int(penalty_order[id(pb)]) for pb in penalties]

        metadata = dict(tb.metadata or {})
        factor_by = dict(metadata.get("factor_by", {}) or {})
        by_name = factor_by.get("source_by", None)
        by_level = factor_by.get("level", None)
        if by_name is None and getattr(tb.by_variable_info, "name", None) is not None:
            raw_by = str(tb.by_variable_info.name)
            if not raw_by.startswith("__gam_by__"):
                by_name = raw_by

        term_spec = dict(metadata.get("term_spec", {}) or {})
        basis_options = dict(term_spec.get("basis_options", {}) or {})

        out.append(
            {
                "class_name": _python_smooth_class_name(tb),
                "special": str(basis_options.get("special", "s")),
                "term": tuple(str(v) for v in tb.feature_info.feature_names),
                "by_name": None if by_name is None else str(by_name),
                "by_level": None if by_level is None else str(by_level),
                "id": (
                    None
                    if getattr(tb, "smoothing_group_id", None) is None
                    else str(tb.smoothing_group_id)
                ),
                "first_para": start,
                "last_para": stop,
                "first_sp": None if not penalty_positions else min(penalty_positions),
                "last_sp": None if not penalty_positions else max(penalty_positions),
                "sp": tuple(float(sp_setup[idx]) for idx in unique_sp_idx),
                "rank": rank,
                "null_space_dim": int(null_space_dim),
                "n_penalties": int(len(penalties)),
                "n_coef": int(width),
                "full": (
                    None
                    if "full" not in basis_options
                    else bool(basis_options.get("full"))
                ),
                "ord": (
                    None
                    if basis_options.get("ord", None) is None
                    else tuple(int(v) for v in basis_options["ord"])
                ),
            }
        )
    return out


def _canonical_expected_smooth_summary(value) -> dict:
    return {
        "class_name": str(value["class_name"]),
        "special": str(value["special"]),
        "term": _as_tuple_or_none(value.get("term", None)) or (),
        "by_name": _coerce_optional_str(value.get("by_name", None)),
        "by_level": _coerce_optional_str(value.get("by_level", None)),
        "id": _coerce_optional_str(value.get("id", None)),
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
        "sp": tuple(_coerce_float_array_1d(value.get("sp", None)).tolist()),
        "rank": tuple(_coerce_int_array_1d(value.get("rank", None)).tolist()),
        "null_space_dim": (
            None
            if value.get("null_space_dim", None) is None
            else int(value["null_space_dim"])
        ),
        "n_penalties": int(value.get("n_penalties", 0)),
        "n_coef": None if value.get("n_coef", None) is None else int(value["n_coef"]),
        "full": None if value.get("full", None) is None else bool(value["full"]),
        "ord": (
            None
            if value.get("ord", None) is None
            else tuple(_coerce_int_array_1d(value["ord"]).tolist())
        ),
    }


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


def _block_change_of_basis(
    model: GAM,
    actual_matrix: np.ndarray,
    expected_matrix: np.ndarray,
    expected_smooth: list[dict],
    *,
    matrix_atol: float,
    use_transform: bool,
) -> np.ndarray:
    actual_matrix = np.asarray(actual_matrix, dtype=np.float64)
    expected_matrix = np.asarray(expected_matrix, dtype=np.float64)
    transform = np.eye(actual_matrix.shape[1], dtype=np.float64)
    if not use_transform:
        return transform

    smooth_terms = [
        (term_index, tb)
        for term_index, tb in enumerate(model.compiled_model_.compiled_terms)
        if str(tb.term_type) != "parametric"
    ]
    assert len(smooth_terms) == len(expected_smooth)

    for (term_index, tb), expected_summary in zip(
        smooth_terms, expected_smooth, strict=True
    ):
        del term_index
        full_idx = np.asarray(
            _full_coef_indices(model, tb.coef_slice), dtype=np.int64
        )
        start = int(full_idx[0])
        stop = int(full_idx[-1]) + 1
        assert expected_summary["first_para"] == start + 1
        assert expected_summary["last_para"] == stop
        transform[start:stop, start:stop] = _solve_basis_change(
            actual_matrix[:, start:stop],
            expected_matrix[:, start:stop],
            atol=matrix_atol,
        )

    return transform


def _fit_nampy_model(data, formula, family, method, *, select=False) -> GAM:
    model = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method=method,
        select=select,
    )
    model.fit(data=data)
    return model


def _penalty_atol(case_id: str) -> float:
    if case_id == "gaussian_fs":
        return 1e-8
    if case_id in {"gaussian_tp_two_dim", "gaussian_ts_two_dim"}:
        return 1e-10
    return 1e-12


def _matrix_atol(case_id: str) -> float:
    return 1e-8 if case_id == "gaussian_fs" else 1e-10


def _assert_gam_setup_assembly_case(case_id, data, formula, family, method, *, select):
    model = _fit_nampy_model(data, formula, family, method, select=select)
    actual_setup = build_estimate_gam_setup_state(model)
    expected = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family=family,
        method=method,
        select=select,
    )

    penalty_atol = _penalty_atol(case_id)
    matrix_atol = _matrix_atol(case_id)

    expected_X = np.asarray(expected["X"], dtype=np.float64)
    expected_Xp = np.asarray(expected["Xp"], dtype=np.float64)
    expected_smooth = [
        _canonical_expected_smooth_summary(item)
        for item in (expected.get("smooth", []) or [])
    ]
    use_transform = gam_setup_uses_invariant_transform(case_id)

    actual_X = np.asarray(actual_setup.X, dtype=np.float64)
    actual_Xp = np.asarray(model.lpmatrix(data), dtype=np.float64)
    T_fit = _block_change_of_basis(
        model,
        actual_X,
        expected_X,
        expected_smooth,
        matrix_atol=matrix_atol,
        use_transform=use_transform,
    )
    T_pred = _block_change_of_basis(
        model,
        actual_Xp,
        expected_Xp,
        expected_smooth,
        matrix_atol=matrix_atol,
        use_transform=use_transform,
    )

    np.testing.assert_allclose(actual_X @ T_fit, expected_X, rtol=0.0, atol=matrix_atol)
    np.testing.assert_allclose(
        actual_Xp @ T_pred,
        expected_Xp,
        rtol=0.0,
        atol=matrix_atol,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_setup.off, dtype=np.int64),
        _coerce_int_array_1d(expected.get("off", None)),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_setup.rank, dtype=np.int64),
        _coerce_int_array_1d(expected.get("rank", None)),
    )

    expected_S = [
        np.asarray(item, dtype=np.float64) for item in (expected.get("S", []) or [])
    ]
    assert (
        len(actual_setup.S)
        == len(expected_S)
        == len(model.compiled_model_.compiled_penalties)
    )
    for pb, actual_penalty, expected_penalty in zip(
        model.compiled_model_.compiled_penalties,
        actual_setup.S,
        expected_S,
        strict=True,
    ):
        full_idx = np.asarray(
            _full_coef_indices(model, pb.coef_slice), dtype=np.int64
        )
        start = int(full_idx[0])
        stop = int(full_idx[-1]) + 1
        T_local = T_fit[start:stop, start:stop]
        actual_penalty_t = (
            T_local.T @ np.asarray(actual_penalty, dtype=np.float64) @ T_local
        )
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

    expected_L = _coerce_optional_matrix(expected.get("L", None))
    if expected_L is None:
        assert actual_setup.L is None
    else:
        np.testing.assert_allclose(
            np.asarray(actual_setup.L, dtype=np.float64),
            expected_L,
            rtol=0.0,
            atol=1e-12,
        )

    np.testing.assert_allclose(
        np.asarray(actual_setup.lsp0, dtype=np.float64),
        _coerce_float_array_1d(expected.get("lsp0", None)),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual_setup.sp, dtype=np.float64),
        _coerce_float_array_1d(expected.get("sp", None)),
        rtol=0.0,
        atol=1e-12,
    )

    expected_P = _coerce_optional_matrix(expected.get("P", None))
    actual_P = _prediction_parameterization_map(model)
    if expected_P is None:
        assert actual_P is None
    else:
        assert actual_P is not None
        T_pred_inv = np.linalg.solve(T_pred, np.eye(T_pred.shape[0], dtype=np.float64))
        actual_P_t = T_pred_inv @ np.asarray(actual_P, dtype=np.float64) @ T_fit
        np.testing.assert_allclose(actual_P_t, expected_P, rtol=0.0, atol=1e-10)

    expected_cmX = _coerce_float_array_1d(expected.get("cmX", None))
    actual_cmX = np.asarray(np.mean(actual_Xp, axis=0), dtype=np.float64)
    np.testing.assert_allclose(actual_cmX @ T_pred, expected_cmX, rtol=0.0, atol=1e-10)

    assert _canonical_assign(_python_assign(model)) == _canonical_assign(
        expected.get("assign", None)
    )
    assert _canonical_xlevels(_python_xlevels(model, data)) == _canonical_xlevels(
        expected.get("xlevels", {})
    )
    expected_y = expected.get("y", None)
    if expected_y is not None:
        np.testing.assert_allclose(
            np.asarray(model.y_, dtype=np.float64),
            _coerce_float_array_1d(expected_y),
            rtol=0.0,
            atol=1e-12,
        )

    if _is_nested_offset_list(expected.get("offset", None)) or isinstance(
        getattr(model, "offset_train_", None), (list, tuple)
    ):
        assert _canonical_offset(
            getattr(model, "offset_train_", None)
        ) == _canonical_offset(expected.get("offset", None))
    else:
        expected_offset = expected.get("offset", None)
        expected_offset_arr = (
            np.zeros(data.shape[0], dtype=np.float64)
            if expected_offset is None
            else _coerce_float_array_1d(expected_offset)
        )
        actual_offset_arr = (
            np.zeros(data.shape[0], dtype=np.float64)
            if model.offset_train_ is None
            else np.asarray(model.offset_train_, dtype=np.float64)
        )
        np.testing.assert_allclose(
            actual_offset_arr,
            expected_offset_arr,
            rtol=0.0,
            atol=1e-12,
        )

    actual_smooth = _python_smooth_summaries(model)
    assert actual_smooth == expected_smooth


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, select, _compare_design_space_only",
    PREOPT_CASES,
    ids=[case[0] for case in PREOPT_CASES],
)
def test_gam_setup_assembly_matches_mgcv(
    case_id,
    data_factory,
    formula,
    family,
    method,
    select,
    _compare_design_space_only,
):
    """Verify that gam setup assembly matches mgcv."""
    del _compare_design_space_only
    _assert_gam_setup_assembly_case(
        case_id,
        data_factory(),
        formula,
        family,
        method,
        select=select,
    )


@pytest.mark.parametrize(
    "case_id, family, formula, data_factory, method, select, _compare_design_space_only",
    GENERAL_PREOPT_CASES,
    ids=[case[0] for case in GENERAL_PREOPT_CASES],
)
def test_general_family_gam_setup_assembly_matches_mgcv(
    case_id,
    family,
    formula,
    data_factory,
    method,
    select,
    _compare_design_space_only,
):
    """Verify that general family gam setup assembly matches mgcv."""
    del _compare_design_space_only
    _assert_gam_setup_assembly_case(
        case_id,
        data_factory(),
        formula,
        family,
        method,
        select=select,
    )


def test_gam_setup_assembly_matches_mgcv_for_transformed_formula_surfaces():
    """Verify that gam setup assembly matches mgcv for transformed formula surfaces."""
    data = pd.DataFrame(
        {
            "y": [1.0, 1.5, 2.0, 2.5, 3.0],
            "x": [0.0, 0.5, 1.0, 1.5, 2.0],
            "o": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )

    _assert_gam_setup_assembly_case(
        "gaussian_transformed_formula_surfaces",
        data,
        "I(y**2) ~ I(x**2) + offset(log(o + 1))",
        "gaussian",
        "fixed",
        select=False,
    )
