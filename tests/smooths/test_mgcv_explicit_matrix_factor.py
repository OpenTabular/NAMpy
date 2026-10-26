"""Explicit pandas categorical rows mirror R factor+dim fs/sz arguments."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.contracts import FeatureSchema
from nampy.gam import GAM
from nampy.gam.data import categorical_matrix_metadata
from nampy.gam.model_state import _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_explicit_matrix_factor"
PROBE = Path(__file__).with_name("probes") / "mgcv_explicit_matrix_factor.R"
ROUTES = [
    ("sz", "cr"),
    ("fs", "ps"),
    ("sz", "ps"),
    ("fs", "sf"),
    ("sz", "sf"),
    ("sz", "sw"),
    ("sz", "so"),
    ("fs", "tp"),
    ("sz", "tp"),
]


def case(
    mode="fs",
    basis="sf",
    method="fixed",
    varying=True,
    kind="varying",
    new_width=3,
    special=None,
):
    frozen = json.loads((ROOT / "inputs.json").read_text())
    data, new = frozen["data"], frozen["newdata"]
    levels = frozen["levels"]
    for target, width in [(data, 3), (new, new_width)]:
        codes = np.array([levels.index(v) for v in target["f"]])[:, None]
        codes = np.repeat(codes, width, axis=1)
        if varying:
            codes[:, 1] = (codes[:, 1] + 1) % len(levels)
        target["f"] = np.asarray(levels)[codes].tolist()
        for name in ("X", "Y", "L"):
            target[name] = [row[:width] for row in target[name]]
        if kind == "unit":
            target["L"] = np.ones_like(target["L"]).tolist()
        if kind == "constant":
            weights = np.array(target["L"])
            weights[weights.sum(axis=1) == 0] = 1
            target["L"] = (weights / weights.sum(axis=1)[:, None]).tolist()
    # Repeated locations exercise ExtractData's packing with varying factor cells.
    for name in ("X", "Y"):
        data[name] = np.resize(np.array(data[name])[:8], np.shape(data[name])).tolist()
    data.pop("noise")
    data["response"] = frozen["responses"][f"{kind}/True"]
    boundary = [frozen["outer"]]
    if basis == "so":
        boundary[0]["f"] = [-1.0, 1.0, 1.0, -1.0, -1.0]
    if special is not None:
        new["f"][0][0] = None if special == "missing" else "unseen"
    return {
        "data": data,
        "newdata": new,
        "levels": levels,
        "knots": frozen["knots"],
        "boundary": boundary,
        "mode": mode,
        "basis": basis,
        "method": method,
        "kind": kind,
        "varying": varying,
        "new_width": new_width,
        "special": special,
    }


def reference(p):
    key = reference_key(
        "explicit-matrix-factor-v1",
        {"case": p, "source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
    )
    ref = load_reference("mgcv_explicit_matrix_factor", key)
    if ref is None:
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = [Path(tmp) / n for n in ("in.json", "out.json")]
            inp.write_text(json.dumps(p))
            run = subprocess.run(
                ["Rscript", str(PROBE), str(inp), str(out)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            ref = json.loads(out.read_text())
        save_reference("mgcv_explicit_matrix_factor", key, ref)
    return ref


def frame(data, levels, ordered=False):
    out = pd.DataFrame({k: list(v) for k, v in data.items()})
    for name in ("f", "g"):
        if name in data:
            out[name] = pd.Series(
                [
                    pd.Categorical(row, categories=levels, ordered=ordered)
                    for row in data[name]
                ],
                dtype=object,
            )
    return out


def model_for(p, ref):
    soap = p["basis"] in {"so", "sf", "sw"}
    xt = {"bs": p["basis"]}
    if soap:
        xt.update(bnd=p["boundary"], nmax=30)
    order = (
        ",m=1"
        if p["basis"] == "tp"
        else ",m=[2,1]"
        if p["basis"] == "ps" and not p.get("default_ps_order")
        else ""
    )
    other_factor = ",g" if "g" in p["data"] else ""
    return GAM(
        formula=f"response~s({'X,Y' if soap else 'X'},f{other_factor},bs='{p['mode']}',k=8,xt={xt!r}{',by=L' if p['kind'] != 'unit' else ''}{order})+offset(o)",
        knots=p["knots"] if soap else None,
        smoothing_method=p["method"],
        optimize_smoothing=p["method"] != "fixed",
        smoothing_params=np.atleast_1d(ref["sp"]) if p["method"] == "fixed" else None,
    )


def assert_invariants(model, p, ref):
    new = frame(p["newdata"], p["levels"])
    fit, se = model.predict(new, return_se=True)
    np.testing.assert_allclose(
        fit, np.asarray(ref["fit"], dtype=float), atol=4e-5, rtol=4e-5
    )
    np.testing.assert_allclose(
        se, np.asarray(ref["se"], dtype=float), atol=4e-5, rtol=4e-5
    )
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=4e-5, rtol=4e-5)
    X, P = model.predict(type="lpmatrix"), model.lpmatrix(new)
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T,
        np.asarray(ref["covariance"], dtype=float),
        atol=4e-5,
        rtol=4e-5,
    )
    transform = np.linalg.lstsq(X[:, 1:], np.asarray(ref["X"])[:, 1:], rcond=None)[0]
    np.testing.assert_allclose(
        X[:, 1:] @ transform, np.asarray(ref["X"])[:, 1:], atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        P[:, 1:] @ transform,
        np.asarray(ref["Xp"], dtype=float)[:, 1:],
        atol=2e-7,
        rtol=2e-7,
    )
    transported = [
        transform.T @ penalty.matrix @ transform
        for penalty in _penalty_blocks_seq(model)
    ]
    for actual, target in zip(transported, ref["S"], strict=True):
        np.testing.assert_allclose(actual, target, atol=3e-6, rtol=3e-6)
    # Use common units for the SP-weighted operator: large selected SPs must not
    # amplify transport roundoff at structurally zero entries into a new metric.
    actual = sum(
        lam * S for lam, S in zip(model.smoothing_params, transported, strict=True)
    )
    expected = sum(
        lam * np.asarray(S)
        for lam, S in zip(np.atleast_1d(ref["sp"]), ref["S"], strict=True)
    )
    scale = max(1.0, np.linalg.norm(expected))
    np.testing.assert_allclose(actual / scale, expected / scale, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode,basis", ROUTES)
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("varying", [False, True])
def test_explicit_matrix_factor_source_fit_invariants(mode, basis, method, varying):
    p = case(mode, basis, method, varying)
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)


@pytest.mark.parametrize("basis", ["cr", "ps"])
def test_matrix_factor_fs_multiple_null_directions_remain_guarded(basis):
    # Actual source accepts these. Source natural-null ordering is nonunique;
    # unequal source SPs expose different fitted penalties under a permutation.
    p = case("fs", basis)
    p.update(fixed_sp=[0.03, 0.2, 2.0], default_ps_order=True)
    ref = reference(p)
    assert ref["accepted"], ref
    with pytest.raises(NotImplementedError, match="natural null directions"):
        model_for(p, ref).fit(data=frame(p["data"], p["levels"]))


@pytest.mark.parametrize(
    "mode,basis",
    [
        ("fs", "ps"),
        ("fs", "sf"),
        ("fs", "tp"),
        ("sz", "sf"),
        ("sz", "sw"),
        ("sz", "so"),
    ],
)
def test_matrix_factor_fixed_distinct_penalty_weights_match_source(mode, basis):
    p = case(mode, basis)
    p["fixed_sp"] = [0.03, 2.0] if mode == "fs" else [0.03, 0.2, 2.0]
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)


@pytest.mark.parametrize(
    "mode,basis",
    [
        ("fs", "ps"),
        ("sz", "cr"),
        ("fs", "sf"),
        ("sz", "sf"),
        ("sz", "sw"),
        ("sz", "so"),
    ],
)
@pytest.mark.parametrize("kind", ["unit", "constant"])
def test_matrix_factor_changed_width_and_functional_weight_routes(mode, basis, kind):
    p = case(mode, basis, "REML", kind=kind, new_width=2)
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)


@pytest.mark.parametrize(
    "mode,basis", [("fs", "sf"), ("sz", "sf"), ("sz", "sw"), ("sz", "so")]
)
def test_matrix_factor_point_sum_levels_and_persistence(mode, basis, tmp_path):
    p = case(mode, basis, "REML", new_width=2)
    ref = reference(p)
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"], ordered=True), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)
    new = frame(p["newdata"], p["levels"])
    X = model.lpmatrix(new)
    summed = np.zeros_like(X)
    for column in range(2):
        point = new.copy()
        for name in ("X", "Y", "f"):
            point[name] = [np.asarray(value)[column] for value in new[name]]
        point["f"] = pd.Categorical(point.f, categories=p["levels"])
        point["L"] = 1.0
        weights = np.array([np.asarray(value)[column] for value in new.L])
        summed += weights[:, None] * model.lpmatrix(point)
    summed[:, 0] = 1.0
    np.testing.assert_allclose(X, summed, atol=2e-7, rtol=2e-7)
    restored = GAM.load_model(model.save_model(tmp_path / "explicit-matrix-factor.pkl"))
    np.testing.assert_allclose(
        restored.predict(new, return_se=True),
        model.predict(new, return_se=True),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(restored.lpmatrix(new), X, atol=1e-12, rtol=1e-12)
    # Metadata at prediction can declare the fitted labels in another order.
    reordered = frame(p["newdata"], list(reversed(p["levels"])))
    np.testing.assert_allclose(model.lpmatrix(reordered), X, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("mode", ["fs", "sz"])
@pytest.mark.parametrize("special", ["missing", "unseen"])
def test_matrix_factor_prediction_invalid_cells_propagate_source_na(mode, special):
    p = case(mode, "sf", special=special)
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)
    assert np.isnan(model.predict(frame(p["newdata"], p["levels"]))[0])


@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_sz_multiple_explicit_matrix_factors(method):
    p = case("sz", "sf", method, new_width=2)
    for name in ("data", "newdata"):
        target = p[name]
        n, q = np.shape(target["f"])
        codes = (
            np.arange(n)[:, None] // len(p["levels"]) + 2 * np.arange(q)[None, :]
        ) % len(p["levels"])
        target["g"] = np.asarray(p["levels"])[codes].tolist()
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(
        data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"]
    )
    assert_invariants(model, p, ref)


def test_matrix_factor_sz_empty_joint_groups_remain_guarded():
    p = case("sz", "sf", new_width=2)
    for name in ("data", "newdata"):
        target = p[name]
        n, q = np.shape(target["f"])
        codes = (np.arange(n)[:, None] + 2 * np.arange(q)[None, :]) % len(p["levels"])
        target["g"] = np.asarray(p["levels"])[codes].tolist()
    ref = reference(p)
    assert ref["accepted"], ref
    with pytest.raises(NotImplementedError, match="empty joint factor groups"):
        model_for(p, ref).fit(data=frame(p["data"], p["levels"]))


def test_matrix_factor_metadata_preserves_declared_order_without_generic_schema_changes():
    levels = ["b", "a", "c"]
    rows = pd.Series(
        [
            pd.Categorical(["a", "b"], categories=levels, ordered=True),
            pd.Categorical(["c", "a"], categories=levels, ordered=True),
        ],
        dtype=object,
    )
    assert categorical_matrix_metadata(rows) == {
        "levels": levels,
        "ordered": True,
        "matrix_factor": True,
    }
    schema = FeatureSchema.from_data(pd.DataFrame({"f": rows}))
    assert schema.dtypes == ("object",)
    schema.validate(
        pd.DataFrame(
            {
                "f": pd.Series(
                    [pd.Categorical(["a", "b", "c"], categories=levels)], dtype=object
                )
            }
        )
    )
    assert categorical_matrix_metadata([["a", "b"], ["b", "a"]]) is None
    assert categorical_matrix_metadata([[1, 2], [2, 1]]) is None


@pytest.mark.parametrize("mode", ["fs", "sz"])
def test_explicit_matrix_factor_declared_unused_levels(mode):
    p = case(mode, "sf")
    p["levels"].append("unused")
    ref = reference(p)
    assert not ref["accepted"], ref
    assert "replacement has 360 rows, data has 120" in ref["error"]
    with pytest.raises(ValueError, match="unused declared levels"):
        model_for(p, {"sp": [0.3, 0.3] if mode == "fs" else [0.3] * 4}).fit(
            data=frame(p["data"], p["levels"])
        )


@pytest.mark.parametrize(
    "malformed",
    [
        "width",
        "levels",
        "ordered",
        "mixed_class",
        "missing",
        "coordinate_shape",
        "classless",
    ],
)
def test_explicit_matrix_factor_input_validation(malformed):
    p = case()
    data = frame(p["data"], p["levels"])
    if malformed == "width":
        data.at[0, "f"] = pd.Categorical(["a", "b"], categories=p["levels"])
    elif malformed == "levels":
        data.at[0, "f"] = pd.Categorical(
            list(data.f.iloc[0]), categories=list(reversed(p["levels"]))
        )
    elif malformed == "ordered":
        data.at[0, "f"] = data.f.iloc[0].as_ordered()
    elif malformed == "mixed_class":
        data.at[0, "f"] = np.asarray(data.f.iloc[0])
    elif malformed == "missing":
        row = list(data.f.iloc[0])
        row[0] = None
        data.at[0, "f"] = pd.Categorical(row, categories=p["levels"])
    elif malformed == "coordinate_shape":
        data["f"] = pd.Series(
            [pd.Categorical(list(row)[:2], categories=p["levels"]) for row in data.f],
            dtype=object,
        )
    else:
        data["f"] = [np.asarray(row) for row in data.f]
    with pytest.raises((ValueError, TypeError)):
        model_for(p, {"sp": [0.3, 0.3]}).fit(data=data)
