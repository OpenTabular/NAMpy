"""Strict k.check statistics and p-values using shared sampling inputs."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from tests.parity._registered_pipeline_reference import pipeline_reference
from tests.parity.test_mgcv_smooth_family_crossings import FAMILIES, fit_crossing_case

pytestmark = [pytest.mark.surface_regression]
BASES = ("cr", "gp", "t2", "re", "mrf", "fs", "sos")
PROBE = Path(__file__).parents[1] / "parity/probes/mgcv_shared_plan_k_check.R"


@pytest.mark.parametrize("code", BASES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("subsample", (56, 5000))
def test_k_check_observed_statistic_and_permutation_probability_match_mgcv(
    code, family, subsample, monkeypatch
):
    case, formula, source, knots = fit_crossing_case(code, family)
    n, repetitions = len(case.data), 37
    rng = np.random.default_rng(28801)
    rows = rng.permutation(n)[:subsample] if n > subsample else np.arange(n)
    permutations = np.column_stack(
        [rng.permutation(len(rows)) for _ in range(repetitions)]
    )
    expected = pipeline_reference(
        code + "-" + family + "-" + str(subsample),
        case.data,
        case.newdata,
        [formula],
        source,
        sp=case.model.smoothing_params,
        knots=knots,
        namespace="mgcv_k_check_shared_plan",
        probe=PROBE,
        extra_payload={
            "subsample": subsample,
            "selected_rows": rows.tolist(),
            "permutations": permutations.T.tolist(),
        },
    )
    module = importlib.import_module("nampy.gam.diagnostics.k_check")

    def plan(seed, n_obs, max_rows, nr, n_perm):
        assert n_obs == n and max_rows == subsample and nr == len(rows)
        assert n_perm in {0, repetitions}
        return rows, permutations[:, :n_perm]

    monkeypatch.setattr(module, "_kcheck_sample_plan", plan)
    actual = case.model.k_check(subsample=subsample, n_rep=repetitions, seed=123)
    values = np.asarray(
        [
            [np.nan if v is None else v for v in row]
            for row in np.atleast_2d(expected["values"])
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        actual.to_numpy(), values, atol=2e-7, rtol=2e-7, equal_nan=True
    )
    numeric = code not in {"re", "mrf", "fs"}
    assert expected["permutation_calls"] == (repetitions if numeric else 0)
    if numeric:
        assert actual.iloc[0].p_value * repetitions == pytest.approx(
            round(actual.iloc[0].p_value * repetitions)
        )
    else:
        assert actual[["k_index", "p_value"]].isna().all().all()


def test_shared_plan_diagnostics_have_reference_provenance():
    import json

    from tests.parity.generate_registered_pipeline_manifest import DIRECTORY, manifest

    actual = manifest("mgcv_k_check_shared_plan")
    stored = json.loads(
        (DIRECTORY.parent / "mgcv_k_check_shared_plan/manifest.json").read_text()
    )
    assert actual == stored
    assert {record["case"] for record in actual["captures"].values()} == {
        code + "-" + family + "-" + str(subsample)
        for code in BASES
        for family in FAMILIES
        for subsample in (56, 5000)
    }
