"""Replay the exact floating-point fixture identities reported by Linux CI."""

import copy
import hashlib
import json
import subprocess

import numpy as np
import pytest

from tests.families import test_mgcv_working_information_hooks as working
from tests.optimization import test_mgcv_ncv_edges as ncv
from tests.reference_fixtures import (
    FIXTURE_ROOT,
    MissingReferenceFixture,
    load_reference,
    reference_key,
)


@pytest.fixture(autouse=True)
def forbid_reference_generation(monkeypatch):
    monkeypatch.delenv("NAMPY_REFRESH_REFERENCE_FIXTURES", raising=False)
    monkeypatch.delenv("NAMPY_REBUILD_REFERENCE_FIXTURES", raising=False)
    monkeypatch.delenv("NAMPY_RECORD_REFERENCE_ALIASES", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixture portability tests must never run R")

    monkeypatch.setattr(subprocess, "run", forbidden)


def assert_ci_alias(namespace, operation, identity, ci_key):
    # These were genuinely missing CI filenames. Do not generate replacements:
    # the portable key must resolve to the original saved R output.
    assert reference_key(operation, identity) == ci_key
    assert not (FIXTURE_ROOT / namespace / f"{ci_key}.json.gz").exists()
    key = reference_key(operation, identity, normalize_floats=True)
    aliases = json.loads((FIXTURE_ROOT / namespace / "aliases.json").read_text())
    assert str(key) in aliases
    return load_reference(namespace, key)


WORKING_CASES = [
    (
        "gaussian",
        [
            3.045995948942526e-09,
            0.10826822658929014,
            2.386919516226033,
            103524.38923559157,
        ],
        "07d83417fe8b03c7feb432e85a679762d8b7129f5478a8c027f479adb218364a",
    ),
    (
        "gamma",
        [
            4.122307244877116e-10,
            0.4390493088752212,
            2.159774092121605,
            34.72856366497406,
        ],
        "ea23e9f08044361c2517eff4739ca8bc1d3d0fd68c1836e3e61c5264fcfba5a7",
    ),
    (
        "inverse_gaussian",
        [
            0.027067056647322535,
            0.4390493088752212,
            2.159774092121605,
            34.72856366497406,
        ],
        "ff6d24def7a0e95b8f5920042745227101f7e5dca7145785f138020c1ac9dd0f",
    ),
]


def working_identity(name, response):
    case = next(p for p in working.CASES if p.id == f"{name}-log-None-normal")
    payload = dict(case.values[1], y=response)
    return {
        "payload": payload,
        "source_sha256": hashlib.sha256(working.SCRIPT.encode()).hexdigest(),
    }


@pytest.mark.parametrize("name,response,ci_key", WORKING_CASES)
def test_working_information_replays_exact_ci_fixture_identity(name, response, ci_key):
    identity = working_identity(name, response)
    expected = assert_ci_alias(
        "mgcv_working_information_hooks",
        "working-information-hooks-v1",
        identity,
        ci_key,
    )
    assert working.reference(identity["payload"]) == expected
    for direction in [-np.inf, np.inf]:
        payload = dict(
            identity["payload"], y=np.nextafter(response, direction).tolist()
        )
        assert working.reference(payload) == expected


NCV_CASES = [
    (
        "gaussian",
        1.0,
        "5178b191b752d480e8750d3cbd0345e9264c24e00b625d7f426c9bb4d9c29d6a",
    ),
    (
        "gaussian",
        1.4,
        "ec7c2bb1c6433d13b18c9a8fa2a61ee2f76b50d626529e0d98c369bcf2675047",
    ),
    (
        "poisson",
        1.0,
        "58ec3b71fb5e66b5b36c0769b6e7175cdd961e23712f4785f906928bdd1be89b",
    ),
    (
        "poisson",
        1.4,
        "063cdcefa4683839f4787cd717b14253e2237ca8a42edb07578618b56bf40f0f",
    ),
    (
        "binomial",
        1.0,
        "0f67528e7e352bbbca23861b5c65b5d22d907b3d5cba38142727246b78ae661f",
    ),
    (
        "binomial",
        1.4,
        "cc61b3cc3a0a3065aeb08dbfe9cfcab2169009b8507d5cbac290e394bc5f56d5",
    ),
]
CI_LOGSP = [-0.35667494393873245, 0.09531017980432495]


@pytest.mark.parametrize("family,gamma,ci_key", NCV_CASES)
def test_ncv_replays_exact_ci_fixture_identity(monkeypatch, family, gamma, ci_key):
    data = ncv.data_frame(family, False, False)
    identity = {
        "payload": {
            "data": data.to_dict("list"),
            "family": ncv.FAMILIES[family][1],
            "nei": None,
            "gamma": gamma,
            "geometry": "mixed",
            "offset": False,
            "logsp": CI_LOGSP,
        },
        "source_sha256": hashlib.sha256(ncv.SCRIPT.encode()).hexdigest(),
    }
    expected = assert_ci_alias(
        "mgcv_ncv_edges", "ncv-edge-selection-v1", identity, ci_key
    )

    def ci_reference_key(operation, payload, **kwargs):
        payload = copy.deepcopy(payload)
        payload["payload"]["logsp"] = CI_LOGSP
        return reference_key(operation, payload, **kwargs)

    # Exercise the real helper's opt-in normalization with the CI math-library
    # result. Without that opt-in it would request the absent historical key.
    monkeypatch.setattr(ncv, "reference_key", ci_reference_key)
    assert ncv.reference(data, family, None, gamma, "mixed", False) == expected


@pytest.mark.parametrize("field", ["y", "eta"])
def test_materially_changed_working_inputs_still_require_a_fixture(field):
    name, response, _ = WORKING_CASES[0]
    payload = copy.deepcopy(working_identity(name, response)["payload"])
    payload[field][1] += 1e-4
    with pytest.raises(MissingReferenceFixture):
        working.reference(payload)
