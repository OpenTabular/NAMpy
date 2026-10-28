"""Every cpois likelihood hook compared with installed mgcv 1.9-4."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families._cpois_helpers import dppois, log_ppois
from nampy.gam.families.cpois import CensoredPoissonFamily
from tests.families.test_cnorm_family_contracts import SCRIPT, _numeric, _response
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

SOURCE = SCRIPT.replace('f <- do.call(cnorm,list(theta=a$theta,link=a$link))', 'f <- do.call(cpois,list(link=a$link))')


def reference(case, source=SOURCE):
    key = reference_key("cpois-leaf-v1", {"case": case, "source": source})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as tmp:
            script, inputs, output = [Path(tmp) / name for name in ("run.R", "in.json", "out.json")]
            script.write_text('args<-commandArgs(TRUE)\nlibrary(mgcv);library(jsonlite)\na<-fromJSON(args[1])\n' + source + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="string",null="null")')
            inputs.write_text(json.dumps(case))
            run = subprocess.run(["Rscript", str(script), str(inputs), str(output)], capture_output=True, text=True, timeout=60)
            assert run.returncode == 0, run.stderr
            save_reference("mgcv_cpois_family", key, json.loads(output.read_text()))
    return load_reference("mgcv_cpois_family", key)


def payload(band="mixed", link="log"):
    case = {"y": [0., 1., 3., 5., 9., 2.],
            "bounds": [0., "-Inf", "Inf", 8., 6., 2.],
            "mu": [.4, 2., 2.5, 6., 7., 1.5], "w": [1.] * 6,
            "scale": 1, "link": link, "eta": [.1, .3, .7, 1., 2., 3.]}
    if band == "vector":
        case["bounds"] = None
    elif band == "weights":
        case["w"] = [0., .25, 1., 2., 5., 3.]
    elif band == "tails":
        case.update(y=[0., 1000., 1000., 200., 5., 1.],
                    bounds=["-Inf", "Inf", 1001., 210., "Inf", "-Inf"],
                    mu=[1000., 1., 1., 1., 1000., .001])
    elif band == "fractional":
        case.update(y=[.1, 1.2, 3.4, 5.3, 9.9, 2.1],
                    bounds=["-Inf", "Inf", 4.4, 8.6, 6.5, "-Inf"])
    elif band == "negative_endpoints":
        case.update(y=[0., 0., 1., 1., 2., 3.], bounds=[-1., "-Inf", -2., 0., "Inf", 3.])
    return case


@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
@pytest.mark.parametrize("band", ["mixed", "vector", "weights", "fractional", "negative_endpoints", "tails"])
def test_cpois_all_kernel_fields_match_source(link, band):
    case = payload(band, link)
    ref = reference(case)
    family = CensoredPoissonFamily(link)
    y, mu, w = _response(case), np.asarray(case["mu"]), np.asarray(case["w"])
    assert family.n_theta == ref["n_theta"] == 0
    assert family.getTheta().size == 0
    with np.errstate(all="ignore"):
        for level in range(3):
            expected = ref[f"dd{level}"]
            assert not expected["error"], expected
            actual = family.Dd(y, mu, family.getTheta(), w, level)
            assert actual.keys() == expected["value"].keys()
            for name, value in actual.items():
                source_value = np.asarray(_numeric(expected["value"][name]))
                ordinary = np.ones(value.shape, dtype=bool)
                if band == "tails":
                    from tests.families.test_cpois_tail_derivative_precision import (
                        high_precision_reference,
                    )

                    # The source subtracts O(1e13) terms for these rare events;
                    # its fourth derivatives already miss the true value by
                    # several units. Preserve that capture as source evidence,
                    # and require stronger accuracy against an independent law.
                    rare = np.array([1, 2, 3])
                    column = {"Dmu": 0, "Dmu2": 1, "EDmu2": 1, "Dmu3": 2, "Dmu4": 3}[name]
                    truth = high_precision_reference()[:3, column]
                    np.testing.assert_allclose(value[rare], truth, rtol=2e-12, atol=2e-12, err_msg=name)
                    assert np.all(np.abs(value[rare] - truth) <= np.abs(source_value[rare] - truth) + 4 * np.abs(np.spacing(truth)))
                    ordinary[rare] = False
                np.testing.assert_allclose(value[ordinary], source_value[ordinary], rtol=2e-9, atol=2e-9, equal_nan=True, err_msg=name)
        for name, value in {"dev": family.deviance_obs(y, mu, w), "aic": family.aic(y, mu, weights=w)}.items():
            np.testing.assert_allclose(value, _numeric(ref[name]["value"]), rtol=2e-10, atol=2e-10, equal_nan=True, err_msg=name)
        for name, value in family.ls(y, w).items():
            np.testing.assert_allclose(np.asarray(value).ravel(), np.asarray(_numeric(ref["ls"]["value"][name])).ravel(), rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(family.initialize_mu(y), _numeric(ref["initialize"]["value"]["mu"]))


def test_cpois_probability_helpers_retain_finite_extreme_log_tails():
    case = {"y0": [-2., 0., 1000., 200., 4., -1., -3.], "y1": [-1., 1., 1001., 210., 6., 0., -1.],
            "mu": [1., 1000., 1., 1., 5., 0., 0.]}
    source = 'ans<-list(interval=mgcv:::dppois(a$y0,a$y1,a$mu),lower=ppois(a$y1,a$mu,log.p=TRUE),upper=ppois(a$y1,a$mu,lower.tail=FALSE,log.p=TRUE))'
    ref = reference(case, source)
    for key, actual in (("interval", dppois(case["y0"], case["y1"], case["mu"])),
                        ("lower", log_ppois(case["y1"], case["mu"])),
                        ("upper", log_ppois(case["y1"], case["mu"], False))):
        np.testing.assert_allclose(actual, _numeric(ref[key]), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("link", ["inverse", "logit"])
def test_cpois_rejects_source_unavailable_links(link):
    with pytest.raises(ValueError, match="link"):
        CensoredPoissonFamily(link)
