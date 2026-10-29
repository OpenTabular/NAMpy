"""Import-boundary contracts for lazy NAMpy backends."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "nampy.distributions", "nampy.diagnostics", "nampy.inspection", "nampy.plotting",
])
def test_shared_task_packages_import_independently_without_neural_runtime(module):
    code = f"""
import importlib
import sys
package = importlib.import_module({module!r})
for name in package.__all__:
    getattr(package, name)
assert 'torch' not in sys.modules
assert 'lightning' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_gam_public_import_does_not_initialize_neural_backend():
    code = """
import sys

from nampy.gam import GAM
from nampy import CalibratedQuantileGAM, CoxGAM, GAMLSS, GAMRegressor, ExpectileGAM, QuantileGAM
from nampy import distributions, calibration, diagnostics, inspection
assert callable(diagnostics.quantile_residuals)
assert callable(inspection.prepare_distributional_effects)
from nampy.calibration import calibrate_conformal
assert calibration.calibrate_conformal is calibrate_conformal
assert calibrate_conformal([0., 1., 2.], [0., 0., 0.], level=.5).correction == 1
from nampy.distributions import DistributionLaw, DistributionSpec
from nampy.distributions import BCCGLaw, BCPELaw, BCTLaw, DistributionPrediction, NormalLaw

assert CalibratedQuantileGAM.__module__ == "nampy.models.calibrated_quantile"
assert CoxGAM.__module__ == "nampy.models.cox"
assert GAM.__module__.startswith("nampy.gam.")
assert GAMRegressor.__module__ == "nampy.models.gam"
assert GAMLSS.__module__ == "nampy.models.gamlss"
assert ExpectileGAM.__module__ == "nampy.models.expectile"
assert QuantileGAM.__module__ == "nampy.models.quantile"
assert distributions.DistributionPrediction is DistributionPrediction
assert DistributionPrediction(NormalLaw(), {"loc": [0.], "scale": [1.]}).cdf(0)[0] == 0.5
assert DistributionPrediction(BCCGLaw(), {"mu": [1.], "sigma": [0.4], "nu": [0.]}).cdf(1)[0] == 0.5
assert DistributionPrediction(BCCGLaw(), {"mu": [1.], "sigma": [0.4], "nu": [1.]}).mean()[0] > 1
for law in (BCTLaw(), BCPELaw()):
    prediction = DistributionPrediction(law, {"mu": [1.], "sigma": [.4], "nu": [0.], "tau": [3.]})
    assert prediction.cdf(1)[0] == .5
assert "nampy.neural" not in sys.modules
assert "torch" not in sys.modules
assert "lightning" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_removed_owners_and_forwarding_apis_are_unavailable():
    code = """
from importlib import import_module

for name in (
    "nampy.gam.extensions",
    "nampy.models._persistence",
    "nampy.neural.families.distributions",
    "nampy.neural.distributions",
    "nampy.neural.families.prediction",
    "nampy.neural.families.quantile",
    "nampy.gam.predict.distributions",
    "nampy.gam.observations",
    "nampy.explanations",
    "nampy.interpretation",
    "nampy.distributions.diagnostics",
    "nampy.plotting.distributional",
    "nampy.plotting.distributions",
):
    try:
        import_module(name)
    except ModuleNotFoundError as error:
        assert error.name == name or name.startswith(error.name + ".")
    else:
        raise AssertionError(f"Removed owner still imports: {name}")

from nampy.gam.fit.state import FitCoreSolution
from nampy.gam.fit.solvers import shape_constrained
from nampy.neural import objectives
from nampy import plotting
from nampy import contracts, distributions
from nampy.neural import families

for name in ("DistributionSpec", "DistributionLaw", "DistributionFitStatus"):
    assert not hasattr(contracts, name)
    assert hasattr(distributions, name)
assert not hasattr(distributions, "quantile_residuals")
assert not hasattr(families, "Quantile")

assert not hasattr(objectives, "objective_from_legacy")
assert not hasattr(shape_constrained, "solve_shape_constrained_fit")
for name in ("get", "__getitem__", "__getattr__"):
    assert not hasattr(FitCoreSolution, name)
for name in ("DistributionalEffects", "prepare_distributional_effects", "combine_distributional_effects"):
    assert not hasattr(plotting, name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
