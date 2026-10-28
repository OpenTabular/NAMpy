"""Fixed-input fixtures for registry-wide GAM pipeline contracts.

Independent reference comparisons live alongside these owner-contract checks;
these fixtures never treat a NAMpy fit as an upstream reference.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nampy import GAMRegressor
from nampy.gam.basis_registry import require_basis_descriptor

REGULAR_BASES = (
    "re",
    "sc",
    "scad",
    "so",
    "sf",
    "sw",
    "ad",
    "cr",
    "cs",
    "cc",
    "ps",
    "cp",
    "bs",
    "ds",
    "gp",
    "sos",
    "mrf",
    "tp",
    "ts",
    "fs",
    "sz",
    "t2",
    "te",
    "ti",
)
MULTIVARIATE_ADAPTIVE_CASES = ("ad__2d", "scad__2d")
SHAPE_BASES = (
    "mpi",
    "mpd",
    "mdcv",
    "mdcx",
    "micv",
    "micx",
    "cv",
    "cx",
    "po",
    "dpo",
    "ipo",
    "miso",
    "mifo",
    "mpiby",
    "mpdby",
    "mdcvby",
    "mdcxby",
    "micvby",
    "micxby",
    "cvby",
    "cxby",
    "cpop",
    "lmpi",
    "lipl",
    "tedmi",
    "tedmd",
    "temicx",
    "temicv",
    "tedecv",
    "tedecx",
    "tecvcv",
    "tecxcx",
    "tecxcv",
    "tescv",
    "tescx",
    "tesmi1",
    "tesmd1",
    "tesmi2",
    "tesmd2",
    "tismi",
    "tismd",
)


@dataclass
class PipelineCase:
    name: str
    data: pd.DataFrame
    newdata: pd.DataFrame
    estimator: object
    formulas: tuple[str, ...]

    @property
    def basis(self):
        return self.name.split("__", 1)[0]

    @property
    def model(self):
        return self.estimator.gam_

    @property
    def compiled(self):
        return self.model.gam_result_.require_compiled_model()


def smooth_inputs(code):
    code, _, variant = code.partition("__")
    rng = np.random.default_rng(281026)
    n = 96
    frame = pd.DataFrame(
        {"x": rng.uniform(0.07, 0.93, n), "z": rng.uniform(0.07, 0.93, n)}
    )
    frame["off"] = 0.07 * frame.x
    frame["w"] = 0.7 + frame.x
    frame["by"] = 1 + frame.z
    frame["group"] = pd.Categorical(np.resize(["a", "b", "c"], n))
    frame["latitude"] = rng.uniform(-55, 55, n)
    frame["longitude"] = rng.uniform(-110, 110, n)
    frame["y"] = (
        0.4 + np.sin(3 * frame.x) + 0.3 * frame.z + frame.off + rng.normal(0, 0.06, n)
    )
    kwargs = {}
    if code in {"te", "ti", "t2"}:
        smooth = f"{code}(x,z,bs=['cr','ps'],k=[5,5])"
    elif code in {"fs", "sz"}:
        smooth = f"s(group,x,bs='{code}',k=5,xt='cr')"
    elif code == "re":
        smooth = "s(group,bs='re')"
    elif code == "mrf":
        smooth = "s(group,bs='mrf',k=3,xt={'nb':{'a':['b'],'b':['a','c'],'c':['b']}})"
    elif code == "sos":
        smooth = "s(latitude,longitude,bs='sos',k=10)"
    elif code in {"so", "sf", "sw"}:
        boundary = [{"x": [0.0, 1.0, 1.0, 0.0, 0.0], "z": [0.0, 0.0, 1.0, 1.0, 0.0]}]
        smooth = f"s(x,z,bs='{code}',k=8,xt={{'bnd':{boundary!r},'nmax':25}})"
        kwargs["knots"] = {"x": [0.3, 0.7, 0.3, 0.7], "z": [0.3, 0.3, 0.7, 0.7]}
    elif code in {"ad", "scad"}:
        smooth = (
            f"s(x,z,bs='{code}',k=[6,6],m=[3,3])"
            if variant == "2d"
            else f"s(x,bs='{code}',k=10,m=3)"
        )
    elif code == "sc":
        smooth = "s(x,bs='sc',k=8,m=[3,2])"
    elif code in SHAPE_BASES:
        descriptor = require_basis_descriptor(code)
        if descriptor.runtime_class.__name__ == "BivariateShapePSplineTerm":
            smooth = f"s(x,z,bs='{code}',k=[5,6],m=[2,1])"
        else:
            by = ",by=by" if code.endswith("by") else ""
            extra = ",xt={'xc':.5}" if code in {"lmpi", "lipl"} else ""
            smooth = f"s(x,bs='{code}',k=8,m=2{by}{extra})"
    else:
        smooth = f"s(x,bs='{code}',k=8)"
    if code in SHAPE_BASES:
        # Freeze admissible responses once. A runtime basis reparameterization
        # must not change the dataset or require a new upstream reference.
        path = (
            Path(__file__).parents[1]
            / "reference_fixtures/registered_seven_stage/smooth_responses.json"
        )
        frame["y"] = json.loads(path.read_text())[code]
    formula = f"y ~ {smooth} + offset(off)"
    newdata = frame.drop(columns=["y", "w"]).iloc[[1, 9, 17, 30, 49, 62, 77]].copy()
    # Evaluate genuinely new metric locations while retaining valid domains/levels.
    newdata["x"] = np.linspace(0.1, 0.9, len(newdata))
    newdata["z"] = np.linspace(0.85, 0.15, len(newdata))
    newdata["off"] = -0.04 + 0.09 * newdata.x
    newdata["by"] = 1 + newdata.z
    return frame, newdata, formula, kwargs


def fit_smooth_case(code):
    frame, newdata, formula, kwargs = smooth_inputs(code)
    estimator = GAMRegressor(
        formula=formula,
        smoothing_params=0.4,
        optimize_smoothing=False,
        smoothing_method="fixed",
        max_irls_iter=300,
        irls_tol=1e-9,
        **kwargs,
    ).fit(frame, sample_weight=frame.w.to_numpy())
    return PipelineCase(code, frame, newdata, estimator, (formula,))
