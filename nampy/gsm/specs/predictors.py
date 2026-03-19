from dataclasses import dataclass, field
from typing import Any


@dataclass
class LinearPredictorSpec:
    """
    Specification for one additive linear predictor.

    Examples
    --------
    LinearPredictorSpec(
        name="eta",
        terms=[SplineTerm1D("x1"), SplineTerm1D("x2")]
    )

    Future extensions
    -----------------
    parameter_name:
        Name of the distribution parameter associated with this predictor
        (e.g. "mu", "sigma", "nu"), for general-family / classical LSS models.
    offset_name:
        Reserved for later support of predictor-specific offsets.
    metadata:
        Free-form metadata for future fitting engines.
    """

    name: str = "eta"
    terms: list = field(default_factory=list)
    parameter_name: str | None = None
    offset_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)