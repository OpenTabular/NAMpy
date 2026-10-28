"""Formula-facing smooth metadata, independent of runtime constructor defaults."""

import inspect

TENSORS = frozenset({"te", "ti", "t2"})
SHAPE_NAMES = {
    "mpi": "Increasing P-spline",
    "mpd": "Decreasing P-spline",
    "micv": "Increasing, concave P-spline",
    "micx": "Increasing, convex P-spline",
    "mdcv": "Decreasing, concave P-spline",
    "mdcx": "Decreasing, convex P-spline",
    "cx": "Convex P-spline",
    "cv": "Concave P-spline",
    "po": "Positive P-spline",
    "ipo": "Positive increasing P-spline",
    "dpo": "Positive decreasing P-spline",
    "cpop": "Cyclic positive P-spline",
    "miso": "Increasing spline, zero start",
    "mifo": "Increasing spline, zero end",
    "lmpi": "Locally increasing P-spline",
    "lipl": "Increasing spline with plateau",
    "tedmi": "Increasing in both covariates",
    "tedmd": "Decreasing in both covariates",
    "temicx": "Increasing / convex tensor spline",
    "temicv": "Increasing / concave tensor spline",
    "tedecx": "Decreasing / convex tensor spline",
    "tedecv": "Decreasing / concave tensor spline",
    "tecxcx": "Convex / convex tensor spline",
    "tecvcv": "Concave / concave tensor spline",
    "tecxcv": "Convex / concave tensor spline",
    "tescx": "Convex second-margin spline",
    "tescv": "Concave second-margin spline",
    "tesmi1": "Increasing first-margin spline",
    "tesmd1": "Decreasing first-margin spline",
    "tesmi2": "Increasing second-margin spline",
    "tesmd2": "Decreasing second-margin spline",
    "tismi": "Increasing tensor interaction",
    "tismd": "Decreasing tensor interaction",
}
SHAPE_NAMES.update(
    {
        name + "by": title + " with by variable"
        for name, title in list(SHAPE_NAMES.items())
        if name in {"mpi", "mpd", "micv", "micx", "mdcv", "mdcx", "cv", "cx"}
    }
)


def smooth_inputs(descriptor):
    name = descriptor.name
    if name in TENSORS:
        return "Two or more covariates, grouped into margins"
    if name in {"fs", "sz"}:
        return "Numerical covariate(s) and categorical factor(s)"
    if (
        descriptor.runtime_class
        and descriptor.runtime_class.__name__ == "BivariateShapePSplineTerm"
    ):
        return "Two covariates"
    lo, hi = descriptor.min_features, descriptor.max_features
    if hi is None:
        return "One or more covariates"
    if lo == hi == 1:
        return "One covariate"
    if lo == hi == 2:
        return "Two covariates"
    if (lo, hi) == (1, 2):
        return "One or two covariates"
    return f"{lo} to {hi} covariates"


def formula_option_rows(descriptor):
    """All public options consumed by specs.build._build_predictor_spec.

    Model-level knots/select and runtime setup arguments are deliberately absent.
    Parser aliases are documented in their canonical option's description.
    """
    from reference_schema import smooth_parameter_meanings

    name = descriptor.name
    tensor = name in TENSORS
    has_m = (
        descriptor.runtime_class
        and "m" in inspect.signature(descriptor.runtime_class).parameters
    )
    multiple_k = (
        tensor
        or name in {"ad", "scad"}
        or (
            descriptor.runtime_class
            and descriptor.runtime_class.__name__ == "BivariateShapePSplineTerm"
        )
    )
    meanings = smooth_parameter_meanings(name)
    rows = [
        (
            "bs",
            "str | sequence[str]" if tensor else "str",
            "'cr' per margin" if tensor else "model basis ('tp' by default)",
            "Marginal basis codes, recycled from a scalar or supplied one per margin; e.g. bs=c('cr','ps'). Alias basis is accepted."
            if tensor
            else f"Select this basis with bs='{name}'. The alias basis is also accepted.",
        ),
        (
            "k",
            "int | sequence[int]" if multiple_k else "int",
            "5**d[j] per margin"
            if tensor
            else "-1 (automatic)"
            if descriptor.dynamic_default_k
            else "model k (10 by default)",
            "Marginal basis dimensions before constraints. A scalar is recycled; a sequence has one value per margin. The unconstrained tensor dimension is their product."
            if tensor
            else meanings.get(
                "k",
                "Requested basis dimension before constraints; automatic resolution depends on the basis and covariate dimension.",
            ),
        ),
        (
            "m",
            "int | sequence | None",
            "None",
            meanings.get(
                "m",
                "Derivative/difference orders or covariance specification for this basis; None selects its defaults.",
            )
            if tensor
            or has_m
            or "m" in descriptor.runtime_options
            or name in {"fs", "sz"}
            else "Accepted by the common parser but not used by this basis.",
        ),
        (
            "by",
            "str | expression | None",
            "None",
            "Numeric multiplier or categorical source column/expression defining varying-coefficient or level-specific effects; evaluated on training and new rows.",
        ),
        (
            "id",
            "str | int | None",
            "None",
            "Share smoothing parameters across compatible terms. Alias smoothing_id is accepted; basis sharing follows id_links_bases.",
        ),
        (
            "sp",
            "float | sequence[float] | None",
            "None",
            "Penalty strengths in term penalty order. Nonnegative supplied values fix strengths; negative entries request estimation when automatic selection is enabled.",
        ),
        (
            "xt",
            "mapping | sequence[mapping | None] | str | None",
            "None",
            meanings.get("xt", "No additional xt setup is consumed by this basis."),
        ),
    ]
    if name != "t2":
        rows.append(
            (
                "fx",
                "bool | sequence[bool]" if tensor else "bool",
                "False",
                "Unpenalized marginal regression bases when True. te/ti accept one flag per margin; a scalar is recycled."
                if tensor
                else "Fit the smooth without its roughness penalty. Model-level null-space selection cannot be combined with a fixed unpenalized term.",
            )
        )
    if descriptor.supports_pc or tensor:
        rows.append(
            (
                "pc",
                "float | sequence[float] | mapping[str,float] | None",
                "None",
                "Homogeneous function constraint f(x0)=0. Supply one coordinate per input covariate, or a named mapping; the basis is reparameterized at the point.",
            )
        )
    if tensor:
        rows.append(
            (
                "d",
                "sequence[int] | None",
                "one covariate per margin",
                "Positive marginal dimensions whose sum equals the covariate count. Groups consecutive covariates into multivariate margins; k is per margin, not per coordinate.",
            )
        )
    if name == "ti":
        rows.append(
            (
                "mc",
                "sequence[bool] | None",
                "all True",
                meanings.get(
                    "mc",
                    "Whether to impose each marginal identifiability constraint. Disabling constraints can retain main-effect directions; specify accompanying main effects deliberately.",
                ),
            )
        )
    if name == "t2":
        rows += [
            (
                "full",
                "bool",
                "False",
                meanings.get(
                    "full",
                    "Retain separate null-space blocks instead of grouping them.",
                ),
            ),
            (
                "ord",
                "int | sequence[int] | None",
                "None",
                meanings.get(
                    "ord",
                    "Retain specified numbers of range-space margins; None retains all orders.",
                ),
            ),
        ]
    return [
        (f"``{key}``", typ, f"``{default}``", meaning)
        for key, typ, default, meaning in rows
    ]
