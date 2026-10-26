"""mgcv sc/scad ordinary-GAM constructors without scasm inequality fitting."""

from ...splines.univariate.sc import shape_constraint_tokens
from ..registry import register_smooth
from .ad import AdaptiveSmoothTerm
from .bs import DerivativeBSplineTerm1D


def _require_unconstrained(xt, basis):
    if set(shape_constraint_tokens(xt)) & {"+", "m+", "m-", "c+", "c-"}:
        raise NotImplementedError(
            f"bs='{basis}' with shape constraints requires mgcv's separate scasm "
            "inequality-constrained fitter, which is not implemented. Ordinary "
            "GAM fitting must not silently ignore these constraints."
        )


@register_smooth("sc")
class ShapeConstrainedBSplineTerm(DerivativeBSplineTerm1D):
    """The exact sc -> bs branch when no shape inequalities are requested."""

    basis_name = "sc"

    def __init__(self, feature, *, xt=None, **kwargs):
        _require_unconstrained(xt, "sc")
        super().__init__(feature, **kwargs)
        self.xt = xt


@register_smooth("scad")
class ShapeConstrainedAdaptiveTerm(AdaptiveSmoothTerm):
    """The exact scad -> ad(ps) branch without shape inequalities."""

    basis_name = "scad"

    def __init__(self, feature, *, xt=None, **kwargs):
        _require_unconstrained(xt, "scad")
        super().__init__(feature, xt={"bs": "ps"}, **kwargs)
        if xt is not None and self.dim > 1:
            raise ValueError("scad is for 1D smoothing only")
        self.xt = xt
