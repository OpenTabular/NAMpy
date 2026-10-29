"""Report existing statistical convergence without changing optimizer behavior."""

from ...distributions.contracts import DistributionFitStatus
from ..model_state import _fit_result


def distribution_fit_status(model):
    """Read coefficient and available outer-optimizer status from a fitted GAM."""
    if model is None or getattr(model, "gam_result_", None) is None:
        return DistributionFitStatus(
            "gam", "unknown", "No successful fitted statistical result is available."
        )
    core = _fit_result(model)
    if core.converged is None:
        return DistributionFitStatus(
            "gam", "unknown", "The coefficient backend did not report convergence."
        )
    if not core.converged or core.failed_step:
        return DistributionFitStatus(
            "gam",
            "nonconverged",
            core.failure_reason or "Coefficient fitting did not converge.",
        )
    outer = getattr(model, "_optim_result", None)
    if outer is not None:
        success = getattr(outer, "success", None)
        if success is None:
            return DistributionFitStatus(
                "gam", "unknown", "The outer optimizer did not report success."
            )
        if not success:
            return DistributionFitStatus(
                "gam",
                "nonconverged",
                str(
                    getattr(
                        outer,
                        "message",
                        "Outer smoothing optimization did not converge.",
                    )
                ),
            )
    return DistributionFitStatus(
        "gam",
        "converged",
        "Coefficient fitting and any reported outer optimization converged.",
    )
