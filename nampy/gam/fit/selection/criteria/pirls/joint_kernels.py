"""Family-strategy dispatch for current-smoothing ``gdi2`` kernels."""

from .....families.family_base import JointOuterStrategy
from .derivatives import _GDI2Kernel, gdi2_theta_joint_kernel
from .family_gamma import gdi2_gamma_joint_kernel
from .family_gaussian import gdi2_gaussian_joint_kernel
from .family_general import gdi2_general_family_kernel
from .family_negbin import gdi2_negbin_joint_kernel


def gdi2_joint_kernel(model, y, sol, sp, *, method, need_hessian) -> _GDI2Kernel:
    """Dispatch extended-family derivative assembly by family capability."""
    strategy = getattr(model.family, "joint_outer_strategy", JointOuterStrategy.NONE)
    if strategy is JointOuterStrategy.GAMMA_SCALE:
        return gdi2_gamma_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if strategy is JointOuterStrategy.GAUSSIAN_SCALE:
        return gdi2_gaussian_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if strategy is JointOuterStrategy.NEGBIN_THETA:
        return gdi2_negbin_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if strategy is JointOuterStrategy.BETAR_THETA:
        return gdi2_theta_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if strategy is JointOuterStrategy.OCAT_THETA:
        return gdi2_theta_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        n_theta = int(getattr(model.family, "n_theta", 0) or 0)
        if n_theta != 0:
            raise NotImplementedError(
                "Generic `gdi2` extended-family port is implemented only for "
                "theta-free general families."
            )
        return gdi2_general_family_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    raise NotImplementedError(
        "Generic `gdi2` extended-family port is not complete yet "
        f"for family={model.family.name!r}."
    )
