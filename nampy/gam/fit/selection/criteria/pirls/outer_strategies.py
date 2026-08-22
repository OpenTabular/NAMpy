"""Registry adapters for family-declared joint outer criteria."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .....families.family_base import JointOuterStrategy
from .tweedie import (
    criterion_gradient_ml_reml_pirls_tweedie_joint,
    criterion_hessian_ml_reml_pirls_tweedie_joint,
    criterion_ml_reml_pirls_tweedie_joint,
)


@dataclass(frozen=True)
class JointOuterHandler:
    """Adapter from optimizer coordinates to a joint-family criterion."""

    value_fn: Callable
    gradient_fn: Callable
    hessian_fn: Callable

    def _split_parameters(self, model, joint_x):
        x = np.asarray(joint_x, dtype=np.float64).ravel()
        ntheta = int(getattr(model.family, "n_theta", 0) or 0)
        if x.size <= ntheta:
            raise ValueError(
                "Joint outer criteria require log scale after the theta/sp block."
            )
        log_theta = float(x[0]) if ntheta else float(model.family.getTheta(False))
        return x[ntheta:-1], log_theta, float(x[-1])

    def value(self, model, y, joint_x, method):
        return self.value_fn(model, y, *self._split_parameters(model, joint_x), method)

    def gradient(self, model, y, joint_x, method):
        return self.gradient_fn(
            model, y, *self._split_parameters(model, joint_x), method
        )

    def hessian(self, model, y, joint_x, method):
        return self.hessian_fn(
            model, y, *self._split_parameters(model, joint_x), method
        )


_JOINT_OUTER_HANDLERS = {
    JointOuterStrategy.TWEEDIE: JointOuterHandler(
        value_fn=criterion_ml_reml_pirls_tweedie_joint,
        gradient_fn=criterion_gradient_ml_reml_pirls_tweedie_joint,
        hessian_fn=criterion_hessian_ml_reml_pirls_tweedie_joint,
    ),
}


def get_joint_outer_handler(family):
    """Return the handler declared by ``family``, if one is registered."""
    strategy = getattr(family, "joint_outer_strategy", JointOuterStrategy.NONE)
    return _JOINT_OUTER_HANDLERS.get(strategy)
