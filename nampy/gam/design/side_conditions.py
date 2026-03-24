"""
Backward-compatibility shim: re-exports ``apply_global_side_conditions`` from
its canonical home in ``gam/constraints/identifiability.py``.

Internal code should import directly from ``gam.constraints.identifiability``.
"""
from ..constraints.identifiability import apply_global_side_conditions

__all__ = ["apply_global_side_conditions"]
