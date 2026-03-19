from dataclasses import dataclass, field
from typing import Any

from .base import BaseTermSpec


@dataclass
class SmoothTermSpec(BaseTermSpec):
    """
    mgcv-like smooth specification object.
    """

    special: str = "s"  # one of: s, te, ti, t2
    bs: Any = "cr"
    features: list[str] = field(default_factory=list)

    k: Any = -1
    fx: bool = False
    select: bool = False
    m: Any = None
    by: Any = None
    xt: Any = None
    id: Any = None
    sp: Any = None
    pc: Any = None
    knots: Any = None

    # constructor-time identifiability override
    constraint_mode: str = "auto"

    # constructor-time shared basis setup (used for linked ids)
    shared_basis_setup: Any = None

    # tensor-specific extras
    mc: Any = None
    full: bool = False
    ord: Any = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.special = str(self.special).lower()
        if self.special not in {"s", "te", "ti", "t2"}:
            raise ValueError(
                f"SmoothTermSpec.special must be one of {{'s','te','ti','t2'}}, "
                f"got {self.special!r}."
            )

        self.features = [str(f) for f in self.features]
        if len(self.features) == 0:
            raise ValueError("SmoothTermSpec requires at least one feature name.")

        self.constraint_mode = str(self.constraint_mode).lower()
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        if self.label is None:
            self.label = f"{self.special}({', '.join(self.features)})"

    @property
    def dim(self) -> int:
        return len(self.features)

    @property
    def fixed(self) -> bool:
        return bool(self.fx)

    @property
    def smoothing_id(self):
        return self.id

    def to_metadata_dict(self) -> dict[str, Any]:
        shared_basis_summary = None
        if isinstance(self.shared_basis_setup, dict):
            shared_basis_summary = {
                "mode": self.shared_basis_setup.get("mode", None),
                "id": self.shared_basis_setup.get("id", None),
                "k": self.shared_basis_setup.get("k", None),
                "fx": self.shared_basis_setup.get("fx", None),
                "n_linked_terms": self.shared_basis_setup.get("n_linked_terms", None),
                "features": self.shared_basis_setup.get("features", None),
            }

        return {
            "special": self.special,
            "bs": self.bs,
            "features": list(self.features),
            "k": self.k,
            "fx": bool(self.fx),
            "select": bool(self.select),
            "m": self.m,
            "by": self.by,
            "xt": self.xt,
            "id": self.id,
            "sp": self.sp,
            "pc": self.pc,
            "knots": self.knots,
            "constraint_mode": self.constraint_mode,
            "shared_basis_setup": shared_basis_summary,
            "mc": self.mc,
            "full": bool(self.full),
            "ord": self.ord,
            "extra": dict(self.extra),
        }
