"""Phase 0.2 — intentional NotImplementedError guardrails."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.smooths.registry import make_smooth_term


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y": rng.normal(size=20),
            "x0": rng.normal(size=20),
            "x1": rng.normal(size=20),
            "f": np.array(list("abababcdabcdababcdab")[:20], dtype=object),
        }
    )


def test_te_non_cr_marginal_raises():
    with pytest.raises(NotImplementedError, match="cr"):
        make_smooth_term(
            "te",
            feature=["x0", "x1"],
            k=[6, 6],
            basis=["tp", "cr"],
            label="te",
        )


def test_runtime_tensor_select_true_raises():
    """Isolate select=True on te(); avoid list-valued bs in the formula string."""
    data = _df()
    with pytest.raises(NotImplementedError, match="select=True"):
        GAM(family="gaussian", formula="y ~ te(x0, x1, k=[5, 5])", select=True).fit(
            data=data
        )


def test_runtime_mrf_select_true_raises():
    d = pd.DataFrame({"y": [0.0, 1.0, 0.2], "r": ["A", "B", "C"]})
    with pytest.raises(NotImplementedError, match="mrf"):
        GAM(
            family="gaussian",
            formula=(
                'y ~ s(r, bs="mrf", k=3, '
                'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
            ),
            select=True,
        ).fit(data=d)


def test_runtime_re_select_true_raises():
    d = pd.DataFrame({"y": [0.0, 1.0], "g": ["a", "b"]})
    with pytest.raises(NotImplementedError, match="re"):
        GAM(family="gaussian", formula='y ~ s(g, bs="re")', select=True).fit(data=d)


def test_runtime_fs_select_true_raises():
    d = pd.DataFrame({"y": [0.0, 1.0], "f": ["a", "b"], "x": [0.1, 0.2]})
    with pytest.raises(NotImplementedError, match="fs"):
        GAM(family="gaussian", formula='y ~ s(f, x, bs="fs", k=5)', select=True).fit(data=d)


def test_runtime_sz_select_true_raises():
    d = pd.DataFrame(
        {
            "y": [0.0, 1.0, 0.5],
            "f1": ["a", "b", "a"],
            "f2": ["x", "x", "y"],
            "x": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(NotImplementedError, match="sz"):
        GAM(family="gaussian", formula='y ~ s(f1, f2, x, bs="sz", k=5)', select=True).fit(data=d)


def test_re_linked_id_raises_at_runtime():
    d = pd.DataFrame({"y": [0.0, 1.0], "g": ["a", "b"]})
    with pytest.raises(NotImplementedError, match="re.*id"):
        GAM(family="gaussian", formula='y ~ s(g, bs="re", id="z")').fit(data=d)


def test_ps_pc_not_implemented():
    data = _df()
    with pytest.raises(NotImplementedError, match="pc"):
        GAM(family="gaussian", formula='y ~ s(x0, bs="ps", pc=0.0)').fit(data=data)


def test_linked_id_incompatible_k_raises():
    data = _df()
    with pytest.raises(NotImplementedError, match="Linked id"):
        GAM(
            family="gaussian",
            formula='y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=8, id="g")',
        ).fit(data=data)


def test_te_ps_marginal_raises_via_make_smooth_term():
    """Tensor marginal restriction via runtime, not parse_gam_formula list-bs."""
    with pytest.raises(NotImplementedError, match="cr"):
        make_smooth_term(
            "te",
            feature=["x0", "x1"],
            k=[5, 5],
            basis=["ps", "cr"],
            label="te",
        )
