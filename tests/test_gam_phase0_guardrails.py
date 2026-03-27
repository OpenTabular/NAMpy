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



def test_linked_id_incompatible_k_harmonizes():
    """mgcv auto-resolves k mismatches by using max k; we should warn and succeed."""
    data = _df()
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        gam = GAM(
            family="gaussian",
            formula='y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=8, id="g")',
        )
        gam.fit(data=data)
    assert any(
        "k" in str(warning.message).lower() for warning in w
    ), "Expected a warning about k harmonisation"
    pred = gam.predict(data)
    assert np.all(np.isfinite(pred))


def test_linked_id_compatible_k_fits():
    """Linked 1D cr smooths with matching k share a basis and fit without error."""
    data = _df()
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")',
    )
    gam.fit(data=data)
    pred = gam.predict(data)
    assert np.all(np.isfinite(pred))


def test_linked_id_shares_smoothing_parameter():
    """Two linked terms share one smoothing parameter; unlinked terms have two."""
    data = _df()
    gam_linked = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")',
    )
    gam_unlinked = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
    )
    gam_linked.fit(data=data)
    gam_unlinked.fit(data=data)
    assert gam_linked.n_smoothing_params_ < gam_unlinked.n_smoothing_params_


def test_linked_id_noncrcs_supported():
    """Non-cr/cs bases with id= link smoothing parameters without error."""
    data = _df()
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="ps", k=6, id="g") + s(x1, bs="ps", k=6, id="g")',
    )
    gam.fit(data=data)
    pred = gam.predict(data)
    assert np.all(np.isfinite(pred))


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
