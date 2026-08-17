import math

import numpy as np
import pytest

from nampy.gam.inference import chi_square_mixtures as csm


def test_psum_chisq_rounds_df_like_mgcv():
    lb = np.array([0.8, 1.7], dtype=np.float64)

    actual = csm.psum_chisq(2.4, lb, df=np.array([1.6, 2.4]))
    expected = csm.psum_chisq(2.4, lb, df=np.array([2, 2]))

    assert actual == pytest.approx(expected, rel=0.0, abs=0.0)


def test_psum_chisq_davies_failure_uses_mgcv_liu_upper_tail(monkeypatch):
    def _failing_davies(self, **_kwargs):
        return -1.0, [0.0] * 7, 1

    monkeypatch.setattr(csm.DaviesAlgorithm, "davies", _failing_davies)

    lb = np.array([0.8, 1.7], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="failure of Davies method"):
        actual = csm.psum_chisq(2.4, lb, df=np.array([2, 2]), lower_tail=True)

    expected = csm.liu2(2.4, lb, df=np.array([2, 2]), lower_tail=False)
    assert actual == pytest.approx(expected, rel=0.0, abs=0.0)


def test_psum_chisq_warns_on_davies_roundoff(monkeypatch):
    def _roundoff_davies(self, **_kwargs):
        return 0.25, [0.0] * 7, 2

    monkeypatch.setattr(csm.DaviesAlgorithm, "davies", _roundoff_davies)

    with pytest.warns(RuntimeWarning, match="danger of round-off error"):
        actual = csm.psum_chisq(2.4, np.array([1.0]), lower_tail=False)

    assert actual == pytest.approx(0.75, rel=0.0, abs=0.0)


def test_davies_trace_counts_helper_cycles():
    solver = csm.DaviesAlgorithm()

    _probability, trace, ifault = solver.davies(
        lb=np.array([0.8, 1.7], dtype=np.float64),
        nc=np.array([0.0, 0.0], dtype=np.float64),
        n=np.array([2, 2], dtype=np.int64),
        r=2,
        sigma=0.0,
        c_val=2.4,
        lim=100000,
        acc=2e-5,
    )

    assert ifault in (0, 2)
    assert trace[6] > 0.0


def test_log1pmx_is_stable_for_small_x():
    x = 1e-16

    assert math.log1p(x) - x == 0.0
    assert csm._log1pmx(x) == pytest.approx(-0.5e-32)


def test_davies_rounding_does_not_emulate_c_integer_overflow():
    """Integration counts remain mathematical integers beyond a C ``int`` range."""
    solver = csm.DaviesAlgorithm()
    assert solver._c_round_int(float(2**40) + 0.75) == 2**40 + 1
