from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.model.api import GAM
from tests.mgcv_parity_utils import _make_fs_data, _make_gaussian_data


def _make_factorized_gaussian_data(seed=101, n=96):
    df = _make_gaussian_data(seed=seed, n=n).copy()
    levels = np.array(["a", "b", "c"], dtype=object)
    df["f"] = levels[np.arange(n) % levels.size]
    return df


@pytest.mark.parametrize(
    ("formula", "data"),
    [
        ('y ~ s(f, x, bs="fs", xt="cs")', _make_fs_data()),
        ('y ~ s(f, x, bs="fs", xt="ts")', _make_fs_data()),
        ('y ~ s(f, x0, x1, bs="fs", xt="ts", k=10)', _make_factorized_gaussian_data()),
    ],
)
def test_fs_full_rank_shrinkage_bases_are_explicitly_unsupported(formula, data):
    """Verify that fs full rank shrinkage bases are explicitly unsupported."""
    with pytest.raises(
        NotImplementedError,
        match=r'`bs="fs"` with base bs=.* is unsupported\. Upstream mgcv::smooth\.construct\.fs\.smooth\.spec also rejects this full-rank shrinkage base',
    ):
        GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)
