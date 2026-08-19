from __future__ import annotations

import numpy as np

from nampy.gam.inference.anova import _edf1_vector, _term_edf1
from nampy.gam.model_state import _coef_column_offset, _H_coef, _term_blocks_seq
from tests.parity.test_mgcv_prediction_inference_diagnostics_parity import _case_bundle


def main() -> None:
    for case_id in (
        "gaussian_te_full_false",
        "factor_smooth_sz",
        "gaussian_fs_select_reml",
    ):
        _data, expected, gam = _case_bundle(case_id)
        actual = gam.anova(freq=False)
        edf1 = np.asarray(_edf1_vector(gam), dtype=np.float64)
        print("==", case_id, "==")
        print("actual table")
        print(actual.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]])
        print("mgcv anova")
        print(expected["parity"]["diagnostics"].get("anova_smooth"))
        print("mgcv smooth_edf1")
        print(expected["parity"]["diagnostics"].get("smooth_edf1"))
        print("sum edf1", float(np.sum(edf1)))
        H = np.asarray(_H_coef(gam), dtype=np.float64)
        h_edf1 = 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)
        print("sum H edf1", float(np.sum(h_edf1)))
        for tb in _term_blocks_seq(gam):
            if str(getattr(tb, "term_type", "")) == "parametric":
                continue
            off = _coef_column_offset(gam)
            h_sl = slice(off + tb.coef_slice.start, off + tb.coef_slice.stop)
            print(
                tb.label,
                tb.coef_slice,
                "basis",
                tb.basis_name,
                "null",
                (getattr(tb, "metadata", {}) or {}).get("null_space_dim"),
                "term_edf1",
                _term_edf1(gam, tb),
                "H_term_edf1",
                float(np.sum(h_edf1[h_sl])),
                "ncoef",
                tb.coef_slice.stop - tb.coef_slice.start,
            )


if __name__ == "__main__":
    main()
