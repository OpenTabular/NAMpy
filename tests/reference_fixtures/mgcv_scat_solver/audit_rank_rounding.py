"""Optional numerical audit, not a portable branch assertion.

Run from the repository root with PYTHONPATH=. and one BLAS thread. This
replays Vb.corr with the actual-R Hessian after one-ULP diagonal perturbations.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky

from nampy.gam.fit.solvers.general_family.newton import _vb_corr_root


def audit():
    path = Path(__file__).with_name("rank_vb_correction.json")
    state = json.loads(path.read_text())["capture"]
    H = np.asarray(state["H"])
    X = np.asarray(state["X"])
    rho = np.atleast_1d(state["rho"])
    Vr = np.atleast_2d(state["Vr"])
    nth = int(state["nth"])
    lam = np.atleast_1d(state["lambda"])
    derivatives = [np.zeros_like(H) for _ in range(nth)]
    for j, (S, offset) in enumerate(
        zip(state["S"], np.atleast_1d(state["off"]), strict=True)
    ):
        S = np.asarray(S)
        section = slice(int(offset) - 1, int(offset) - 1 + S.shape[0])
        block = np.zeros_like(H)
        block[section, section] = lam[j + nth] * S
        derivatives.append(block)
    records = []
    for index in [1, 2]:
        for direction in [-np.inf, np.inf]:
            altered = H.copy()
            altered[index, index] = np.nextafter(altered[index, index], direction)
            try:
                cholesky(altered, lower=False)
                success = True
            except np.linalg.LinAlgError:
                success = False
            correction = _vb_corr_root(
                np.zeros_like(X),
                L=None,
                lsp0=None,
                S_blocks=[],
                off=None,
                rho=rho,
                Vr=Vr,
                penalty_matrix=altered,
                dH_blocks=derivatives,
            )
            records.append(
                {
                    "index": index,
                    "direction": str(direction),
                    "H_delta": float(altered[index, index] - H[index, index]),
                    "chol_ok": success,
                    "correction_max": float(np.max(abs(correction))),
                    "AIC_uncapped_correction": float(
                        2 * np.sum(correction * (X.T @ X))
                    ),
                }
            )
    return records


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
