"""Execute independent R references; no upstream source is copied into NAMpy.

Usage: python tests/parity/generate_neighbor_references.py --upstream-dir DIR
  --r-library DIR
DIR contains git clones named expectreg, qgam, gratia, dlnm and pammtools.
R library needs mgcv, jsonlite, quantreg, quadprog, dlnm and survival.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--r-library", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(624)
    x = np.linspace(-1.4, 1.6, 45)
    X = np.column_stack([np.ones(len(x)), x, x**2])
    y = 0.3 + 0.8 * x + 0.25 * x**2 + rng.standard_t(5, len(x)) * (0.3 + 0.2 * x**2)
    V = np.array([[0.4, 0.1, -0.04], [0.1, 0.3, 0.02], [-0.04, 0.02, 0.2]])
    eig, vectors = np.linalg.eigh(V)
    Z = rng.normal(size=(1200, 3))
    data = {
        "design": X.tolist(),
        "response": y.tolist(),
        "penalty": np.diag([0, 0.4, 0.8]).tolist(),
        "weights": rng.uniform(0.3, 2, len(x)).tolist(),
        "levels": [0.2, 0.5, 0.8],
        "bandwidth": 0.15,
        "history": rng.normal(size=(15, 4)).tolist(),
        "series": [1, 2, 3, 4, 5, 6, 7, 8],
        "groups": [1, 1, 1, 1, 2, 2, 2, 2],
        "beta": [0.2, 0.6, -0.1],
        "covariance": V.tolist(),
        "functional": X[::4].tolist(),
        "standard_draws": Z.tolist(),
        "deviations": (Z @ ((vectors * np.sqrt(eig)) @ vectors.T).T).tolist(),
        "survival": {
            "subject": [1, 2, 3, 4],
            "entry": [0, 0.5, 0, 1.2],
            "time": [1, 2.7, 3, 2],
            "event": [1, 0, 1, 1],
        },
        "cuts": [0, 1, 2, 3],
    }
    family_x = np.linspace(-1, 1, 120)
    family_mu = 0.3 + 0.6 * family_x
    family_sigma = np.exp(-0.6 + 0.2 * family_x)
    beta_mu = 1 / (1 + np.exp(-family_mu))
    beta_phi = np.exp(2 + 0.2 * family_x)
    nb_mu = np.exp(family_mu)
    nb_alpha = np.exp(-0.5 + 0.3 * family_x)
    data["families"] = {
        "x": family_x.tolist(),
        "student_lss": (
            family_mu + family_sigma * rng.standard_t(5, len(family_x))
        ).tolist(),
        "beta_lss": rng.beta(beta_mu * beta_phi, (1 - beta_mu) * beta_phi).tolist(),
        "nb_lss": rng.negative_binomial(
            1 / nb_alpha, 1 / (1 + nb_alpha * nb_mu)
        ).tolist(),
    }
    boundary = [
        (mu, y, alpha)
        for mu in [0.2, 2.0, 10000.0]
        for y in [0, 1, 5, 10, 10000]
        for alpha in [0.01000001, 0.00999999, 1e-4, 1e-8, 1e-12, 1e-16]
    ]
    data["nb_boundary"] = {
        "mu": [row[0] for row in boundary],
        "y": [row[1] for row in boundary],
        "alpha": [row[2] for row in boundary],
    }
    design_x = np.linspace(-1, 1, 60)
    data["design_data"] = {
        "x": design_x.tolist(),
        "y": (0.4 + 0.7 * design_x + 0.1 * np.cos(np.arange(60))).tolist(),
        "off": (0.2 * design_x).tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "input.json", Path(tmp) / "output.json"
        src.write_text(json.dumps(data))
        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("neighbor_reference.R")),
                str(src),
                str(args.upstream_dir),
                str(args.r_library),
                str(dst),
            ],
            check=True,
        )
        result = json.loads(dst.read_text())
    commits = {}
    import mpmath as mp

    exact = []
    with mp.workdps(100):
        for mean, count, dispersion in boundary:
            mu, y, alpha = (mp.mpf(str(value)) for value in [mean, count, dispersion])
            size = 1 / alpha
            exact.append(
                float(
                    mp.loggamma(size + y)
                    - mp.loggamma(size)
                    - mp.loggamma(y + 1)
                    + size * (mp.log(size) - mp.log(size + mu))
                    + y * (mp.log(mu) - mp.log(size + mu))
                )
            )
    result["nb_high_precision"] = {
        "digits": 100,
        "mpmath": mp.__version__,
        "log_density": exact,
    }
    for name in ["expectreg", "qgam", "gratia", "dlnm", "pammtools", "gamlss.dist"]:
        commits[name] = subprocess.check_output(
            ["git", "-C", str(args.upstream_dir / name), "rev-parse", "HEAD"], text=True
        ).strip()
    out = ROOT / "tests/reference_fixtures/extensions/neighbors.json"
    # Publish only a complete reference, so readers never see truncated JSON.
    with tempfile.NamedTemporaryFile(
        "w", dir=out.parent, delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            {"inputs": data, "upstream": result, "commits": commits}, handle, indent=2
        )
        handle.write("\n")
    try:
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
