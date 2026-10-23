"""Generate a representative optimizer matrix by fitting the advertised routes."""

from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam import GAM

ROOT = Path(__file__).resolve().parents[2]


def main():
    rng = np.random.default_rng(117)
    x = np.linspace(-1, 1, 36)
    data = pd.DataFrame(
        {"x": x, "y": 1 + np.exp(0.35 * x) + rng.normal(0, 0.06, len(x))}
    )
    rows = []
    cases = []
    for criterion, optimizer in [
        ("gcv", "outer_newton"),
        ("reml", "outer_newton"),
        ("reml", "efs"),
        ("reml", "optim"),
    ]:
        cases.append(
            ("Ordinary", "gaussian", criterion, optimizer, "newton", 0.0, True, "")
        )
    for optimizer in ["bfgs", "efs", "optim", "nlm", "nlm.fd"]:
        cases.append(("SCAM", "gaussian", "gcv", optimizer, "newton", 0.0, True, ""))
    cases += [
        ("SCAM", "gaussian", "gcv", "efs", "bfgs", 0.0, True, ""),
        ("SCAM", "gaussian", "reml", "outer_newton", "newton", 0.0, False, "GCV/UBRE"),
        ("Transformed LSS", "gaulss", "fixed", "outer_newton", "newton", 0.0, True, ""),
        (
            "Transformed LSS",
            "gaulss",
            "reml",
            "outer_newton",
            "newton",
            0.0,
            False,
            "transformed Laplace",
        ),
        ("Ordinary", "gaussian", "reml", "outer_newton", "newton", 0.3, True, ""),
        (
            "Ordinary",
            "poisson",
            "reml",
            "outer_newton",
            "newton",
            0.3,
            False,
            "Gaussian",
        ),
    ]
    for (
        kind,
        family,
        criterion,
        optimizer,
        coefficient,
        rho,
        supported,
        reason,
    ) in cases:
        formula = (
            "y ~ s(x, bs='cr', k=6)"
            if kind == "Ordinary"
            else "y ~ s(x, bs='mpi', k=6)"
        )
        if kind == "Transformed LSS":
            formula = [formula, "~ s(x, bs='cr', k=5)"]
        kwargs = {
            "formula": formula,
            "family": family,
            "optimize_smoothing": criterion != "fixed",
            "smoothing_method": criterion,
            "smoothing_optimizer": optimizer,
            "coefficient_optimizer": coefficient,
        }
        if criterion == "fixed":
            kwargs["smoothing_params"] = [0.5, 0.5]
        if rho:
            kwargs["ar1_rho"] = rho
        try:
            model = GAM(**kwargs).fit(data=data)
        except (NotImplementedError, ValueError) as error:
            if supported or reason.lower() not in str(error).lower():
                raise
            status = "Rejected as intended"
        else:
            assert supported, (kind, family, criterion, optimizer)
            assert np.isfinite(model.fit_result().coef_full).all()
            status = "Finite fit"
        rows.append(
            [
                kind,
                family,
                criterion,
                optimizer,
                coefficient,
                "AR(1)" if rho else "Independent",
                status,
            ]
        )
    text = ".. list-table:: Representative runtime-checked routes\n   :header-rows: 1\n   :class: api-reference-table\n\n"
    for row in [
        [
            "Model",
            "Family",
            "Criterion",
            "Outer optimizer",
            "Coefficient solver",
            "Correlation",
            "Result",
        ],
        *rows,
    ]:
        text += (
            "\n".join(
                ("   * - " if i == 0 else "     - ") + str(value)
                for i, value in enumerate(row)
            )
            + "\n"
        )
    output = ROOT / "docs/_generated/reference/optimizer_routes.inc"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n")
    print("PASS", len(rows), "optimizer route probes")


if __name__ == "__main__":
    main()
