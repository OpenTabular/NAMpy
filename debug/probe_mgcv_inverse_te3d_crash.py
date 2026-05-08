from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gam_cartesian_matrix import make_data


FORMULA = (
    'y ~ te(x0, x1, x2, bs=c("cr","cr","cr"), '
    "k=c(5,5,5), sp=c(1.0,1.2,1.4))"
)

OPS = {
    "fit_only": "invisible(fit)",
    "loglik_aic": "print(logLik(fit)); print(AIC(fit))",
    "predict": (
        "print(head(predict(fit, type='response'))); "
        "print(dim(predict(fit, type='terms'))); "
        "print(dim(predict(fit, type='lpmatrix')))"
    ),
    "predict_se": "print(head(predict(fit, type='link', se.fit=TRUE)$se.fit))",
    "concurvity_full": "print(concurvity(fit, full=TRUE))",
    "concurvity_pairwise": "print(concurvity(fit, full=FALSE))",
    "sp_vcov": "print(sp.vcov(fit, edge.correct=FALSE))",
    "vcov_sandwich": (
        "print(dim(vcov(fit, sandwich=TRUE, freq=FALSE))); "
        "print(dim(vcov(fit, sandwich=TRUE, freq=TRUE)))"
    ),
    "gam_vcomp": "print(gam.vcomp(fit, rescale=FALSE))",
    "residuals": (
        "print(head(residuals(fit, type='response'))); "
        "print(head(residuals(fit, type='deviance')))"
    ),
    "k_check": "set.seed(0); print(k.check(fit, subsample=120, n.rep=8))",
    "anova": "print(anova(fit, freq=FALSE))",
    "summary_edf": "print(summary(fit)$edf)",
    "smooth_blocks": (
        "pt <- predict(fit, type='terms'); lp <- predict(fit, type='lpmatrix'); "
        "for (sm in fit$smooth) { ind <- sm$first.para:sm$last.para; "
        "print(sm$label); print(dim(fit$Vp[ind, ind, drop=FALSE])); "
        "print(dim(lp[, ind, drop=FALSE])); }"
    ),
}


def main() -> None:
    data = make_data("positive")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        csv_path = tmpdir / "data.csv"
        data.to_csv(csv_path, index=False)
        for name, expr in OPS.items():
            script = tmpdir / f"{name}.R"
            script.write_text(
                "\n".join(
                    [
                        "suppressPackageStartupMessages(library(mgcv))",
                        f"data <- read.csv({str(csv_path)!r}, stringsAsFactors=FALSE)",
                        "for (nm in names(data)) if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])",
                        f"fit <- gam(as.formula({FORMULA!r}), data=data, family=gaussian(link='inverse'), method='REML')",
                        "cat('fit ok\\n')",
                        expr,
                        "cat('op ok\\n')",
                    ]
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["Rscript", str(script)],
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            print(f"## {name}: returncode={proc.returncode}")
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)


if __name__ == "__main__":
    main()
