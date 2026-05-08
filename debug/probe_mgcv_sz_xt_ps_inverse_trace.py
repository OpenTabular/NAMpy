from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gam_cartesian_matrix import make_data


FORMULA = (
    'y ~ s(f, x0, bs="sz", k=7, m=2, xt=list(bs="ps"), '
    "sp=c(1.0,1.2,1.4,1.6))"
)


def main() -> None:
    data = make_data("positive")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        csv_path = tmpdir / "data.csv"
        script_path = tmpdir / "trace.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(
            "\n".join(
                [
                    "suppressPackageStartupMessages(library(mgcv))",
                    f"data <- read.csv({str(csv_path)!r}, stringsAsFactors=FALSE)",
                    "for (nm in names(data)) if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])",
                    "ctrl <- gam.control(trace=TRUE)",
                    f"fit <- gam(as.formula({FORMULA!r}), data=data, family=gaussian(link='inverse'), method='REML', control=ctrl)",
                    "cat('iter', fit$iter, '\\n')",
                    "cat('deviance', fit$deviance, '\\n')",
                    "cat('logLik', as.numeric(logLik(fit)), '\\n')",
                    "cat('coef', paste(format(unname(coef(fit)), digits=17), collapse=' '), '\\n')",
                    "cat('fitted', paste(format(head(fitted(fit), 8), digits=17), collapse=' '), '\\n')",
                ]
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["Rscript", str(script_path)],
            text=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        print("returncode", proc.returncode)
        print(proc.stdout)
        print(proc.stderr)


if __name__ == "__main__":
    main()
