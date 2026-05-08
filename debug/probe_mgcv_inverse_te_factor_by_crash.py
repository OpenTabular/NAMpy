from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gam_cartesian_matrix import make_data


FORMULA = (
    'y ~ f + te(x0, x1, by=f, bs=c("cr","cr"), '
    "k=c(5,5), sp=c(1.0,1.2))"
)

OPS = {
    "fit_only": "invisible(fit)",
    "loglik": "print(logLik(fit))",
    "predict": "print(head(predict(fit, type='response')))",
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
