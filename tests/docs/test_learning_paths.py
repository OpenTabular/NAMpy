"""Learning tracks, application claims, and the mgcv landing-page snapshot."""

import posixpath
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _toctree_targets():
    found = []
    for path in DOCS.rglob("*.rst"):
        lines = path.read_text().splitlines()
        index = 0
        while index < len(lines):
            if not lines[index].startswith(".. toctree::"):
                index += 1
                continue
            index += 1
            while index < len(lines) and (
                lines[index].startswith("   ") or lines[index].strip() == ""
            ):
                entry = lines[index].strip()
                index += 1
                if not entry or entry.startswith(":") or entry.startswith(".."):
                    continue
                target = entry.split("<")[-1].rstrip(">").strip() if "<" in entry else entry
                parent = path.parent.relative_to(DOCS).as_posix()
                base = "" if parent == "." else parent
                resolved = posixpath.normpath(posixpath.join(base, target))
                found.append((str(path.relative_to(DOCS)), resolved))
    return found


def test_every_tutorial_has_one_track_and_the_gallery_is_linked():
    targets = [target for _, target in _toctree_targets()]
    duplicates = sorted({item for item in targets if targets.count(item) > 1})
    assert duplicates == []
    for page in (
        "tutorials/track_new",
        "tutorials/track_statistical",
        "tutorials/track_neural",
        "applications/index",
        "from_mgcv",
    ):
        assert page in targets
    applications = {
        target
        for source, target in _toctree_targets()
        if source == "applications/index.rst"
    }
    assert applications == {
        "applications/claim_counts",
        "applications/energy_demand",
        "applications/monotone_risk",
        "applications/housing_effects",
        "applications/loss_distribution",
        "applications/retention",
        "applications/spatial_pollution",
        "applications/quantile_demand",
    }
    openers = {
        "notebooks/01_getting_started": "Discover nonlinear effects and check whether you can trust them",
        "notebooks/smoothness_selection": "Which parts of a curve should the penalty keep?",
    }
    for name, title in openers.items():
        text = (DOCS / f"{name}.ipynb").read_text()
        assert text.split('"source"', 1)[1].startswith(f""": [\n    "# {title}""") or f"# {title}" in text
    assert "How does demand change with temperature and time of day?" in (
        DOCS / "tutorials/first_gam.rst"
    ).read_text()
    assert "Do a GAM and a NAM recover the same effects?" in (
        DOCS / "tutorials/gam_nam.rst"
    ).read_text()


def test_published_application_numbers_still_match_their_sources():
    energy = (DOCS / "_includes/first_gam_walkthrough.inc").read_text()
    assert "RMSE: 1.95" in energy and "0.989" in energy
    scores = (DOCS / "_generated/gam_nam_scores.inc").read_text()
    assert "0.975" in scores and "0.971" in scores
    citations = {
        "notebooks/overdispersed_counts.ipynb": ("2.429", "2.662", "2.207"),
        "notebooks/gamma_location_scale.ipynb": ("0.584", "0.843"),
        "notebooks/survival_censoring.ipynb": ("0.188", "0.669"),
        "notebooks/regional_smoothing.ipynb": ("0.947", "0.946"),
        "notebooks/04_distributional_models.ipynb": ("77.5%", "91.25%"),
    }
    for name, phrases in citations.items():
        text = (DOCS / name).read_text()
        for phrase in phrases:
            assert phrase in text
    quantile = (DOCS / "in_depth/quantile_calibration.rst").read_text()
    for phrase in ("3.9%", "97.5%", "93.6%", "39.3%"):
        assert phrase in quantile
    pages = {
        "applications/claim_counts.rst": ("2.429", "2.662", "2.207"),
        "applications/energy_demand.rst": ("1.95", "0.989", "15.46", "0.286"),
        "applications/housing_effects.rst": ("0.975", "0.971", "0.752"),
        "applications/loss_distribution.rst": ("0.584", "0.843"),
        "applications/retention.rst": ("0.188", "0.669"),
        "applications/spatial_pollution.rst": ("0.947", "0.946"),
        "applications/quantile_demand.rst": ("77.5%", "91.25%", "93.6%", "39.3%"),
    }
    for name, phrases in pages.items():
        text = (DOCS / name).read_text()
        for phrase in phrases:
            assert phrase in text, (name, phrase)


def _energy_linear_baseline():
    rng = np.random.default_rng(21)
    n = 900
    temperature = rng.uniform(-5.0, 35.0, n)
    hour = rng.uniform(0.0, 24.0, n)
    demand = (
        45.0
        + 0.12 * (temperature - 18.0) ** 2
        + 5.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
        + rng.normal(0.0, 2.0, n)
    )
    data = pd.DataFrame({"demand": demand, "temperature": temperature, "hour": hour})
    train, test = train_test_split(data, test_size=0.2, random_state=21)
    model = LinearRegression().fit(train[["temperature", "hour"]], train["demand"])
    predicted = model.predict(test[["temperature", "hour"]])
    rmse = float(np.sqrt(np.mean((test["demand"] - predicted) ** 2)))
    return rmse, float(r2_score(test["demand"], predicted))


def _housing_linear_r2():
    rng = np.random.default_rng(41)
    n = 1_200
    data = pd.DataFrame(
        {
            "median_income": rng.uniform(1.0, 10.0, n),
            "house_age": rng.uniform(1.0, 52.0, n),
            "latitude": rng.uniform(32.0, 42.0, n),
            "rooms_per_household": rng.lognormal(1.5, 0.35, n),
        }
    )
    data["value"] = (
        2.2 * np.tanh((data["median_income"] - 5.0) / 1.7)
        + 0.9 * np.sin((data["house_age"] - 5.0) / 8.0)
        - 0.55 * (data["latitude"] - 37.0) ** 2 / 6.0
        + 0.75 * np.log1p(data["rooms_per_household"])
        + rng.normal(0.0, 0.3, n)
    )
    train, test = train_test_split(data, test_size=0.2, random_state=41)
    columns = [
        "median_income",
        "house_age",
        "latitude",
        "rooms_per_household",
    ]
    predicted = LinearRegression().fit(train[columns], train["value"]).predict(test[columns])
    return float(r2_score(test["value"], predicted))


def test_application_linear_baselines_use_the_published_splits():
    rmse, energy_r2 = _energy_linear_baseline()
    assert round(rmse, 2) == 15.46
    assert round(energy_r2, 3) == 0.286
    assert round(_housing_linear_r2(), 3) == 0.752


def test_mgcv_landing_page_matches_the_committed_snapshot():
    module = runpy.run_path(str(DOCS / "scripts/generate_mgcv_parity_figure.py"))
    comparison = module["gaussian_cr_comparison"]()
    response_gap = np.max(np.abs(comparison["response_nampy"] - comparison["response_mgcv"]))
    term_gap = np.max(np.abs(comparison["terms_nampy"] - comparison["terms_mgcv"]))
    assert response_gap < 1e-9
    assert term_gap < 1e-9
    assert abs(comparison["edf_total_nampy"] - comparison["edf_total_mgcv"]) < 1e-8
    assert abs(comparison["reml_nampy"] - comparison["reml_mgcv"]) < 1e-8
    table = (DOCS / "_generated/mgcv_parity_gaussian_cr.inc").read_text()
    assert "REML" in table
    for line in table.splitlines():
        if line.startswith("   Max absolute"):
            assert float(line.rsplit(",", 1)[-1]) < 1e-9
    image = DOCS / "_static/mgcv_parity_gaussian_cr.png"
    assert image.stat().st_size > 10_000
    page = (DOCS / "from_mgcv.rst").read_text()
    assert "library(mgcv)" in page and "GAMRegressor" in page
    assert "mgcv_parity_gaussian_cr.png" in page
