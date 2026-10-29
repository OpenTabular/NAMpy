"""The task guide distinguishes training support from probability queries."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_family_support_table_keeps_quantile_loss_separate_from_probability_laws():
    support = runpy.run_path(str(ROOT / "docs/_ext/family_support.py"))
    rows = {row[0]: row for row in support["support_rows"]()}
    assert rows["normal"][1:3] == ("gaulss", "normal")
    assert "log_prob" in rows["normal"][3] and "log_prob" in rows["normal"][4]
    assert rows["quantile"][2] == "quantile"
    assert rows["quantile"][3:] == ("Unavailable", "Unavailable")
    assert "cdf_left" in rows["beinf"][3] and "cdf_left" in rows["beinf"][4]
    for name in ("bct", "bcpe", "lognormal_observation", "weibull_observation"):
        assert rows[name][1:3] == (name + "_lss", name)
        assert "log_prob" in rows[name][3] and "log_prob" in rows[name][4]
    for name in ("lognormal_observation", "weibull_observation"):
        assert "variance" not in rows[name][3] and "variance" not in rows[name][4]


def test_generated_support_table_matches_the_owning_registries(tmp_path):
    support = runpy.run_path(str(ROOT / "docs/_ext/family_support.py"))
    support["generate"](tmp_path)
    relative = "docs/_generated/family_support.inc"
    assert (tmp_path / relative).read_text() == (ROOT / relative).read_text()
    page = (ROOT / "docs/user_guide/api_and_families.rst").read_text()
    assert "../_generated/family_support.inc" in page
