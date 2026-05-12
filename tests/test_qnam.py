from nampy.models import QNAM
from nampy.models.sklearn_lss import SklearnBaseLSS


def test_qnam_fit_returns_self_and_uses_default_quantile_family(monkeypatch):
    captured = {}

    def fake_fit(self, X, y, family, distributional_kwargs=None, **kwargs):
        captured["family"] = family
        captured["distributional_kwargs"] = distributional_kwargs
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(SklearnBaseLSS, "fit", fake_fit)

    model = QNAM()
    result = model.fit([[0.0]], [1.0], max_epochs=1)

    assert result is model
    assert captured["family"] == "quantile"
    assert captured["distributional_kwargs"] == {"quantiles": [0.25, 0.5, 0.75]}
    assert captured["kwargs"]["max_epochs"] == 1
