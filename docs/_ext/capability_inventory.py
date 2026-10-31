"""Render support declarations and scoped test evidence as separate inventories.

Runtime registries own support. This documentation adapter adds task labels and
links to evidence; it never fits models or upgrades a declaration into a claim
that all configurations have been validated.
"""

from __future__ import annotations

import json
from pathlib import Path

# These are evidence annotations, not a second capability registry. A missing
# annotation means no evidence is indexed here, not that support is unavailable.
EVIDENCE = {
    "statistical_gam": ("Selected Gaussian/binomial interface and prediction cases", "tests/public_api/test_gam_adapters.py"),
    "distributional": ("Selected Student-t, beta and negative-binomial joint likelihood fits", "tests/extensions/test_distributional.py"),
    "quantile": ("Selected fixed-penalty scalar quantile loss fits", "tests/extensions/test_neighbors.py"),
    "expectile": ("Selected weighted expectile fits and conditional-mean limit checks", "tests/extensions/test_neighbors.py"),
    "calibrated_quantile": ("qgam fixed-penalty profiles, endpoints and a specified coverage simulation", "tests/extensions/test_calibrated_quantile.py"),
    "adaptive_quantile": ("Selected SHASH bandwidth and automatic-smoothing fixtures", "tests/extensions/test_quantile_bandwidth.py"),
    "survival": ("Selected piecewise-exponential statistical fits", "tests/survival/test_statistical.py"),
    "parametric_survival": ("Selected censored parametric statistical fits", "tests/survival/test_censored_fitting.py"),
    "cox": ("Selected Cox likelihood, coefficients, covariance and survival fixtures", "tests/survival/test_cox_fixed.py"),
    "cox_smoothing": ("Selected supported automatic-smoothing Cox fits", "tests/survival/test_cox_smoothing.py"),
    "neural_cox": ("Selected neural Cox loss, fitting and prediction cases", "tests/survival/test_neural_cox.py"),
    "neural_survival": ("Selected neural full-likelihood survival cases", "tests/survival/test_neural.py"),
    "semistructured": ("Selected spline/neural orthogonality and prediction cases", "tests/neural/test_semistructured.py"),
    "bct": ("Selected fixed-smoothing BCT/BCPE fits and likelihood derivatives", "tests/extensions/test_boxcox_tail_fitting.py"),
    "bcpe": ("Selected fixed-smoothing BCT/BCPE fits and likelihood derivatives", "tests/extensions/test_boxcox_tail_fitting.py"),
    "neural_bct": ("Selected neural BCT/BCPE densities and autograd cases", "tests/neural/test_boxcox_tail_objectives.py"),
    "neural_bcpe": ("Selected neural BCT/BCPE densities and autograd cases", "tests/neural/test_boxcox_tail_objectives.py"),
    "neural_normal_observation": ("Selected neural normal observation losses and fits", "tests/extensions/test_normal_observations.py"),
    "neural_lognormal_observation": ("Selected neural positive observation losses and fits", "tests/extensions/test_positive_observations.py"),
    "neural_weibull_observation": ("Selected neural positive observation losses and fits", "tests/extensions/test_positive_observations.py"),
    "bccg": ("Selected BCCG likelihood fits", "tests/extensions/test_bccg_fitting.py"),
    "betabinomial": ("Selected beta-binomial likelihood and trial-context cases", "tests/extensions/test_betabinomial_lss.py"),
    "beinf": ("Selected inflated-beta likelihood and atom cases", "tests/extensions/test_inflated_beta.py"),
    "zaga": ("Selected zero-adjusted Gamma fits and atom cases", "tests/extensions/test_zero_adjusted_gamma.py"),
    "normal_observation": ("Selected exact/censored/truncated normal likelihood cases", "tests/extensions/test_normal_observations.py"),
    "lognormal_observation": ("Selected censored/truncated positive likelihood cases", "tests/extensions/test_positive_observations.py"),
    "weibull_observation": ("Selected censored/truncated positive likelihood cases", "tests/extensions/test_positive_observations.py"),
}


def _evidence(*keys):
    return [{"scope": EVIDENCE[key][0], "test_module": EVIDENCE[key][1]} for key in keys if key in EVIDENCE]


def _capabilities(model):
    return {
        name: {"status": status.state, "conditions": list(status.conditions), "reason": status.reason}
        for name, status in sorted(model.capabilities().items())
    }


def inventory():
    """Return JSON-safe declarations, without training or inspecting test results."""
    from nampy import (
        GAMLSS,
        CalibratedQuantileGAM,
        CoxGAM,
        ExpectileGAM,
        GAMClassifier,
        GAMRegressor,
        NAMSurvival,
        QuantileGAM,
        SemiStructuredRegressor,
        SurvivalGAM,
        SurvivalObjectiveConfig,
    )
    from nampy.gam.families.registry import GENERAL_FAMILIES
    from nampy.neural.families.registry import FAMILY_REGISTRY

    routes = (
        (GAMRegressor(), "Exact scalar response", "Statistical penalized fitting", "Configured statistical criterion", ("statistical_gam",)),
        (GAMClassifier(), "Binary or grouped binomial", "Statistical penalized fitting", "Configured statistical criterion", ("statistical_gam",)),
        (GAMLSS(), "Family-dependent", "Joint statistical likelihood", "Family-declared criteria/optimizers", ("distributional",)),
        (ExpectileGAM(), "Exact scalar response; optional observation weights", "Asymmetric least squares (LAWS)", "Supplied penalties; no automatic selection", ("expectile",)),
        (QuantileGAM(), "Exact scalar response", "Quantile loss", "Supplied penalties", ("quantile",)),
        (CalibratedQuantileGAM(), "Exact scalar response", "ELF and scalar calFast calibration", "Supplied penalties or REML at each scale", ("calibrated_quantile", "adaptive_quantile")),
        (SurvivalGAM(), "Right-censored; delayed entry", "Piecewise-exponential full likelihood", "Poisson statistical engine", ("survival",)),
        (SurvivalGAM(likelihood="weibull"), "Right-censored; delayed entry", "Parametric full likelihood (Weibull shown)", "Fixed or eligible joint REML", ("parametric_survival",)),
        (CoxGAM(), "Static right-censored; optional strata", "Breslow partial likelihood", "Fixed or eligible REML; stratified automatic requires BFGS", ("cox", "cox_smoothing")),
        (NAMSurvival(), "Right-censored; full-likelihood route", "Neural survival objective", "Gradient training", ("neural_survival",)),
        (NAMSurvival(objective=SurvivalObjectiveConfig(likelihood="cox")), "Static right-censored, one stratum", "Neural Breslow partial likelihood", "Gradient training on full cohort", ("neural_cox",)),
        (SemiStructuredRegressor(), "Exact scalar response", "Identified spline and neural composition", "Configured penalties and gradient training", ("semistructured",)),
    )
    models = []
    for model, observations, fitting, smoothing, evidence in routes:
        capabilities = _capabilities(model)
        models.append({
            "model": type(model).__name__,
            "observation_type": observations,
            "fitting_route": fitting,
            "smoothing_selection": smoothing,
            "probability_operations": {k: v for k, v in capabilities.items() if k.startswith(("predict_", "distribution.", "survival."))},
            "uncertainty_methods": {k: v for k, v in capabilities.items() if k.startswith(("uncertainty.", "fisher."))},
            "conditions": sorted({c for v in capabilities.values() if v["status"] != "unsupported" for c in v["conditions"]}),
            "validation_evidence": _evidence(*evidence),
        })
    families = []
    for backend, registry in (("statistical", GENERAL_FAMILIES), ("neural", FAMILY_REGISTRY)):
        for name, spec in sorted(registry.items()):
            cls = spec.family_type if backend == "statistical" else spec.distribution
            law = spec.law_factory() if spec.law_factory is not None else None
            family_name = (spec.public_name or name) if backend == "statistical" else name
            observation_target = bool(getattr(cls, "supports_observation_target", False))
            families.append({
                "backend": backend,
                "family": name,
                "observation_type": ("Exact/censored observations with configured truncation" if observation_target else law.spec.response_kind if law is not None else "Family-specific; no shared probability law declared"),
                "fitting_route": "Registered statistical family" if backend == "statistical" else "Registered neural objective",
                "smoothing_selection": sorted(getattr(cls, "supported_smoothing_optimizers", ()) or ()) if backend == "statistical" else [],
                "probability_operations": sorted(law.spec.operations) if law is not None else [],
                "uncertainty_methods": "Use configured/fitted model.capabilities(); not implied by probability operations",
                "conditions": list(getattr(cls, "prediction_context_names", law.spec.context_names if law is not None else ())),
                "validation_evidence": _evidence(family_name if backend == "statistical" else "neural_" + family_name),
            })
    return {"models": models, "families": families}


def _table(headers, rows):
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + headers[0]]
    lines.extend("     - " + cell for cell in headers[1:])
    for row in rows:
        lines.append("   * - " + row[0])
        lines.extend("     - " + cell for cell in row[1:])
    return "\n".join(lines) + "\n"


def _supported(operations):
    return "; ".join(
        name + (" (conditional)" if detail["conditions"] else "")
        for name, detail in operations.items() if detail["status"] != "unsupported"
    ) or "None declared"


def generate(root):
    data = inventory()
    target = Path(root) / "docs/_generated/capabilities"
    target.mkdir(parents=True, exist_ok=True)
    outputs = {
        "models.inc": _table(
            ["Model", "Observation type", "Fitting route", "Smoothing selection"],
            [(r["model"], r["observation_type"], r["fitting_route"], r["smoothing_selection"]) for r in data["models"]],
        ),
        "operations.inc": _table(
            ["Model / route", "Prediction and probability operations", "Uncertainty methods", "Conditions"],
            [(r["model"] + " / " + r["fitting_route"], _supported(r["probability_operations"]), _supported(r["uncertainty_methods"]), ", ".join(r["conditions"]) or "None declared") for r in data["models"]],
        ),
        "families.inc": _table(
            ["Backend / family", "Observation type", "Declared smoothing optimizers", "Probability operations", "Prediction context"],
            [(r["backend"] + " / " + r["family"], r["observation_type"], ", ".join(r["smoothing_selection"]) or "No declaration at family level", ", ".join(r["probability_operations"]) or "No shared probability adapter", ", ".join(r["conditions"]) or "None") for r in data["families"]],
        ),
        "evidence.inc": _table(
            ["Evidence scope", "Test module"],
            [(scope, f"`{path} <https://github.com/OpenTabular/NAMpy/blob/main/{path}>`_") for scope, path in sorted(set(EVIDENCE.values()))],
        ),
        "inventory.json": json.dumps(data, indent=2, sort_keys=True) + "\n",
    }
    for name, content in outputs.items():
        path = target / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    return data


def setup(app):
    app.connect("builder-inited", lambda app: generate(Path(app.srcdir).parent))
    return {"parallel_read_safe": True, "parallel_write_safe": True}
