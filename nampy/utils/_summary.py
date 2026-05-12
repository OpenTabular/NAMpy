"""Internal model summary and diagnostics helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import torch


def _class_name(value):
    if value is None:
        return None
    if isinstance(value, type):
        return value.__name__
    return value.__class__.__name__


def _unwrap_model(estimator):
    fitted_model = getattr(estimator, "model", None)
    if fitted_model is not None and hasattr(fitted_model, "model"):
        return fitted_model.model
    return fitted_model if isinstance(fitted_model, torch.nn.Module) else estimator


def _config_dict(config):
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value for key, value in vars(config).items() if not key.startswith("_")
        }
    return {}


def _parameter_summary(model):
    if not isinstance(model, torch.nn.Module):
        return {"total": 0, "trainable": 0}

    params = list(model.parameters())
    return {
        "total": int(sum(param.numel() for param in params)),
        "trainable": int(sum(param.numel() for param in params if param.requires_grad)),
    }


def _term_summary(model):
    terms = {
        "numerical": [],
        "categorical": [],
        "interactions": [],
    }
    if hasattr(model, "num_feature_networks"):
        terms["numerical"] = list(model.num_feature_networks.keys())
    if hasattr(model, "cat_feature_networks"):
        terms["categorical"] = list(model.cat_feature_networks.keys())
    if hasattr(model, "interaction_networks"):
        terms["interactions"] = list(model.interaction_networks.keys())
    terms["total"] = (
        len(terms["numerical"]) + len(terms["categorical"]) + len(terms["interactions"])
    )
    return terms


def _diagnostics(estimator):
    """
    Return model-specific diagnostics for a fitted NAMpy estimator or base model.
    """
    model = _unwrap_model(estimator)
    info = {}

    if hasattr(model, "get_spline_diagnostics"):
        info["spline"] = model.get_spline_diagnostics()
    if hasattr(model, "get_group_norms"):
        info["group_norms"] = model.get_group_norms()
    if hasattr(model, "selected_groups"):
        info["selected_groups"] = model.selected_groups()

    return info


def _summary_dict(estimator):
    """
    Build a structured summary dictionary for a NAMpy estimator or base model.
    """
    fitted_model = getattr(estimator, "model", None)
    task_model = fitted_model if hasattr(fitted_model, "task_kind") else None
    base_model = _unwrap_model(estimator)
    data_module = getattr(estimator, "data_module", None)
    config = getattr(estimator, "config", None)

    fitted = fitted_model is not None and data_module is not None
    base_model_class = _class_name(base_model)
    if (not isinstance(base_model, torch.nn.Module)) and hasattr(
        estimator, "base_model"
    ):
        base_model_class = _class_name(estimator.base_model)

    num_feature_info = (
        getattr(data_module, "num_feature_info", None)
        if data_module is not None
        else {}
    ) or {}
    cat_feature_info = (
        getattr(data_module, "cat_feature_info", None)
        if data_module is not None
        else {}
    ) or {}

    feature_names_in = getattr(estimator, "feature_names_in_", None)
    feature_names = list(feature_names_in) if feature_names_in is not None else []
    if not feature_names:
        feature_names = list(num_feature_info) + list(cat_feature_info)

    info = {
        "estimator": _class_name(estimator),
        "base_model": base_model_class,
        "fitted": bool(fitted),
        "task": getattr(task_model, "task_kind", None),
        "output_dim": getattr(task_model, "output_dim", None),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "numerical_features": list(num_feature_info.keys()),
        "categorical_features": list(cat_feature_info.keys()),
        "terms": _term_summary(base_model),
        "parameters": _parameter_summary(
            fitted_model if isinstance(fitted_model, torch.nn.Module) else base_model
        ),
        "config": _config_dict(config),
        "diagnostics": _diagnostics(estimator),
    }

    if hasattr(estimator, "classes_"):
        info["classes"] = list(estimator.classes_)
    if hasattr(estimator, "family"):
        info["family"] = _class_name(estimator.family)

    return info


def _format_summary(info):
    """Format a summary dictionary as compact human-readable text."""
    fitted_text = "yes" if info["fitted"] else "no"
    parameters = info["parameters"]
    terms = info["terms"]

    lines = [
        "NAMpy Model Summary",
        f"Estimator: {info['estimator']}",
        f"Base model: {info['base_model']}",
        f"Fitted: {fitted_text}",
    ]
    if info.get("task") is not None:
        lines.append(f"Task: {info['task']}")
    if info.get("family") is not None:
        lines.append(f"Family: {info['family']}")
    if info.get("classes") is not None:
        lines.append(f"Classes: {info['classes']}")

    lines.extend(
        [
            f"Features: {info['n_features']} total "
            f"({len(info['numerical_features'])} numerical, "
            f"{len(info['categorical_features'])} categorical)",
            f"Terms: {terms['total']} total "
            f"({len(terms['numerical'])} numerical, "
            f"{len(terms['categorical'])} categorical, "
            f"{len(terms['interactions'])} interactions)",
            f"Parameters: {parameters['trainable']} trainable / "
            f"{parameters['total']} total",
        ]
    )

    diagnostics_info = info.get("diagnostics", {})
    if "spline" in diagnostics_info:
        spline = diagnostics_info["spline"]
        lines.append(
            "Spline: "
            f"{spline['n_knots']} knots, "
            f"learn_knots={spline['learn_knots']}, "
            f"smoothing={spline['smoothing']}"
        )
    if "group_norms" in diagnostics_info:
        lines.append(f"Sparse groups: {len(diagnostics_info['group_norms'])}")

    return "\n".join(lines)


def _summary(estimator, *, print_fn=print):
    """
    Print and return a structured summary for a NAMpy estimator or base model.

    Pass ``print_fn=None`` to suppress printing and only receive the dictionary.
    """
    info = _summary_dict(estimator)
    if print_fn is not None:
        print_fn(_format_summary(info))
    return info
