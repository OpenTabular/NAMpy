"""Parameter-diagonal neural Fisher information without dense parameter matrices.

Likelihood algebra lives in fisher_likelihood; common normalization and sampling
uncertainty aggregation live in diagnostics.fisher. This module owns only Torch
execution, parameter grouping and bounded preprocessing.
"""

import random
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..diagnostics.fisher import aggregate_fisher
from .fisher_likelihood import FisherLikelihood
from .offsets import predictor_offset
from .prediction import _prediction_batches


@dataclass(frozen=True)
class ParameterGroups:
    names: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    grouping: str

    @classmethod
    def from_module(cls, module, group_by):
        named = tuple(
            (name, p)
            for name, p in module.named_parameters()
            if p.requires_grad and p.numel()
        )
        if not named:
            raise ValueError("Fisher diagnostics require fitted parameter coordinates.")
        slices, cursor = {}, 0
        for name, parameter in named:
            slices[name] = np.arange(cursor, cursor + parameter.numel())
            cursor += parameter.numel()
        if group_by == "tensor":
            mapping, actual = {name: [name] for name, _ in named}, "tensor"
        elif group_by == "layer":
            mapping, actual = {}, "layer"
            for name, _ in named:
                owner = (
                    name.rsplit(".", 1)[0]
                    if name.rsplit(".", 1)[-1] in {"weight", "bias"}
                    else name
                )
                mapping.setdefault(owner, []).append(name)
        else:
            hook = getattr(
                getattr(module, "model", module), "fisher_parameter_groups", None
            )
            if hook is None:
                warnings.warn(
                    "No term Fisher groups are declared; using layer groups.",
                    UserWarning,
                    stacklevel=2,
                )
                return cls.from_module(module, "layer")
            declared = hook()
            if not isinstance(declared, Mapping):
                raise TypeError("fisher_parameter_groups must return a mapping.")
            mapping, actual, assigned = {}, "term", set()
            for group, names in declared.items():
                resolved = []
                for name in names:
                    candidate = name if name in slices else "model." + name
                    if candidate not in slices or candidate in assigned:
                        raise ValueError(
                            f"Unknown or overlapping Fisher coordinate {name!r}."
                        )
                    assigned.add(candidate)
                    resolved.append(candidate)
                if resolved:
                    mapping[str(group)] = resolved
            remaining = [name for name, _ in named if name not in assigned]
            if remaining:
                mapping.setdefault("__global__", []).extend(remaining)
        return cls(
            tuple(mapping),
            tuple(
                np.concatenate([slices[n] for n in names]) for names in mapping.values()
            ),
            actual,
        )


_NONSMOOTH = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.Threshold,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.ELU,
    nn.CELU,
    nn.SELU,
    nn.Hardshrink,
    nn.Softshrink,
    nn.Softsign,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
)


def validate_fisher_model(module, config):
    """Reject unsupported sampling experiments before fitting/collection."""
    likelihood = FisherLikelihood.from_objective(module.objective, config)
    if config.estimator == "negative_hessian":
        architecture = getattr(module, "model", module)
        if type(architecture).__dict__.get("supports_fisher_hessian") is not True:
            raise NotImplementedError(
                "This architecture does not declare smooth Hessian Fisher support."
            )
        if any(isinstance(child, _NONSMOOTH) for child in architecture.modules()):
            raise NotImplementedError(
                "Negative-Hessian Fisher requires a smooth model; nonsmooth activations/routing are unsupported."
            )
    return likelihood


@contextmanager
def _diagnostic_state(module, seed, include_frozen):
    """Restore observable Torch/RNG state even when collection fails."""
    parameters = tuple(module.parameters())
    modes = [(child, child.training) for child in module.modules()]
    flags = [p.requires_grad for p in parameters]
    values = [p.detach().clone() for p in parameters]
    gradients = [
        (p.grad, None if p.grad is None else p.grad.detach().clone())
        for p in parameters
    ]
    buffers = [(buffer, buffer.detach().clone()) for buffer in module.buffers()]
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        if include_frozen:
            for p in parameters:
                if p.is_floating_point():
                    p.requires_grad_(True)
        if any(
            p.requires_grad and p.dtype not in {torch.float32, torch.float64}
            for p in parameters
        ):
            raise NotImplementedError(
                "Fisher diagnostics require float32 or float64 parameter coordinates."
            )
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        module.eval()
        yield
        if any(
            not torch.equal(p.detach(), value)
            for p, value in zip(parameters, values, strict=True)
        ):
            raise RuntimeError(
                "Diagnostic forward mutated model parameters; initialize the architecture on training data first."
            )
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        with torch.no_grad():
            for p, value, flag, (reference, gradient) in zip(
                parameters, values, flags, gradients, strict=True
            ):
                if not torch.equal(p.detach(), value):
                    if p.shape != value.shape:
                        p.data = value
                    else:
                        p.copy_(value)
                p.requires_grad_(flag)
                p.grad = reference
                if reference is not None:
                    assert gradient is not None
                    if not torch.equal(reference, gradient):
                        reference.copy_(gradient)
            for buffer, value in buffers:
                if not torch.equal(buffer, value):
                    buffer.copy_(value)
        for child, training in modes:
            child.training = training


def _flat_grad(value, parameters, *, create_graph=False):
    if value.requires_grad:
        gradients = torch.autograd.grad(
            value,
            parameters,
            allow_unused=True,
            create_graph=create_graph,
            retain_graph=True,
        )
    else:
        gradients = (None,) * len(parameters)
    return torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def _exact_diagonal(likelihood, raw, parameters):
    if likelihood.kind == "regression":
        assert likelihood.regression_scale is not None
        diagonal = torch.zeros(
            sum(p.numel() for p in parameters), device=raw.device, dtype=torch.float64
        )
        for mean in raw.reshape(-1):
            derivative = _flat_grad(mean, parameters).detach().double()
            diagonal += derivative.square() / likelihood.regression_scale**2
        return diagonal.cpu().numpy()
    if likelihood.kind == "binary":
        derivative = _flat_grad(raw.sum(), parameters).detach().double()
        logit = raw.detach().double().reshape(())
        variance = torch.sigmoid(logit) * torch.sigmoid(-logit)
        return (variance * derivative.square()).cpu().numpy()
    if likelihood.kind in {"categorical", "multiclass"}:
        distribution = likelihood.distribution(raw)
        class_diagonal: torch.Tensor | None = None
        for target, probability in enumerate(distribution.probs[0]):
            log_probability = distribution.log_prob(
                torch.tensor([target], device=raw.device)
            ).sum()
            score = _flat_grad(log_probability, parameters).detach().double()
            contribution = probability.detach().double() * score.square()
            class_diagonal = (
                contribution
                if class_diagonal is None
                else class_diagonal + contribution
            )
        assert class_diagonal is not None
        return class_diagonal.cpu().numpy()
    coordinates, information = likelihood.coordinates_and_information(raw)
    jacobian = torch.stack(
        [_flat_grad(value, parameters).detach().double() for value in coordinates]
    )
    information = information.detach().double()
    # The first axis is the small response-coordinate axis, never parameters.
    factor = torch.linalg.cholesky(information)
    return (factor.T @ jacobian).square().sum(dim=0).cpu().numpy()


def _sample_diagonal(likelihood, raw, parameters, target, estimator):
    log_probability = likelihood.log_prob(raw, target).sum()
    if not torch.isfinite(log_probability):
        raise FloatingPointError("Fisher log probability is not finite.")
    gradient = _flat_grad(
        log_probability, parameters, create_graph=estimator == "negative_hessian"
    )
    if estimator == "score_outer_product":
        return gradient.detach().double().square().cpu().numpy()
    # Each reverse pass extracts one diagonal element. There is no PxP tensor.
    diagonal = [
        -_flat_grad(value, parameters)[i].detach().double()
        for i, value in enumerate(gradient)
    ]
    return torch.stack(diagonal).cpu().numpy()


def estimate_fisher_information(
    module,
    preprocessing,
    X,
    config,
    *,
    offset=None,
    context=None,
    random_state=None,
    split=None,
    population_size=None,
    include_frozen=False,
    batch_size=None,
):
    """Evaluate all supplied reference rows, with no observed targets.

    Offsets are in raw predictor coordinates and stay fixed under parameter
    differentiation. The caller owns schema and required-offset validation.
    """
    if context:
        raise NotImplementedError(
            "These neural Fisher likelihoods do not accept observation context."
        )
    likelihood = validate_fisher_model(module, config)
    offsets = predictor_offset(offset, len(X), module.objective.output_dim)
    seed = random_state if random_state is not None else config.random_state
    seed = (
        int(np.random.SeedSequence().generate_state(1)[0])
        if seed is None
        else int(seed)
    )
    with (
        _diagnostic_state(module, seed, include_frozen),
        torch.inference_mode(False),
        torch.enable_grad(),
    ):
        named = tuple(
            (name, p)
            for name, p in module.named_parameters()
            if p.requires_grad and p.numel()
        )
        if not named:
            raise ValueError("Fisher diagnostics require fitted parameter coordinates.")
        names, parameters = zip(*named, strict=True)
        groups = ParameterGroups.from_module(module, config.group_by)
        shapes = tuple(tuple(p.shape) for p in parameters)
        stops = np.cumsum([0] + [p.numel() for p in parameters])
        slices = tuple(zip(stops[:-1].tolist(), stops[1:].tolist(), strict=True))
        device = parameters[0].device

        def rows():
            start = 0
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", enabled=False
            ):
                for cat, num, count in _prediction_batches(
                    preprocessing, X, batch_size, device
                ):
                    for row in range(count):
                        result = module(
                            num_features={k: v[row : row + 1] for k, v in num.items()},
                            cat_features={k: v[row : row + 1] for k, v in cat.items()},
                        )
                        raw = result["output"]
                        if offsets is not None:
                            raw = raw + torch.as_tensor(
                                offsets[start + row : start + row + 1],
                                device=raw.device,
                                dtype=raw.dtype,
                            )
                        if not torch.isfinite(raw).all():
                            raise FloatingPointError(
                                "Fisher raw predictors must be finite."
                            )
                        if config.estimator == "exact":
                            yield (
                                _exact_diagonal(likelihood, raw, parameters),
                                None,
                                None,
                            )
                            continue
                        targets = likelihood.distribution(raw.detach()).sample(
                            (config.n_draws,)
                        )
                        mean = np.zeros(stops[-1])
                        m2 = np.zeros_like(mean)
                        group_mean = np.zeros(len(groups.names))
                        group_m2 = np.zeros_like(group_mean)
                        for draw in range(config.n_draws):
                            value = _sample_diagonal(
                                likelihood,
                                raw,
                                parameters,
                                targets[draw],
                                config.estimator,
                            )
                            if not np.isfinite(value).all():
                                raise FloatingPointError(
                                    "Fisher sample derivatives must be finite."
                                )
                            trace = np.array(
                                [value[index].sum() for index in groups.indices]
                            )
                            delta, group_delta = value - mean, trace - group_mean
                            mean += delta / (draw + 1)
                            group_mean += group_delta / (draw + 1)
                            m2 += delta * (value - mean)
                            group_m2 += group_delta * (trace - group_mean)
                        yield (
                            mean,
                            m2 / (config.n_draws - 1),
                            group_m2 / (config.n_draws - 1),
                        )
                    start += count

        return aggregate_fisher(
            rows(),
            config=config,
            parameter_names=names,
            parameter_shapes=shapes,
            parameter_slices=slices,
            groups=tuple(zip(groups.names, groups.indices, strict=True)),
            backend="neural",
            estimator=config.estimator,
            n_inputs=len(X),
            split=split,
            grouping=groups.grouping,
            population_size=population_size,
            return_diagonal=config.record_diagonal != "never",
            held_fixed=likelihood.held_fixed,
        )
