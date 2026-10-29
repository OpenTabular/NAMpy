"""Configuration for Interpretable Generalized Additive Neural Networks."""

from dataclasses import dataclass


@dataclass
class DefaultIGANNConfig:
    """Reference-oriented IGANN hyperparameters.

    ``n_estimators`` counts sequential ELM boosting stages. ``sparse`` enables
    the IGANN-Sparse best-subset pass and denotes the maximum number of atomic
    transformed features retained; it requires the optional ``abess`` package.
    """

    n_hid: int = 10
    n_estimators: int = 5000
    elm_scale: float = 1.0
    activation: str = "elu"
    elm_random_state: int = 0


__all__ = ["DefaultIGANNConfig"]
