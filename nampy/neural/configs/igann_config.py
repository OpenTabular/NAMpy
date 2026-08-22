"""Configuration for Interpretable Generalized Additive Neural Networks."""

from dataclasses import dataclass


@dataclass
class DefaultIGANNConfig:
    """Reference-oriented IGANN hyperparameters.

    ``n_estimators`` counts sequential ELM boosting stages. ``sparse`` enables
    the IGANN-Sparse best-subset pass and denotes the maximum number of atomic
    transformed features retained; it requires the optional ``abess`` package.
    """

    # Present for the shared TaskModule constructor; native IGANN fitting does
    # not use a gradient optimizer or learning-rate scheduler.
    lr: float = 0.1
    lr_patience: int = 10
    weight_decay: float = 0.0
    lr_factor: float = 0.1
    lr_schedule: str = "none"
    solver: str = "auto"

    n_hid: int = 10
    n_estimators: int = 5000
    boost_rate: float = 0.1
    init_reg: float = 1.0
    elm_scale: float = 1.0
    elm_alpha: float = 1.0
    activation: str = "elu"
    early_stopping: int = 50
    elm_random_state: int = 0
    sparse: int = 0
    device: str = "cpu"
    clip_predictions: float = 100.0


__all__ = ["DefaultIGANNConfig"]
