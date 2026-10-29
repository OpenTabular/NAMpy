"""Transient fit-time Fisher probes and scheduling, independent of optimization."""

from dataclasses import replace

import lightning as pl
import numpy as np

from ...inspection.fisher import FisherSnapshot, normalize_fisher_diagnostics
from ..configuration import GradientTrainingConfig
from ..fisher import estimate_fisher_information, validate_fisher_model
from ..offsets import predictor_offset


class FisherCollector(pl.Callback):
    """Hold a fixed input probe only while one transactional fit is running."""

    def __init__(self, estimator, training, *, include_frozen=False):
        self.estimator = estimator
        self.config = normalize_fisher_diagnostics(estimator.fisher_diagnostics)
        self.include_frozen = include_frozen
        self.history = []
        self.preprocessing = estimator.data_module
        data = estimator.data_module
        config = self.config
        if (
            not isinstance(training, GradientTrainingConfig)
            and config.every_n_epochs is not None
        ):
            raise ValueError("Periodic Fisher diagnostics require gradient training.")
        validate_fisher_model(estimator.model, config)
        if config.include_initial and not getattr(
            estimator.model.model, "supports_initial_fisher", True
        ):
            raise ValueError(
                "Initial Fisher diagnostics require initialized parameter coordinates; "
                "set include_initial=False for this training route."
            )
        split = "val" if config.split == "validation" else "train"
        X = getattr(data, f"X_{split}")
        if X is None or not len(X):
            raise ValueError(
                f"No rows are available for Fisher split {config.split!r}."
            )
        self.seed = (
            config.random_state
            if config.random_state is not None
            else int(
                np.random.SeedSequence(
                    [int(training.random_state), 0xF15E2]
                ).generate_state(1)[0]
            )
        )
        indices = np.sort(
            np.random.default_rng(self.seed).choice(
                len(X), min(config.n_inputs, len(X)), replace=False
            )
        )
        self.indices = tuple(int(index) for index in indices)
        self.X = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        offset = getattr(data, f"offset_{split}")
        self.offset = (
            None
            if offset is None
            else predictor_offset(offset, len(X), estimator.model.objective.output_dim)[
                indices
            ]
        )
        if config.input_sampling == "without_replacement":
            if config.population_size is not None and config.population_size != len(X):
                raise ValueError(
                    "Fit-time Fisher population_size must match the selected split."
                )
            self.config = replace(config, population_size=len(X))

    def collect(self, stage, *, epoch=None, optimizer_step=0):
        config = self.config
        if config.record_diagonal != "all" and stage != "final":
            config = replace(config, record_diagonal="never")
        result = estimate_fisher_information(
            self.estimator.model,
            self.preprocessing,
            self.X,
            config,
            offset=self.offset,
            random_state=self.seed,
            split=self.config.split,
            include_frozen=self.include_frozen,
        )
        self.history.append(
            FisherSnapshot(
                stage=stage,
                epoch=epoch,
                optimizer_step=optimizer_step,
                groups=result.groups,
                diagonal=result.diagonal,
            )
        )
        return result

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.config.every_n_epochs == 0:
            self.collect("epoch", epoch=epoch, optimizer_step=int(trainer.global_step))

    def finish(self):
        estimator = self.estimator
        if self.config.include_final:
            epoch = getattr(estimator, "n_epochs_", None)
            if (
                self.history
                and self.history[-1].stage == "epoch"
                and self.history[-1].epoch == epoch
            ):
                self.history.pop()
            estimator.fisher_information_ = self.collect(
                "final",
                epoch=epoch,
                optimizer_step=getattr(estimator, "n_optimizer_steps_", 0),
            )
        estimator.fisher_history_ = tuple(self.history)

    def release(self):
        self.estimator = None
        self.preprocessing = None
        self.X = None
        self.offset = None
        self.history.clear()


def prepare_fisher(estimator, training, *, include_frozen=False):
    """Skip all diagnostic preparation when disabled."""
    config = normalize_fisher_diagnostics(estimator.fisher_diagnostics)
    estimator.fisher_config_ = config
    estimator.fisher_history_ = ()
    estimator.fisher_information_ = None
    estimator.fisher_seed_ = None
    estimator.fisher_probe_indices_ = ()
    if config is None:
        return None
    collector = FisherCollector(estimator, training, include_frozen=include_frozen)
    estimator.fisher_config_ = collector.config
    estimator.fisher_seed_ = collector.seed
    estimator.fisher_probe_indices_ = collector.indices
    try:
        if config.include_initial:
            collector.collect("initial")
    except Exception:
        collector.release()
        raise
    return collector
