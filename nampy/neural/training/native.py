"""Architecture-native and exact fixed-design execution."""

from ..linear_solver import solve_fixed_linear_regression
from .common import _training_tensor


def fit_native(estimator, plan, train_num, train_cat, val_num, val_cat, training):
    estimator.native_training_info_ = dict(
        estimator.model.model.fit_native(
            train_num_features=train_num,
            train_cat_features=train_cat,
            train_targets=_training_tensor(estimator.data_module.y_train),
            val_num_features=val_num,
            val_cat_features=val_cat,
            val_targets=_training_tensor(estimator.data_module.y_val),
            objective=plan.objective,
            train_offset=_training_tensor(estimator.data_module.offset_train),
            val_offset=_training_tensor(estimator.data_module.offset_val),
            train_sample_weight=_training_tensor(
                estimator.data_module.sample_weight_train
            ),
            val_sample_weight=_training_tensor(estimator.data_module.sample_weight_val),
            random_state=training.random_state,
        )
    )
    estimator.best_model_path_ = None
    estimator.averaged_checkpoints_ = 0
    estimator.linear_solver_info_ = None
    estimator._record_model_metadata()
    return


def fit_fixed_design(estimator, train_num, train_cat, training):

    estimator.linear_solver_info_ = solve_fixed_linear_regression(
        estimator.model.model,
        num_features=train_num,
        cat_features=train_cat,
        targets=estimator.data_module.y_train,
        training=training,
        sample_weight=estimator.data_module.sample_weight_train,
        offset=estimator.data_module.offset_train,
    )
    estimator.best_model_path_ = None
    estimator.averaged_checkpoints_ = 0
    estimator._record_model_metadata()
    return
