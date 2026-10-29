"""Complete trajectories, native likelihood reuse and subject-level data roles."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    NAMSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.survival import (
    LongitudinalSurvivalTarget,
    PiecewiseExponentialDesign,
    SurvivalTarget,
    SurvivalTrajectory,
    predict_trajectory,
)


@pytest.fixture
def longitudinal():
    root = Path(__file__).parents[1]/"reference_fixtures"
    frame = pd.read_csv(root/"survival_longitudinal_input.csv")
    y = LongitudinalSurvivalTarget(frame.start, frame.stop, frame.subject, frame.event)
    return frame[["x"]], y, frame.weight.to_numpy(), frame.offset.to_numpy()


def test_record_contract_contiguity_terminal_events_and_whole_subject_selection():
    y = LongitudinalSurvivalTarget([0,.5,0], [.5,1,2], ["a","a","b"], [0,1,0])
    assert y.n_subjects == 2
    assert_array_equal(y.normalization, [.5,.5,1])
    assert_array_equal(y.subset(["a"]).stop, [.5,1])
    assert not y.stop.flags.writeable
    with pytest.raises(ValueError, match="gaps/overlaps"):
        SurvivalTrajectory([0,.6], [.5,1], ["a","a"])
    with pytest.raises(ValueError, match="contiguous ordered block"):
        SurvivalTrajectory([0,0,1], [1,1,2], ["a","b","a"])
    with pytest.raises(NotImplementedError, match="terminal event"):
        LongitudinalSurvivalTarget([0,.5], [.5,1], ["a","a"], [1,0])


def test_statistical_longitudinal_fit_and_trajectory_match_independent_r(longitudinal, tmp_path):
    X,y,weight,offset = longitudinal
    model = SurvivalGAM(cuts=[0,1,3], formula="~x", optimize_smoothing=False,
                        irls_tol=1e-10).fit(X,y,sample_weight=weight,offset=offset)
    assert model.n_subjects_ == 8
    root = Path(__file__).parents[1]/"reference_fixtures"
    expected = pd.read_csv(root/"survival_longitudinal.csv")
    assert_allclose(model.survival_log_likelihood_, expected.loglik.iloc[0], atol=2e-9)
    assert_allclose(model.score(X,y,weight,offset=offset), expected.score.iloc[0], atol=2e-9)
    path = SurvivalTrajectory([0,.7,0,1.4], [.7,3,1.4,3], ["A","A","B","B"])
    features = pd.DataFrame({"x":[-.5,.8,-.2,.4]})
    prediction = predict_trajectory(model,features,path,offset=np.zeros(4))
    assert_array_equal(prediction.cuts,[0,.7,1,1.4,3])
    assert_allclose(prediction.hazard_rates, expected[["rate1","rate2","rate3","rate4"]].iloc[[0,4]],atol=2e-9)
    assert_allclose(prediction.survival([.5,1,2,3]).ravel(), expected.survival,atol=2e-9)
    restored = SurvivalGAM.load_model(model.save_model(tmp_path/"paths.nampy"))
    assert_array_equal(predict_trajectory(restored,features,path,offset=np.zeros(4)).hazard_rates,prediction.hazard_rates)
    changed = weight.copy()
    changed[0] += 1
    with pytest.raises(ValueError,match="constant within"):
        model.fit(X,y,sample_weight=changed)
    assert_array_equal(predict_trajectory(model,features,path,offset=np.zeros(4)).hazard_rates,prediction.hazard_rates)


def test_splitting_constant_records_preserves_statistical_likelihood_and_endpoint():
    X = pd.DataFrame({"x":np.tile([-.5,.5],12)})
    time = np.linspace(.3,2.9,24)
    event = np.resize([1,0,1],24)
    static = SurvivalTarget(time,event)
    target = LongitudinalSurvivalTarget(np.ravel(np.column_stack([np.zeros(24),time/2])),
        np.ravel(np.column_stack([time/2,time])), np.repeat(np.arange(24),2),
        np.ravel(np.column_stack([np.zeros(24),event])))
    a = SurvivalGAM(cuts=[0,1,3],formula="~x",optimize_smoothing=False).fit(X,static)
    b = SurvivalGAM(cuts=[0,1,3],formula="~x",optimize_smoothing=False).fit(X.iloc[np.repeat(np.arange(24),2)].reset_index(drop=True),target)
    assert_allclose(a.predict_link(X),b.predict_link(X),atol=2e-10)
    assert_allclose(a.score(X,static),b.score(X.iloc[np.repeat(np.arange(24),2)].reset_index(drop=True),target),atol=2e-10)


def test_longitudinal_gradient_loss_divides_by_subject_weight(longitudinal):
    import torch

    from nampy.neural.survival import LongitudinalPiecewiseObjective
    X,y,weight,_ = longitudinal
    design = PiecewiseExponentialDesign(y,[0,1,3])
    packed = np.column_stack([design.exposure,design.events,y.normalization])
    objective = LongitudinalPiecewiseObjective(2)
    objective.configure_data(SimpleNamespace(y_train=packed,y_val=packed,
        sample_weight_train=weight,sample_weight_val=weight))
    eta = torch.zeros((len(y),2),dtype=torch.float64,requires_grad=True)
    loss = objective.compute_stage_loss(eta,torch.tensor(packed),torch.tensor(weight),stage="train")
    expected = -weight@design.log_likelihood(np.zeros((len(y),2)))/(weight@y.normalization)
    assert_allclose(loss.detach(),expected,atol=2e-15)
    loss.backward()
    assert torch.isfinite(eta.grad).all()


@pytest.mark.parametrize("cls",[LinRegSurvival,NAMSurvival])
def test_neural_whole_subject_validation_paths_persistence_and_rejections(cls,longitudinal,tmp_path):
    X,y,weight,offset = longitudinal
    extra = {}
    if cls is NAMSurvival:
        from nampy.neural.configs.nam_config import DefaultNAMConfig
        extra["architecture_config"] = DefaultNAMConfig(layer_sizes=[4],dropout=0)
    model = cls(**extra,objective=SurvivalObjectiveConfig(cuts=[0,1,3]),training=GradientTrainingConfig(
        max_epochs=1,batch_size=64,random_state=4,checkpoint_path=str(tmp_path),
        trainer_kwargs={"logger":False,"enable_progress_bar":False,"enable_model_summary":False}))
    with pytest.raises(NotImplementedError,match="whole-subject"):
        model.fit(X,y)
    # Three validation subjects, with distinct original IDs and both intervals.
    train = y.subset(list("abcde"))
    validation = y.subset(list("fgh"))
    train_rows = np.isin(y.subject_id,train.subject_ids)
    val_rows = ~train_rows
    model.fit(X.loc[train_rows],train,X_val=X.loc[val_rows],y_val=validation,
        sample_weight=weight[train_rows],sample_weight_val=weight[val_rows],
        offset=offset[train_rows],offset_val=offset[val_rows])
    assert model.n_subjects_ == 5
    path = SurvivalTrajectory([0,.7],[.7,3],["new","new"])
    features = pd.DataFrame({"x":[-.3,.8]})
    pred = predict_trajectory(model,features,path,offset=np.zeros(2))
    eta = model.predict_link(features,offset=np.zeros(2))
    assert_allclose(pred.hazard_rates, np.exp(eta[[0,1,1],[0,0,1]])[None,:])
    restored = cls.load_model(model.save_model(tmp_path/"longitudinal.nampy"))
    assert_array_equal(predict_trajectory(restored,features,path,offset=np.zeros(2)).hazard_rates,pred.hazard_rates)
    with pytest.raises(ValueError,match="overlap"):
        model.fit(X.loc[train_rows],train,X_val=X.loc[train_rows],y_val=train)


def test_explicit_unsupported_longitudinal_routes(longitudinal):
    X,y,_,_ = longitudinal
    with pytest.raises(NotImplementedError,match="parameter paths"):
        SurvivalGAM(likelihood="weibull").fit(X,y)
    model = SurvivalGAM(cuts=[0,1,3],formula="~x",optimize_smoothing=False).fit(X,y)
    with pytest.raises(NotImplementedError,match="from zero"):
        predict_trajectory(model,X.iloc[:1],SurvivalTrajectory([.2],[1.],["a"]))
    with pytest.raises(NotImplementedError,match="beyond"):
        predict_trajectory(model,X.iloc[:1],SurvivalTrajectory([0],[4.],["a"]))
