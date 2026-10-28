"""Seven-stage contracts for the two extended and six general family ports.

Every case uses frozen workflow inputs and a direct actual-mgcv prediction
capture. The established single-predictor pipeline fixture remains independent.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy import GAMLSS, GAMRegressor
from nampy.gam import GAM
from nampy.gam.families.bcg import CensoredBoxCoxGaussianFamily
from nampy.gam.families.cpois import CensoredPoissonFamily
from nampy.gam.families.gamlss.gevlss import GevlssFamily
from nampy.gam.families.gamlss.gumbls import GumblsFamily
from nampy.gam.families.gamlss.multinom import MultinomFamily
from nampy.gam.families.gamlss.shash import ShashFamily
from nampy.gam.families.gamlss.twlss import TwlssFamily
from nampy.gam.families.gamlss.ziplss import ZiplssFamily
from nampy.gam.fit.state import FitCoreSolution
from nampy.gam.smooths.univariate.cr import CubicSplineTerm
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = [pytest.mark.surface_regression]
ROOT = Path(__file__).parents[1] / 'reference_fixtures'
NAMES = ['cpois','bcg','gumbls','ziplss','multinom','gevlss','shash','twlss']

SOURCE=r'''
library(mgcv);library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);nd<-as.data.frame(p$newdata)
if('bound'%in%names(d))d$bound<-as.numeric(d$bound)
f<-switch(p$name,cpois=cpois(),bcg=bcg(theta=c(.7,.8)),gumbls=gumbls(b=-3),ziplss=ziplss(),
multinom=multinom(2),gevlss=gevlss(),shash=shash(),twlss=twlss())
if(p$name=='bcg')f$putTheta(c(.7,log(.8)))
form<-lapply(p$formula,as.formula);if(length(form)==1)form<-form[[1]]
if(p$name=='twlss')form<-list(y~s(x,bs='cr',k=6,sp=.7),~1,~s(x,bs='cr',k=5,sp=.9))
args<-list(formula=form,data=d,family=f,method='REML',optimizer=if(p$name=='twlss')'efs' else c('outer','newton'),
control=gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10)))
if(p$name!='twlss')args$sp<-p$sp
b<-do.call(gam,args)
pr<-predict(b,nd,type='response',se.fit=TRUE)
write_json(list(pred=pr$fit,se=pr$se.fit,coef=coef(b),Vp=b$Vp,edf=sum(b$edf),
eta=predict(b,nd,type='link'),converged=b$converged),a[2],digits=17,auto_unbox=TRUE)
'''


def _read(namespace):
    return json.loads((ROOT/namespace/'inputs.json').read_text())


def configuration(name):
    formulas=['y~s(x,bs="cr",k=6)']
    sp=[.7]
    if name=='cpois':
        data=_read('mgcv_cpois_workflows')
        formulas=['cbind(y,bound)~s(x,bs="cr",k=7)+offset(off)']
        family=CensoredPoissonFamily()
    elif name=='bcg':
        data=_read('mgcv_gevlss_workflows')
        family=CensoredBoxCoxGaussianFamily(theta=[.7,.8])
        # Source fixed constructor stores an erroneous third entry. Its own
        # putTheta method supplies the documented explicit valid fixed state.
        family.putTheta([.7,-0.2231435513142097])
    elif name in ('gumbls','ziplss'):
        data=_read(f'mgcv_{name}')['data']
        formulas=['y~s(x,bs="cr",k=5)+offset(o1)','~s(z,bs="cr",k=4)+offset(o2)']
        sp=[.4,.7]
        family=GumblsFamily(b=-3) if name=='gumbls' else ZiplssFamily()
    elif name=='multinom':
        data=_read('mgcv_multinom_workflows')['2']
        formulas+=['~s(x,bs="cr",k=6)']
        sp=[.7,.7]
        family=MultinomFamily(2)
    elif name=='gevlss':
        data=_read('mgcv_gevlss_workflows')
        formulas+=['~1','~1']
        family=GevlssFamily()
    elif name=='shash':
        data=_read('mgcv_shash')
        formulas+=['~s(x,bs="cr",k=5)','~1','~1']
        sp=[.7,.9]
        family=ShashFamily()
    else:
        data=_read('mgcv_twlss')
        formulas+=['~1','~s(x,bs="cr",k=5)']
        sp=[.7,.9]
        family=TwlssFamily()
    return data,formulas,sp,family


@dataclass
class FamilyPipeline:
    name: str
    data: pd.DataFrame
    newdata: pd.DataFrame
    formulas: list
    smoothing_params: list
    model: GAM
    expected: dict


def reference(payload):
    key=reference_key('remaining-eight-family-pipeline',{'payload':payload,'source':SOURCE})
    result=load_reference('mgcv_remaining_family_pipeline',key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source,inputs,output=[Path(directory)/name for name in ('reference.R','inputs.json','output.json')]
            source.write_text(SOURCE)
            inputs.write_text(json.dumps(payload))
            run=subprocess.run(['Rscript',str(source),str(inputs),str(output)],capture_output=True,text=True,timeout=120)
            assert run.returncode==0,run.stderr
            result=json.loads(output.read_text())
        save_reference('mgcv_remaining_family_pipeline',key,result)
    return result


@pytest.fixture(scope='module',params=NAMES)
def family_pipeline(request):
    name=request.param
    data,formulas,sp,family=configuration(name)
    frame=pd.DataFrame(data)
    if 'bound' in frame:
        frame['bound']=frame['bound'].astype(float)
    newdata=frame.drop(columns=['y','bound'],errors='ignore').iloc[:7].copy()
    newdata['x']=[-.8,-.5,-.2,0,.25,.5,.8]
    # Keep raw frozen JSON in the fixture key; converting infinities through
    # DataFrame changes representation even though their source meaning agrees.
    payload={'name':name,'data':data,'newdata':newdata.to_dict('list'),'formula':formulas,'sp':sp}
    expected=reference(payload)
    model=GAM(family=family,formula=formulas if len(formulas)>1 else formulas[0],
              smoothing_params=sp,optimize_smoothing=False,smoothing_method='fixed',
              smoothing_optimizer='efs' if name=='twlss' else 'outer_newton',
              control={'epsilon':1e-10,'newton':{'conv.tol':1e-10}}).fit(data=frame)
    return FamilyPipeline(name,frame,newdata,formulas,sp,model,expected)


def _compiled(case):
    return case.model.gam_result_.compiled_model


def _source_tolerance(case):
    # Retain the corresponding family-workflow tolerances instead of applying
    # the most permissive family's endpoint budget to every distribution.
    return {
        'gumbls': (3e-6, 3e-5), 'ziplss': (3e-6, 3e-5),
        'multinom': (4e-6, 4e-6), 'gevlss': (5e-6, 5e-6),
    }.get(case.name, (2e-7, 2e-7))


def test_remaining_stage_1_formula_and_specs(family_pipeline):
    case=family_pipeline
    specs=case.model.predictor_specs
    assert case.model.formula_.nlp==len(case.formulas)
    assert len(specs)==len(case.formulas)
    assert [item.name for item in specs]==[f'eta{k+1}' for k in range(len(specs))]
    smooths=[term.smooth_spec for spec in specs for term in spec.terms if term.kind=='smooth']
    assert len(smooths)==len(case.smoothing_params)
    assert all(term.bs=='cr' for term in smooths)
    assert all(term.k>=4 for term in smooths)
    assert 'x' in case.model.formula_used_columns_
    assert specs[0].has_intercept
    assert case.model.formula_.response_name.replace(' ','')==('cbind(y,bound)' if case.name=='cpois' else 'y')


def test_remaining_stage_2_runtime_basis_ownership(family_pipeline):
    case=family_pipeline
    smooths=[term for term in _compiled(case).compiled_terms if term.term_type=='smooth']
    assert smooths
    for term in smooths:
        runtime=term.predict_fn.__self__
        assert isinstance(runtime,CubicSplineTerm)
        assert runtime.basis_train.shape[0]==len(case.data)
        np.testing.assert_allclose(term.predict_fn(case.model.X_),runtime.basis_train,atol=1e-13,rtol=0)
        assert len(runtime.get_penalty_definitions())==1


def test_remaining_stage_3_constructed_basis_penalty_pair(family_pipeline):
    case=family_pipeline
    compiled=_compiled(case)
    assert len(compiled.compiled_penalties)==len(case.smoothing_params)
    for penalty in compiled.compiled_penalties:
        term=compiled.compiled_terms[penalty.term_index]
        width=term.coef_slice.stop-term.coef_slice.start
        assert penalty.coef_slice==term.coef_slice
        assert penalty.matrix.shape==(width,width)
        np.testing.assert_allclose(term.predict_matrix(case.model.X_),term.basis_train,atol=1e-12,rtol=0)
        np.testing.assert_allclose(penalty.matrix,term.penalty_specs[0].matrix,atol=1e-13,rtol=0)


def test_remaining_stage_4_predictor_blocks_and_smoothing_maps(family_pipeline):
    case=family_pipeline
    compiled=_compiled(case)
    assert len(compiled.predictors)==len(case.formulas)
    assert compiled.n_smoothing_params==len(case.smoothing_params)
    np.testing.assert_array_equal(case.model.smoothing_params,case.smoothing_params)
    assert [p.smoothing_index for p in compiled.compiled_penalties]==list(range(len(case.smoothing_params)))
    for predictor in compiled.predictors:
        basis=[np.asarray(term.basis_train) for term in predictor.compiled_terms]
        expected=np.column_stack(basis) if basis else np.empty((len(case.data),0))
        np.testing.assert_array_equal(predictor.design_matrix,expected)
    full_indices=compiled.predictor_full_indices
    assert len(full_indices)==len(case.formulas)
    assert sorted(np.concatenate(full_indices).tolist())==list(range(len(case.model.fit_result().coef_full)))


def test_remaining_stage_5_identifiability_and_prediction_pairing(family_pipeline):
    case=family_pipeline
    compiled=_compiled(case)
    assert len(compiled.side_condition_reports)==len(case.formulas)
    for predictor,report in zip(compiled.predictors,compiled.side_condition_reports,strict=True):
        assert report['predictor']==predictor.name
        np.testing.assert_allclose(predictor.build_new_matrix(case.model.X_),predictor.design_matrix,atol=1e-12,rtol=0)
        if predictor.compiled_terms:
            assert predictor.side_condition_Q is not None
            assert predictor.side_condition_Q.shape[1]==predictor.n_coef
    lpmatrix=case.model.predict(case.newdata,type='lpmatrix')
    assert lpmatrix.shape==(len(case.newdata),len(case.model.fit_result().coef_full))


def test_remaining_stage_6_solution_and_source_fit(family_pipeline):
    case=family_pipeline
    solution=case.model.gam_result_.fit_core_solution
    assert isinstance(solution,FitCoreSolution)
    assert solution.fit_state.X is not None
    result=case.model.fit_result()
    atol,rtol=_source_tolerance(case)
    # Coefficients and Vp depend on the chosen basis. Observable prediction
    # covariance kernels and likelihood criteria are compared independently in
    # test_recent_family_seven_stage_pipeline's established-family cases.
    assert np.isfinite(result.coef_full).all()
    assert np.isfinite(case.model.vcov()).all()
    np.testing.assert_allclose(result.edf_total,case.expected['edf'],atol=atol,rtol=rtol)
    assert result.coef_full.shape==(case.model.lpmatrix(case.data).shape[1],)
    assert solution.fit_result.eta.shape[0]==len(case.data)
    lpmatrix=case.model.lpmatrix(case.data)
    eta=[]
    for indices,spec in zip(_compiled(case).predictor_full_indices,case.model.predictor_specs,strict=True):
        value=lpmatrix[:,indices]@result.coef_full[indices]
        if spec.offset_name is not None:
            value+=case.data[spec.offset_name].to_numpy()
        eta.append(value)
    expected_eta=np.column_stack(eta)
    if case.name in ('cpois','bcg'):
        expected_eta=expected_eta[:,0]
    np.testing.assert_allclose(result.eta,expected_eta,atol=2e-10,rtol=0)


def test_remaining_stage_7_source_prediction_public_components_and_persistence(family_pipeline,tmp_path):
    case=family_pipeline
    prediction,se=case.model.predict(case.newdata,return_se=True)
    atol,rtol=_source_tolerance(case)
    np.testing.assert_allclose(prediction,case.expected['pred'],atol=atol,rtol=rtol)
    np.testing.assert_allclose(se,case.expected['se'],atol=atol,rtol=rtol)
    np.testing.assert_allclose(case.model.predict(case.newdata,type='link'),case.expected['eta'],atol=atol,rtol=rtol)
    general=case.name not in ('cpois','bcg')
    adapter_type=GAMLSS if general else GAMRegressor
    estimator=adapter_type(family=case.model.family,formula=case.formulas if general else case.formulas[0],
                          smoothing_params=case.smoothing_params,optimize_smoothing=False,
                          smoothing_optimizer='efs' if case.name=='twlss' else 'outer_newton',irls_tol=1e-10)
    estimator.fit(case.data)
    components=estimator.predict_components(case.newdata)
    components.validate_additive_reconstruction()
    np.testing.assert_allclose(estimator.predict_link(case.newdata),case.model.predict(case.newdata,type='link'),atol=atol,rtol=rtol)
    path=estimator.save_model(tmp_path/f'{case.name}.nampy')
    restored=adapter_type.load_model(path)
    np.testing.assert_array_equal(restored.predict(case.newdata),estimator.predict(case.newdata))
    restored.predict_components(case.newdata).validate_additive_reconstruction()
