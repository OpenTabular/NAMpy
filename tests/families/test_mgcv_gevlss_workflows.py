"""GEV fit, uncertainty and residual parity against actual mgcv captures."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.gevlss import GevlssFamily
from tests.families.test_gevlss_family_contracts import reference

SOURCE=r'''
d<-data.frame(x=unlist(a$x),y=unlist(a$y));link<-unlist(a$link)
sp<-if(a$selected)NULL else .7
b<-gam(list(y~s(x,k=6,bs='cr'),~1,~1),data=d,family=gevlss(link),method='REML',sp=sp)
nd<-data.frame(x=c(-.9,-.6,-.2,.1,.5,.8));pr<-predict(b,nd,type='response',se.fit=TRUE)
ans<-list(version=as.character(packageVersion('mgcv')),coef=coef(b),Vp=b$Vp,edf=b$edf,sp=b$sp,
score=unname(b$gcv.ubre),eta=b$linear.predictors,fitted=b$fitted.values,pred=pr$fit,se=pr$se.fit,
residuals=lapply(c('deviance','pearson','response'),function(type)residuals(b,type=type)),deviance=b$deviance)
write_json(ans,args[2],digits=17,auto_unbox=TRUE)
'''


def data():
    path=Path(__file__).resolve().parents[1]/'reference_fixtures/mgcv_gevlss_workflows/inputs.json'
    return json.loads(path.read_text())


@pytest.mark.parametrize('location,selected',[('identity',False),('log',False),('identity',True),('log',True)])
def test_gevlss_fixed_selected_fit_prediction_and_persistence(location,selected):
    link=[location,'identity','logit']
    ref=reference(dict(link=link,selected=selected,**data()),SOURCE)
    model=GAM(family=GevlssFamily(link),formula=['y~s(x,k=6,bs="cr")','~1','~1'],
              optimize_smoothing=selected,smoothing_params=None if selected else [.7],smoothing_method='REML')
    model.fit(data=pd.DataFrame(data()))
    for key,value in [('coef',model.fit_result().coef_full),('Vp',model.vcov())]:
        np.testing.assert_allclose(value,ref[key],atol=5e-6,rtol=5e-6,err_msg=key)
    np.testing.assert_allclose(model.fit_result().edf_total,np.sum(ref['edf']),atol=5e-6,rtol=5e-6)
    new=pd.DataFrame({'x':[-.9,-.6,-.2,.1,.5,.8]})
    pred,se=model.predict(new,return_se=True)
    np.testing.assert_allclose(pred,ref['pred'],atol=5e-6,rtol=5e-6)
    np.testing.assert_allclose(se,ref['se'],atol=5e-6,rtol=5e-6)
    for rtype,expected in zip(('deviance','pearson','response'),ref['residuals'],strict=True):
        np.testing.assert_allclose(model.residuals(type=rtype),expected,atol=5e-6,rtol=5e-6)
    if selected:
        np.testing.assert_allclose(model.smoothing_score_,ref['score'],atol=5e-6,rtol=5e-6)
        np.testing.assert_allclose(model.smoothing_params,np.atleast_1d(ref['sp']),atol=2e-5,rtol=2e-5)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(new),pred)
