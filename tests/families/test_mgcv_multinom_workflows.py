"""Fixed and selected multinomial fits against committed actual-R captures."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.multinom import MultinomFamily
from tests.families.test_multinom_family_contracts import reference

SOURCE = r'''
d<-data.frame(x=unlist(a$x),y=a$y)
fml<-c(list(y~s(x,k=6,bs='cr')),rep(list(~s(x,k=6,bs='cr')),a$K-1))
sp<-if(a$selected)NULL else rep(.7,a$K)
b<-gam(fml,data=d,family=multinom(a$K),method='REML',sp=sp)
nd<-data.frame(x=c(-.9,-.6,-.2,.1,.5,.8))
pr<-predict(b,nd,type='response',se.fit=TRUE)
ans<-list(version=as.character(packageVersion('mgcv')),coef=coef(b),Vp=b$Vp,edf=b$edf,sp=b$sp,
score=unname(b$gcv.ubre),eta=b$linear.predictors,fitted=b$fitted.values,
pred=pr$fit,se=pr$se.fit,residuals=residuals(b),null=b$null.deviance,deviance=b$deviance)
write_json(ans,args[2],digits=17,auto_unbox=TRUE)
'''


def data(K):
    path = Path(__file__).resolve().parents[1] / 'reference_fixtures/mgcv_multinom_workflows/inputs.json'
    return json.loads(path.read_text())[str(K)]


@pytest.mark.parametrize('K,selected', [(1, False), (2, False), (3, False), (2, True)])
def test_multinom_fits_predicts_and_persists_against_mgcv(K, selected):
    payload = dict(K=K, selected=selected, **data(K))
    ref = reference(payload, SOURCE)
    frame = pd.DataFrame(data(K))
    formula = ['y ~ s(x,k=6,bs="cr")'] + ['~ s(x,k=6,bs="cr")'] * (K-1)
    model = GAM(formula=formula, family=MultinomFamily(K), smoothing_method='REML',
                optimize_smoothing=selected, smoothing_params=None if selected else [0.7]*K)
    model.fit(data=frame)
    np.testing.assert_allclose(model.fit_result().edf_total, np.sum(ref['edf']), atol=4e-6, rtol=4e-6)
    for field, actual in [('coef',model.fit_result().coef_full), ('Vp',model.vcov())] :
        np.testing.assert_allclose(actual, ref[field], atol=4e-6, rtol=4e-6, err_msg=field)
    new = pd.DataFrame({'x':[-.9,-.6,-.2,.1,.5,.8]})
    pred, se = model.predict(new, type='response', return_se=True)
    np.testing.assert_allclose(pred, ref['pred'], atol=4e-6, rtol=4e-6)
    np.testing.assert_allclose(se, ref['se'], atol=4e-6, rtol=4e-6)
    if selected:
        np.testing.assert_allclose(model.smoothing_score_, ref['score'], atol=4e-6, rtol=4e-6)
        np.testing.assert_allclose(model.smoothing_params, np.atleast_1d(ref['sp']), atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(model.family.null_deviance(frame.y, None, None), ref['null'], atol=2e-12)
    np.testing.assert_allclose(model.residuals(), ref['residuals'], atol=4e-6, rtol=4e-6)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(new), pred, atol=0, rtol=0)
