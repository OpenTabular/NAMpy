"""Actual mgcv 1.9-4 multinom kernels, all derivative index patterns."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.gamlss.multinom import MultinomFamily
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

SOURCE = r'''
f<-multinom(a$K); X<-do.call(rbind,a$X);jj<-lapply(a$jj,function(z)as.integer(z)+1L)
attr(X,'lpi')<-jj; co<-a$coef; y<-a$y; w<-a$w
raw<-new.env(parent=environment(f$ll))
raw$gamlss.gH<-function(X,jj,l1,l2,i2,l3,i3,l4,i4,...)list(l1=l1,l2=l2,l3=l3,l4=l4)
environment(f$ll)<-raw
ans<-list(version=as.character(packageVersion('mgcv')),raw=f$ll(y,X,co,w,f,offset=a$offset,deriv=4,ncv=TRUE))
eta<-sapply(seq_along(jj),function(k)drop(X[,jj[[k]],drop=FALSE]%*%co[jj[[k]]])+a$offset[[k]])
if(a$K==1)eta<-matrix(eta,ncol=1)
ans$pred<-f$predict(f,eta=eta)[[1]]
ans$pred_se<-f$predict(f,X=X,beta=co,off=a$offset,se=TRUE,Vb=do.call(rbind,a$Vb))
obj<-list(y=y,fitted.values=eta,family=f);ans$residuals<-f$residuals(obj)
e<-new.env(parent=asNamespace('mgcv'));e$y<-y;e$nobs<-length(y);e$start<-NULL;e$x<-X;e$E<-do.call(rbind,a$E);e$family<-f
eval(f$initialize,e);ans$initial<-e$start
write_json(ans,args[2],digits=17,auto_unbox=TRUE)
'''


def reference(payload, source=SOURCE):
    key = reference_key("multinom-leaf", {"payload": payload, "source": source})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            script, inputs, output = [Path(directory) / name for name in ("ref.R", "inputs.json", "output.json")]
            script.write_text('args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH");if(nzchar(lib)).libPaths(c(lib,.libPaths()))\nsuppressPackageStartupMessages(library(mgcv));library(jsonlite)\na<-fromJSON(args[1],simplifyVector=FALSE)\na$K<-as.integer(a$K);a$coef<-unlist(a$coef);a$y<-unlist(a$y);a$w<-unlist(a$w);a$offset<-lapply(a$offset,unlist);a$X<-lapply(a$X,unlist);a$Vb<-lapply(a$Vb,unlist);a$E<-lapply(a$E,unlist)\n' + source)
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(["Rscript", str(script), str(inputs), str(output)], capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            save_reference("mgcv_multinom_family", key, json.loads(output.read_text()))
    return load_reference("mgcv_multinom_family", key)


def case(K=2):
    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, -0.6, 0.6])
    X = np.column_stack([column for _ in range(K) for column in (np.ones(12), x)])
    co = np.array([value for k in range(K) for value in (0.1 * (k + 1), -0.2 + 0.1 * k)])
    Vb = np.eye(2 * K) * 0.03 + np.full((2 * K, 2 * K), -0.001)
    return {"K": K, "X": X.tolist(), "coef": co.tolist(), "jj": [list(range(2*k, 2*k+2)) for k in range(K)],
            "y": [i % (K + 1) for i in range(12)], "w": [0.5, 1, 2] * 4,
            "offset": [[0.01 * (k+1)] * 12 for k in range(K)], "E": (np.eye(2*K)*0.2).tolist(), "Vb": Vb.tolist()}


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_multinom_source_complete_derivatives_prediction_and_initialization(K):
    payload = case(K)
    ref = reference(payload)
    f = MultinomFamily(K)
    X, coef = np.asarray(payload['X']), np.asarray(payload['coef'])
    jj = [np.asarray(cols) for cols in payload['jj']]
    actual = f.ll(payload['y'], X, jj, coef, payload['w'], offset=payload['offset'], deriv=4,
                  d1b=np.zeros((len(coef), 1)), d2b=np.zeros((len(coef), 1)), fh=np.eye(len(coef)), D=np.ones(len(coef)), ncv=True)
    assert ref['version'] == '1.9.4'
    for key in ('l', 'l1', 'l2', 'l3', 'l4'):
        expected = np.asarray(ref['raw'][key])
        np.testing.assert_allclose(actual[key], expected.reshape(np.asarray(actual[key]).shape), atol=4e-13, rtol=4e-13, err_msg=key)
    pred, se = f.predict(X=X, jj=jj, coef=coef, offset=payload['offset'], se=True, Vb=payload['Vb'])
    np.testing.assert_allclose(pred, ref['pred'], atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(se, ref['pred_se']['se.fit'], atol=2e-15, rtol=2e-15)
    eta = f.predict_fitted(X, jj, coef, offset=payload['offset'])
    np.testing.assert_allclose(f.residuals(payload['y'], eta), ref['residuals'], atol=3e-15)
    np.testing.assert_allclose(f.initialize(payload['y'], X, jj, E=np.asarray(payload['E'])), ref['initial'], atol=2e-14)
    # Normalize public density, preserving the source baseline offset only in ll.
    normalized = f.logpdf_from_parameters(payload['y'], pred)
    ordinary = f.ll(payload['y'], X, jj, coef, payload['w'], offset=payload['offset'])
    np.testing.assert_allclose(ordinary['l'], np.sum(normalized) + np.sum(np.asarray(payload['y']) == 0), atol=1e-13)
    np.testing.assert_array_equal(f.point_prediction_from_parameters(pred), np.argmax(pred, axis=1))


@pytest.mark.parametrize('K', [0, -1, 1.5, np.nan])
def test_multinom_rejects_invalid_category_count(K):
    with pytest.raises(ValueError):
        MultinomFamily(K)


def test_multinom_rounding_validation_and_probability_jacobian():
    f = MultinomFamily(2)
    for y in ([0, -2], [3], [np.nan]):
        with pytest.raises(ValueError):
            f.validate_y(y)
    eta = np.array([[0.2, -0.3], [1.2, 0.8]])
    probability = f.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(f.distribution_parameter_jacobian(eta).sum(axis=1), 0, atol=1e-16)
    assert probability.shape == (2, 3)
    with pytest.raises(ValueError, match='deviance'):
        f.residuals([0, 1], eta, rtype='pearson')
