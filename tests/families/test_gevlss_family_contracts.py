"""Source GEV analytic derivatives and initialization, mgcv 1.9-4."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.gamlss.gevlss import GevlssFamily
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

SOURCE = r'''
f<-gevlss(link=unlist(a$link));X<-do.call(rbind,lapply(a$X,unlist));jj<-lapply(a$jj,function(x)as.integer(x)+1L)
attr(X,'lpi')<-jj;co<-unlist(a$coef);y<-unlist(a$y);w<-unlist(a$w);off<-lapply(a$offset,unlist)
raw<-new.env(parent=environment(f$ll));raw$gamlss.gH<-function(X,jj,l1,l2,i2,l3,i3,l4,i4,...)list(l1=l1,l2=l2,l3=l3,l4=l4)
environment(f$ll)<-raw
eta<-sapply(seq_along(jj),function(k)drop(X[,jj[[k]],drop=FALSE]%*%co[jj[[k]]])+off[[k]])
fitted<-sapply(1:3,function(k)f$linfo[[k]]$linkinv(eta[,k]))
obj<-list(y=y,fitted=fitted,family=f)
ans<-list(version=as.character(packageVersion('mgcv')),raw=f$ll(y,X,co,w,f,off,deriv=4,ncv=TRUE),fitted=fitted,
residuals=lapply(c('deviance','pearson','response'),function(type)f$residuals(obj,type=type)))
e<-new.env(parent=asNamespace('mgcv'));e$y<-y;e$nobs<-length(y);e$start<-NULL;e$x<-X;e$E<-do.call(rbind,lapply(a$E,unlist));e$family<-f;e$weights<-w;e$offset<-off
eval(f$initialize,e);ans$initial<-e$start
ans$links<-lapply(f$linfo,function(info){mu<-info$linkinv(c(-40,-2,0,1,40));list(mu=mu,mu_eta=info$mu.eta(c(-40,-2,0,1,40)),d2=info$d2link(mu),d3=info$d3link(mu),d4=info$d4link(mu))})
write_json(ans,args[2],digits=17,auto_unbox=TRUE,na='string')
'''


def reference(payload, source=SOURCE):
    key = reference_key('gevlss-source', {'payload':payload, 'source':source})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            script, inputs, output = [Path(directory) / name for name in ('ref.R','in.json','out.json')]
            script.write_text('args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH");if(nzchar(lib)).libPaths(c(lib,.libPaths()))\nsuppressPackageStartupMessages(library(mgcv));library(jsonlite)\na<-fromJSON(args[1],simplifyVector=FALSE)\n'+source)
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(['Rscript',str(script),str(inputs),str(output)], capture_output=True,text=True)
            assert run.returncode == 0, run.stderr
            save_reference('mgcv_gevlss_family',key,json.loads(output.read_text()))
    return load_reference('mgcv_gevlss_family',key)


def case(link):
    x = np.array([-.9,-.7,-.5,-.3,-.1,.1,.3,.5,.7,.9,-.6,.6])
    X = np.column_stack([np.ones(len(x)),x]*3)
    return {'link':link,'X':X.tolist(),'jj':[[0,1],[2,3],[4,5]],'coef':[.7,-.1,.2,.05,.15,-.08],
            'y':[.4,1.1,.6,1.8,.9,2.1,.2,1.7,.6,1.2,.3,1.9],
            'w':[.5,1,2]*4,'offset':[[.01]*12,[-.02]*12,[.03]*12], 'E':(np.eye(6)*.2).tolist()}


@pytest.mark.parametrize('location,shape',[('identity','logit'),('log','logit'),('identity','identity'),('log','identity')])
def test_gevlss_all_derivatives_links_residuals_and_initializer_match_source(location,shape):
    payload = case([location,'identity',shape])
    ref = reference(payload)
    f = GevlssFamily(link=payload['link'])
    X, coef = np.asarray(payload['X']),np.asarray(payload['coef'])
    jj = [np.array(j) for j in payload['jj']]
    actual = f.ll(payload['y'],X,jj,coef,payload['w'],offset=payload['offset'],deriv=4,
                  d1b=np.zeros((6,1)),d2b=np.zeros((6,1)),fh=np.eye(6),D=np.ones(6),ncv=True)
    for key in ('l','l0','l1','l2','l3','l4'):
        expected=np.asarray(ref['raw'][key],float).reshape(np.asarray(actual[key]).shape)
        np.testing.assert_allclose(actual[key],expected,atol=5e-10,rtol=5e-10,err_msg=key)
    fitted=f.predict_fitted(X,jj,coef,offset=payload['offset'])
    np.testing.assert_allclose(fitted,ref['fitted'],atol=1e-15)
    for rtype,expected in zip(('deviance','pearson','response'),ref['residuals'],strict=True):
        np.testing.assert_allclose(f.residuals(payload['y'],fitted,rtype=rtype),expected,atol=4e-13,rtol=4e-13)
    np.testing.assert_allclose(f.initialize(payload['y'],X,jj,offset=payload['offset'],weights=payload['w'],E=np.asarray(payload['E'])),ref['initial'],atol=1e-12,rtol=1e-12)
    eta=f._stacked_eta(X,jj,coef,payload['offset'])
    params=f.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(f.logpdf_from_parameters(payload['y'],params),actual['l0'],atol=3e-14,rtol=3e-14)
    for info,expected in zip(f.linfo,ref['links'],strict=True):
        mu=info.linkinv(np.array([-40,-2,0,1,40]))
        for value,key in [(mu,'mu'),(info.mu_eta(np.array([-40,-2,0,1,40])),'mu_eta'),(info.d2link(mu),'d2'),(info.d3link(mu),'d3'),(info.d4link(mu),'d4')]:
            np.testing.assert_allclose(value,np.asarray(expected[key],float),atol=1e-13,rtol=2e-14)


@pytest.mark.parametrize('link',[['identity'],['log','log','logit'],['identity','identity','log'],['sqrt','identity','logit']])
def test_gevlss_invalid_links_rejected(link):
    with pytest.raises(ValueError):
        GevlssFamily(link)


def test_gevlss_support_and_null_deviance():
    f=GevlssFamily()
    assert np.isneginf(f.logpdf_from_parameters([10],[[0,1,-.5]])[0])
    assert np.isnan(f.null_deviance([0],np.zeros((1,3)),[1]))
    np.testing.assert_array_equal(f._source_shape([0,1e-8,-1e-8,.2]),[1e-7,1e-7,-1e-7,.2])
    with pytest.raises(ValueError):
        f.validate_y([np.inf])
