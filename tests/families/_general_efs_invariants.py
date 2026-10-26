"""Function-space checks for common-start general-family EFS captures."""

import numpy as np

from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
)
from nampy.gam.model_state import _predictor_full_indices

INITIAL_SP = [100.0, 100.0]

# Keep the original source string/keys for existing kernel and fixed captures.
# Only the explicitly common-start EFS experiment adds this instrumentation.
SOURCE_SETUP = """
 if(!is.null(p$initial_sp))args$in.out<-list(sp=unlist(p$initial_sp),scale=1)
"""
SOURCE_OUTPUT = """
if(!is.null(p$initial_sp)) {
 lp<-predict(fit,nd,type='lpmatrix');jj<-attr(lp,'lpi')
 design<-do.call(rbind,lapply(jj,function(j){z<-lp*0;z[,j]<-lp[,j];z}))
 penalties<-lapply(fit$smooth,function(sm){
  z<-matrix(0,length(coef(fit)),length(coef(fit)))
  j<-sm$first.para:sm$last.para;z[j,j]<-sm$S[[1]];z
 })
 ans$efs_invariants<-list(design=design,penalties=penalties)
}
"""


def capture_source(source, payload):
    if "initial_sp" not in payload:
        return source
    return source.replace(" fit<-do.call(gam,args)", SOURCE_SETUP + " fit<-do.call(gam,args)").replace(
        "write_json(ans", SOURCE_OUTPUT + "\nwrite_json(ans"
    )


def assert_efs_function_space(model, newdata, reference):
    """Transport design/penalties, then compare full prediction covariance."""
    lp = model.predict(newdata, type="lpmatrix")
    blocks = []
    for columns in _predictor_full_indices(model):
        block = np.zeros_like(lp)
        block[:, columns] = lp[:, columns]
        blocks.append(block)
    design = np.vstack(blocks)
    source = reference["efs_invariants"]
    source_design = np.asarray(source["design"])
    transport = np.linalg.lstsq(design, source_design, rcond=None)[0]
    np.testing.assert_allclose(design @ transport, source_design, atol=2e-10, rtol=2e-10)
    setup = build_general_family_setup_state(model, model.smoothing_params)
    for penalty, expected in zip(setup.S_blocks, source["penalties"], strict=True):
        np.testing.assert_allclose(transport.T @ penalty @ transport, expected, atol=2e-9, rtol=2e-9)
    for frequentist, key in ((False, "Vp"), (True, "Ve")):
        covariance = design @ model.vcov(freq=frequentist) @ design.T
        source_covariance = source_design @ np.asarray(reference[key]) @ source_design.T
        np.testing.assert_allclose(covariance, source_covariance, atol=2e-7, rtol=2e-7)
