import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from nampy.basemodels.gam import GAM
from nampy.gam.inference.anova import _smooth_test_stat, _term_edf1
from tests.mgcv_parity_utils import _run_mgcv_snapshot

rng = np.random.default_rng(302); n=160
x0=rng.uniform(-2.,2.,size=n); x1=rng.uniform(-1.5,1.5,size=n)
y=np.sin(1.2*x0)+0.4*x1**2+rng.normal(scale=0.15,size=n)
data=pd.DataFrame({'y':y,'x0':x0,'x1':x1})
formula='y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

gam=GAM(family='gaussian',formula=formula,optimize_smoothing=False,smoothing_method='fixed')
gam.fit(data=data)

snap=_run_mgcv_snapshot(data,formula,'gaussian','fixed')
sti=snap['parity']['diagnostics']['smooth_test_inputs']

print('r_blocks type:', type(sti['r_blocks']))
print('parity keys:', list(snap['parity'].keys()))

r_block_0=np.asarray(sti['r_blocks'][0], dtype=np.float64)
r_coef_0=np.asarray(sti['coef_blocks'][0], dtype=np.float64)
r_edf1_0=float(sti['edf1'][0])
r_resdf=float(sti['residual_df'])
tb0=gam.term_blocks_[0]; sl0=tb0.coef_slice
r_Vb_0=gam.Vp_[sl0.start:sl0.stop, sl0.start:sl0.stop]  # same Vp for both

print('R block shape:', r_block_0.shape, 'Py shape:', gam._summary_R_[:, :7].shape)
d_r, rdf_r, _ = _smooth_test_stat(r_coef_0, r_block_0, r_Vb_0, rank=r_edf1_0, residual_df=r_resdf)
print(f'R r_block + same Vp -> F={d_r/rdf_r:.4f} (want 623.49)')

py_beta=gam.coef_[sl0]; py_V=gam.Vp_[sl0,sl0]; py_Ri=gam._summary_R_[:,sl0]
py_edf1=_term_edf1(gam,tb0); py_resdf=gam.n_samples_-1-gam.edf_
d_py, rdf_py, _ = _smooth_test_stat(py_beta, py_Ri, py_V, rank=min(7,max(py_edf1,1.)), residual_df=py_resdf)
print(f'Py r_block + same Vp -> F={d_py/rdf_py:.4f} (Python=1172.43)')
print(f'coef match: {np.allclose(r_coef_0, py_beta)}')
print('R block norms:', np.linalg.norm(r_block_0, axis=0))
print('Py R norms:', np.linalg.norm(py_Ri, axis=0))
print()
# Test with mixed: R r_block, Python coef and Vp
d_mix, rdf_mix, _ = _smooth_test_stat(py_beta, r_block_0, py_V, rank=r_edf1_0, residual_df=r_resdf)
print(f'R r_block + Py coef/Vp -> F={d_mix/rdf_mix:.4f}')
