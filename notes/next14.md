High: newdata prediction parity duplicated in same file. tests/
     test_mgcv_output_parity.py:237 already fits model, calls predict(...,
     return_se=True), and asserts both predictions and standard errors. tests/
     test_mgcv_output_parity.py:269 does same work again with near-identical
     assertion body. Best merge: one parametrized test with fixed/REML cases
     together.
  2. High: term-parity test split into three copies, two fully redundant.
     tests/test_mgcv_output_parity.py:354 checks type="terms" values. tests/
     test_mgcv_output_parity.py:398 refits same cases and rechecks same term
     values, plus SE. tests/test_mgcv_parity_failing_and_warnings.py:184 is
     same SE test again for fs. Best merge: single terms + optional SE test,
     with fs carried as marked param instead of separate function/file.
  3. Medium: trace parity has same expensive non-Gaussian fits three times per
     family. tests/test_mgcv_trace_parity.py:272 only checks gradient/hessian
     exposed and finite. tests/test_mgcv_trace_parity.py:292 and tests/
     test_mgcv_trace_parity.py:313 already require same objects and stronger
     value parity. Merge into one test per family: exposed + shape + gradient
     parity + hessian parity. Same story for Gaussian pair tests/
     test_mgcv_trace_parity.py:327 and tests/test_mgcv_trace_parity.py:345.
  4. Medium: k_check suite re-runs full snapshot-style parity for many models,
     then reasserts data snapshot suite already covers. Example fits in tests/
     test_mgcv_k_check_parity.py:167, tests/test_mgcv_k_check_parity.py:275,
     tests/test_mgcv_k_check_parity.py:342. Overlap examples already exist in
     snapshot parity at tests/_mgcv_snapshot_parity_shared.py:1489, tests/
     _mgcv_snapshot_parity_shared.py:1656, tests/
     _mgcv_snapshot_parity_shared.py:1670, tests/
     _mgcv_snapshot_parity_shared.py:1699, tests/
     _mgcv_snapshot_parity_shared.py:1387. Distinct k_check behavior mostly
     k_index/p_value; k_prime and much of edf check look redundant. Best trim:
     keep few representative k_check cases per smooth class, stop rechecking
     snapshot-covered fields.
  5. Low: Gaussian smoothness postprocess tests refit same 1-smooth Gaussian
     REML model several times for closely related invariants: tests/
     test_gam_gaussian_smoothness_postprocess_parity.py:44, tests/
     test_gam_gaussian_smoothness_postprocess_parity.py:110, tests/
     cheap.

  Best first cuts

  1. Merge newdata_predictions + newdata_standard_errors.
  2. Merge all terms parity into one parametrized test.
  3. Collapse trace gradient/hessian/schema tests.
  4. Slim k_check to diagnostic-only assertions.