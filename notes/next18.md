What Remains To Do Next

- The repo still treats the cases in tests/test_mgcv_known_gaps.py:18 as not yet
promotable to stable parity. The main themes are:
- strict tensor parity for PS-based te/ti/t2
- stricter Poisson/binomial residual parity
- tighter negative-binomial estimate_theta=True REML endpoint parity
- The quarantined bucket in tests/test_mgcv_parity_failing_and_warnings.py:1 is
the next obvious cleanup target. It isolates cases like ti(..., mc=...),
t2(..., full=False), near-separation binomial TP, lattice MRF, SZ interaction,
FS select=True, TP negbin theta estimation, and strict factor-by link parity.
- Some explicit unsupported surfaces remain in code and are the real
implementation backlog:
- formula gaps: . shorthand, removing terms via -, multiple predictor-specific
offsets, unsupported RHS calls in nampy/gam/formula/parse.py:219
- smooth/runtime gaps: random effects with id=, fs extra by= / cross-term id=,
some cyclic-cubic shared/factor-by paths, select=True plus term-level sp
- general-family gaps: only families with analytic outer derivatives are
allowed; finite-difference fallback was intentionally removed
- diagnostics gaps: exact k.check() parity still depends on Rscript, and
shashlss still has explicit unsupported residual / k_check surfaces

  Most pragmatic next step: promote anything in the quarantined buckets that now
  passes in isolation, then focus on the remaining true parity gaps in smooth.r-
  driven tensor/factor-smooth behavior and gam.fit3.r negative-binomial endpoint
  behavior.