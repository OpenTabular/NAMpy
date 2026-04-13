Proposed Plan


  # Split gamlss.py Into a gamlss/ Package While Preserving mgcv Parity

  ## Summary

  Refactor nampy/gam/families/gamlss.py from a 4k-line monolith into a nampy/gam/families/gamlss/ package with one shared core module plus family-specific modules for gaulss, gammals, ziplss, gevlss, and shashlss. Preserve
  exact behavioral parity with mgcv/R/gamlss.r by moving code without algebraic rewrites, keeping existing control flow and derivative assembly intact, and validating with the smallest family-specific test slices.

  ## Implementation Changes

  - Replace the file nampy/gam/families/gamlss.py with a package directory nampy/gam/families/gamlss/.
  - Keep nampy.gam.families.gamlss as the import anchor by making its __init__.py re-export the existing public family factories/classes:
    GaulssFamily, gaulss, GammalsFamily, gammals, ZiplssFamily, ziplss, GevlssFamily, gevlss, ShashlssFamily, shashlss.
  - Do not keep broad package-level exports for private helpers. Move those to explicit submodules and update repo tests/imports to target them directly.
  - Create one shared internal module for common GAMLSS infrastructure:
    base GamlssFamily, generic eta stacking / predictor extraction helpers, shared link objects used by multiple families, and any common constants or utility routines that are not family-specific.
  - Create one family module per upstream mgcv implementation block:
    gaulss, gammals, ziplss, gevlss, shashlss.
    Each module should contain only:
    the family class, its factory, and family-local helper/link routines.
  - Keep the heaviest derivative logic co-located with the family it belongs to, especially gevlss and shashlss, so the mapping back to mgcv/R/gamlss.r remains obvious.
  - Add brief comments only where needed to identify upstream routine correspondence when that is no longer obvious after the split.
  - Update nampy/gam/families/__init__.py and nampy/gam/families/registry.py to import from the new package structure directly, not through temporary shims.
  - Update repo-internal imports in tests to use the new explicit submodule paths for private helpers such as _SoftplusBLinkInfo, _ShiftedLogitLinkInfo, _LogEBLinkInfo, _l1ee, _lee1, _ldg, _lde, and _zipll.

  ## Public Interfaces / Import Surface

    and the corresponding family classes.
    private helper imports no longer come from the package root; tests should import them from their family-specific modules.
  ## Test Plan

  - Split tests/test_gamlss_families.py into focused modules aligned with the new code layout:
    one shared/core test module for gamlss_utils-adjacent family plumbing and one module each for gaulss, gammals, ziplss, gevlss, shashlss.
    and where needed exact functions for the moved private-helper tests.
  - Run the targeted parity coverage already exercising these families after the split:
    tests/test_general_family_mgcv_parity.py with family-specific -k filters for the affected family.
  - Acceptance criteria:
    imports resolve from the new package layout,
    public family construction remains unchanged,
    all moved helper tests pass,
    existing mgcv parity tests for each touched family hold without snapshot/tolerance churn.

  ## Assumptions

  - Keep nampy.gam.families.gamlss as the stable public import surface, but as a package instead of a single file.
  - Split tests by family/helper area instead of keeping one monolithic GAMLSS test file.
  - Refactor scope is structural cleanup only: move code, tighten import boundaries, and preserve current behavior unless a parity issue is uncovered during the split.
  - Upstream behavioral reference remains mgcv/R/gamlss.r; implementation should preserve existing ordering and formulas verbatim where possible rather than “cleaning up” internals mathematically.


  Split nampy/gam/families/gamlss.py by copying existing mgcv-aligned blocks into a package, not rewriting them. The current file already has natural boundaries: shared link/base code at the top, then contiguous blocks for
  gaulss, gammals, ziplss, gevlss, and shashlss. That makes this a structural move rather than a behavioral rewrite.

  Target layout:

  nampy/gam/families/gamlss/
    __init__.py
    _base.py
    gaulss.py
    gammals.py
    ziplss.py
    gevlss.py
    shashlss.py

  What goes where:

  - _base.py
      - GamlssFamily
      - shared predictor helpers such as _stacked_eta / _predict_response_from_eta
      - shared link helpers used across families only if they are actually cross-family:
          - _IdentityLinkInfo
          - _AdaptedLinkInfo
      - imports shared across multiple family modules
  - gaulss.py
      - _LogBLinkInfo
      - GaulssFamily
      - gaulss
  - gammals.py
      - _SoftplusBLinkInfo
      - GammalsFamily
      - gammals
  - ziplss.py
      - _l1ee, _lee1, _ldg, _lde, _zipll
      - ZiplssFamily
      - ziplss
  - gevlss.py
      - _ShiftedLogitLinkInfo
      - GevlssFamily
      - gevlss
  - shashlss.py
      - _LogEBLinkInfo
      - ShashlssFamily
      - shashlss

  Execution Order

  1. Create gamlss/__init__.py and re-export only the public surface:
      - GaulssFamily, gaulss
      - GammalsFamily, gammals
      - ZiplssFamily, ziplss
      - GevlssFamily, gevlss
      - ShashlssFamily, shashlss
  2. Move shared base code into _base.py with minimal edits.
  3. Move each family block verbatim into its module, keeping derivative assembly beside the family.
  4. Update nampy/gam/families/__init__.py and nampy/gam/families/registry.py to import from the new package directly.
  5. Update internal imports in tests to explicit module paths for private helpers.
  6. Split tests/test_gamlss_families.py into focused files aligned with the new modules.

  Recommended test split:

  tests/test_mgcv_gamlss_core.py
  tests/test_mgcv_gamlss_gaulss.py
  tests/test_mgcv_gamlss_gammals.py
  tests/test_mgcv_gamlss_ziplss.py
  tests/test_mgcv_gamlss_gevlss.py
  tests/test_mgcv_gamlss_shashlss.py

  This naming matters: current pyproject.toml only auto-collects test_mgcv_*.py, so tests/test_gamlss_families.py is not part of default discovery unless invoked directly.

  Parity Rules During Refactor

  - Move code block-for-block first, then do only the import surgery needed to make it run.
  - Do not re-derive link derivatives, Hessians, or helper formulas while splitting.
  - Keep heavy gevlss and shashlss derivative logic in their own files even if long.
  - Add only short comments where the mgcv mapping stops being obvious after the move.
  - Preserve function/class names for parity-sensitive routines so existing helper tests remain meaningful.

  Validation Plan

  Run only targeted slices after each moved family:

  - pytest tests/test_mgcv_gamlss_gammals.py -v
  - pytest tests/test_mgcv_gamlss_gaulss.py -v
  - pytest tests/test_mgcv_gamlss_ziplss.py -v

  Then parity smoke per affected family:
  - pytest tests/test_general_family_mgcv_parity.py -k ziplss -v
  - pytest tests/test_general_family_mgcv_parity.py -k gevlss -v
  - pytest tests/test_general_family_mgcv_parity.py -k shashlss -v

  Focused helper-import updates:

  - _SoftplusBLinkInfo from nampy.gam.families.gamlss.gammals
  - _ShiftedLogitLinkInfo from nampy.gam.families.gamlss.gevlss

  - Packaging: converting gamlss.py into gamlss/ is fine for repo imports, but the current setuptools config should be verified so the new nested package is included in built distributions.
  - Circular imports: keep _base.py free of family-specific imports.
  - Test breakage from private helper imports: update tests in the same change, not later.

  If you want, I can implement this refactor next in small parity-safe commits, family by family.