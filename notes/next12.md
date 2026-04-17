 # Collapse GAM Front-End Into parse -> extract -> build -> construct

  Refactor the GAM front-end to match the upstream mgcv stage split more closely:

  - interpret.gam equivalent: formula parsing and per-linear-predictor term extraction
  - gam.setup equivalent: spec building plus data attachment/preprocessing for all predictors
  - smoothCon equivalent: smooth construction from one canonical smooth spec type

  Target pipeline:

  1. gam/formula/parse.py: formula text/list -> parsed formula AST
  2. gam/formula/extract.py: parsed AST -> per-predictor declarative term requests
  3. gam/specs/build.py: term requests + data/knots/defaults -> canonical predictor/smooth specs
  4. gam/smooths/construct.py: smooth specs -> constructed smooth objects with X, S, coefficient maps, metadata

  Delete gam/design/ in the same change. design responsibilities move either into smooths/construct.py or into a renamed predictor assembly module outside design/.

  ## Implementation Changes

  ### 1. Replace the current formula layer with a strict two-step front-end

  Create:

  - nampy/gam/formula/parse.py
  - nampy/gam/formula/extract.py

  Move responsibilities as follows:

  - parser.py becomes parse.py and keeps only syntax parsing into a typed AST.
  - formula/compiler.py is removed.
  - formula/extract.py is repurposed from data extraction into AST-to-term extraction.

  Define stable intermediate contracts:

  - ParsedGAMFormula
  - ParsedPredictorFormula
  - ExtractedPredictor
  - ExtractedTerm
  - ExtractedParametricTerm
  - ExtractedSmoothTerm

  Rules:

  - parse.py does not know about defaults, k, bs, knots, data columns, SmoothSpec, or runtime classes.
  - extract.py does not attach data or build final TermSpec; it only preserves formula intent per LP.
  - Multi-formula list handling stays here, so list(y ~ ..., ~ ...) becomes multiple extracted predictors without threading nlp through downstream modules manually.
  - apply_drop_intercept moves beside extraction logic or becomes part of extraction options.

  Upstream reference:

  - mgcv/R/mgcv.r: interpret.gam0, interpret.gam

  ### 2. Move all spec/default/data-attachment work into specs/build.py

  Create:

  - nampy/gam/specs/build.py

  This module becomes the single owner for turning extracted formula terms into canonical predictor specs.

  It should absorb the current responsibilities scattered across:

  - formula/compiler.py
  - formula/preprocess.py
  - parts of current formula/extract.py

  Responsibilities:

  - basis/default resolution (s default to tp, tensor defaults, k defaults)
  - fx, select, id, sp, m, xt, pc, mc, full, ord coercion
  - by-variable validation
  - knot attachment
  - parametric expansion for factors/interactions
  - factor-by expansion
  - hidden-column generation and preprocess state needed for prediction
  - used-column collection and response/offset extraction

  Define canonical outputs:

  - BuiltPredictorSpec
  - BuiltTermSpec
  - BuiltSmoothSpec or reuse LinearPredictorSpec/TermSpec if names are retained
  - FormulaBuildResult containing:
      - predictor specs
      - transformed working data / feature matrix inputs
      - used columns
      - response
      - offsets
      - preprocess state for prediction

  Key rule:

  - after build.py, downstream code should not need access to parsed AST or raw formula terms.

  Upstream reference:

  - mgcv/R/mgcv.r: gam.setup, gam.setup.list
  - especially list-of-formula LP bookkeeping and shared setup semantics

  ### 3. Narrow specs/smooth.py to typed smooth spec definitions only

  Keep nampy/gam/specs/smooth.py, but reduce it to one job:

  - dataclasses / validation / replacement helpers for canonical smooth specs

  Move builder logic out:

  - build_smooth_spec
  - default merging
  - basis dispatch assembly

  New ownership:

  - specs/build.py decides which smooth spec to create
  - specs/smooth.py only defines the types and lightweight helpers

  If helpful, rename the exported constructor to something explicit like make_smooth_spec internally, but keep only one authoritative spec creation path.

  ### 4. Replace design/constructors.py with smooths/construct.py

  Create:

  - nampy/gam/smooths/construct.py

  Move into it:

  - runtime instantiation
  - term fitting
  - penalty extraction
  - explicit constraint absorption
  - prediction coefficient map handling
  - by-handling metadata capture

  This module should become the direct smoothCon analogue for one term/spec.

  Define one canonical constructed object, likely by moving/renaming ConstructedTerm out of design/constructed.py into smooths/construct.py or smooths/constructed.py.

  Rules:

  - construction is per term/smooth, not per whole predictor
  - basis-family dispatch remains delegated to existing smooths/* runtime classes
  - constructor output must already carry all term-local transforms needed by predictor assembly and prediction

  Upstream reference:

  - mgcv/R/smooth.r: smoothCon
  - plus existing per-basis smooth.construct.* routines already mirrored in nampy/gam/smooths/*

  ### 5. Remove gam/design/ and re-home predictor assembly under a non-overlapping name

  Current design/compiler.py is doing three things:

  - linked-basis metadata handling
  - runtime instantiation / construction loop
  - predictor assembly into CompiledPredictor

  After the refactor:

  - term construction leaves design entirely
  - only predictor assembly remains as a separate stage

  Because you want design/ deleted, move predictor assembly to a clearer home such as:

  - nampy/gam/runtime/compile.py, or
  - nampy/gam/predictors/compile.py

  Recommended split:

  - smooths/construct.py: one built term -> one constructed term
  - new predictor compiler module: list of constructed terms -> CompiledPredictor

  Move these there:

  - current compile_predictor_designs
  - linked_basis.py if still needed before construction
  - structures.py if they remain predictor-assembly types

  Rename compile_predictor_designs to something stage-accurate, e.g. compile_predictors or assemble_predictors.

  Delete:

  - nampy/gam/design/compiler.py
  - nampy/gam/design/constructors.py
  - nampy/gam/design/constructed.py
  - nampy/gam/design/__init__.py

  ### 6. Rewire model entrypoints to the new pipeline in one shot

  Update gam/model/gam_specs.py so formula preparation becomes:

  1. parse formulas
  2. extract per-LP terms
  3. build predictor specs + data/preprocess state
  4. compile predictors from built specs using the new assembly module

  Update all direct imports in:

  - model code
  - parity helpers
  - tests

  Hard-cut rules from your chosen strategy:

  - no deprecated wrappers
  - no compatibility exports from old formula/compiler.py or design/*
  - tests and internal callers move immediately to the new names

  ### 7. Keep semantics unchanged during the structural move

  This refactor should not change numerical behavior.

  Guardrails:

  - preserve existing TermSpec / LinearPredictorSpec semantics unless renaming is necessary
  - preserve term_id, smoothing_id, penalty ordering, coefficient slice ordering, and side-condition inputs exactly
  - preserve factor-by and parametric hidden-column recipes exactly unless there is an upstream parity reason to alter them
  - preserve linked-id handling and shared-basis metadata timing
  - do not mix in formula feature additions beyond the structural re-home, except where required to support list-of-formula LP threading cleanly

  ## Public / Internal Interface Changes

  Hard cutover means these import/API changes happen in the same change:

  - Replace nampy.gam.formula.parser with nampy.gam.formula.parse
  - Replace compile_predictor_specs_from_formula(...) with a two-step parse/extract/build sequence or a new top-level formula-build entrypoint backed by those stages
  - Replace nampy.gam.design.compiler.compile_predictor_designs with the new predictor assembly module/function
  - Replace nampy.gam.design.constructors.construct_terms with nampy.gam.smooths.construct.construct_smooth or equivalent
  - Remove ConstructedTerm from nampy.gam.design.*; re-export from its new home only if still needed internally

  Recommended top-level internal API shape:

  - parse_gam_formula(formula) -> ParsedGAMFormula
  - extract_formula_terms(parsed, *, drop_intercept=...) -> list[ExtractedPredictor]
  - build_formula_model(extracted, data, *, y, knots, defaults...) -> FormulaBuildResult
  - compile_predictors(X, feature_names, predictor_specs) -> list[CompiledPredictor]
  - construct_smooth(term_spec, X, feature_names, ...) -> ConstructedSmooth

  ## Test Plan

  Run only narrow slices that cover the moved stages.

  Primary targeted slices:

  1. Formula/extraction/build structure
      - pytest tests/test_mgcv_smoothcon_parity.py -k "cr or ps or tp or fs or mrf" -v
  3. Multi-predictor compilation shape
      - pytest tests/test_mgcv_gamlss_gaulss.py::test_gam_public_api_gaulss_formula_list_fit -v
  5. If linked id or shared-basis code moves materially
      - pytest tests/test_mgcv_pc_id_parity.py -k "linked_id" -v

  Acceptance criteria:

  - list-of-formula models still produce one predictor spec per LP with stable ordering
  - same used columns / offset behavior as before
  - same number/order of constructed terms and penalties
  - same coefficient slices and smoothing-id grouping
  - no regression in targeted smoothCon parity tests

  ## Assumptions And Defaults

  - Hard cutover is intentional: old formula/*compiler* and design/* import paths are removed in the same change.
  - formula/preprocess.py should not survive as a separate post-pass; its logic belongs in specs/build.py so the new pipeline really has one owner per concern.
  - Predictor-wide side conditions remain outside this refactor; they stay where they are and consume the same compiled predictor structure.
  - The refactor is structural first, parity-preserving second; no opportunistic behavioral expansion beyond what is needed to make multi-LP/list formulas flow through one canonical pipeline.
  - Upstream mapping to preserve in comments/docs:
      - parse + extract -> interpret.gam
      - build -> gam.setup / gam.setup.list
      - smooths/construct -> smoothCon