# PreTab enhancement review and future PR plan

## Purpose and current status

This document records optional PreTab enhancements developed while integrating
several neural tabular model families. It is intended as the design and review
basis for future pull requests to the PreTab repository.

## Executive assessment

| Enhancement | General PreTab value | Example consumers | Recommendation |
|---|---|---|---|
| Method-wide `representation_params` | High | Quantile transforms, one-hot options, splines, embeddings, TF-IDF | Submit as foundational API |
| Parameterized per-feature overrides | High | Different transforms or hyperparameters per column | Submit with one canonical syntax |
| TF-IDF representation | High but separable | Free-text columns, concept models, linear baselines | Submit as its own representation PR |
| Configurable output ordering | Moderate to high | Positional model contracts and reproducible exports | Submit in an output-layout PR |
| Atomic encoded-column output | High | Scalar-concept models, feature subnetworks, interpretable exports | Submit after lineage edge cases are resolved |
| Post-representation scaling | Moderate to high | Models requiring a common encoded range | Submit with explicit sparse policy |
| Quantile count/distribution controls | Useful, but overlaps generic parameters | Quantile preprocessing and NodeGAM-like pipelines | Prefer generic routing; decide on aliases |
| Quantile fit-only noise | Specialized but legitimate | Robust empirical quantile fitting, NodeGAM-style preprocessing | Submit separately as an opt-in fit policy |

All proposed defaults preserve pristine PreTab behavior:

- no post-representation scaler;
- source-feature block output;
- numerical-first block order;
- no quantile noise;
- no representation-specific overrides unless supplied;
- no TF-IDF unless selected explicitly.

## Existing pristine PreTab contract

Pristine PreTab already provides a strong generic contract:

- `Preprocessor.fit(X, y)` and `Preprocessor.transform(X)`;
- dictionary output grouped by source feature, such as `num_age` and
  `cat_city`;
- `return_array=True` for a single stacked matrix;
- `get_feature_names_out()` for encoded column names;
- `get_feature_info(verbose=False)` with source-feature metadata such as
  `dimension`, `categories`, and `preprocessing`;
- basic per-feature method selection through `feature_preprocessing`;
- output budgets, dense-memory safeguards, sparse output, missing-value policy,
  serialization, presets, and extension registration.

The proposals extend this contract in areas pristine PreTab cannot currently
express. They should compose with existing safeguards rather than bypass or
duplicate them.

---

## Proposal 1: generic representation parameters

### Problem

PreTab can select a representation globally or per feature, but many useful
constructor options cannot be expressed through its current high-level API.
Examples include:

- `QuantileTransformer(n_quantiles=..., output_distribution=...)`;
- `OneHotEncoder(drop=..., min_frequency=..., handle_unknown=...)`;
- `MinMaxScaler(feature_range=...)`;
- polynomial `degree`, `interaction_only`, and `include_bias`;
- language embedding model selection;
- future TF-IDF vocabulary and normalization options.

Adding a top-level `Preprocessor` argument for every transformer option would
make the constructor large, representation-specific, and difficult to extend.

### Prototype API

The audit branch adds method-wide parameters:

```python
preprocessor = Preprocessor(
    numerical_method="quantile",
    representation_params={
        "quantile": {
            "n_quantiles": 200,
            "output_distribution": "normal",
        }
    },
)
```

The mapping is keyed by normalized representation name. Each value is a mapping
of arguments accepted by that representation's registered transformer.

### Parameter resolution

The prototype resolves configuration in this order:

1. representation implementation defaults;
2. existing shared PreTab controls;
3. method-wide `representation_params[method]`;
4. feature-specific parameters.

Later layers override earlier layers. This makes a global policy cheap while
still allowing exceptional columns.

```python
Preprocessor(
    numerical_method="quantile",
    representation_params={
        "quantile": {
            "n_quantiles": 100,
            "output_distribution": "normal",
        }
    },
    feature_preprocessing={
        "income": {
            "method": "quantile",
            "n_quantiles": 500,
        }
    },
)
```

Every quantile feature uses a normal output distribution and 100 quantiles,
except `income`, which uses 500 quantiles.

### Validation behavior

The audit implementation:

- requires `representation_params` to be a mapping;
- requires every method entry to contain a parameter mapping;
- normalizes aliases to canonical registry names;
- rejects parameters not listed in `TransformerSpec.allowed_args`;
- reports valid parameters in the typed invalid-parameter error;
- preserves scikit-learn `get_params`, `set_params`, and `clone` behavior because
  the constructor value remains an estimator attribute.

Strict rejection is important. Silently dropping misspelled or unsupported
arguments would make preprocessing configurations difficult to audit.

### Prototype implementation locations

- `pretab/compose/config.py`
  - validates and normalizes `representation_params`;
  - merges method-wide and feature-specific parameters through
    `representation_for()`.
- `pretab/compose/factory.py`
  - validates arguments against the representation registry;
  - forwards filtered parameters to numerical and categorical constructors.
- `pretab/compose/registry.py`
  - expands `allowed_args` for min-max scaling, one-hot encoding, language
    embeddings, quantile transformation, and TF-IDF.
- `pretab/preprocessor.py`
  - exposes and documents the constructor argument.

### Open API decision

Method names such as `none` can exist in multiple feature-kind namespaces. The
PR should define whether method-wide parameters are keyed only by canonical
method name or by a typed key such as `numerical:quantile`. The prototype tries
numerical resolution first and then categorical resolution. That is convenient
but could become ambiguous as the registry grows.

### Required tests

- aliases normalize to the same parameter namespace;
- unsupported parameters fail before fitting;
- method-wide settings reach every matching feature;
- feature-specific settings override method-wide settings;
- values survive `clone`, `get_params`, `set_params`, and serialization;
- extension representations receive only their declared arguments;
- numerical and categorical methods with similar names cannot be confused.

---

## Proposal 2: parameterized per-feature representations

### Problem

Pristine `feature_preprocessing` can select different methods by feature, but it
cannot cleanly express different constructor parameters for two features using
the same method.

### Prototype API

The audit branch accepts the existing string form:

```python
feature_preprocessing={"age": "quantile"}
```

and a structured form:

```python
feature_preprocessing={
    "age": {
        "method": "quantile",
        "n_quantiles": 50,
        "output_distribution": "uniform",
    },
    "income": {
        "method": "quantile",
        "n_quantiles": 500,
        "output_distribution": "normal",
    },
}
```

The prototype also accepts a nested `params` mapping. Supporting both flat and
nested forms creates unnecessary ambiguity.

### Recommended API before submission

Choose one canonical structured syntax. The flat form is shorter:

```python
{"method": "quantile", "n_quantiles": 50}
```

The nested form separates routing from constructor parameters more clearly:

```python
{"method": "quantile", "params": {"n_quantiles": 50}}
```

The recommendation is the nested form because it leaves room for future
feature-level policy keys without colliding with transformer constructor names.
The existing string form should remain supported for backward compatibility.

### Required tests

- string overrides retain existing behavior;
- a structured override requires `method`;
- `params` must be a mapping;
- feature parameters override method-wide parameters;
- invalid parameters identify both feature and method;
- resolved configuration survives serialization and cloning.

---

## Proposal 3: TF-IDF categorical/text representation

### Problem

PreTab supports embeddings and categorical encoders but lacks a lightweight,
sparse, deterministic representation for ordinary free-text columns. Users must
currently vectorize text outside PreTab, separating that feature from PreTab's
lineage, budgets, output-format handling, and serialization.

### Prototype API

Global selection:

```python
Preprocessor(
    categorical_method="tfidf",
    representation_params={
        "tfidf": {
            "max_features": 5_000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
        }
    },
    output_format="sparse",
)
```

Per-feature selection:

```python
Preprocessor(
    categorical_method="one-hot",
    feature_preprocessing={
        "review_text": {
            "method": "tfidf",
            "params": {"lowercase": False, "norm": None},
        }
    },
)
```

### Transformer behavior

The prototype `TfidfTransformer`:

- wraps `sklearn.feature_extraction.text.TfidfVectorizer`;
- accepts exactly one source column per instance;
- accepts one- or two-dimensional inputs;
- returns a sparse matrix;
- exposes scikit-learn-compatible `fit`, `transform`, and
  `get_feature_names_out`;
- prefixes vocabulary terms with the source feature name;
- exposes the main vectorizer controls:
  - `lowercase`;
  - `max_features`;
  - `ngram_range`;
  - `min_df` and `max_df`;
  - `norm`;
  - `use_idf` and `smooth_idf`;
  - `sublinear_tf`;
  - `dtype`.

It is registered as categorical-only with `sparse_output=True`, allowing
capability discovery through `list_representations(sparse_output=True)`.

### Missing-value policy that must be resolved

The prototype converts values to strings immediately before vectorization. If
categorical imputation is disabled, unresolved missing values can become the
literal token `"nan"`. That should not be an accidental public contract.

Before submission, choose and test one policy:

1. require missing values to be resolved by PreTab's categorical imputer;
2. map missing values to an explicit documented empty document or sentinel;
3. raise when missing values reach `TfidfTransformer`.

The first or third option is preferable because it keeps missing-state handling
explicit.

### Other edge cases

- empty vocabulary after filtering should produce a feature-specific error;
- `min_df`/`max_df` validation should remain sklearn-compatible;
- duplicate vocabulary-derived output names must be impossible or rejected;
- sparse output must remain sparse unless a later operation explicitly
  requires densification;
- portable serialization must preserve vocabulary, IDF state, and dtype;
- lineage should identify the original text column for every token.

### Recommended PR boundary

Submit TF-IDF independently after generic representation parameters are
accepted. This keeps text-specific review separate from output layout and
scaling changes.

---

## Proposal 4: configurable output ordering

### Problem

Pristine PreTab stacks numerical blocks before categorical blocks. That is a
reasonable default, but downstream estimators and exported artifacts may have a
different positional contract:

- categorical blocks first;
- original input-column order;
- stable reproduction of an external preprocessing pipeline.

Reordering outside PreTab is error-prone because the transformed array,
dictionary order, names, lineage, and metadata must remain synchronized.

### Prototype API

```python
Preprocessor(output_order="numerical-first")   # existing behavior
Preprocessor(output_order="categorical-first")
Preprocessor(output_order="input")
```

### Semantics

- `numerical-first`: numerical features in discovered order, followed by
  categorical features in discovered order;
- `categorical-first`: categorical features first, then numerical features;
- `input`: transformed blocks follow the original DataFrame column order.

The factory builds `ColumnTransformer.transformers` in the requested order so
every downstream surface inherits one canonical layout. This is preferable to
rearranging only the final array.

### Compatibility

The default remains `numerical-first`, so existing pipelines do not change.
Ordering becomes fitted state through the resulting `ColumnTransformer` layout.

### Edge cases to specify

- passthrough/remainder columns and whether they always remain last;
- duplicate input column names;
- multivariate representations consuming more than one source column;
- external embedding blocks;
- columns excluded by policy or preprocessing selection;
- serialization across scikit-learn versions.

### Required tests

For every order, verify agreement among:

- `transform(..., return_array=True)` positions;
- dictionary insertion order;
- `get_feature_names_out()`;
- feature-lineage output indices;
- `get_feature_info()`;
- serialized/restored preprocessors.

---

## Proposal 5: atomic encoded-column output

### Problem

Pristine dictionary output is grouped by source feature. For example, a
three-level one-hot encoding returns:

```text
cat_city -> array with shape (n_rows, 3)
```

Some consumers operate on scalar transformed concepts and otherwise must split
the block themselves while reconstructing names, lineage, and order.

### Prototype API

```python
Preprocessor(output_granularity="feature")  # existing behavior
Preprocessor(output_granularity="column")
```

With column granularity, the example becomes:

```text
cat_city_Berlin -> shape (n_rows, 1)
cat_city_Paris  -> shape (n_rows, 1)
cat_city_Rome   -> shape (n_rows, 1)
```

### Metadata contract

For every atomic output, the prototype reports:

- `dimension: 1`;
- `output_index`: position in the stacked encoded matrix;
- `source_feature`: original source column;
- inherited preprocessing and category metadata.

Atomic dictionary names come from cleaned `ColumnTransformer` feature names.
Duplicate names raise explicitly instead of overwriting a dictionary entry.

### Implementation approach

- `get_output_slices(..., granularity="column")` returns one width-one slice per
  encoded column;
- `build_feature_info(..., output_granularity="column")` uses fitted lineage to
  synthesize atomic metadata;
- `Preprocessor.transform()` and `get_feature_info()` use the same fitted
  granularity so tensors and metadata cannot disagree.

### Important lineage limitation

The prototype assigns an atomic output to `record.source_features[0]`. That is
not sufficient for multivariate representations such as tensor products or
explicit interactions. Before submission, atomic metadata must define how
outputs with multiple sources are represented. Possible contracts include:

- keep multivariate outputs grouped even under column granularity;
- expose `source_features` as a tuple rather than singular `source_feature`;
- allow atomic output while retaining every contributing source feature.

The third option is the most general and aligns with existing lineage concepts.

### Required tests

- one-hot, spline, polynomial, Fourier, and passthrough outputs;
- missing-indicator columns;
- sparse output blocks;
- duplicate cleaned names;
- multivariate representations;
- embedding outputs;
- consistency with output ordering;
- portable serialization round-trip.

---

## Proposal 6: post-representation scaling

### Problem

Pristine `scaling` is applied inside numerical pipelines before or as part of
representation construction. It does not scale the complete encoded matrix.
Some models require all encoded outputs, including one-hot or basis columns, to
share a common range.

This is a distinct operation:

```text
input scaling -> representation -> post-representation scaling
```

It should not be overloaded onto the existing `scaling` option.

### Prototype API

```python
Preprocessor(
    scaling=None,
    output_scaling="minmax",
    output_range=(0.0, 1.0),
)
```

Supported prototype values are:

- `output_scaling=None`;
- `output_scaling="minmax"`;
- a finite increasing two-value `output_range`.

### Fitting and transform semantics

The audit implementation:

1. fits all per-feature representations;
2. checks the fitted output-width budget;
3. transforms clean training data through those representations;
4. fits `MinMaxScaler` over the complete encoded matrix;
5. applies the fitted scaler on every later transform;
6. casts to the requested dtype after scaling.

When quantile fit noise is active, the post-scaler is fitted on clean input
transformed through the noisy-fitted quantile map. It is not fitted on another
noisy sample.

### Sparse-output consequence

The prototype densifies a sparse encoded matrix before fitting or applying
`MinMaxScaler`. This must be an explicit contract, not an implementation
surprise.

Before submission, decide whether to:

- reject `output_scaling` when the encoded representation is sparse;
- allow it only after checking `max_dense_memory` against actual shape and
  dtype;
- introduce sparse-safe scalers where their mathematical semantics match.

At minimum, documentation and warnings must state that min-max post-scaling can
destroy sparsity. A user requesting `output_format="sparse"` should not incur an
unannounced dense intermediate.

### Budget ordering

The audit fix deliberately calls `_enforce_output_budget()` before materializing
the encoded training matrix for the post-scaler. This prevents an obviously
over-wide representation from being densified before width policy rejects it.
Dense-memory policy still needs explicit coverage for accepted-width cases.

### Required tests

- exact comparison with sklearn `MinMaxScaler` on the stacked representation;
- one-hot plus numerical blocks;
- custom ranges and constant columns;
- unseen categorical levels;
- output dtype preservation;
- sparse rejection or documented densification;
- output-budget and dense-memory checks before allocation;
- cloning and portable serialization.

---

## Proposal 7: quantile transformer controls

### Problem

Pristine PreTab exposes `numerical_method="quantile"` but not the most commonly
needed `QuantileTransformer` controls through a convenient high-level surface.

### Prototype API

```python
Preprocessor(
    numerical_method="quantile",
    quantile_n_quantiles=2_000,
    quantile_output_distribution="normal",
    random_state=42,
)
```

Validation requires:

- `quantile_n_quantiles` to be an integer at least one;
- `quantile_output_distribution` to be `"uniform"` or `"normal"`;
- booleans to be rejected as integer values.

Values are forwarded only to representations whose registry allows the
corresponding arguments.

### Duplication with `representation_params`

Once generic method parameters exist, the same configuration is expressible as:

```python
Preprocessor(
    numerical_method="quantile",
    representation_params={
        "quantile": {
            "n_quantiles": 2_000,
            "output_distribution": "normal",
        }
    },
    random_state=42,
)
```

Maintaining both APIs creates precedence questions and expands the constructor.
The recommended upstream design is:

- use `representation_params` as the canonical mechanism;
- keep top-level quantile parameters only as documented convenience aliases if
  PreTab intentionally supports common-method shortcuts;
- if aliases remain, specify that feature-specific values override method-wide
  values, which override top-level convenience values.

### Required tests

- typed validation errors;
- forwarding of count, distribution, and random state;
- precedence relative to method-wide and feature-specific parameters;
- scikit-learn cloning;
- no effect on non-quantile representations.

---

## Proposal 8: quantile fit-only noise

### Problem

Some empirical quantile pipelines add small random jitter while fitting the
quantile map to reduce problems caused by repeated values. Noise must affect
only the fitted empirical distribution, never caller data or later transforms.

This policy cannot be represented solely as a `QuantileTransformer` constructor
argument, so it remains distinct from generic representation parameters.

### Prototype API

```python
Preprocessor(
    numerical_method="quantile",
    quantile_noise=1e-3,
    random_state=7,
)
```

`quantile_noise=0.0` preserves pristine behavior.

### Prototype algorithm

For each numerical feature whose resolved method is quantile:

1. copy the fit frame;
2. identify finite values, leaving missing and non-finite values untouched for
   normal policy and imputation handling;
3. compute population standard deviation `s` from finite values;
4. calculate `noise_scale = noise / max(s, noise)`;
5. draw Gaussian noise with `numpy.random.RandomState(random_state)`;
6. fit the preprocessing pipeline on the noisy copy;
7. transform the original clean input thereafter.

The caller's DataFrame is never mutated. Repeated `transform()` calls are
deterministic because noise is applied only while fitting.

### Why it belongs at Preprocessor fit level

- it changes the data presented to a representation during fitting;
- it must select only features whose resolved method is quantile;
- it must cooperate with missing-value handling;
- it must not run during transform;
- it should share PreTab's reproducibility policy.

### Design questions before submission

- Should it remain quantile-specific or become a representation fit-policy
  hook?
- Should random draws use legacy `RandomState` for reference reproducibility or
  `numpy.random.Generator` for new code?
- Should noise be independent per feature or use one matrix-shaped stream?
- How should all-missing and zero-variance columns behave?
- Should the fitted policy appear explicitly in lineage and serialized specs?

### Required tests

- exact seeded comparison with sklearn `QuantileTransformer`;
- no mutation of caller data;
- deterministic repeated transforms;
- noise affects only quantile-selected features;
- missing values still reach the configured imputer;
- zero noise is identical to pristine behavior;
- invalid negative, non-finite, boolean, and non-numeric values fail clearly;
- clone and serialization preserve the setting.

---

## Model-consumer map

The table records motivating consumers without making them PreTab dependencies.

| Model family | Useful enhancement | Why useful | Behavior without enhancement |
|---|---|---|---|
| NAM | atomic output, categorical-first order, post-scaling to `[-1, 1]` | Reproduces scalar-network-per-encoded-column preprocessing | A valid grouped categorical NAM consumes pristine blocks |
| NBM | atomic output, input order, post-scaling to `[0, 1]`, TF-IDF | Names every encoded column as a concept | NBM flattens pristine grouped blocks internally; text must be handled separately |
| SPAM | atomic naming and stable order | Gives polynomial terms deterministic scalar identities | SPAM flattens grouped blocks internally |
| NBM-SPAM | same as NBM and SPAM | Stabilizes hybrid block assembly and attribution | Core model remains functional with grouped blocks |
| NodeGAM | quantile count, normal output, fit-only noise | Reproduces a common empirical quantile map | Basic pristine quantile preprocessing remains available |
| Linear/additive baselines | representation parameters and TF-IDF | Keeps heterogeneous processing in one fitted estimator | Only currently exposed global controls are available |
| Any positional consumer | output ordering and atomic metadata | Avoids bespoke reordering and metadata reconstruction | Consumer uses numerical-first feature blocks |

## Cross-cutting compatibility requirements

Every future PR should preserve these PreTab invariants.

### Scikit-learn estimator contract

- constructor arguments are assigned without hidden mutation;
- `get_params`, `set_params`, and `clone` work;
- fitted-state checks remain standard;
- feature names and `n_features_in_` remain correct.

### Output contract

- dictionary, array, pandas, polars, and sparse outputs remain consistent;
- names, metadata, lineage, and array positions describe the same layout;
- dtype requests are honored;
- sparse representations remain sparse unless a documented operation requires
  densification.

### Safety policies

- output-width budgets run before expensive materialization;
- dense-memory limits cover every new dense intermediate;
- missing-value policy is not bypassed;
- unsupported parameters raise rather than being ignored;
- optional dependencies remain discoverable and explicit.

### Presets and `UNSET`

New fields must preserve PreTab's `UNSET` and preset resolution rules. Explicit
user values override presets, while omitted fields inherit the preset or stable
default. The audit branch was rebased specifically to retain the release
branch's newer preset behavior.

### Serialization

New configuration and fitted state must survive `to_spec()` and `from_spec()`.
The audit branch also includes a portability fix for scikit-learn's private
`_RemainderColsList`, encoding it as a plain list. That fix is not one of the
model features, but it was needed for portable round trips on current release
code and should be considered as a small independent bug-fix PR.

## Proposed PR sequence

Do not submit the audit commits as one PR. A more reviewable sequence is:

### PR 1: generic representation parameter routing

- add `representation_params`;
- choose one structured per-feature syntax;
- expand and validate registry `allowed_args`;
- cover cloning, extensions, presets, and serialization;
- do not add TF-IDF or output-layout changes yet.

### PR 2: TF-IDF representation

- add `TfidfTransformer` and its registry entry;
- resolve missing-value semantics;
- preserve sparse output;
- add lineage, capability-discovery, serialization, and text-specific tests;
- use PR 1 for configuration instead of TF-IDF-specific top-level arguments.

### PR 3: output ordering

- add `output_order` with numerical-first as the default;
- define remainder, embedding, and multivariate behavior;
- test every output surface for consistent order.

### PR 4: atomic output granularity

- add `output_granularity` with feature blocks as the default;
- resolve multivariate lineage;
- ensure unique names and width-one metadata;
- test sparse blocks, missing indicators, embeddings, and serialization.

### PR 5: post-representation scaling

- add `output_scaling` and `output_range`;
- define sparse and dense-memory behavior;
- enforce budgets before materialization;
- test constant columns, unknown categories, dtype, and round trips.

### PR 6: quantile fit-only noise

- use generic representation parameters for `n_quantiles` and
  `output_distribution` unless maintainers prefer convenience aliases;
- add only fit-time jitter as a distinct Preprocessor concern;
- document reproducibility and missing-value semantics.

The `_RemainderColsList` portability fix can be submitted before or with PR 1
as a narrowly scoped bug fix.

## Prototype file inventory

| Area | Files | Responsibility |
|---|---|---|
| Public API and fit flow | `pretab/preprocessor.py` | Constructor controls, quantile noise, output scaler, fitted granularity |
| Configuration | `pretab/compose/config.py` | Validation, normalization, precedence, output controls |
| Pipeline construction | `pretab/compose/factory.py` | Output ordering and parameter forwarding |
| Registry | `pretab/compose/registry.py` | Allowed arguments, TF-IDF registration, sparse capability |
| Metadata and slicing | `pretab/compose/inspection.py` | Atomic blocks, output indices, source metadata |
| Serialization | `pretab/compose/serialize.py` | Portable remainder-column state |
| TF-IDF | `pretab/transformers/categorical/tfidf.py` and exports | Sparse text transformer |
| Integration tests | `tests/integration/test_model_preprocessing_contracts.py` | Composition, scaling, ordering, atomization, TF-IDF, budgets, round trips |
| Quantile tests | `tests/integration/test_quantile_controls.py` | Forwarding, jitter, missing values, validation, cloning |
| Registry tests | compose and extension test files | Availability and sparse discovery |

## Validation already performed on the audit branch

Historical audit results were:

- focused latest-release PreTab integration: `503 passed, 60 skipped`;
- full PreTab suite: `1303 passed, 72 skipped, 7 xfailed, 2 failed`;
- the two full-suite failures were pre-existing uses of `numpy.trapezoid` under
  NumPy 1.26, despite the project declaring NumPy 1.24+ support;
- targeted NAMpy integration against audited PreTab passed quantile, ordering,
  atomic concept, TF-IDF, and preprocessing-parity checks.

Since NAMpy now deliberately targets pristine PreTab, current NAMpy tests are
not acceptance tests for these enhancements. Each PreTab PR must carry its own
package-level tests and be rebased onto the then-current PreTab branch.

## Decisions required before opening PRs

1. Is `representation_params` keyed only by canonical method name, or by method
   plus feature kind?
2. Should structured `feature_preprocessing` use flat parameters or nested
   `params`?
3. Are quantile count/distribution top-level aliases worth the duplicate API?
4. What is the explicit missing-value contract for TF-IDF?
5. Should post-scaling reject sparse input or permit guarded densification?
6. How does atomic granularity report multivariate source lineage?
7. How are remainder columns and embeddings ordered under `output_order="input"`?
8. Should quantile noise remain specialized or use a generic fit-policy hook?
9. Which random-number API is the stable reproducibility contract?
10. Should `output_range` be validated when `output_scaling=None`, or only when
    the scaler is enabled?

## Submission checklist

For each future PR:

- fetch and rebase onto the latest PreTab target branch;
- confirm the proposal still fills a current public-API gap;
- keep one conceptual feature per PR;
- add API documentation and changelog entries;
- add examples that do not mention or depend on NAMpy;
- preserve defaults and preset/`UNSET` semantics;
- add focused unit and integration tests;
- run PreTab's configured lint and type checks;
- run focused tests first and the full PreTab suite before submission;
- report pre-existing failures separately;
- verify wheel/sdist metadata and optional dependencies;
- document sparse behavior, memory allocation, serialization, and backward
  compatibility explicitly;
- avoid presenting downstream-reference parity as the primary justification.

## Final recommendation

The prototype demonstrates that these capabilities can be integrated without
changing PreTab's default behavior. The strongest broadly useful pieces are
generic representation parameters, one canonical per-feature parameter syntax,
TF-IDF, and synchronized output-layout metadata. Post-scaling and fit-only
quantile noise are useful too, but require especially clear memory and
reproducibility contracts.

Keep the local audit branch as a reference until the decisions above are
resolved. Future PRs should extract and refine individual features rather than
merge the audit branch wholesale.
