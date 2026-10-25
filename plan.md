# Implementation progress

- [x] 1. Tutorials index — keep one collection
- [x] 2. Your first additive models — remove material that belongs later
- [x] 3. Classification with missing values and categorical features — keep the complete workflow
- [x] 4. Fit and diagnose a GAM — keep one dataset throughout
- [x] 5. Compare GAM and NAM feature effects — retain, but emphasize its different purpose
- [x] 6. Compare neural additive architectures — delete the second NAM/NBM example
- [x] 7. Interaction selection — move to Tutorials and simplify the output
- [x] 8. GAMs with groups, offsets, and changing variance — give it a more accurate title
- [x] 9. Model a changing response distribution — merge and redirect
- [x] 10. Compare distributions, quantiles, and ensemble spread — retain its three-way comparison
- [x] 11. Conditional quantiles — preserve the deeper assessment
- [x] 12. Additive trees — keep the distinct example
- [x] 13. Four modeling examples — remove the standalone duplication
- [x] 14. How-to index — make the task list smaller
- [x] 15. Add an interaction — retain a concise recipe
- [x] 16. Monotonicity and shape constraints — merge, then correct the comparison
- [x] 17. Predict with an offset — keep both examples
- [x] 18. Save and reload — retain the recipe, shorten the explanation
- [x] 19. Prediction outputs and interpretability — keep separate responsibilities
- [x] 20. Quick-fit recipes — consolidate into one page
- [x] 21. Training, preprocessing, and compatibility — remove repetition without losing contracts
- [x] 22. Statistical overview and formula reference — keep the useful tables
- [x] 23. Add a group-smooth tutorial — highest-priority mgcv-related addition
- [x] 24. Add a smoothness and term-selection tutorial
- [x] 25. Add derivatives, contrasts, and uncertainty
- [x] 26. Add a diagnostic tutorial that actually changes a model
- [x] 27. Add non-Gaussian examples with a reason for each family
- [x] 28. Expand statistical interactions beyond a syntax comparison
- [x] 29. Secondary demonstrations — exact scope, after the six above
- [x] 30. Fix the development validation notebook before reusing its examples

---

Below are the edits I would make to the pages reviewed in the previous turn. **Keep the useful plots, fitted summaries, and model comparisons. Remove repeated explanations and examples that answer the same question twice.**

Replacement passages are ready to paste. For proposed new demonstrations, I give the exact introduction, section order, and required outputs; their numerical conclusions must come from running the examples.

**1. Tutorials index — keep one collection**

Page: [Tutorials](https://nampy.readthedocs.io/en/latest/tutorials/index.html)

Keep the unified collection. Replace its introduction with:

The tutorials work through complete modeling examples, from fitting a first additive model to interpreting interactions, response distributions, and uncertainty. Start with your first additive models, or choose an example that matches your data.

Use this order for the existing material after the consolidations below:

1. Your first additive models
2. Classification with missing values and categorical features
3. Fit and diagnose a GAM
4. Compare GAM and NAM feature effects
5. Compare neural additive architectures
6. Select interactions from validation data
7. Model counts and changing variance
8. Compare distributions, quantiles, and ensemble spread
9. Fit and assess conditional quantiles
10. Recover step-shaped effects with additive trees

Insert new statistical tutorials near the relevant topic as they become available. Do not list unfinished pages.

Remove the cards for:

- **Four modeling examples**, after retaining its useful material on the homepage.
- **Model a changing response distribution**, after merging its explanation into the distributional notebook.

Keep redirects from their old URLs.

---

**2. Your first additive models — remove material that belongs later**

Page: [Your first additive models](https://nampy.readthedocs.io/en/latest/notebooks/01_getting_started.html)

This page currently fits regression and classification models, checks calibration, and runs cross-validation. It also generates heteroscedastic and interaction datasets that it never analyzes.

**Delete or move:**

- Delete the generation of `sigma`, `y_hetero`, `X_sparse`, and `y_sparse`.
- Replace the opening four-panel figure with the regression scatterplot alone.
- Move **Check predicted probabilities** into the classification tutorial if that tutorial does not already contain an equivalent calibration plot.
- Move **Cross-validation**, including its code, into the training guide.
- Remove duplicate `logging` and `warnings` imports and repeated warning configuration.
- Remove unused imports after these changes.
- Keep reconstruction assertions silent. Do not restore a “Checked” table.

**Keep:**

- GAM/NAM test-score table.
- Observed-versus-predicted figure.
- Residual figure.
- Numerical and categorical effect plots.
- The short classification/link-function example and inverse-logit plot.

Replace the opening paragraph with:

An additive model builds a prediction from a baseline and a contribution from each feature. Those contributions can be nonlinear.

We will fit a GAM and a neural additive model to the same simulated data, compare their predictions, and inspect the relationships they learn. A short classification example then shows why additive contributions are interpreted on the link scale.

Replace the caption under the simplified opening plot with:

The response depends on two numerical features and a group effect. This plot shows its relationship with x1; variation around the curve also reflects the other inputs and random noise.

End after the inverse-logit explanation with:

The regression examples add contributions directly to the predicted response. In logistic classification, contributions add to log-odds, and the inverse-logit function converts the total to a probability.

The classification tutorial applies this workflow to data with missing values and categorical features.

**Why:** this remains a substantial, visual first tutorial without becoming an introduction to every later topic.

---

**3. Classification with missing values and categorical features — keep the complete workflow**

Page: [Classification workflow](https://nampy.readthedocs.io/en/latest/notebooks/05_real_data_classification.html)

Do not remove the preprocessing or save/reload sections merely because separate how-to pages exist. Here they complete the analysis.

Make these edits:

- Explain splitting before imputation and encoding once, immediately before preprocessing.
- Keep the actual preprocessing code.
- Keep the probability comparison, calibration figure, and feature-effect plots.
- If importing calibration material from the first notebook, adapt it to this dataset; do not add a second nearly identical calibration figure.
- Keep one save/reload example and one prediction-equivalence assertion.
- Link to the persistence guide for format and compatibility details.

Before the probability comparison, use:

We compare predicted probabilities on the held-out rows. Discrimination measures whether the model ranks positive cases above negative cases; calibration measures whether its probabilities agree with observed frequencies. A model can perform well on one and poorly on the other.

Before saving, use:

To reuse this fit, we need the model and the preprocessing learned from the training data. Save any transformations performed outside the estimator alongside it, then apply the same transformations to new observations.

Delete any additional general explanation of serialization that repeats the persistence guide.

---

**4. Fit and diagnose a GAM — keep one dataset throughout**

Page: [Fit and diagnose a GAM](https://nampy.readthedocs.io/en/latest/tutorials/first_gam.html)

Keep everything through **Inspect residuals and basis dimensions**.

Remove these sections from this page:

- **Other smooth structures**
- **Fit a monotone effect**
- **Fit a spatial interaction**

Move any unique figures and their generating code into the shape-constraint and interaction material. Do not paste entire duplicate examples into destinations that already contain equivalent demonstrations.

The energy-demand example already teaches continuous and cyclic smooths, REML, prediction, effect interpretation, and diagnostics. That is sufficient.

Replace the ending with:

The fitted model captures the temperature relationship and daily cycle, and the residual and basis checks do not indicate a clear problem in this example.

If a smooth appeared too restrictive, we would increase its basis dimension and refit. If residual structure remained after allowing sufficient flexibility, we would reconsider the model terms or response distribution.

**One additional check:** confirm that the cyclic smooth uses the intended period endpoints, 0 and 24, through the supported knot/boundary configuration. Randomly sampled minimum and maximum hours should not silently define a slightly different daily period.

Keep the printed statistical summary here. It teaches readers what a GAM summary contains.

---

**5. Compare GAM and NAM feature effects — retain, but emphasize its different purpose**

Page: [Compare GAM and NAM feature effects](https://nampy.readthedocs.io/en/latest/tutorials/gam_nam.html)

This overlaps with the first tutorial, but the four known effects give it a useful purpose: **effect recovery**, beyond introductory fitting.

Replace the introduction with:

Similar prediction scores do not guarantee similar fitted relationships. Here we compare a GAM and a NAM against four known effects, including saturation, curvature, and diminishing returns.

The data are simulated, so we can assess both held-out prediction accuracy and how closely each fitted effect follows the function that generated it.

Keep:

- The short test-score table.
- All four effect panels with centered truth.
- The shared centering reference.

Remove repeated explanations of what an additive model is. Keep only the sentence explaining spline smooths versus feature networks.

Change **Compare the feature effects** to **Compare the fitted effects with the truth**.

Do not add another residual dashboard here unless it reveals something that affects this comparison.

---

**6. Compare neural additive architectures — delete the second NAM/NBM example**

Page: [Compare neural additive architectures](https://nampy.readthedocs.io/en/latest/notebooks/03_neural_models.html)

Delete **A second NAM/NBM example** in full, including its code, plots, training logs, and repeated imports.

Keep:

- NAM/NBM accuracy and parameter-count table.
- The first effect-recovery figure.
- NAMformer component table.
- The controlled-input plot.

The first comparison already answers the NAM/NBM question.

Replace the paragraph before the parameter table with:

Both models use the same training, validation, and test rows. We compare test accuracy with the number of trainable parameters to see how sharing basis functions changes the size of the fitted model.

For the transformer section, retain the distinction between feature heads and a global contribution. Its current small contribution should not be presented as a successful interaction demonstration.

Use:

The global contribution is small in this fit. The example shows which inputs each component can depend on, but it does not demonstrate a substantial improvement from the transformer. The interaction-selection tutorial examines a dataset with an explicit interaction signal.

**Fix, rather than merely hide, the emitted non-writable NumPy-array warning.** Rerun the notebook after resolving the underlying conversion or input issue. Suppress routine progress displays through fitting options, but do not globally silence meaningful warnings.

---

**7. Interaction selection — move to Tutorials and simplify the output**

Current page: [Select interactions](https://nampy.readthedocs.io/en/latest/howto/interaction_selection.html)

Move this into Tutorials because it contains a complete comparison, validation analysis, selection procedure, and test evaluation.

Rename it:

> **Select interactions from validation data**

Keep a redirect from the old URL.

Replace its introduction with:

An additive model can miss a relationship that depends jointly on two inputs. We will first fit a main-effects model, inspect its validation residuals, and then use SIAN to select interaction terms.

The test outcomes are reserved for the final comparison; they are not used to choose interactions.

Use these headings:

1. Generate data with an interaction
2. Fit a main-effects model
3. Inspect validation residuals
4. Select interaction terms
5. Compare test predictions
6. Inspect the selected interaction

Reduce the visible selection table to:

| Interaction | Selected | Selection score |
|---|---:|---:|

Remove `order`, inclusion/removal details, `signed_score`, `total_effect`, and `n_contrasts` from this tutorial’s displayed table unless you explicitly interpret them. Retain the full result object in code and document its fields in the reference.

Before the table, add:

The selection table records which candidate interactions were retained. The score summarizes the discovery procedure’s evidence for each candidate; it is not a p-value or a measure of causal importance.

Keep the validation-residual plot, test-metric table, and selected-interaction surface.

---

**8. GAMs with groups, offsets, and changing variance — give it a more accurate title**

Page: [Statistical GAM notebook](https://nampy.readthedocs.io/en/latest/notebooks/02_statistical_gams.html)

Rename it:

> **Model counts and changing variance**

The current group term is a group shift, not a demonstration of group-specific smooths or partial pooling.

Replace its introduction with:

The response distribution determines what a fitted mean tells us. We first model counts whose expected value depends on exposure, then fit a continuous response whose variability increases with a predictor.

These examples show how an offset changes the expected count and how a distributional GAM can model the mean and standard deviation separately.

For a cleaner sequence:

1. Create the data
2. Fit a continuous-response baseline
3. Account for exposure in count data
4. Check whether the spread changes
5. Model the mean and standard deviation

Replace **Choose a basis dimension** with **Fit a continuous-response baseline**.

Remove the `k=8` versus `k=12` competition from this page and use one adequate basis, such as the existing `k=12` candidate. The current near-tie does not teach much about basis selection. Move that topic into the dedicated smoothness tutorial below.

Keep:

- Baseline fitted effect.
- Exposure-level prediction curves.
- Squared-residual plot.
- Fitted and true mean/standard-deviation panels.

After changing the baseline, rerun all downstream results and revise their numerical descriptions.

Under the exposure equation, replace \(u_{\mathrm{group}}\) with a clearly named fixed group coefficient, such as \(\beta_{\mathrm{group}}\), unless the example is changed to a random effect.

Use:

The group term shifts the linear predictor by a constant. It does not give each group its own curve or estimate a partially pooled random effect.

This sentence belongs next to the formula, not in a repeated warning box.

---

**9. Model a changing response distribution — merge and redirect**

Page: [Changing response distribution](https://nampy.readthedocs.io/en/latest/tutorials/changing_variability.html)

Remove this as a standalone tutorial.

Its main lesson and plotting call already occur in the distributional notebook. Preserve the explanation of the distributional-effect display there:

The central curve is the point effect, centered relative to the training data. The surrounding curves lie one fitted conditional standard deviation above and below it. Their separation describes response variability; it is not a confidence interval for the central curve.

Do not append the entire second dataset to the destination notebook.

If its two-feature figure adds a useful distinction missing from the existing example, retain it as a short supplementary example. Otherwise the existing fitted-mean, fitted-scale, and interval figures provide better coverage without duplication.

---

**10. Compare distributions, quantiles, and ensemble spread — retain its three-way comparison**

Page: [Distributional models](https://nampy.readthedocs.io/en/latest/notebooks/04_distributional_models.html)

Keep all three substantive comparisons:

- Normal location and scale.
- Normal versus quantile prediction intervals.
- Disagreement between independently fitted models.

Keep the coverage-and-width table. Unlike a “Reconstruction: Checked” table, it directly supports a statistical conclusion.

Make these edits:

- Incorporate the display explanation from item 9.
- Keep the separate mean and standard-deviation panels: they answer a different question from the centered effect plot.
- Keep the observed overcoverage of the quantile model. Do not adjust the narrative to make every model appear successful.
- Explain ensemble disagreement once before its figure; shorten the repeated caption.

Replace that caption with:

The band spans one standard deviation of member predictions around their mean. The wider scatter of test observations includes response noise that this band does not represent.

Do not add another general uncertainty taxonomy table. The three fitted examples already explain the distinction.

---

**11. Conditional quantiles — preserve the deeper assessment**

Page: [Fit and assess conditional quantiles](https://nampy.readthedocs.io/en/latest/in_depth/quantile_calibration.html)

Keep this separate from the preceding comparison. Its purpose should be assessing quantiles, not introducing NAMLSS again.

Replace its introduction with:

A fitted quantile should have a corresponding proportion of observations below it. We will compare estimated quantiles with their known values and examine coverage across the predictor range.

Ordered quantiles avoid crossing, but ordering alone does not ensure accurate coverage.

Keep the quantile curves, coverage results, and effect plots.

If the page currently reports only overall coverage, add coverage by a small number of predefined predictor bins. Display the number of test observations in each bin so readers can judge how noisy the estimates are.

Remove repeated explanations of normal-distribution parameters or ensemble disagreement. Those belong in the comparison tutorial.

---

**12. Additive trees — keep the distinct example**

Page: [Recover step-shaped effects](https://nampy.readthedocs.io/en/latest/in_depth/tree_effects.html)

Keep this tutorial. Step-shaped effects give the tree models a distinct modeling question.

Use this introduction:

Some relationships change abruptly rather than smoothly. We will simulate step-shaped feature effects, fit an additive tree model and a joint ensemble, and compare their predictions and recovered effects.

Keep:

- Truth-versus-fit effect plots.
- Held-out prediction results.
- The explanation of how the ensemble is trained.

Delete any repeated general explanation of additive reconstruction.

Describe a jointly trained ensemble accurately; do not reuse the independent-fit disagreement interpretation from the distributional notebook.

---

**13. Four modeling examples — remove the standalone duplication**

Page: [Four modeling examples](https://nampy.readthedocs.io/en/latest/tutorials/modeling_workflows.html)

Retain the corresponding examples on the homepage, consistent with your preference for a substantial landing page.

Before removing this tutorial:

1. Compare each figure, table, and code example against the homepage.
2. Move any genuinely unique and useful result to its matching homepage example.
3. Remove the tutorial card and internal links.
4. Redirect the old URL to the homepage’s relevant examples section.

Do not copy all four examples into other tutorials as well. The homepage provides the overview; the individual tutorials provide the analysis.

---

**14. How-to index — make the task list smaller**

Page: [How-to Guides](https://nampy.readthedocs.io/en/latest/howto/index.html)

Replace the introduction with:

Use these guides to complete a specific task with NAMpy. For a worked analysis with data, fitted results, and interpretation, see the tutorials.

Make these navigation edits:

- Move **Select interactions** to Tutorials.
- Replace **Impose monotonicity** and **Choose other shape constraints** with one entry: **Constrain the shape of an effect**.
- Replace the three quick-fit links with **Fit a regression, classification, or distributional model**.
- Move historical PreTab migration material out of the main task list.
- Keep preprocessing, training, interactions, offsets, prediction outputs, interpretability, and persistence.

The statistical and distributional overview pages can remain linked, but label them as broader guides rather than implying they are short recipes.

---

**15. Add an interaction — retain a concise recipe**

Page: [Add an Interaction](https://nampy.readthedocs.io/en/latest/howto/interactions.html)

Keep the statistical and neural examples. They use different interfaces.

Replace the introduction with:

Add an interaction when the effect of one input is allowed to depend on another. In a statistical GAM, use a joint smooth or combine main effects with a tensor interaction. In a neural additive model, specify the feature combination supported by the architecture.

Immediately before the GAM example, use:

Use `s(x) + s(z) + ti(x, z)` to represent separate main effects and an additional interaction. Use `te(x, z)` for a full joint surface. `t2(x, z)` provides an alternative tensor construction with a different penalty decomposition.

Keep one working example per interface.

Do not add the full SIAN selection workflow here. End with a link to its tutorial.

---

**16. Monotonicity and shape constraints — merge, then correct the comparison**

Pages: [Monotonicity](https://nampy.readthedocs.io/en/latest/howto/monotonicity.html) and [Shape-constrained GAMs](https://nampy.readthedocs.io/en/latest/user_guide/shape_constrained_gams.html)

Redirect the monotonicity page to the shape-constrained guide.

The current guide says:

> “The only formula change below is the basis code.”

That is too narrow for the displayed comparison: it changes the spline construction, and the fitting configuration also changes from REML to GCV.

Replace that paragraph with:

We compare an ordinary cubic regression spline with a monotone-increasing P-spline. The constrained fit uses the supported GCV fitting route. This illustrates the effect of imposing monotonicity, but the two fits also differ in their basis and smoothing-selection method.

For a more controlled comparison, change the ordinary fit to `bs="ps"` and use compatible GCV settings for both models. Then describe the shared settings explicitly after verifying them.

Also:

- Keep the curve comparison and derivative plot.
- Reduce **Univariate bases** and **Bivariate bases** to representative examples and links to the exhaustive API inventories.
- Move detailed coefficient-space implementation explanations to developer/reference material.
- Move the general AR(1) workflow into its own guide when that example exists.
- Retain a short, visible statement of fitting restrictions.

Use:

For transformed shape-constrained models, use fixed smoothing or a supported GCV/UBRE route. Automatic ML, REML, and LAML selection is not available for these models. The compatibility reference lists restrictions for multi-parameter models and other combinations.

Keep shape constraints identified as SCAM-related functionality rather than folding them into a blanket mgcv capability claim.

---

**17. Predict with an offset — keep both examples**

Page: [Predict with an Offset](https://nampy.readthedocs.io/en/latest/howto/offsets.html)

This is useful overlap, not safe deletion. The statistical and neural prediction contracts differ.

Use this opening:

An offset is a known contribution to the linear predictor with coefficient fixed at one. It must be included consistently during fitting and prediction.

Keep:

- The statistical offset prediction example.
- The neural binary example that adds the known offset on the logit scale before applying the inverse link.
- The numerical comparison showing the consequence of omitting the offset.

Delete a repeated full derivation of exposure scaling if it duplicates the count tutorial. Link to that tutorial for the modeling explanation.

Label intermediate variables clearly: **model contribution**, **known offset**, **full linear predictor**, and **probability**.

---

**18. Save and reload — retain the recipe, shorten the explanation**

Page: [Save and Reload a Model](https://nampy.readthedocs.io/en/latest/howto/persistence.html)

Keep one neural and one statistical example.

Use this introduction:

Save the fitted estimator together with any preprocessing performed outside it. After loading, check that predictions agree on the same input rows before applying the model to new data.

Keep the equivalence assertion silent.

Delete success tables such as “Loaded: True” or “Predictions match: Checked.” A failed assertion is useful during execution; a table of successful checks adds little to the rendered guide.

Link to the classification tutorial for a complete workflow.

---

**19. Prediction outputs and interpretability — keep separate responsibilities**

Pages: [Prediction Outputs](https://nampy.readthedocs.io/en/latest/howto/prediction_outputs.html) and [Interpretability](https://nampy.readthedocs.io/en/latest/user_guide/interpretability.html)

**Prediction Outputs should own:**

- Method names.
- Output shapes and parameter ordering.
- Labels versus probabilities.
- Natural-scale parameters versus link-scale contributions.
- Appropriate display for each output type.

Keep its lookup table.

**Interpretability should own:**

- The additive decomposition.
- Centering and reference data.
- Main effects, interactions, and global terms.
- The meaning of effect magnitude.
- Interpretation of distributional envelopes and repeated-fit variation.

Replace the prediction-output introduction with:

Choose the prediction method according to the quantity you need: a response prediction, class probability, distribution parameter, quantile, or additive contribution. The table below summarizes the returned values and their shapes.

Remove repeated explanations of centering from Prediction Outputs and link to Interpretability.

In tutorials, retain the one or two sentences needed to interpret the figure at hand. Do not replace all local explanations with links.

---

**20. Quick-fit recipes — consolidate into one page**

Current pages:

- `examples/basic_regression.html`
- `examples/basic_classification.html`
- `examples/distributional_regression.html`

Create one page titled:

> **Fit a regression, classification, or distributional model**

Use three sections:

1. Predict a continuous response
2. Predict class probabilities
3. Predict a response distribution

Use this introduction:

These examples show the minimum fitting and prediction calls for three common tasks. Each assumes that training and test data have already been separated. The linked tutorials show how to prepare the data and assess the fitted models.

Preserve any unique multi-output example as a short fourth section.

Remove repeated installation instructions, generic training advice, and repeated import blocks where they can be shared.

Redirect the three existing URLs to their corresponding sections.

---

**21. Training, preprocessing, and compatibility — remove repetition without losing contracts**

For [Training Models](https://nampy.readthedocs.io/en/latest/user_guide/training.html):

- Keep fitting controls, early stopping, scheduling, validation, regularization, sample weights, and architecture-specific differences.
- Consolidate **Basic Training** and **Complete Example** if they repeat the same fit.
- Replace overlapping **Training Tips** and **Common Issues** lists with one troubleshooting table: symptom, checks, possible adjustment.
- Add the cross-validation example moved from the first tutorial.
- Remove its redundant one-step `Pipeline` wrapper unless a real external preprocessing step is added.
- State which features the CV example uses; it currently drops the group feature.

Use this introduction to the CV section:

Cross-validation evaluates a fitting procedure across several train–validation splits. Any transformation learned from the data must be fitted separately within each training fold.

Use cross-validation to choose settings on the development data, then evaluate the selected procedure on a separate test set.

For [Preprocessing](https://nampy.readthedocs.io/en/latest/user_guide/preprocessing.html):

- Keep the high-level configuration example.
- Keep access to the fitted preprocessor.
- Retain the distinction between numerical, categorical, and embedding inputs.
- Move detailed extension contracts to the custom-model reference if they interrupt the everyday workflow.

For [PreTab compatibility](https://nampy.readthedocs.io/en/latest/user_guide/pretab_compatibility.html):

- Move **Removed requirements** to migration/development documentation.
- Keep the current supported contract.
- Keep architecture-specific consequences where they explain how encoded inputs become model terms.
- Remove duplicated statements from **Behavioral consequences** when already explained under **Model impact**.

For [Custom Models](https://nampy.readthedocs.io/en/latest/user_guide/custom_models.html):

- Use one complete example introduced incrementally.
- Remove the second full copy under **Complete Example** if it repeats those steps.
- Keep forward-pass requirements and the tests that establish the extension contract; these are appropriate technical content here.

---

**22. Statistical overview and formula reference — keep the useful tables**

Pages: [Statistical GAMs](https://nampy.readthedocs.io/en/latest/user_guide/statistical_gams.html) and [Formula visual reference](https://nampy.readthedocs.io/en/latest/user_guide/formula_visual_reference.html)

Do **not** remove every large table. The formula, prediction, family, and diagnostic tables are useful lookup material.

Make these targeted edits:

- Keep one exhaustive basis inventory in the API reference.
- Keep the overview’s representative formula and method tables.
- Move **Advanced public utilities** out of the main user guide into extension/reference documentation.
- Move `parity_snapshot` into numerical validation documentation.
- Keep optimizer names and restrictions in the reference; do not repeat them in tutorials.
- Fix the self-link at the end of **Choose basis capacity and smoothness**, which currently sends readers back to **Statistical GAMs in Python** for fitting controls already on the page.

Replace that closing sentence with:

The first GAM tutorial shows a complete fit and diagnostic check. The sections below summarize available formulas and fitting controls; the API reference gives their full arguments.

The formula visual reference already has fitted examples for cyclic effects, groups, random effects, and tensor surfaces. Keep those. Add links from each to the deeper demonstrations below rather than duplicating their entire analyses.

---

**23. Add a group-smooth tutorial — highest-priority mgcv-related addition**

Proposed title:

> **Fit group-specific curves with partial pooling**

Opening:

Groups may differ in their average response, in the shape of a relationship, or both. We will simulate groups with unequal sample sizes and compare models that allow different kinds of variation.

The small groups are particularly informative: they show how pooling changes a fitted curve when little data are available.

Use these sections:

1. Generate groups with unequal sample sizes
2. Fit a common curve with group intercepts
3. Fit separate factor-by smooths
4. Share smoothing parameters across groups
5. Fit penalized group curves
6. Separate a global curve from group deviations
7. Compare predictions and fitted curves

Demonstrate these distinct constructions:

| Construction | What to demonstrate |
|---|---|
| Parametric group term | Constant shifts with a shared curve |
| `bs="re"` | Penalized random intercepts |
| Factor `by` smooths | Different curves for different levels |
| Shared `id` | Shared smoothing parameters, not identical curves |
| `bs="fs"` | Fully penalized group curves |
| `bs="sz"` | Group deviations constrained around a global smooth |

Include the appropriate group main effects and identifiability constraints for each construction; do not compare formulas that accidentally omit needed group intercepts.

**Required outputs:**

- Group sample-size table.
- Faceted fitted-versus-true curves.
- A focused small-group versus large-group comparison.
- Global curve and deviation panels.
- Test performance under a clearly stated split.

Use this explanation:

Sharing a smoothing parameter gives the group curves a common penalty strength. It does not force their coefficients or shapes to match. Penalizing group effects and defining deviations around a global curve impose different structures, so these models should be compared by what they assume as well as by prediction accuracy.

Use held-out observations from known groups initially. Treat prediction for entirely new groups as a separate example only after verifying the supported prediction behavior.

---

**24. Add a smoothness and term-selection tutorial**

Proposed title:

> **Choose smoothness and shrink unnecessary terms**

Opening:

A smooth needs enough basis functions to represent the relationship, but it also needs a penalty to avoid fitting noise. These are separate choices.

We will simulate a simple effect, a more irregular effect, and an irrelevant predictor. The example compares limited basis capacity, stronger smoothing, and penalties that can shrink an entire term toward zero.

Sections:

1. Generate two effects and an irrelevant predictor
2. Fit a basis that is too small
3. Increase the basis dimension
4. Compare fixed and estimated smoothing
5. Penalize unnecessary terms
6. Compare effect recovery and prediction

Demonstrate ordinary supported configurations using:

- An ordinary smooth.
- A shrinkage basis such as `ts` or `cs`.
- `select=True`.

Do not combine a shrinkage basis and `select=True` in the sole example: show their roles separately.

**Required outputs:**

- True and fitted curves.
- Basis diagnostic results.
- EDF and smoothing-parameter table.
- Irrelevant-predictor effect panel.
- Held-out prediction metric.

Use:

An ordinary roughness penalty can leave an unpenalized component of a smooth. Increasing its penalty therefore need not remove the term entirely. Shrinkage bases and additional null-space penalties allow the whole effect to be penalized.

Do not rank GCV and REML by placing their raw objective values in one “best score” column.

---

**25. Add derivatives, contrasts, and uncertainty**

Proposed title:

> **Estimate rates of change and differences between predictions**

Opening:

A fitted curve often leaves a more specific question unanswered: where is the response increasing, and how much does the predicted response change between two conditions?

We will estimate a smooth’s derivative and use the linear-predictor matrix to calculate contrasts with uncertainty.

Sections:

1. Fit a smooth relationship
2. Estimate its derivative
3. Compare two predictor settings
4. Calculate uncertainty for the contrast
5. Account for smoothing-parameter uncertainty where supported

Start with an ordinary P-spline whose derivative route is supported.

For prediction-matrix rows \(X_a\) and \(X_b\), show:

\[
C=X_a-X_b,\qquad
\widehat{\Delta}=C\widehat{\beta},\qquad
\operatorname{Var}(\widehat{\Delta})=CVC^\mathsf{T}.
\]

**Required outputs:**

- Fitted curve with pointwise intervals.
- Derivative with a zero line and pointwise intervals.
- Contrast curve or a small contrast table.
- Conditional versus smoothing-adjusted intervals, only where available.

Use:

These intervals describe uncertainty at each evaluated point. They do not provide simultaneous coverage for the entire curve.

The prediction-matrix contrast is on the linear-predictor scale. Include any difference in known offsets separately; for a nonlinear link, a response-scale contrast requires transforming the predictions and accounting for that transformation.

Do not obtain contrast uncertainty by adding two prediction variances while ignoring their covariance.

---

**26. Add a diagnostic tutorial that actually changes a model**

Proposed title:

> **Use GAM diagnostics to improve a model**

Opening:

Diagnostics are most useful when they lead to a specific modeling change. We will examine examples with insufficient basis capacity, a missing interaction, and correlated predictors, then check what changes after refitting.

Use three short cases rather than one model with every possible problem:

1. **Insufficient basis capacity:** same formula and data, changing only the relevant `k`.
2. **Missing interaction:** compare main effects with main effects plus an interaction.
3. **Concurvity:** compare predictive stability with the stability of individual effects.

**Required outputs:**

- Before/after residual plots.
- Before/after fitted effects.
- A compact basis-diagnostic table.
- A displayed concurvity result.
- Test performance where relevant.

Use:

A diagnostic result suggests what to investigate; it does not identify the cause by itself. We change one aspect of the model at a time and check whether the fitted relationship and remaining residual structure change as expected.

Do not choose a random seed merely to obtain a small diagnostic p-value. If the diagnostic does not detect a simulated deficiency, report that limitation.

---

**27. Add non-Gaussian examples with a reason for each family**

Start with:

> **Model overdispersed counts**

Opening:

Count data can vary more than a Poisson model allows. We will simulate overdispersed counts, fit Poisson and negative-binomial GAMs with the same mean structure, and compare their residuals and predictions.

Use genuinely overdispersed data, not Poisson-generated counts.

Keep the mean structure and exposure treatment comparable across fits.

**Required outputs:**

- Fitted mean curves.
- Pearson or other appropriate residual diagnostics.
- The estimated negative-binomial dispersion parameter, with its parameterization explained.
- A comparable held-out measure.
- Predictive intervals only if implemented through a verified route.

Then add other families as distinct examples when their data features justify them:

| Family | Demonstration |
|---|---|
| Gamma | Positive, right-skewed responses |
| Beta | Continuous responses strictly between zero and one |
| Tweedie | A mass at zero with positive continuous values |
| Ordered categorical | Ordered class probabilities |
| Gamma location-scale | Variation in both mean and dispersion |

Do not create a tutorial that fits every family and only reports `finite=True`.

---

**28. Expand statistical interactions beyond a syntax comparison**

Proposed title:

> **Model interactions with tensor-product smooths**

Opening:

When the effect of one predictor changes with another, separate smooths may leave systematic residual structure. We will fit a main-effects model, add a tensor interaction, and examine both the interaction contribution and the complete predicted surface.

Sections:

1. Generate a response with a known interaction
2. Fit separate main effects
3. Inspect residual structure
4. Add `ti(x, z)`
5. Inspect the interaction and total surface
6. Compare with a full tensor smooth

**Required outputs:**

- Residual surface from the main-effects model.
- Isolated interaction surface.
- Total predicted surface.
- Conditional slices.
- Test-metric comparison.

Use:

The interaction contribution is the additional joint pattern beyond the fitted lower-order effects. The total predicted surface includes those effects and the intercept. Plotting both avoids mistaking an interaction component for the complete response relationship.

Keep the visual reference as a compact introduction to `te`, `ti`, and `t2`. This new tutorial should explain their consequences through the analysis.

---

**29. Secondary demonstrations — exact scope, after the six above**

These capabilities deserve examples, but they should not delay the more common workflows.

| Proposed page | Exact demonstration | Essential figure/output |
|---|---|---|
| **Smooth responses over neighboring regions** | Fit an MRF using an explicit adjacency graph; compare with independent region effects | Adjacency map, true/fitted regional effects, comparison |
| **Fit a GAM with correlated Gaussian errors** | Ordinary Gaussian identity model with known AR(1) correlation and explicit series starts | Raw and standardized residual ACFs; fitted effect |
| **Choose smoothness with neighborhood cross-validation** | Define neighborhoods and explain which observations are excluded together | Neighborhood diagram and held-out comparison |
| **Allow smoothness to vary across a predictor** | Compare ordinary and adaptive smooths on a function whose roughness changes | Full curve and close-up of the irregular region |
| **Fit a smooth on the sphere** | Use broad geographic coverage and correct latitude/longitude units | Global fitted surface and seam/pole behavior |
| **Model weighted functional inputs** | Fit a response depending on weighted trajectories | Input trajectories, fitted coefficient function, prediction results |
| **Constrain curvature or positivity** | Compare unconstrained and justified constrained fits | Curves and the relevant derivative or positivity display |

For AR(1), explicitly state that the example supplies a known correlation parameter. Do not describe it as estimating a general correlation model.

For spherical smooths, use data that make spherical geometry relevant; a small local patch is an unconvincing demonstration.

---

**30. Fix the development validation notebook before reusing its examples**

Page: `development/notebooks/02_statistical_gams.html`

This is the final cleanup I would make:

| Current item | Exact edit |
|---|---|
| `language_examples` only constructs models and prints names | Label it a construction check. Fit selected examples in the new tutorials before describing them as demonstrations. |
| MRF and spherical examples only construct models | Keep them as API checks; add fitted examples separately. |
| “Poor” basis example also omits another predictor | Retain the same model terms and change only `k`. |
| Basis check does not flag the purported poor fit | Describe the actual result; do not claim it detected insufficient capacity. |
| `concurvity()` computed but not shown | Display and interpret it, or remove the unused call from the rendered walkthrough. |
| Prediction band joins rows sorted by one input while others vary | Build an explicit prediction grid holding other inputs fixed. |
| Summary followed by a large result-object representation | Retain the formatted summary; suppress the duplicate object representation. |
| Different smoothing criteria in one score column | Separate criterion-specific values; compare methods using common prediction or recovery metrics. |
| Family checks report finite predictions | Keep these in development validation, not the learner-facing family examples. |

Keep the automated checks. Remove their decorative success output from tutorials.

**Implementation order:** make the consolidations and wording corrections first, then add group smooths, smoothness/selection, derivatives/contrasts, diagnostics, overdispersed counts, and tensor interactions. That preserves the documentation’s visual richness while giving the statistical capabilities substantially more meaningful coverage.