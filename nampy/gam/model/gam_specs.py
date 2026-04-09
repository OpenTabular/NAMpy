# gam/model/gam_specs.py
"""Predictor spec building helpers for the GAM class."""
import warnings

import numpy as np

from ..formula import (
    apply_drop_intercept,
    compile_predictor_specs_from_formula,
    extract_formula_data,
    parse_gam_formula,
)
from ..formula.preprocess import preprocess_formula_predictor_specs
from ..specs import LinearPredictorSpec, TermSpec, build_smooth_spec


class _GAMSpecsMixin:
    def _make_tensor_term(self, spec, *, knots=None):
        if hasattr(spec, "fit") and callable(spec.fit):
            return spec

        if (
            isinstance(spec, (tuple, list))
            and len(spec) >= 2
            and not isinstance(spec, dict)
        ):
            features = list(spec)
            return TermSpec(
                kind="smooth",
                features=tuple(str(f) for f in features),
                by_variable=None,
                smooth_spec=build_smooth_spec(
                    special="te",
                    bs=self.basis,
                    k=self.k,
                    knots=self._knots_for_features(features, knots=knots),
                    select=bool(self.select),
                ),
                smoothing_id=None,
                label=f"te({', '.join(map(str, features))})",
                metadata={},
            )

        if not isinstance(spec, dict):
            raise TypeError(
                "Each tensor term specification must be either a term object, "
                "a tuple/list of features, or a dict."
            )

        features = spec.get("features", None)
        if features is None:
            raise ValueError(
                "Tensor term dicts must contain a 'features' entry, "
                "e.g. {'features': ('x1', 'x2')}"
            )

        kind = str(spec.get("kind", "te")).lower()
        k = spec.get("k", self.k)
        basis = spec.get("basis", self.basis)
        label = spec.get("label", f"{kind}({', '.join(map(str, features))})")
        smoothing_id = spec.get("smoothing_id", None)
        metadata = spec.get("metadata", None)
        by = spec.get("by", None)
        mc = spec.get("mc", None)
        full = bool(spec.get("full", False))
        ord_ = spec.get("ord", None)
        fixed = bool(spec.get("fixed", spec.get("fx", False)))
        select = bool(spec.get("select", self.select))
        sp = spec.get("sp", None)
        term_knots = spec.get("knots", self._knots_for_features(features, knots=knots))

        if kind == "ti" and mc is not None:
            mc_vals = [bool(v) for v in ([mc] if np.isscalar(mc) else mc)]
            if len(mc_vals) == 0:
                raise ValueError("mc must not be empty.")
            if not any(mc_vals):
                warnings.warn(
                    f"{label}: all ti() marginal constraints are turned off (mc={mc_vals}). "
                    "This leaves the term without identifiability constraints unless the rest "
                    "of the model provides them.",
                    stacklevel=2,
                )

        if kind == "t2":
            return TermSpec(
                kind="smooth",
                features=tuple(str(f) for f in features),
                by_variable=by,
                smooth_spec=build_smooth_spec(
                    special=kind,
                    bs=basis,
                    k=k,
                    full=full,
                    ord_=ord_,
                    fx=fixed,
                    select=select,
                    sp=sp,
                    knots=term_knots,
                ),
                smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
                label=label,
                metadata=dict(metadata or {}),
            )

        return TermSpec(
            kind="smooth",
            features=tuple(str(f) for f in features),
            by_variable=by,
            smooth_spec=build_smooth_spec(
                special=kind,
                bs=basis,
                k=k,
                mc=mc,
                fx=fixed,
                select=select,
                sp=sp,
                knots=term_knots,
            ),
            smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
            label=label,
            metadata=dict(metadata or {}),
        )

    def _make_predictor_specs(self, feature_names, *, knots=None):
        terms = []

        if self.main_effects:
            basis = str(self.basis).lower()
            main_terms = []

            for name in feature_names:
                term_knots = self._knots_for_feature(name, knots=knots)

                if basis in {"cr", "cs", "cc"}:
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            smooth_spec=build_smooth_spec(
                                special="s",
                                bs=basis,
                                k=self.k,
                                sp=None,
                                fx=False,
                                select=bool(self.select),
                                knots=term_knots,
                            ),
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis == "ps":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            smooth_spec=build_smooth_spec(
                                special="s",
                                bs="ps",
                                k=self.k,
                                m=None,
                                sp=None,
                                fx=False,
                                select=bool(self.select),
                                knots=term_knots,
                            ),
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis in {"tp", "ts"}:
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            smooth_spec=build_smooth_spec(
                                special="s",
                                bs=basis,
                                k=self.k,
                                m=None,
                                sp=None,
                                fx=False,
                                select=bool(self.select),
                                knots=term_knots,
                                xt=None,
                            ),
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis == "gp":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            smooth_spec=build_smooth_spec(
                                special="s",
                                bs="gp",
                                k=self.k,
                                m=None,
                                sp=None,
                                fx=False,
                                select=bool(self.select),
                                pc=None,
                                knots=term_knots,
                                xt=None,
                            ),
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        ),
                    )
                elif basis == "re":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            smooth_spec=build_smooth_spec(
                                special="s",
                                bs="re",
                                sp=None,
                                xt=None,
                            ),
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"Automatic main-effect construction currently supports "
                        f"basis in {{'cr','cs','cc','ps','tp','ts','gp','re'}}, got {self.basis!r}."
                    )

            terms.extend(main_terms)

        has_full_te = False
        has_t2 = False
        if self.tensor_terms is not None:
            for spec in self.tensor_terms:
                term = self._make_tensor_term(spec, knots=knots)
                if getattr(term, "term_type", None) == "tensor_smooth":
                    has_full_te = True
                if getattr(term, "term_type", None) == "tensor_anova":
                    has_t2 = True
                terms.append(term)

        if self.main_effects and has_full_te:
            warnings.warn(
                "Model contains both main-effect smooths and full te() tensor-product terms. "
                "This is identifiable via side conditions, but is typically less stable and less "
                "interpretable than a ti() ANOVA-style decomposition.",
                stacklevel=2,
            )

        if self.main_effects and has_t2:
            warnings.warn(
                "Model contains both separate main-effect smooths and t2() terms. "
                "t2() already contains ANOVA-style lower-order components, so this combination "
                "can create strong overlap in the current framework.",
                stacklevel=2,
            )

        return [LinearPredictorSpec(name="eta", terms=terms)]

    def _prepare_formula_inputs(
        self, data, formula, y=None, knots=None, drop_intercept=None
    ):
        parsed = parse_gam_formula(formula)
        parsed = apply_drop_intercept(parsed, drop_intercept=drop_intercept)

        predictor_specs = compile_predictor_specs_from_formula(
            parsed,
            default_k=self.k,
            default_basis=self.basis,
            default_select=self.select,
            knots=knots,
            available_columns=(None if data is None else data.columns),
        )

        predictor_specs, data_work, preprocess_state = (
            preprocess_formula_predictor_specs(
                parsed=parsed,
                predictor_specs=predictor_specs,
                data=data,
            )
        )

        X_np, feature_names, y_out, used_cols, offset_out = extract_formula_data(
            data=data_work,
            parsed=parsed,
            predictor_specs=predictor_specs,
            y=y,
        )
        preprocess_state = dict(preprocess_state)
        preprocess_state["used_columns"] = list(used_cols)
        preprocess_state["offset_name"] = parsed.predictors[0].offset_name
        return (
            parsed,
            predictor_specs,
            X_np,
            feature_names,
            y_out,
            used_cols,
            offset_out,
            preprocess_state,
        )
