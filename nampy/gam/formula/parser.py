"""
Formula parser for GAM model specifications (Stage 1 of the pipeline).

Parses string formulas of the form::

    "y ~ x1 + s(x2) + s(x3, x4, bs='te') + offset(z)"

into structured :class:`ParsedGAMFormula` objects used by the formula compiler
to produce :class:`~nampy.gam.specs.LinearPredictorSpec` instances.

Supported term types
--------------------
- Intercept ``1`` / no-intercept ``-1`` or ``0``.
- Parametric terms: bare variable names or ``x1 + x2 + ...``.
- Smooth terms: ``s(vars, bs=..., k=..., by=..., ...)``.
- Offset: ``offset(varname)``.
- Multi-predictor (GAMLSS-style) lists of formulas.
"""

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedParametricTerm:
    variables: tuple[str, ...]
    raw_label: str


@dataclass
class ParsedSmoothTerm:
    kind: str
    features: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)
    raw_label: str = ""


@dataclass
class ParsedPredictorFormula:
    response_name: str | None
    intercept: bool
    terms: list[Any]
    offset_name: str | None = None
    raw_formula: str = ""


@dataclass
class ParsedGAMFormula:
    response_name: str | None
    predictors: list[ParsedPredictorFormula]
    raw: Any = None


def _contains_standalone_dot(rhs: str) -> bool:
    import re

    return re.search(r"(^|[~+\-(),\s])\.([~+\-(),\s]|$)", rhs) is not None


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    return None


def _source_segment(src: str, node, default: str):
    try:
        out = ast.get_source_segment(src, node)
        return out if out is not None else default
    except Exception:
        return default


def _ast_to_value(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_ast_to_value(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v) for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Call):
        func_name = _call_name(node.func)
        if func_name == "list":
            if len(node.args) != 0:
                raise NotImplementedError(
                    "positional arguments to list(...) are not supported in formulas."
                )
            return {kw.arg: _ast_to_value(kw.value) for kw in node.keywords}
        if func_name == "c":
            if node.keywords:
                raise NotImplementedError(
                    "keyword arguments to c(...) are not supported."
                )
            return [_ast_to_value(arg) for arg in node.args]
        raise NotImplementedError(
            f"Unsupported formula value expression: {ast.dump(node, include_attributes=False)}"
        )
    if isinstance(node, ast.Set):
        out = {}
        for elt in node.elts:
            if not (isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.MatMult)):
                raise NotImplementedError(
                    "Unsupported set-shaped dictionary literal in formula."
                )
            key = _ast_to_value(elt.left)
            value = _ast_to_value(elt.right)
            out[key] = value
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_to_value(node.operand)
        if isinstance(val, (int, float)):
            return -val
    raise NotImplementedError(
        f"Unsupported formula value expression: {ast.dump(node, include_attributes=False)}"
    )


def _ast_to_feature_name(node):
    val = _ast_to_value(node)
    if isinstance(val, str):
        return val
    raise NotImplementedError(
        "Smooth covariates must currently be bare variable names."
    )


def _flatten_additive_terms(node, sign=1):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_additive_terms(node.left, sign=sign) + _flatten_additive_terms(
            node.right, sign=sign
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _flatten_additive_terms(node.left, sign=sign) + _flatten_additive_terms(
            node.right, sign=-sign
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _flatten_additive_terms(node.operand, sign=-sign)
    return [(sign, node)]


def _parse_smooth_call(node, rhs_src: str):
    kind = _call_name(node.func)
    if kind not in {"s", "te", "ti", "t2"}:
        raise ValueError(f"Unknown smooth special {kind!r}.")

    features = [_ast_to_feature_name(arg) for arg in node.args]
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise NotImplementedError(
                "**kwargs style smooth specification is not supported."
            )
        kwargs[kw.arg] = _ast_to_value(kw.value)

    raw_label = _source_segment(rhs_src, node, f"{kind}({', '.join(features)})")
    return ParsedSmoothTerm(
        kind=kind,
        features=features,
        kwargs=kwargs,
        raw_label=raw_label,
    )


def _parse_offset_call(node):
    if _call_name(node.func) != "offset":
        raise ValueError("Expected an offset(...) call.")
    if len(node.args) != 1 or len(node.keywords) != 0:
        raise NotImplementedError(
            "offset(...) currently supports exactly one bare variable name."
        )
    return _ast_to_feature_name(node.args[0])


def _ordered_union(a, b):
    out = []
    for v in list(a) + list(b):
        if v not in out:
            out.append(v)
    return tuple(out)


def _parametric_termsets(node):
    if isinstance(node, ast.Name):
        return {(node.id,)}

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _parametric_termsets(node.left)
        right = _parametric_termsets(node.right)
        out = set(left) | set(right)
        for left_term in left:
            for right_term in right:
                out.add(_ordered_union(left_term, right_term))
        return out

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left = _parametric_termsets(node.left)
        right = _parametric_termsets(node.right)
        out = set()
        for left_term in left:
            for right_term in right:
                out.add(_ordered_union(left_term, right_term))
        return out

    raise NotImplementedError(
        f"Unsupported parametric formula term: {ast.dump(node, include_attributes=False)}"
    )


def _parse_single_formula(formula: str) -> ParsedPredictorFormula:
    if "~" not in formula:
        response_name = None
        rhs = formula.strip()
    else:
        lhs, rhs = formula.split("~", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        response_name = lhs if lhs != "" else None

    if rhs == "":
        rhs = "1"

    if _contains_standalone_dot(rhs):
        raise NotImplementedError("'.' is not supported in GAM formulas.")

    rhs_ast = rhs.replace(":", "@")
    expr = ast.parse(rhs_ast, mode="eval")
    items = _flatten_additive_terms(expr.body)

    intercept = True
    offset_name = None
    parsed_terms = []
    seen_parametric = set()

    for sign, node in items:
        if isinstance(node, ast.Constant) and node.value in {0, 1}:
            if node.value == 0:
                intercept = False
            elif node.value == 1 and sign < 0:
                intercept = False
            continue

        if isinstance(node, ast.Call):
            fname = _call_name(node.func)
            if fname in {"s", "te", "ti", "t2"}:
                if sign < 0:
                    raise NotImplementedError(
                        f"Removing smooth terms via '-' is not yet supported: {fname}(...)"
                    )
                parsed_terms.append(_parse_smooth_call(node, rhs))
                continue

            if fname == "offset":
                if sign < 0:
                    raise NotImplementedError(
                        "Negative offset terms are not supported."
                    )
                off = _parse_offset_call(node)
                if offset_name is not None and offset_name != off:
                    raise NotImplementedError(
                        "Only one offset(...) term is currently supported per predictor."
                    )
                offset_name = off
                continue

            raise NotImplementedError(f"Unsupported call in formula RHS: {fname}(...)")

        try:
            termsets = _parametric_termsets(node)
        except NotImplementedError as err:
            raise NotImplementedError(
                f"Unsupported formula term: {ast.dump(node, include_attributes=False)}"
            ) from err

        if sign < 0:
            raise NotImplementedError(
                "Removing parametric terms via '-' is not yet supported."
            )

        for vars_ in sorted(termsets, key=lambda z: (len(z), z)):
            if vars_ in seen_parametric:
                continue
            seen_parametric.add(vars_)
            parsed_terms.append(
                ParsedParametricTerm(
                    variables=tuple(str(v) for v in vars_),
                    raw_label=":".join(vars_),
                )
            )

    return ParsedPredictorFormula(
        response_name=response_name,
        intercept=bool(intercept),
        terms=parsed_terms,
        offset_name=offset_name,
        raw_formula=formula,
    )


def parse_gam_formula(formula) -> ParsedGAMFormula:
    if isinstance(formula, str):
        parsed = [_parse_single_formula(formula)]
        response_name = parsed[0].response_name
        return ParsedGAMFormula(
            response_name=response_name,
            predictors=parsed,
            raw=formula,
        )

    if isinstance(formula, (list, tuple)):
        if len(formula) == 0:
            raise ValueError("Formula list must not be empty.")

        parsed = [_parse_single_formula(f) for f in formula]

        response_name = None
        for pf in parsed:
            if pf.response_name is not None:
                if response_name is None:
                    response_name = pf.response_name
                elif pf.response_name != response_name:
                    raise ValueError(
                        f"All formulae must refer to the same response. "
                        f"Got {response_name!r} and {pf.response_name!r}."
                    )

        return ParsedGAMFormula(
            response_name=response_name,
            predictors=parsed,
            raw=formula,
        )

    raise TypeError("formula must be a string or a list/tuple of strings.")


__all__ = [
    "ParsedParametricTerm",
    "ParsedSmoothTerm",
    "ParsedPredictorFormula",
    "ParsedGAMFormula",
    "parse_gam_formula",
]
