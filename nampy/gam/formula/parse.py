"""
Typed formula parsing for GAM model specifications.

This stage mirrors the syntax-level bookkeeping in mgcv::interpret.gam:
formula text/list -> split formula components plus per-linear-predictor term
aggregation, without any data attachment or runtime/spec construction.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

_FORMULA_DOT_SENTINEL = "__GAM_DOT__"


@dataclass
class ParsedParametricTerm:
    variables: tuple[str, ...]
    raw_label: str
    factor_labels: tuple[str, ...] = ()


@dataclass
class ParsedSmoothTerm:
    kind: str
    features: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    raw_label: str = ""
    label: str = ""
    by_variable: str | None = None


@dataclass
class ParsedFormulaComponent:
    response_name: str | None
    intercept: bool
    terms: list[Any]
    offset_names: tuple[str, ...] = ()
    raw_formula: str = ""
    pf: str = ""
    pfok: int = 1
    fake_formula: str = ""
    fake_names: tuple[str, ...] = ()
    pred_names: tuple[str, ...] = ()
    pred_formula: str = "~ 1"
    lpi: tuple[int, ...] = ()
    is_base_formula: bool = True

    @property
    def offset_name(self) -> str | None:
        if len(self.offset_names) == 1:
            return self.offset_names[0]
        return None


@dataclass
class ParsedPredictorFormula:
    response_name: str | None
    intercept: bool
    terms: list[Any]
    offset_names: tuple[str, ...] = ()
    raw_formula: str = ""
    source_component_indices: tuple[int, ...] = ()

    @property
    def offset_name(self) -> str | None:
        if len(self.offset_names) == 1:
            return self.offset_names[0]
        return None


@dataclass
class ParsedGAMFormula:
    response_name: str | None
    predictors: list[ParsedPredictorFormula]
    components: list[ParsedFormulaComponent] = field(default_factory=list)
    fake_formula: str = ""
    pred_formula: str = "~ 1"
    nlp: int = 1
    raw: Any = None


def strip_offset_wrapper(expr: str) -> str:
    expr = str(expr).strip()
    if expr.startswith("offset(") and expr.endswith(")"):
        return expr[7:-1].strip()
    return expr


def _contains_standalone_dot(rhs: str) -> bool:
    return re.search(r"(^|[~+\-(),=\s])\.([~+\-(),=\s]|$)", rhs) is not None


def _restore_formula_surface(expr: str) -> str:
    return str(expr).replace("@", ":").replace(_FORMULA_DOT_SENTINEL, ".")


def _replace_formula_colons(expr: str) -> str:
    out = expr.replace(":", "@")
    if _contains_standalone_dot(out):
        out = re.sub(
            r"(^|[~+\-(),=\s])\.([~+\-(),=\s]|$)",
            rf"\1{_FORMULA_DOT_SENTINEL}\2",
            out,
        )
    return out


def _parse_formula_expr(expr: str):
    return ast.parse(_replace_formula_colons(expr), mode="eval")


def _split_formula_text(formula: str) -> tuple[str | None, str]:
    text = str(formula).strip()
    if "~" not in text:
        return None, text
    lhs, rhs = text.split("~", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    return (lhs if lhs else None), rhs


def _source_segment(src: str, node, default: str):
    try:
        out = ast.get_source_segment(src, node)
        return _restore_formula_surface(out if out is not None else default)
    except Exception:
        return _restore_formula_surface(default)


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    return None


def _freeze_formula_value(value):
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_formula_value(v))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_formula_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_formula_value(v) for v in value)
    return value


def _ordered_unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _as_r_vector(value):
    if isinstance(value, np.ndarray):
        return list(np.asarray(value).ravel(order="F"))
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _r_matrix_value(node: ast.Call):
    args = [_ast_to_value(arg) for arg in node.args]
    kwargs = {kw.arg: _ast_to_value(kw.value) for kw in node.keywords if kw.arg}

    data = kwargs.pop("data", args[0] if args else np.nan)
    nrow = kwargs.pop("nrow", args[1] if len(args) > 1 else None)
    ncol = kwargs.pop("ncol", args[2] if len(args) > 2 else None)
    byrow = bool(kwargs.pop("byrow", args[3] if len(args) > 3 else False))
    kwargs.pop("dimnames", None)
    if kwargs:
        names = ", ".join(sorted(str(k) for k in kwargs))
        raise NotImplementedError(f"Unsupported matrix(...) argument(s): {names}.")

    values = np.asarray(_as_r_vector(data))
    if nrow is None and ncol is None:
        nrow = int(values.size)
        ncol = 1
    elif nrow is None:
        ncol = int(ncol)
        if ncol <= 0 or values.size % ncol != 0:
            raise ValueError("matrix(...) ncol must divide the data length.")
        nrow = int(values.size // ncol)
    elif ncol is None:
        nrow = int(nrow)
        if nrow <= 0 or values.size % nrow != 0:
            raise ValueError("matrix(...) nrow must divide the data length.")
        ncol = int(values.size // nrow)
    else:
        nrow = int(nrow)
        ncol = int(ncol)

    if nrow <= 0 or ncol <= 0:
        raise ValueError("matrix(...) dimensions must be positive.")
    if values.size != nrow * ncol:
        raise ValueError(
            "matrix(...) data recycling is not supported; provide exactly "
            "nrow * ncol values."
        )

    order: Literal["C", "F"] = "C" if byrow else "F"
    return values.reshape((nrow, ncol), order=order)


def _r_diag_value(node: ast.Call):
    args = [_ast_to_value(arg) for arg in node.args]
    kwargs = {kw.arg: _ast_to_value(kw.value) for kw in node.keywords if kw.arg}
    if kwargs:
        names = ", ".join(sorted(str(k) for k in kwargs))
        raise NotImplementedError(f"Unsupported diag(...) argument(s): {names}.")
    if len(args) != 1:
        raise NotImplementedError("Only diag(x) formula values are supported.")

    x = args[0]
    if isinstance(x, (int, np.integer)) and int(x) == x:
        return np.eye(int(x), dtype=np.float64)
    values = np.asarray(_as_r_vector(x))
    return np.diag(values)


def all_vars1(expr: str) -> tuple[str, ...]:
    """Mirror mgcv's all_vars1 on the supported Python formula surface."""

    if _contains_standalone_dot(str(expr)):
        return ()

    parsed = _parse_formula_expr(str(expr))
    names: list[str] = []

    def visit(node, *, is_func: bool = False):
        if isinstance(node, ast.Name):
            if not is_func and node.id not in {"True", "False", "None"}:
                names.append(node.id)
            return

        if isinstance(node, ast.Attribute):
            visit(node.value)
            names.append(node.attr)
            return

        if isinstance(node, ast.Call):
            visit(node.func, is_func=True)
            for arg in node.args:
                visit(arg)
            for kw in node.keywords:
                visit(kw.value)
            return

        if isinstance(node, ast.keyword):
            visit(node.value)
            return

        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(parsed.body)
    return tuple(_ordered_unique(names))


def get_numeric_response_labels(formula_or_lhs: str) -> tuple[int, ...]:
    """Mirror mgcv::getNumericResponse on the Python formula surface."""

    lhs, _rhs = _split_formula_text(str(formula_or_lhs))
    if lhs is None and "~" in str(formula_or_lhs):
        return ()
    lhs_text = str(lhs if lhs is not None else formula_or_lhs).strip()
    if lhs_text == "":
        return ()

    try:
        _parse_formula_expr(lhs_text)
    except SyntaxError:
        return ()

    has_vars = len(all_vars1(lhs_text)) > 0
    if has_vars:
        return ()

    nums = [int(tok) for tok in re.findall(r"[0-9]+", lhs_text)]
    return tuple(_ordered_unique(nums))


def _ast_to_value(node):
    if isinstance(node, ast.Name):
        name = str(node.id)
        if name == "TRUE":
            return True
        if name == "FALSE":
            return False
        return name
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_ast_to_value(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Call):
        func_name = _call_name(node.func)
        if func_name == "list":
            if node.args:
                if not node.keywords:
                    return [_ast_to_value(arg) for arg in node.args]
                # Mirror R's mixed list(...) shape while staying usable by current
                # Python smooth builders: unnamed entries keep source order under
                # integer keys, named entries keep their string keys.
                out: dict[Any, Any] = {}
                for i, arg in enumerate(node.args):
                    out[i] = _ast_to_value(arg)
                for kw in node.keywords:
                    if kw.arg is None:
                        expanded = _ast_to_value(kw.value)
                        if not isinstance(expanded, dict):
                            raise NotImplementedError(
                                "**kwargs style list(...) specification requires "
                                "a dictionary value."
                            )
                        for key, value in expanded.items():
                            out[key] = value
                        continue
                    out[kw.arg] = _ast_to_value(kw.value)
                return out
            if any(kw.arg is None for kw in node.keywords):
                out: dict[Any, Any] = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        expanded = _ast_to_value(kw.value)
                        if not isinstance(expanded, dict):
                            raise NotImplementedError(
                                "**kwargs style list(...) specification requires "
                                "a dictionary value."
                            )
                        for key, value in expanded.items():
                            out[key] = value
                    else:
                        out[kw.arg] = _ast_to_value(kw.value)
                return out
            return {kw.arg: _ast_to_value(kw.value) for kw in node.keywords}
        if func_name == "c":
            if node.keywords:
                raise NotImplementedError(
                    "keyword arguments to c(...) are not supported."
                )
            return [_ast_to_value(arg) for arg in node.args]
        if func_name == "matrix":
            return _r_matrix_value(node)
        if func_name == "diag":
            return _r_diag_value(node)
        raise NotImplementedError(
            "Unsupported formula value expression: "
            f"{ast.dump(node, include_attributes=False)}"
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
        "Unsupported formula value expression: "
        f"{ast.dump(node, include_attributes=False)}"
    )


def _ast_to_expr_label(node, rhs_src: str) -> str:
    return str(
        _source_segment(rhs_src, node, ast.dump(node, include_attributes=False))
    ).strip()


def _numeric_constant_value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _numeric_constant_value(node.operand)
        if value is not None:
            return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _numeric_constant_value(node.operand)
    return None


def _is_formula_binary(node) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Pow),
    )


def _is_offset_call(node) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) == "offset"


def _is_smooth_call(node) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) in {
        "s",
        "te",
        "ti",
    }


def _is_negative_formula_expr(node) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and _numeric_constant_value(node) is None
    )


def _canonical_term(
    factors: tuple[str, ...] | list[str], factor_order: dict[str, int]
) -> tuple[str, ...]:
    unique = _ordered_unique(str(factor) for factor in factors)
    return tuple(sorted(unique, key=lambda label: factor_order.get(label, 10**9)))


def _term_sort_key(term: tuple[str, ...], factor_order: dict[str, int]):
    return (len(term), tuple(factor_order.get(label, 10**9) for label in term))


def _unique_sorted_terms(
    terms: list[tuple[str, ...]], factor_order: dict[str, int]
) -> list[tuple[str, ...]]:
    seen: dict[tuple[str, ...], tuple[str, ...]] = {}
    for term in terms:
        canon = _canonical_term(term, factor_order)
        if len(canon) == 0:
            continue
        seen.setdefault(canon, canon)
    out = list(seen.values())
    out.sort(key=lambda term: _term_sort_key(term, factor_order))
    return out


def _term_interactions(
    left: list[tuple[str, ...]],
    right: list[tuple[str, ...]],
    factor_order: dict[str, int],
) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for left_term in left:
        for right_term in right:
            out.append(tuple(left_term) + tuple(right_term))
    return _unique_sorted_terms(out, factor_order)


def _term_union(
    *term_lists: list[tuple[str, ...]], factor_order: dict[str, int]
) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for term_list in term_lists:
        out.extend(term_list)
    return _unique_sorted_terms(out, factor_order)


def _term_difference(
    left: list[tuple[str, ...]],
    right: list[tuple[str, ...]],
    factor_order: dict[str, int],
) -> list[tuple[str, ...]]:
    right_keys = {tuple(term) for term in right}
    out = [term for term in left if tuple(term) not in right_keys]
    return _unique_sorted_terms(out, factor_order)


def _nested_left_term(
    left: list[tuple[str, ...]], factor_order: dict[str, int]
) -> tuple[str, ...] | None:
    if len(left) == 0:
        return None
    merged: list[str] = []
    for term in left:
        merged.extend(term)
    return _canonical_term(merged, factor_order)


def _formula_power(node) -> int:
    value = _numeric_constant_value(node)
    if value is None or value < 1:
        raise ValueError("invalid power in formula")
    power = int(value)
    if power < 1:
        raise ValueError("invalid power in formula")
    if power == 1:
        raise ValueError("invalid power in formula")
    return power


def _collect_factor_order_labels(node, rhs_src: str, labels: list[str]) -> None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _collect_factor_order_labels(node.left, rhs_src, labels)
        return

    if _is_formula_binary(node):
        _collect_factor_order_labels(node.left, rhs_src, labels)
        _collect_factor_order_labels(node.right, rhs_src, labels)
        return

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _numeric_constant_value(node)
        if value is None:
            _collect_factor_order_labels(node.operand, rhs_src, labels)
        return

    if _is_offset_call(node):
        return

    value = _numeric_constant_value(node)
    if value is not None:
        if value in {0.0, 1.0}:
            return
        raise ValueError("invalid model formula in ExtractVars")

    label = _ast_to_expr_label(node, rhs_src)
    if label == "":
        raise NotImplementedError("Empty formula factor encountered during parsing.")
    if label not in labels:
        labels.append(label)


def _collect_special_nodes(node, rhs_src: str, out: dict[str, ast.Call]) -> None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _collect_special_nodes(node.left, rhs_src, out)
        return

    if _is_formula_binary(node):
        _collect_special_nodes(node.left, rhs_src, out)
        _collect_special_nodes(node.right, rhs_src, out)
        return

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _numeric_constant_value(node)
        if value is None:
            _collect_special_nodes(node.operand, rhs_src, out)
        return

    if isinstance(node, ast.Call) and _call_name(node.func) == "t2":
        raise NotImplementedError(
            "t2(...) tensor product smooths are not supported; "
            "use te(...) or ti(...)."
        )

    if _is_smooth_call(node):
        label = _ast_to_expr_label(node, rhs_src)
        out.setdefault(label, node)


def _collect_offset_exprs(node, rhs_src: str, out: list[str]) -> None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _collect_offset_exprs(node.left, rhs_src, out)
        return

    if _is_formula_binary(node):
        _collect_offset_exprs(node.left, rhs_src, out)
        _collect_offset_exprs(node.right, rhs_src, out)
        return

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _numeric_constant_value(node)
        if value is None:
            _collect_offset_exprs(node.operand, rhs_src, out)
        return

    if _is_offset_call(node):
        expr, _label = _parse_offset_call(node, rhs_src)
        if expr not in out:
            out.append(expr)


def _expand_formula_terms(
    node,
    rhs_src: str,
    factor_order: dict[str, int],
) -> list[tuple[str, ...]]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _term_union(
            _expand_formula_terms(node.left, rhs_src, factor_order),
            _expand_formula_terms(node.right, rhs_src, factor_order),
            factor_order=factor_order,
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _term_difference(
            _expand_formula_terms(node.left, rhs_src, factor_order),
            _expand_formula_terms(node.right, rhs_src, factor_order),
            factor_order=factor_order,
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if _is_negative_formula_expr(node.left):
            return []
        left = _expand_formula_terms(node.left, rhs_src, factor_order)
        right = _expand_formula_terms(node.right, rhs_src, factor_order)
        return _term_union(
            left,
            right,
            _term_interactions(left, right, factor_order),
            factor_order=factor_order,
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left = _expand_formula_terms(node.left, rhs_src, factor_order)
        right = _expand_formula_terms(node.right, rhs_src, factor_order)
        return _term_interactions(left, right, factor_order)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if _is_negative_formula_expr(node.left):
            return []
        left = _expand_formula_terms(node.left, rhs_src, factor_order)
        right = _expand_formula_terms(node.right, rhs_src, factor_order)
        nested = _nested_left_term(left, factor_order)
        if nested is None:
            return left
        return _term_union(
            left,
            _term_interactions([nested], right, factor_order),
            factor_order=factor_order,
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        left = _expand_formula_terms(node.left, rhs_src, factor_order)
        power = _formula_power(node.right)
        if len(left) == 0:
            return []
        out: list[tuple[str, ...]] = []
        max_size = min(power, len(left))
        for size in range(1, max_size + 1):
            for combo in itertools.combinations(left, size):
                merged: list[str] = []
                for term in combo:
                    merged.extend(term)
                out.append(tuple(merged))
        return _unique_sorted_terms(out, factor_order)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _numeric_constant_value(node)
        if value is not None:
            return []
        return []

    if _is_offset_call(node):
        return []

    value = _numeric_constant_value(node)
    if value is not None:
        if value in {0.0, 1.0}:
            return []
        raise ValueError("invalid model formula in ExtractVars")

    label = _ast_to_expr_label(node, rhs_src)
    if label == "":
        raise NotImplementedError("Empty formula factor encountered during parsing.")
    return [(label,)]


def _evaluate_intercept(node, state: bool = True, sign: int = 1) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        state = _evaluate_intercept(node.left, state=state, sign=sign)
        return _evaluate_intercept(node.right, state=state, sign=sign)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        state = _evaluate_intercept(node.left, state=state, sign=sign)
        return _evaluate_intercept(node.right, state=state, sign=-sign)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _evaluate_intercept(node.left, state=state, sign=sign)

    if _is_formula_binary(node):
        state = _evaluate_intercept(node.left, state=state, sign=sign)
        return _evaluate_intercept(node.right, state=state, sign=sign)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _numeric_constant_value(node)
        if value is not None:
            if value == -1.0:
                return False if sign > 0 else True
            if value == 0.0:
                return True if sign > 0 else False
            return state
        return _evaluate_intercept(node.operand, state=state, sign=-sign)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _evaluate_intercept(node.operand, state=state, sign=sign)

    value = _numeric_constant_value(node)
    if value == 1.0:
        return True if sign > 0 else False
    if value == 0.0:
        return False if sign > 0 else True
    return state


def _smooth_label(kind: str, features: tuple[str, ...], textra: str | None) -> str:
    suffix = "" if textra is None else str(textra)
    return f"{kind}{suffix}({','.join(str(f) for f in features)})"


def _parse_smooth_call(node, rhs_src: str, textra: str | None):
    kind = _call_name(node.func)
    if kind not in {"s", "te", "ti"}:
        raise ValueError(f"Unknown smooth special {kind!r}.")

    features = tuple(_ast_to_expr_label(arg, rhs_src) for arg in node.args)
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise NotImplementedError(
                "**kwargs style smooth specification is not supported."
            )
        if kw.arg == "by":
            kwargs[kw.arg] = _ast_to_expr_label(kw.value, rhs_src)
        else:
            kwargs[kw.arg] = _ast_to_value(kw.value)

    raw_label = _source_segment(rhs_src, node, f"{kind}({', '.join(features)})")
    if re.search(rf"^{kind}\s*\(\s*\.", str(raw_label)):
        # The parser uses a temporary sentinel for formula dot shorthand. For
        # unsupported smooth-dot inputs such as s(.), source offsets can include
        # the closing parenthesis in the argument label; normalize it so the
        # builder can raise the intended mgcv-style unsupported error.
        features = tuple(
            "." if str(f).rstrip().rstrip(")") == "." else f for f in features
        )
    by_value = kwargs.get("by", None)
    by_variable = None if by_value is None else str(by_value)
    return ParsedSmoothTerm(
        kind=kind,
        features=features,
        kwargs=kwargs,
        raw_label=raw_label,
        label=_smooth_label(kind, features, textra),
        by_variable=by_variable,
    )


def _parse_offset_call(node, rhs_src: str) -> tuple[str, str]:
    if _call_name(node.func) != "offset":
        raise ValueError("Expected an offset(...) call.")
    if len(node.args) != 1 or len(node.keywords) != 0:
        raise NotImplementedError("offset(...) currently supports one expression.")
    expr = _ast_to_expr_label(node.args[0], rhs_src)
    return expr, f"offset({expr})"


def _smooth_term_key(term: ParsedSmoothTerm) -> str:
    frozen = _freeze_formula_value(term.kwargs)
    return json.dumps(
        [term.kind, list(term.features), frozen],
        default=str,
        sort_keys=True,
    )


def _build_formula_text(response_name: str | None, rhs: str) -> str:
    rhs_txt = str(rhs).strip()
    if response_name is None:
        return f"~{rhs_txt}" if rhs_txt else "~"
    return f"{response_name} ~ {rhs_txt}" if rhs_txt else f"{response_name} ~"


def _build_pf_text(
    response_name: str | None,
    parametric_labels: list[str],
    offset_exprs: list[str],
    intercept: bool,
) -> tuple[str, int]:
    items = list(parametric_labels) + [f"offset({expr})" for expr in offset_exprs]
    if len(items) == 0:
        rhs = "1" if intercept else "-1"
        return _build_formula_text(response_name, rhs), int(intercept)

    rhs = " + ".join(items)
    if not intercept:
        rhs = f"{rhs} - 1"
    return _build_formula_text(response_name, rhs), 1


def _build_fake_formula_text(
    response_name: str | None,
    parametric_labels: list[str],
    offset_exprs: list[str],
    smooth_terms: list[ParsedSmoothTerm],
    intercept: bool,
) -> str:
    base, _pfok = _build_pf_text(
        response_name,
        parametric_labels,
        offset_exprs,
        intercept,
    )
    fake_formula = base
    for term in smooth_terms:
        ff1 = " + ".join(str(feature) for feature in term.features)
        fake_formula = f"{fake_formula} + {ff1}"
        if term.by_variable is not None:
            fake_formula = f"{fake_formula} + {term.by_variable}"
    return fake_formula


def _build_pred_names(fake_names: list[str]) -> tuple[str, ...]:
    pred_names: list[str] = []
    for item in fake_names:
        stripped = str(strip_offset_wrapper(item)).strip()
        if stripped == "" or stripped.rstrip(")") == ".":
            continue
        pred_names.extend(all_vars1(stripped))
    return tuple(_ordered_unique(pred_names))


def _build_pred_formula(pred_names: tuple[str, ...]) -> str:
    if len(pred_names) == 0:
        return "~1"
    return f"~{' + '.join(pred_names)}"


def _parse_formula_component(
    formula: str,
    *,
    textra: str | None,
    lpi: tuple[int, ...],
    is_base_formula: bool,
) -> ParsedFormulaComponent:
    response_name, rhs = _split_formula_text(formula)
    rhs = rhs.strip()
    if rhs == "":
        rhs = "1"

    expr = _parse_formula_expr(rhs)
    rhs_src = _replace_formula_colons(rhs)
    factor_labels: list[str] = []
    _collect_factor_order_labels(expr.body, rhs_src, factor_labels)
    factor_order = {label: idx for idx, label in enumerate(factor_labels)}

    termsets = _expand_formula_terms(expr.body, rhs_src, factor_order)
    intercept = _evaluate_intercept(expr.body, state=True, sign=1)

    offset_exprs: list[str] = []
    _collect_offset_exprs(expr.body, rhs_src, offset_exprs)
    if len(offset_exprs) > 1:
        # Mirror mgcv::interpret.gam0 (mgcv/R/mgcv.r:387-389): every offset
        # label is assigned into the single slot av[kp], so base R keeps only
        # the first offset and raises this warning; model.offset() then never
        # sees the later terms. Verified against mgcv 1.9-4 in
        # debug/multi_offset_probe.R.
        warnings.warn(
            "number of items to replace is not a multiple of replacement length",
            stacklevel=2,
        )
        offset_exprs = offset_exprs[:1]

    special_nodes: dict[str, ast.Call] = {}
    _collect_special_nodes(expr.body, rhs_src, special_nodes)

    parametric_terms: list[ParsedParametricTerm] = []
    smooth_terms: list[ParsedSmoothTerm] = []

    for term in termsets:
        smooth_labels = [label for label in term if label in special_nodes]
        if smooth_labels:
            if len(term) != 1 or len(smooth_labels) != 1:
                raise NotImplementedError(
                    "Interactions involving smooth specials are not yet supported "
                    "beyond exact formula parsing."
                )

            smooth = _parse_smooth_call(special_nodes[smooth_labels[0]], rhs, textra)
            key = _smooth_term_key(smooth)
            if key not in [_smooth_term_key(existing) for existing in smooth_terms]:
                smooth_terms.append(smooth)
            continue

        raw_label = ":".join(str(label) for label in term)
        src_vars: list[str] = []
        for label in term:
            if label == ".":
                src_vars.append(".")
            else:
                src_vars.extend(all_vars1(label))

        parametric_terms.append(
            ParsedParametricTerm(
                variables=tuple(_ordered_unique(src_vars)),
                raw_label=raw_label,
                factor_labels=tuple(str(label) for label in term),
            )
        )

    parametric_labels = [term.raw_label for term in parametric_terms]
    fake_names = list(parametric_labels) + [f"offset({expr})" for expr in offset_exprs]
    for smooth in smooth_terms:
        fake_names.extend(str(feature) for feature in smooth.features)
        if smooth.by_variable is not None:
            fake_names.append(str(smooth.by_variable))

    pred_names = _build_pred_names(fake_names)
    pf, pfok = _build_pf_text(response_name, parametric_labels, offset_exprs, intercept)
    fake_formula = _build_fake_formula_text(
        response_name, parametric_labels, offset_exprs, smooth_terms, intercept
    )

    return ParsedFormulaComponent(
        response_name=response_name,
        intercept=bool(intercept),
        terms=[*parametric_terms, *smooth_terms],
        offset_names=tuple(offset_exprs),
        raw_formula=str(formula),
        pf=pf,
        pfok=int(pfok),
        fake_formula=fake_formula,
        fake_names=tuple(fake_names),
        pred_names=pred_names,
        pred_formula=_build_pred_formula(pred_names),
        lpi=tuple(int(v) for v in lpi),
        is_base_formula=bool(is_base_formula),
    )


def _insert_response_into_pf(pf: str, response_name: str | None) -> str:
    if response_name is None:
        return pf
    lhs, rhs = _split_formula_text(pf)
    if lhs is not None:
        return pf
    return _build_formula_text(response_name, rhs)


def _component_extra_response_name(component: ParsedFormulaComponent) -> str | None:
    if component.response_name is None or component.is_base_formula is False:
        return None
    return component.response_name


def _top_level_fake_formula(
    components: list[ParsedFormulaComponent], response_name: str | None
) -> str:
    av: list[str] = []
    for i, component in enumerate(components):
        av.extend(component.fake_names)
        if i > 0:
            extra_response = _component_extra_response_name(component)
            if extra_response is not None:
                av.append(extra_response)

    av = _ordered_unique(av)
    if len(av) == 0:
        return components[0].fake_formula
    return _build_formula_text(response_name, " + ".join(av))


def _top_level_pred_formula(components: list[ParsedFormulaComponent]) -> str:
    pred_names: list[str] = []
    for component in components:
        pred_names.extend(component.pred_names)
    pred_unique = tuple(_ordered_unique(pred_names))
    return _build_pred_formula(pred_unique)


def _aggregate_predictors(
    components: list[ParsedFormulaComponent], *, nlp: int
) -> list[ParsedPredictorFormula]:
    base_component_by_lp: dict[int, ParsedFormulaComponent] = {}
    for component in components:
        if component.is_base_formula and len(component.lpi) == 1:
            base_component_by_lp[int(component.lpi[0])] = component

    predictors: list[ParsedPredictorFormula] = []
    for lp in range(1, nlp + 1):
        base_component = base_component_by_lp.get(lp)
        response_name = None if base_component is None else base_component.response_name
        intercept = True if base_component is None else bool(base_component.intercept)
        terms: list[Any] = []
        offsets: list[str] = []
        source_component_indices: list[int] = []

        for i, component in enumerate(components):
            if lp in component.lpi:
                source_component_indices.append(i)
                terms.extend(component.terms)
                offsets.extend(component.offset_names)

        predictors.append(
            ParsedPredictorFormula(
                response_name=response_name,
                intercept=intercept,
                terms=list(terms),
                offset_names=tuple(_ordered_unique(offsets)),
                raw_formula=(
                    "" if base_component is None else str(base_component.raw_formula)
                ),
                source_component_indices=tuple(source_component_indices),
            )
        )

    return predictors


def parse_gam_formula(formula) -> ParsedGAMFormula:
    if isinstance(formula, str):
        component = _parse_formula_component(
            formula,
            textra=None,
            lpi=(),
            is_base_formula=True,
        )
        predictors = [
            ParsedPredictorFormula(
                response_name=component.response_name,
                intercept=bool(component.intercept),
                terms=list(component.terms),
                offset_names=tuple(component.offset_names),
                raw_formula=str(formula),
                source_component_indices=(0,),
            )
        ]
        return ParsedGAMFormula(
            response_name=component.response_name,
            predictors=predictors,
            components=[component],
            fake_formula=component.fake_formula,
            pred_formula=component.pred_formula,
            nlp=1,
            raw=formula,
        )

    if isinstance(formula, (list, tuple)):
        if len(formula) == 0:
            raise ValueError("Formula list must not be empty.")

        components: list[ParsedFormulaComponent] = []
        response_name: str | None = None
        nlp = 0

        for i, component_formula in enumerate(formula, start=1):
            lhs, rhs = _split_formula_text(str(component_formula))
            textra = None if i == 1 else f".{i-1}"
            explicit_lpi = get_numeric_response_labels(str(component_formula))
            if len(explicit_lpi) == 1:
                warnings.warn(
                    "single linear predictor indices are ignored",
                    stacklevel=2,
                )

            if len(explicit_lpi) > 0:
                if rhs == "":
                    rhs = "1"
                parse_formula = _build_formula_text(None, rhs)
                component = _parse_formula_component(
                    parse_formula,
                    textra=textra,
                    lpi=explicit_lpi,
                    is_base_formula=False,
                )
            else:
                nlp += 1
                component = _parse_formula_component(
                    str(component_formula),
                    textra=textra,
                    lpi=(nlp,),
                    is_base_formula=True,
                )
                if response_name is None and component.response_name is not None:
                    response_name = component.response_name

            components.append(component)

        if response_name is None and len(components) > 0:
            response_name = components[0].response_name

        for component in components:
            component.pf = _insert_response_into_pf(component.pf, response_name)
            if len(component.lpi) > 0:
                if min(component.lpi) < 1 or max(component.lpi) > nlp:
                    raise ValueError("linear predictor labels out of range")

        predictors = _aggregate_predictors(components, nlp=nlp)
        return ParsedGAMFormula(
            response_name=response_name,
            predictors=predictors,
            components=components,
            fake_formula=_top_level_fake_formula(components, response_name),
            pred_formula=_top_level_pred_formula(components),
            nlp=nlp,
            raw=formula,
        )

    raise TypeError("formula must be a string or a list/tuple of strings.")


__all__ = [
    "ParsedParametricTerm",
    "ParsedSmoothTerm",
    "ParsedFormulaComponent",
    "ParsedPredictorFormula",
    "ParsedGAMFormula",
    "all_vars1",
    "get_numeric_response_labels",
    "parse_gam_formula",
    "strip_offset_wrapper",
]
