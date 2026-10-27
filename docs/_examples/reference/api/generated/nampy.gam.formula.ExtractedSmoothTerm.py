"""Runnable reference example: api/generated/nampy.gam.formula.ExtractedSmoothTerm."""

from nampy.gam.formula import (
    parse_gam_formula, extract_formula_terms, all_vars1, get_numeric_response_labels,
    strip_offset_wrapper, ParsedParametricTerm, ParsedSmoothTerm,
    ExtractedParametricTerm, ExtractedSmoothTerm,
)

parsed = parse_gam_formula("y ~ z + s(x, bs='cr', k=6) + offset(exposure)")
extracted = extract_formula_terms(parsed)
result = next(term for term in extracted[0].terms if isinstance(term, ExtractedSmoothTerm))
print(result)
