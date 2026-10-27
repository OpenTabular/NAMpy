"""Runnable reference example: api/generated/nampy.gam.formula.strip_offset_wrapper."""

from nampy.gam.formula import (
    parse_gam_formula, extract_formula_terms, all_vars1, get_numeric_response_labels,
    strip_offset_wrapper, ParsedParametricTerm, ParsedSmoothTerm,
    ExtractedParametricTerm, ExtractedSmoothTerm,
)

parsed = parse_gam_formula("y ~ z + s(x, bs='cr', k=6) + offset(exposure)")
extracted = extract_formula_terms(parsed)
result = strip_offset_wrapper('offset(exposure)')
print(result)
