"""Accessible HTML result summaries, with type/shape metadata beside values."""

import html
import math
import re


def escape(value):
    return html.escape(str(value), quote=True)


def number(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".5g")
    return str(value)


def dimensions(shape):
    if isinstance(shape, list):
        return " × ".join(map(str, shape)) or "scalar"
    if shape == "()":
        return "scalar"
    if re.fullmatch(r"\([\d, ]+\)", str(shape)):
        return " × ".join(re.findall(r"\d+", str(shape)))
    return str(shape)


def short_type(value):
    return re.sub(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)", r"\1", value)


def structured(value):
    return isinstance(value, dict) and (
        (
            value.get("kind") in ("array", "table")
            and "shape" in value
            and "values" in value
        )
        or (
            value.get("kind") in ("list", "tuple")
            and "items" in value
            and "length" in value
        )
        or ("fields" in value and "omitted_fields" in value)
    )


def schema_index(rows):
    return {path: (short_type(kind), dimensions(shape)) for path, kind, shape in rows}


def sample_values(value, columns=None):
    """Render a vector or a matrix sample without flattening its axes."""
    if not isinstance(value, list):
        return (
            '<p class="example-result-note">Values omitted; dimensions shown above.</p>'
        )
    if not value:
        return '<p class="example-result-note">Empty array.</p>'
    if not isinstance(value[0], list):
        return (
            '<div class="example-result-values">'
            + "".join(
                '<span class="example-result-value">' + escape(number(v)) + "</span>"
                for v in value
            )
            + "</div>"
        )
    rows = []
    for index, row in enumerate(value):
        cells = []
        for col, item in enumerate(row):
            label = (
                columns[col] if columns and col < len(columns) else f"column {col + 1}"
            )
            body = (
                sample_values(item)
                if isinstance(item, list)
                else "<span>" + escape(number(item)) + "</span>"
            )
            cells.append(
                '<div class="example-result-coordinate"><dt>'
                + escape(label)
                + "</dt><dd>"
                + body
                + "</dd></div>"
            )
        rows.append(
            '<li><span class="example-result-row-label">Row '
            + str(index + 1)
            + '</span><dl class="example-result-coordinates">'
            + "".join(cells)
            + "</dl></li>"
        )
    return '<ol class="example-result-rows">' + "".join(rows) + "</ol>"


def value_html(value, path, schemas, depth=0):
    if not isinstance(value, dict):
        if isinstance(value, list):
            return sample_values(value)
        return (
            '<span class="example-result-scalar">' + escape(number(value)) + "</span>"
        )
    if depth > 3:
        return '<p class="example-result-note">See the returned object for further detail.</p>'
    kind = value.get("kind") if structured(value) else None
    if kind in {"array", "table"}:
        result = sample_values(value.get("values"), value.get("columns"))
        shape = value.get("shape", [])
        omitted = value.get("omitted_by_axis", [])
        if omitted and any(omitted):
            shown = [size - hidden for size, hidden in zip(shape, omitted, strict=True)]
            if len(shape) == 1:
                note = f"Showing {shown[0]} of {shape[0]} values."
            else:
                note = f"Showing {shown[0]} of {shape[0]} rows"
                if omitted[1]:
                    note += f" and {shown[1]} of {shape[1]} columns"
                if len(shape) > 2:
                    note += (
                        "; sampled extent "
                        + dimensions(shown)
                        + " of "
                        + dimensions(shape)
                    )
                note += "."
            result += '<p class="example-result-note">' + escape(note) + "</p>"
        return result
    if kind in {"list", "tuple"}:
        items = value.get("items", [])
        result = (
            '<ol class="example-result-sequence">'
            + "".join(
                "<li>"
                + value_html(item, f"{path}[{index}]", schemas, depth + 1)
                + "</li>"
                for index, item in enumerate(items)
            )
            + "</ol>"
        )
        if value.get("omitted"):
            result += (
                '<p class="example-result-note">Showing '
                + str(len(items))
                + " of "
                + str(value["length"])
                + " items.</p>"
            )
        return result
    fields = (
        value.get("fields", {})
        if "fields" in value and kind
        else {k: v for k, v in value.items() if k != "omitted_fields"}
    )
    result = '<dl class="example-result-fields">'
    for name, item in fields.items():
        child = f"{path}.{name}" if "fields" in value and kind else f"{path}[{name!r}]"
        meta = metadata(item, child, schemas)
        result += (
            "<div><dt>"
            + escape(name)
            + meta
            + "</dt><dd>"
            + value_html(item, child, schemas, depth + 1)
            + "</dd></div>"
        )
    result += "</dl>"
    omitted = value.get("omitted_fields", [])
    if omitted:
        result += (
            '<p class="example-result-note">'
            + str(len(omitted))
            + " additional fields available on the returned object.</p>"
        )
    return result


def metadata(value, path, schemas):
    kind, shape = schemas.get(path, ("", ""))
    if isinstance(value, dict):
        shape = dimensions(value["shape"]) if "shape" in value else shape
        if not kind:
            kind = value.get("kind", "record")
    text = " · ".join(part for part in (kind, shape) if part)
    return (
        '<span class="example-result-meta">' + escape(text) + "</span>" if text else ""
    )


def result_rst(preview, rows):
    schemas = schema_index(rows)
    if isinstance(preview, dict) and not structured(preview):
        items = [
            (name, value, f"result[{name!r}]")
            for name, value in preview.items()
            if name != "omitted_fields"
        ]
        # Natural outputs lead; raw network predictors remain inspectable.
        items.sort(key=lambda item: item[0] == "raw")
    else:
        items = [("result", preview, "result")]
    cards = []
    for name, value, path in items:
        complex_value = isinstance(value, dict) and value.get("kind") not in {
            "array",
            "table",
        }
        collapsed = name == "raw" or (complex_value and name != "result")
        tag = "details" if collapsed else "section"
        heading = "summary" if collapsed else "div"
        cards.append(
            f'<{tag} class="example-result-item">'
            + f'<{heading} class="example-result-heading"><span class="example-result-label">'
            + escape(name)
            + "</span>"
            + metadata(value, path, schemas)
            + f"</{heading}>"
            + '<div class="example-result-body">'
            + value_html(value, path, schemas)
            + "</div>"
            + f"</{tag}>"
        )
    omitted = preview.get("omitted_fields", []) if isinstance(preview, dict) else []
    footer = (
        '<p class="example-result-footnote">Sample values are rounded for display.</p>'
    )
    if omitted:
        footer += (
            '<p class="example-result-note">'
            + str(len(omitted))
            + " additional output fields are available in the returned result.</p>"
        )
    rendered = (
        '<div class="example-result" aria-label="Example result">'
        + "".join(cards)
        + footer
        + "</div>"
    )
    return (
        "\n\nExample result\n~~~~~~~~~~~~~~\n\n.. raw:: html\n\n   " + rendered + "\n\n"
    )
