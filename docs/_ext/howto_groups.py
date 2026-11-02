"""Group related how-to guides into plus-menus without dropping their text.

The source files stay where they are so relative includes keep working. This
directive reads one source, turns its section headings into rubrics, and
parses the result as a ``dropdown``. The source path is excluded from the
document set; ``page_redirects`` sends the old URL to the menu.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from docutils import nodes
from docutils.nodes import make_id
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

# Page -> source documents, in reading order. Titles come from each source.
GROUPS = {
    "howto/fit_first_model": [
        "examples/fit_models",
        "user_guide/preprocessing",
        "user_guide/formula_visual_reference",
        "user_guide/api_and_families",
    ],
    "howto/fit_and_diagnose": [
        "user_guide/model_workflow",
        "user_guide/statistical_gams",
    ],
    "howto/read_the_fit": [
        "user_guide/interpretability",
        "howto/prediction_outputs",
        "user_guide/fisher_diagnostics",
    ],
    "howto/calibrate": [
        "user_guide/calibration",
        "user_guide/quantile_inference",
        "user_guide/calibration_shift",
        "user_guide/calibration_groups",
        "user_guide/calibration_survival",
        "user_guide/calibration_contract",
    ],
    "howto/change_predictor": [
        "howto/offsets",
        "user_guide/shape_constrained_gams",
        "howto/interactions",
    ],
    "howto/location_and_spread": [
        "user_guide/gamlss",
        "user_guide/distributional_families",
        "user_guide/distributional_observations",
        "user_guide/distributional_boxcox",
        "user_guide/distributional_comparison",
        "user_guide/distributional_diagnostics",
        "user_guide/distributional_probabilities",
    ],
    "howto/survival": [
        "user_guide/survival",
        "user_guide/survival_likelihoods",
        "user_guide/survival_cox",
        "user_guide/survival_longitudinal",
        "user_guide/survival_evaluation",
        "user_guide/survival_uncertainty",
        "user_guide/survival_calibration",
    ],
    "howto/neural": [
        "user_guide/training",
        "user_guide/semistructured",
        "user_guide/custom_models",
    ],
}

FOLDED_DOCNAMES = [doc for docs in GROUPS.values() for doc in docs]
_DOC_TO_PAGE = {doc: page for page, docs in GROUPS.items() for doc in docs}

# Heading ids that legacy URLs address, and that a rubric would otherwise drop.
_HEADING_ANCHORS = {
    "predict-a-continuous-response",
    "predict-class-probabilities",
    "predict-a-response-distribution",
    "compare-an-unconstrained-and-monotone-fit",
}

_ADORN = set("=-~^\"'`:+#*")
_DOC_ROLE = re.compile(r":doc:`([^`]+)`", re.DOTALL)


def dropdown_name(docname: str) -> str:
    return "howto-" + docname.replace("/", "-").replace("_", "-")


def page_redirects() -> dict[str, str]:
    return {
        doc: f"{page}#{dropdown_name(doc)}" for doc, page in _DOC_TO_PAGE.items()
    }


def _is_adornment(line: str, title: str) -> bool:
    if not line or not title or line[0] not in _ADORN:
        return False
    return len(line) >= len(title) and set(line) == {line[0]}


def _rewrite_doc_roles(
    text: str, source: Path, srcdir: Path, *, absolutize: bool = False
) -> str:
    def resolve(raw: str) -> str | None:
        raw = raw.strip()
        if raw.startswith("<") or raw.startswith("{"):
            return None
        if raw.startswith("/"):
            return posixpath.normpath(raw[1:])
        if source.suffix not in {".rst", ".inc"}:
            return None
        try:
            base = source.parent.relative_to(srcdir).as_posix()
        except ValueError:
            return None
        joined = raw if base == "." else posixpath.join(base, raw)
        return posixpath.normpath(joined)

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        label = None
        target = inner
        if inner.endswith(">") and "<" in inner:
            label, target = inner[:-1].rsplit("<", 1)
            label = label.strip()
            target = target.strip()
        path, _, fragment = target.partition("#")
        docname = resolve(path)
        if docname in _DOC_TO_PAGE:
            ref = dropdown_name(docname)
            if label:
                return f":ref:`{label} <{ref}>`"
            return f":ref:`{ref}`"
        if absolutize and docname and not path.strip().startswith("/"):
            absolute = "/" + docname + (f"#{fragment}" if fragment else "")
            if label:
                return f":doc:`{label} <{absolute}>`"
            return f":doc:`{absolute}`"
        return match.group(0)

    return _DOC_ROLE.sub(replace, text)


def _without_headings(lines: list[str]) -> list[str]:
    """Turn section titles into rubrics so they can live inside a dropdown."""
    body: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(".. toctree::"):
            index += 1
            while index < len(lines) and (
                lines[index].strip() == "" or lines[index][:1].isspace()
            ):
                index += 1
            continue
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            line
            and not line[:1].isspace()
            and not line.startswith("..")
            and _is_adornment(nxt, line)
        ):
            slug = make_id(line)
            if slug in _HEADING_ANCHORS:
                body.append(f".. _{slug}:")
                body.append("")
            body.append(f".. rubric:: {line}")
            body.append("")
            index += 2
            continue
        body.append(line)
        index += 1
    return body


_INCLUDE = re.compile(r"^(\s*)\.\. include::\s+(\S+)\s*$")


def _expand_includes(
    lines: list[str], origin: Path, srcdir: Path, stack: set[Path]
) -> list[str]:
    expanded: list[str] = []
    for line in lines:
        match = _INCLUDE.match(line)
        if match is None:
            expanded.append(line)
            continue
        indent, raw = match.group(1), match.group(2)
        path = srcdir / raw.lstrip("/") if raw.startswith("/") else (origin.parent / raw).resolve()
        if path in stack:
            raise ValueError(f"include cycle at {path}")
        included = _rewrite_doc_roles(
            path.read_text(), path, srcdir, absolutize=True
        ).splitlines()
        included = _expand_includes(included, path, srcdir, stack | {path})
        included = _without_headings(included)
        for item in included:
            expanded.append("" if item.strip() == "" else indent + item)
    return expanded


def _dropdown_rst(text: str, docname: str, source: Path, srcdir: Path) -> tuple[str, list[str]]:
    text = _rewrite_doc_roles(text, source, srcdir, absolutize=True)
    lines = text.splitlines()
    labels: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(".. _") and line.rstrip().endswith(":"):
            labels.append(line)
            index += 1
            continue
        if line.strip() == "":
            index += 1
            continue
        break
    if index >= len(lines):
        raise ValueError(f"{source} has no title")
    title = lines[index]
    underline = lines[index + 1] if index + 1 < len(lines) else ""
    if _is_adornment(underline, title):
        index += 2
    elif (
        set(title) == {title[0]}
        and title[0] in _ADORN
        and index + 2 < len(lines)
        and _is_adornment(lines[index + 2], lines[index + 1])
    ):
        title = lines[index + 1]
        index += 3
    else:
        raise ValueError(f"{source} does not start with a section title")
    if index < len(lines) and lines[index].strip() == "":
        index += 1

    body = _expand_includes(_without_headings(lines[index:]), source, srcdir, {source.resolve()})
    indented = ["" if item.strip() == "" else f"   {item}" for item in body]
    name = dropdown_name(docname)
    generated = (
        f".. dropdown:: {title}\n   :icon: plus\n   :name: {name}\n\n"
        + "\n".join(indented).rstrip()
        + "\n"
    )
    return generated, labels


class HowToDropdown(SphinxDirective):
    required_arguments = 1
    final_argument_whitespace = True
    has_content = False

    def run(self) -> list[nodes.Node]:
        srcdir = Path(self.env.srcdir)
        relative = self.arguments[0].strip().lstrip("/")
        source = srcdir / relative
        docname = relative[: -len(".rst")] if relative.endswith(".rst") else relative
        if docname not in _DOC_TO_PAGE:
            raise self.error(f"{relative} is not a grouped how-to source")
        self.env.note_dependency(str(source))
        generated, labels = _dropdown_rst(source.read_text(), docname, source, srcdir)
        wrapper = nodes.Element()
        self.state.nested_parse(
            StringList(generated.splitlines(), source=str(source)), 0, wrapper
        )
        dropdown = next(
            (
                child
                for child in wrapper.children
                if child.get("design_component") == "dropdown"
            ),
            None,
        )
        if dropdown is not None and labels and dropdown.children:
            title = dropdown[0]
            if not title.get("ids"):
                self.state.document.set_id(title)
            refid = title["ids"][0]
            for raw in labels:
                name = nodes.fully_normalize_name(raw[4:].rstrip()[:-1].strip())
                target = nodes.target()
                target["names"].append(name)
                target["refid"] = refid
                dropdown.append(target)
                self.state.document.note_explicit_target(target)
        return wrapper.children


def setup(app):
    app.add_directive("howto-dropdown", HowToDropdown)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def rewrite_sources(root: Path | None = None) -> list[Path]:
    """Point :doc: roles at the grouped menus. Safe to run more than once."""
    root = root or Path(__file__).resolve().parents[1]
    changed: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or "_build" in path.parts:
            continue
        if path.suffix not in {".rst", ".py", ".inc"}:
            continue
        original = path.read_text()
        updated = _rewrite_doc_roles(original, path, root)
        if updated != original:
            path.write_text(updated)
            changed.append(path)
    return changed
