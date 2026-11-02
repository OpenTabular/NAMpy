"""Build an Rd-style reference manual for the public NAMpy interfaces.

The CRAN mgcv manual is one PDF of help topics: a name and description, then
usage, arguments, details, and examples. This script reads the generated
reference pages and writes that kind of manual. Narrative tutorials stay in
the separate NAMpy.pdf book.

Run from the repository root:

    python docs/scripts/build_reference_manual.py
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
LOGO_SVG = DOCS / "_static" / "logo-sidebar-refined.svg"
ROOTS = (
    DOCS / "statistical_reference" / "index.rst",
    DOCS / "neural_reference" / "index.rst",
    DOCS / "extension_reference" / "index.rst",
)
SECTION_TITLES = {
    "Signature": "Usage",
    "Parameters": "Arguments",
    "Returns": "Value",
    "Runnable example": "Examples",
}
UNDERLINE = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5}
PREFERRED_COLUMNS = (
    "description",
    "meaning",
    "column meaning",
    "main distinction",
    "value",
)
COLUMN_LABELS = {
    "type / input shape": "type",
    "default": "default",
    "shape": "shape",
    "inputs": "inputs",
    "value": "value",
}


@dataclass
class Paragraph:
    text: str


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Code:
    text: str
    language: str = ""


@dataclass
class Math:
    text: str


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass
class Toc:
    entries: list[str]


@dataclass
class Page:
    title: str
    path: Path
    items: list = field(default_factory=list)

    @property
    def is_topic(self) -> bool:
        return not any(isinstance(item, Page) for item in self.items)


def package_version() -> str:
    root = str(DOCS.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import nampy

    return nampy.__version__


def resolve_doc(current: Path, target: str) -> Path:
    target = target.strip().split("<")[-1].rstrip(">").strip()
    path = DOCS / target[1:] if target.startswith("/") else current.parent / target
    if path.suffix != ".rst":
        path = path.parent / f"{path.name}.rst"
    return path.resolve()


def resolve_source(current: Path, target: str) -> Path:
    target = target.strip()
    path = DOCS / target[1:] if target.startswith("/") else current.parent / target
    return path.resolve()


def _directive(line: str):
    match = re.match(r"\.\. +(\S.*?)::[ \t]*(.*)$", line)
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _read_directive(lines: list[str], index: int):
    parsed = _directive(lines[index])
    if parsed is None:
        return index + 1, None
    name, argument = parsed
    index += 1
    options = {}
    while index < len(lines):
        match = re.match(r"   :([^:]+):[ \t]*(.*)$", lines[index])
        if match is None:
            break
        options[match.group(1).strip()] = match.group(2).strip()
        index += 1
    if (
        index < len(lines)
        and lines[index].strip() == ""
        and index + 1 < len(lines)
        and lines[index + 1].startswith(" ")
    ):
        index += 1
    body = []
    if index < len(lines) and lines[index].startswith(" "):
        indent = len(lines[index]) - len(lines[index].lstrip(" "))
        while index < len(lines):
            line = lines[index]
            if line.strip() == "":
                body.append("")
                index += 1
                continue
            if line.startswith(" " * indent):
                body.append(line[indent:])
                index += 1
                continue
            break
        while body and body[-1] == "":
            body.pop()
    return index, (name, argument, options, body)


def _parse_table(body: list[str]) -> Table:
    rows: list[list[str]] = []
    for line in body:
        if not line.strip():
            continue
        if line.startswith("* - "):
            rows.append([line[4:].strip()])
        elif line.startswith("  - "):
            if not rows:
                raise ValueError(f"table cell without a row: {line}")
            rows[-1].append(line[4:].strip())
        elif rows and rows[-1]:
            rows[-1][-1] = f"{rows[-1][-1]} {line.strip()}"
    if not rows:
        return Table([], [])
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    return Table(padded[0], padded[1:])


def _toc_entries(body: list[str]) -> list[str]:
    return [
        line.strip()
        for line in body
        if line.strip() and not line.strip().startswith(":")
    ]


def parse_blocks(text: str, source: Path, stack: tuple[Path, ...]) -> list:
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.strip() == ":orphan:":
            index += 1
            continue
        if line.startswith(".."):
            index, directive = _read_directive(lines, index)
            if directive is None:
                continue
            name, argument, _options, body = directive
            if name == "include":
                included = resolve_source(source, argument)
                if included in stack:
                    raise ValueError(f"include cycle at {included}")
                blocks.extend(
                    parse_blocks(included.read_text(), included, stack + (included,))
                )
            elif name in {"code-block", "code", "sourcecode"}:
                blocks.append(Code("\n".join(body), argument.strip()))
            elif name == "math":
                formula = argument if argument and not body else "\n".join(body)
                blocks.append(Math(formula))
            elif name == "list-table":
                blocks.append(_parse_table(body))
            elif name == "literalinclude":
                included = resolve_source(source, argument)
                language = "python" if included.suffix == ".py" else ""
                blocks.append(Code(included.read_text().rstrip("\n"), language))
            elif name == "toctree":
                entries = _toc_entries(body)
                if entries:
                    blocks.append(Toc(entries))
            continue
        if (
            index + 1 < len(lines)
            and not line.startswith(" ")
            and _underline_level(lines[index + 1], line) is not None
        ):
            blocks.append(Heading(_underline_level(lines[index + 1], line), line.strip()))
            index += 2
            continue
        if line.startswith("* ") or line.startswith("- "):
            items = []
            while index < len(lines) and (
                lines[index].startswith("* ") or lines[index].startswith("- ")
            ):
                item = [lines[index][2:].strip()]
                index += 1
                while index < len(lines) and lines[index].startswith("  ") and lines[index].strip():
                    item.append(lines[index].strip())
                    index += 1
                items.append(" ".join(item))
            blocks.append(Bullets(items))
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not _starts_block(lines, index):
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append(Paragraph(" ".join(paragraph)))
    return blocks


def _underline_level(underline: str, title: str):
    token = underline.strip()
    if len(token) < 3 or len(set(token)) != 1 or token[0] not in UNDERLINE:
        return None
    if len(token) < len(title.strip()):
        return None
    return UNDERLINE[token[0]]


def _starts_block(lines: list[str], index: int) -> bool:
    line = lines[index]
    if line.startswith("..") or line.startswith("* ") or line.startswith("- "):
        return True
    return (
        index + 1 < len(lines)
        and not line.startswith(" ")
        and _underline_level(lines[index + 1], line) is not None
    )


def _import_object(qualname: str):
    module_name, _, attribute = qualname.rpartition(".")
    while module_name:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_name, separator, nested = module_name.rpartition(".")
            if not separator:
                raise
            attribute = f"{nested}.{attribute}"
            continue
        object_ = module
        for part in attribute.split("."):
            object_ = getattr(object_, part)
        return object_
    raise ModuleNotFoundError(qualname)


def _page_from_object(qualname: str, path: Path) -> Page:
    """Autosummary targets are generated during the HTML build."""
    object_ = _import_object(qualname)
    name = getattr(object_, "__name__", qualname.rsplit(".", 1)[-1])
    try:
        signature = str(inspect.signature(object_))
    except (TypeError, ValueError):
        signature = ""
    items = []
    if signature:
        items.append(Heading(2, "Usage"))
        items.append(Code(f"{name}{signature}"))
    docstring = inspect.getdoc(object_) or ""
    if docstring:
        items.append(Paragraph(docstring.split("\n\n", 1)[0].replace("\n", " ")))
        rest = docstring.split("\n\n", 1)[1] if "\n\n" in docstring else ""
        if rest.strip():
            items.append(Code(rest.strip()))
    return Page(name, path, items)


def parse_page(path: Path, seen: set[Path]) -> Page | None:
    path = path.resolve()
    if path in seen:
        return None
    if not path.is_file():
        qualname = path.name[: -len(".rst")] if path.name.endswith(".rst") else path.name
        if qualname.startswith("nampy."):
            seen.add(path)
            return _remember(_page_from_object(qualname, path))
        raise FileNotFoundError(path)
    seen.add(path)
    blocks = parse_blocks(path.read_text(), path, (path,))
    title = path.stem
    if blocks and isinstance(blocks[0], Heading):
        title = blocks[0].text
        blocks = blocks[1:]
    items = []
    for block in blocks:
        if not isinstance(block, Toc):
            items.append(block)
            continue
        for entry in block.entries:
            child = parse_page(resolve_doc(path, entry), seen)
            if child is not None:
                items.append(child)
    return _remember(Page(title, path, items))


DOC_PAGES: dict[str, tuple[str, str]] = {}
_CURRENT_PAGE: Path | None = None
_EXAMPLE_BANNER = re.compile(
    r'"""Runnable reference example: (?P<path>.+?)\."""[ \t]*(?:\r?\n|\Z)'
)


def _anchor(page: Page) -> str:
    relative = page.path.relative_to(DOCS).as_posix()
    return "ref-" + re.sub(r"[^A-Za-z0-9]+", "-", relative).strip("-")


def _remember(page: Page) -> Page:
    """Index a page so a documentation path can become a phrase link."""
    relative = page.path.relative_to(DOCS).as_posix()
    stem = relative[:-4] if relative.endswith(".rst") else relative
    record = (page.title, _anchor(page))
    DOC_PAGES[stem] = record
    if stem.endswith("/index"):
        DOC_PAGES[stem[: -len("/index")]] = record
    return page


def build_manual() -> list[Page]:
    seen: set[Path] = set()
    DOC_PAGES.clear()
    pages = []
    for root in ROOTS:
        page = parse_page(root, seen)
        if page is not None:
            pages.append(page)
    return pages


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CODE_TOKEN = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\n|."
)


def code_span(text: str) -> str:
    """Set one inline code span, quotation marks included, in typewriter."""
    pieces = []
    for token in _CODE_TOKEN.findall(text):
        pieces.append(escape_latex(token).replace(r"\_", r"\_\allowbreak{}"))
        if token in ".,|/:":
            pieces.append(r"\allowbreak{}")
    return r"\texttt{" + "".join(pieces) + "}"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _typographic_quotes(text: str) -> str:
    """Pair straight quotes. Apostrophes inside words stay apostrophes."""
    opening_before = set(" \t\n({[")
    pieces = []
    for index, char in enumerate(text):
        if char not in {"'", '"'}:
            pieces.append(char)
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if char == "'" and previous.isalpha() and following.isalpha():
            pieces.append("'")
            continue
        if previous in opening_before or previous == "":
            pieces.append("``" if char == '"' else "`")
        else:
            pieces.append("''" if char == '"' else "'")
    return "".join(pieces)


def latex_inline(text: str) -> str:
    """Convert the inline markup used by the reference pages."""
    pieces: list[str] = []

    def take(pattern: str, repl, source: str) -> str:
        def substitute(match: re.Match) -> str:
            pieces.append(repl(match))
            return f"\ue000{len(pieces) - 1}\ue001"

        return re.sub(pattern, substitute, source)

    text = take(r"``(.+?)``", lambda match: code_span(match.group(1)), text)
    text = take(
        r":math:`([^`]*)`",
        lambda match: f"${match.group(1)}$",
        text,
    )
    text = take(
        r":(doc|ref|download):`(?:([^`<]+)\s*<([^`>]+)>|([^`]+))`",
        lambda match: _linked_role(
            match.group(1),
            match.group(2),
            match.group(4),
        ),
        text,
    )
    text = take(
        r":(?:mod|class|func|meth|attr|py:mod|py:class|py:func|py:meth):`([^`]*)`",
        lambda match: code_span(match.group(1)),
        text,
    )
    text = take(
        r"\*\*(.+?)\*\*",
        lambda match: rf"\textbf{{{escape_latex(match.group(1))}}}",
        text,
    )
    text = take(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        lambda match: rf"\emph{{{escape_latex(match.group(1))}}}",
        text,
    )
    text = take(
        r"`([^`<]+)\s*<([^`>]+)>`_",
        lambda match: rf"\href{{{match.group(2).strip()}}}{{{escape_latex(match.group(1).strip())}}}",
        text,
    )
    text = take(
        r"https?://[^\s<>]+",
        lambda match: rf"\url{{{match.group(0).rstrip('.,)')}}}",
        text,
    )
    text = take(r"`([^`]+)`", lambda match: code_span(match.group(1)), text)

    rendered = escape_latex(_typographic_quotes(text))
    return re.sub(
        r"\ue000(\d+)\ue001",
        lambda match: pieces[int(match.group(1))],
        rendered,
    )


def _role_text(visible: str) -> str:
    """A short code name stays typewriter; a phrase stays in the text font."""
    visible = visible.strip()
    if _IDENTIFIER.match(visible):
        return code_span(visible)
    return escape_latex(visible.replace("-", " "))


def _linked_role(kind: str, title: str | None, bare: str | None) -> str:
    """Use an authored link title, or a phrase that names the documentation page."""
    if title:
        return _role_text(title)
    target = (bare or "").strip()
    if kind != "doc":
        return _role_text(target)
    resolved = _resolve_doc(target)
    if resolved is not None:
        phrase, anchor = resolved
        return rf"\Rdlink{{{anchor}}}{{{latex_inline(phrase)}}}"
    if target.startswith("/") or "/" in target:
        return _external_doc(target)
    return _role_text(target)


def _resolve_doc(target: str) -> tuple[str, str] | None:
    """Return the page title and PDF anchor for a documentation path."""
    for key in _doc_keys(target):
        found = DOC_PAGES.get(key)
        if found is not None:
            return found
    return None


def _doc_keys(target: str) -> list[str]:
    target = target.strip().strip("<>").strip()
    if target.startswith("/"):
        target = target[1:]
    elif _CURRENT_PAGE is not None:
        try:
            target = (_CURRENT_PAGE.parent / target).resolve().relative_to(DOCS).as_posix()
        except ValueError:
            pass
    if target.endswith(".rst"):
        target = target[:-4]
    keys = [target]
    if target.endswith("/index"):
        keys.append(target[: -len("/index")])
    else:
        keys.append(f"{target}/index")
    return keys


def _external_doc(target: str) -> str:
    """Link a page outside this manual to its online documentation."""
    cleaned = target.strip().strip("<>").strip().lstrip("/")
    if cleaned.endswith(".rst"):
        cleaned = cleaned[:-4]
    phrase = cleaned.split("/")[-1].replace("_", " ").replace("-", " ")
    url = f"https://nampy.readthedocs.io/en/latest/{cleaned}.html"
    return rf"\Rdext{{{url}}}{{{escape_latex(phrase)}}}"


def plain_text(text: str) -> str:
    """Heading text for a PDF bookmark, with markup removed."""
    text = re.sub(r"``(.+?)``", r"\1", text)
    text = re.sub(r":math:`([^`]*)`", r"\1", text)
    text = re.sub(
        r":[A-Za-z]+:`(?:([^`<]+)\s*<[^`>]+>|([^`]+))`",
        lambda match: (match.group(1) or match.group(2) or "").strip(),
        text,
    )
    text = re.sub(r"`([^`<]+)\s*<([^`>]+)>`_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    return escape_latex(text.strip())


def latex_heading(text: str) -> str:
    """Headings use typewriter for code, so backticks do not become quotes."""
    visual = latex_inline(text)
    plain = plain_text(text)
    if visual == plain:
        return visual
    return rf"\texorpdfstring{{{visual}}}{{{plain}}}"


def plain_label(label: str) -> str:
    """Argument names are set in the text font, as in an R reference manual."""
    text = re.sub(r"``(.+?)``", r"\1", label)
    text = re.sub(r":[^:`]+:`([^`]*)`", r"\1", text)
    text = re.sub(r":math:`([^`]*)`", r"\1", text)
    return escape_latex(text.strip())


def _row_fields(
    headers: list[str], cells: list[str]
) -> tuple[str, str, list[tuple[str, str]]]:
    """Split a reference row into its name, description, and quiet metadata."""
    label = cells[0] if cells else ""
    named = {
        header.strip().lower(): cell
        for header, cell in zip(headers[1:], cells[1:])
    }
    description = ""
    for key in PREFERRED_COLUMNS:
        if named.get(key, "").strip():
            description = named.pop(key)
            break
    meta = [
        (COLUMN_LABELS.get(header, header), cell)
        for header, cell in named.items()
        if cell.strip() and header not in {"name"}
    ]
    if not description and not meta:
        description = " ".join(cell for cell in cells[1:] if cell.strip())
    return label, description, meta


def _is_argument_table(headers: list[str]) -> bool:
    names = {header.strip().lower() for header in headers}
    return bool(names & {"parameter", "default", "type / input shape"})


def _render_table(table: Table) -> str:
    argument_names = _is_argument_table(table.headers)
    lines = []
    for row in table.rows:
        label, description, meta = _row_fields(table.headers, row)
        label_tex = plain_label(label) if argument_names else latex_inline(label.strip())
        description_tex = latex_inline(description) if description else ""
        meta_tex = r"\hspace{1.25em}".join(
            rf"{escape_latex(name)}\hspace{{0.35em}}{latex_inline(value)}"
            for name, value in meta
        )
        lines.append(rf"\Rdarg{{{label_tex}}}{{{description_tex}}}{{{meta_tex}}}")
    return "\n".join(lines)


def _split_banner(text: str) -> tuple[str, str]:
    """Drop the generated example banner and keep its path as a caption."""
    match = _EXAMPLE_BANNER.match(text)
    if match is None:
        return text, ""
    return text[match.end() :].lstrip("\n"), match.group("path").strip()


def _pygments_formatter():
    from pygments.formatters import LatexFormatter
    from pygments.style import Style
    from pygments.token import (
        Comment,
        Keyword,
        Name,
        Number,
        Operator,
        Punctuation,
        String,
        Token,
    )

    class _ReferenceStyle(Style):
        styles = {
            Token: "#1C2834",
            Comment: "italic #6A7886",
            Keyword: "#2C5F73",
            Name: "#1C2834",
            String: "#3E5166",
            Number: "#1C2834",
            Operator: "#1C2834",
            Punctuation: "#1C2834",
        }

    return LatexFormatter(
        style=_ReferenceStyle,
        commandprefix="RDpy",
        nobackground=True,
    )


_FORMATTER = None


def _formatter():
    global _FORMATTER
    if _FORMATTER is None:
        _FORMATTER = _pygments_formatter()
    return _FORMATTER


def pygments_preamble() -> str:
    try:
        return _formatter().get_style_defs()
    except Exception:
        return ""


def _highlight_python(code: str) -> str | None:
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer

        rendered = highlight(code, PythonLexer(), _formatter())
    except Exception:
        return None
    rendered = re.sub(r"^\\begin\{Verbatim\}\[[^\]]*\]\n", "", rendered)
    return re.sub(r"\\end\{Verbatim\}\s*$", "", rendered).rstrip()


def _render_code(block: Code) -> str:
    body, caption = _split_banner(block.text)
    parts = []
    if caption:
        parts.append(rf"\Rdcaption{{{code_span(caption)}}}")
    highlighted = None
    if block.language.lower() in {"python", "py", "python3", "ipython"}:
        highlighted = _highlight_python(body)
    if highlighted:
        parts.append("\\begin{Rdpy}\n" + highlighted + "\n\\end{Rdpy}")
    else:
        parts.append("\\begin{Rdcode}\n" + body + "\n\\end{Rdcode}")
    return "\n".join(parts)


def _topic_items(blocks: list) -> list:
    kept = [
        block
        for block in blocks
        if not (
            isinstance(block, Paragraph)
            and block.text.strip().startswith(":download:")
        )
    ]
    rendered = []
    index = 0
    while index < len(kept):
        block = kept[index]
        if isinstance(block, Heading) and block.text == "Example result":
            end = index + 1
            while end < len(kept) and not isinstance(kept[end], Heading):
                end += 1
            if any(isinstance(item, (Code, Table, Math)) for item in kept[index + 1 : end]):
                rendered.extend(kept[index:end])
            index = end
            continue
        rendered.append(block)
        index += 1
    return rendered


def render_blocks(blocks: list, *, topic: bool) -> list[str]:
    rendered = []
    items = _topic_items(blocks) if topic else list(blocks)
    if topic and items and isinstance(items[0], Paragraph):
        rendered.append(rf"\Rdlead{{{latex_inline(items[0].text)}}}")
        items = items[1:]
    for block in items:
        if isinstance(block, Paragraph):
            rendered.append(rf"{latex_inline(block.text)}\par")
        elif isinstance(block, Heading):
            title = SECTION_TITLES.get(block.text, block.text)
            rendered.append(rf"\Rdsection{{{latex_inline(title)}}}")
        elif isinstance(block, Bullets):
            lines = [r"\begin{itemize}[leftmargin=1.35em,itemsep=0.22em,topsep=0.4em,parsep=0.05em]"]
            lines.extend(rf"\item {latex_inline(item)}" for item in block.items)
            lines.append(r"\end{itemize}")
            rendered.append("\n".join(lines))
        elif isinstance(block, Code):
            rendered.append(_render_code(block))
        elif isinstance(block, Math):
            rendered.append("\\[\n" + block.text.strip() + "\n\\]")
        elif isinstance(block, Table) and block.rows:
            rendered.append(_render_table(block))
        elif isinstance(block, Page):
            raise TypeError("pages are rendered by the manual walker")
    return rendered


def _contents(title: str, level: str) -> str:
    safe = latex_heading(title)
    command = {"part": "Rdpart", "section": "Rdgroup"}.get(level, "Rdminor")
    return f"\\{command}{{{safe}}}\n"


def _render_topic(page: Page, depth: int, group: str) -> str:
    level = "subsection" if depth > 1 else "section"
    name = latex_heading(page.title)
    eyebrow = latex_heading(group) if group and group != page.title else ""
    mark_right = eyebrow or name
    parts = [
        r"\par\Needspace{8\baselineskip}",
        r"\phantomsection",
        rf"\hypertarget{{{_anchor(page)}}}{{}}",
        rf"\addcontentsline{{toc}}{{{level}}}{{{name}}}",
        rf"\markboth{{{name}}}{{{mark_right}}}",
        rf"\Rdtopic{{{name}}}{{{eyebrow}}}",
    ]
    parts.extend(render_blocks(page.items, topic=True))
    return "\n".join(parts) + "\n"


def _render_group(page: Page, depth: int) -> str:
    name = latex_heading(page.title)
    parts = [
        r"\phantomsection",
        rf"\hypertarget{{{_anchor(page)}}}{{}}",
        rf"\markboth{{{name}}}{{{name}}}",
        _contents(page.title, "section"),
    ]
    intro = []
    for item in page.items:
        if isinstance(item, Page):
            if intro:
                parts.extend(render_blocks(intro, topic=False))
                intro = []
            parts.append(render_page(item, depth + 1, page.title))
        elif isinstance(item, Heading) and item.level == 2:
            if intro:
                parts.extend(render_blocks(intro, topic=False))
                intro = []
            parts.append(_contents(item.text, "subsection"))
        else:
            intro.append(item)
    if intro:
        parts.extend(render_blocks(intro, topic=False))
    return "\n".join(parts) + "\n"


def _render_part(page: Page) -> str:
    parts = [_contents(page.title, "part"), rf"\hypertarget{{{_anchor(page)}}}{{}}"]
    leading = []
    for item in page.items:
        if isinstance(item, Page):
            break
        leading.append(item)
    parts.extend(render_blocks(leading, topic=False))
    children = [item for item in page.items if isinstance(item, Page)]
    if children:
        lines = [
            r"\begin{itemize}[leftmargin=1.35em,itemsep=0.18em,topsep=0.55em,parsep=0pt]"
        ]
        lines.extend(
            rf"\item \Rdlink{{{_anchor(child)}}}{{{latex_heading(child.title)}}}"
            for child in children
        )
        lines.append(r"\end{itemize}")
        parts.append("\n".join(lines))
        parts.append(r"\clearpage")
    intro = []
    for item in page.items[len(leading) :]:
        if isinstance(item, Page):
            if intro:
                parts.extend(render_blocks(intro, topic=False))
                intro = []
            parts.append(render_page(item, 1, page.title))
        elif isinstance(item, Heading) and item.level == 2:
            if intro:
                parts.extend(render_blocks(intro, topic=False))
                intro = []
            parts.append(_contents(item.text, "subsection"))
        else:
            intro.append(item)
    if intro:
        parts.extend(render_blocks(intro, topic=False))
    return "\n".join(parts) + "\n"


def render_page(page: Page, depth: int, group: str = "") -> str:
    global _CURRENT_PAGE
    previous = _CURRENT_PAGE
    _CURRENT_PAGE = page.path
    try:
        if page.is_topic:
            return _render_topic(page, depth, group)
        if depth == 0:
            return _render_part(page)
        return _render_group(page, depth)
    finally:
        _CURRENT_PAGE = previous


def _index_letter(title: str) -> str:
    for char in title:
        if char.isalnum():
            return char.upper()
    return ""


def render_index(pages: list[Page]) -> str:
    entries = []

    def walk(page: Page, group: str):
        if page.is_topic:
            entries.append((page.title.lower(), page.title, group, _anchor(page)))
            return
        for item in page.items:
            if isinstance(item, Page):
                walk(item, page.title)

    for page in pages:
        walk(page, page.title)
    lines = [
        r"\Rdpart{Index}",
        r"\begin{multicols}{2}",
        r"\raggedcolumns",
    ]
    letter = ""
    for _key, title, group, anchor in sorted(entries):
        current = _index_letter(title)
        if current and current != letter:
            letter = current
            lines.append(
                rf"\par\vspace{{0.85em}}\Needspace{{5\baselineskip}}"
                rf"\noindent{{\color{{accent}}\bfseries {escape_latex(letter)}}}\par"
                rf"\nopagebreak\vspace{{0.12em}}"
            )
        lines.append(
            rf"\par\vspace{{0.32em}}\noindent\hyperlink{{{anchor}}}{{{latex_heading(title)}}}\par"
            rf"\nopagebreak\vspace{{0.02em}}\noindent{{\color{{mist}}\footnotesize {escape_latex(group)}}}\par"
        )
    lines.append(r"\end{multicols}")
    return "\n".join(lines) + "\n"


_RD_PYTHON = r"""
\newenvironment{Rdpy}
  {\VerbatimEnvironment
    \begin{snugshade}
    \begin{Verbatim}[commandchars=\\\{\},frame=leftline,framerule=1.8pt,framesep=0.95em,rulecolor=\color{coderule},xleftmargin=0.2em]}
  {\end{Verbatim}\end{snugshade}}
"""


def render_manual(pages: list[Page], version: str, built_on: str) -> str:
    body = "\n".join(render_page(page, 0) for page in pages)
    index = render_index(pages)
    return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage{{fontspec}}
\\usepackage{{unicode-math}}
\\usepackage{{microtype}}
\\setmainfont{{lmroman10-regular.otf}}[
  BoldFont=lmroman10-bold.otf,
  ItalicFont=lmroman10-italic.otf,
  BoldItalicFont=lmroman10-bolditalic.otf,
  Ligatures=TeX,
  Scale=1.01
]
\\setmonofont{{lmmono10-regular.otf}}[
  ItalicFont=lmmono10-italic.otf,
  Scale=MatchLowercase
]
\\setmathfont{{latinmodern-math.otf}}
\\usepackage[margin=2.8cm,top=2.35cm,bottom=2.4cm,headheight=16.5pt,headsep=9pt]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{enumitem}}
\\providecommand{{\\UseTaggingSocket}}[1]{{}}
\\providecommand{{\\SuspendTagging}}[1]{{}}
\\providecommand{{\\ResumeTagging}}[1]{{}}
\\usepackage[titles]{{tocloft}}
\\usepackage{{fancyhdr}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{fancyvrb}}
\\usepackage{{fvextra}}
\\usepackage{{framed}}
\\usepackage{{multicol}}
\\usepackage{{needspace}}
\\definecolor{{ink}}{{HTML}}{{1C2834}}
\\definecolor{{slate}}{{HTML}}{{3E5166}}
\\definecolor{{mist}}{{HTML}}{{6A7886}}
\\definecolor{{accent}}{{HTML}}{{2C5F73}}
\\definecolor{{rule}}{{HTML}}{{D7DEE5}}
\\definecolor{{shadecolor}}{{HTML}}{{F4F7F8}}
\\definecolor{{coderule}}{{HTML}}{{2C5F73}}
\\hypersetup{{colorlinks=true,linkcolor=ink,urlcolor=accent,pdftitle={{NAMpy Reference Manual}}}}
{pygments_preamble()}
\\fvset{{fontsize=\\small,breaklines=true,breakindent=1.7em,breaksymbol=\\textcolor{{mist}}{{\\scriptsize\\ensuremath{{\\hookrightarrow}}}}}}
\\newenvironment{{Rdcode}}
  {{\\VerbatimEnvironment
    \\begin{{snugshade}}
    \\begin{{Verbatim}}[frame=leftline,framerule=1.8pt,framesep=0.95em,rulecolor=\\color{{coderule}},xleftmargin=0.2em]}}
  {{\\end{{Verbatim}}\\end{{snugshade}}}}
{_RD_PYTHON}
\\urlstyle{{tt}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0.34em plus 0.08em minus 0.04em}}
\\setlength{{\\emergencystretch}}{{2em}}
\\setlength{{\\abovedisplayskip}}{{0.85em}}
\\setlength{{\\belowdisplayskip}}{{0.75em}}
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{\\footnotesize\\color{{slate}}\\nouppercase{{\\leftmark}}}}
\\fancyhead[R]{{\\footnotesize\\color{{mist}}\\nouppercase{{\\rightmark}}}}
\\fancyfoot[C]{{\\footnotesize\\color{{mist}}\\thepage}}
\\renewcommand{{\\headrulewidth}}{{0pt}}
\\renewcommand{{\\headrule}}{{\\color{{rule}}\\hrule width\\headwidth height 0.45pt\\vskip-0.45pt}}
\\renewcommand{{\\cfttoctitlefont}}{{\\color{{ink}}\\Large\\bfseries}}
\\renewcommand{{\\cftpartfont}}{{\\color{{accent}}\\bfseries}}
\\renewcommand{{\\cftsecfont}}{{\\color{{ink}}}}
\\renewcommand{{\\cftsubsecfont}}{{\\color{{slate}}}}
\\setlength{{\\cftbeforepartskip}}{{1.35em}}
\\setlength{{\\cftbeforesecskip}}{{0.22em}}
\\setlength{{\\cftaftertoctitleskip}}{{0.8em}}
\\newcommand{{\\Rdpart}}[1]{{\\clearpage\\phantomsection\\markboth{{#1}}{{}}\\addcontentsline{{toc}}{{part}}{{#1}}\\vspace*{{0.4em}}{{\\color{{accent}}\\LARGE\\bfseries #1}}\\par\\vspace{{0.32em}}{{\\color{{accent}}\\hrule height 0.8pt}}\\par\\vspace{{0.7em}}}}
\\newcommand{{\\Rdgroup}}[1]{{\\par\\vspace{{1.15em}}\\phantomsection\\addcontentsline{{toc}}{{section}}{{#1}}{{\\color{{ink}}\\Large\\bfseries #1}}\\par\\vspace{{0.22em}}{{\\color{{rule}}\\hrule height 0.45pt}}\\par\\vspace{{0.55em}}}}
\\newcommand{{\\Rdminor}}[1]{{\\par\\vspace{{0.85em}}\\phantomsection\\addcontentsline{{toc}}{{subsection}}{{#1}}\\noindent{{\\color{{slate}}\\bfseries #1}}\\par\\vspace{{0.15em}}}}
\\newcommand{{\\Rdtopic}}[2]{{\\vspace{{1.05em}}\\noindent{{\\color{{ink}}\\large\\bfseries #1}}\\par\\nopagebreak\\vspace{{0.22em}}{{\\color{{accent}}\\hrule height 0.65pt}}\\par\\nopagebreak\\if\\relax\\detokenize{{#2}}\\relax\\else\\vspace{{0.16em}}{{\\color{{mist}}\\footnotesize #2}}\\par\\nopagebreak\\fi\\vspace{{0.38em}}}}
\\newcommand{{\\Rdlead}}[1]{{\\noindent{{\\color{{ink}}#1}}\\par}}
\\newcommand{{\\Rdmeta}}[2]{{\\textcolor{{mist}}{{\\textbf{{#1}}}} & #2\\\\[0.36em]}}
\\newcommand{{\\Rdarg}}[3]{{\\par\\vspace{{0.42em}}\\noindent{{\\bfseries\\color{{ink}}#1}}\\par\\nopagebreak\\if\\relax\\detokenize{{#2}}\\relax\\else\\vspace{{0.08em}}\\noindent #2\\par\\nopagebreak\\fi\\if\\relax\\detokenize{{#3}}\\relax\\else\\vspace{{0.05em}}{{\\color{{mist}}\\footnotesize #3\\par}}\\fi}}
\\newcommand{{\\Rdlink}}[2]{{\\hyperlink{{#1}}{{\\textcolor{{accent}}{{#2}}}}}}
\\newcommand{{\\Rdext}}[2]{{\\href{{#1}}{{\\textcolor{{accent}}{{#2}}}}}}
\\newcommand{{\\Rdcaption}}[1]{{\\par\\vspace{{0.45em}}\\noindent{{\\color{{mist}}\\footnotesize #1}}\\par\\nopagebreak\\vspace{{0.18em}}}}
\\newcommand{{\\Rdsection}}[1]{{\\par\\vspace{{1.05em}}\\noindent{{\\color{{accent}}\\bfseries #1}}\\par\\nopagebreak\\vspace{{0.28em}}}}
\\setcounter{{tocdepth}}{{1}}
\\begin{{document}}
\\thispagestyle{{empty}}
\\vspace*{{0.2em}}
\\begin{{center}}
\\begin{{tabular}}{{@{{}}r@{{ \\hspace{{0.9em}}}}c@{{}}}}
\\begin{{tabular}}{{@{{}}c@{{}}}}
{{\\color{{ink}}\\Huge\\bfseries NAMpy}}\\\\[0.32em]
{{\\color{{accent}}\\rule{{2.6cm}}{{0.8pt}}}}\\\\[0.36em]
{{\\color{{slate}}\\large Reference Manual}}
\\end{{tabular}}
&
\\includegraphics[height=2.35cm]{{nampy-logo.pdf}}
\\end{{tabular}}
\\end{{center}}
\\vspace{{1.7em}}
\\noindent\\begin{{tabular}}{{@{{}}l@{{ \\hspace{{0.9em}}}}p{{12.15cm}}@{{}}}}
\\Rdmeta{{Version}}{{{escape_latex(version)}}}
\\Rdmeta{{Date}}{{{escape_latex(built_on)}}}
\\Rdmeta{{Title}}{{Interpretable statistical and neural additive models}}
\\Rdmeta{{Author}}{{Ananyapam De and Anton Thielmann}}
\\Rdmeta{{Maintainer}}{{Ananyapam De and Anton Thielmann}}
\\Rdmeta{{Description}}{{Reference manual for the public estimators, formulas, smooths, families, fitting methods, prediction, and extension interfaces.}}
\\Rdmeta{{License}}{{MIT}}
\\Rdmeta{{URL}}{{\\url{{https://nampy.readthedocs.io}}}}
\\end{{tabular}}\\par
\\vspace{{1.15em}}
\\noindent{{\\color{{slate}}Each topic states the call, its arguments, and the returned value. The separate NAMpy book contains the tutorials and task guides.}}\\par
\\vspace{{0.85em}}
\\makeatletter
\\begingroup
\\let\\@mkboth\\@gobbletwo
\\markboth{{Contents}}{{}}
\\tableofcontents
\\endgroup
\\makeatother
\\clearpage
{body}
{index}
\\end{{document}}
"""


def export_logo(destination: Path) -> None:
    """Write the title-page mark as a PDF drawn from the source SVG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rsvg = shutil.which("rsvg-convert")
    if rsvg is not None:
        subprocess.run(
            [rsvg, "-f", "pdf", "-o", str(destination), str(LOGO_SVG)],
            check=True,
        )
        return
    inkscape = shutil.which("inkscape")
    if inkscape is not None:
        subprocess.run(
            [
                inkscape,
                str(LOGO_SVG),
                "--export-type=pdf",
                f"--export-filename={destination}",
            ],
            check=True,
        )
        return
    try:
        import cairosvg
    except ImportError:
        cairosvg = None
    if cairosvg is not None:
        cairosvg.svg2pdf(url=str(LOGO_SVG), write_to=str(destination))
        return
    raise RuntimeError(
        "The title page uses docs/_static/logo-sidebar-refined.svg. Install "
        "rsvg-convert (librsvg2-bin), Inkscape, or cairosvg so the build can "
        "convert that SVG."
    )


def compile_manual(tex_path: Path) -> Path:
    engine = shutil.which("latexmk")
    if engine is None:
        raise RuntimeError(
            "latexmk is not installed. The TeX source was written; install "
            "LuaLaTeX and latexmk, then rerun this script."
        )
    subprocess.run(
        [
            engine,
            "-lualatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={tex_path.parent}",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        check=True,
    )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS / "_build" / "reference-manual",
        help="Directory for the TeX source and PDF",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Write the TeX source without running LuaLaTeX",
    )
    args = parser.parse_args()
    pages = build_manual()
    topics = []

    def count(page: Page):
        if page.is_topic:
            topics.append(page)
        else:
            for item in page.items:
                if isinstance(item, Page):
                    count(item)

    for page in pages:
        count(page)
    tex = render_manual(
        pages,
        package_version(),
        datetime.now().strftime("%B %-d, %Y"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    export_logo(args.output / "nampy-logo.pdf")
    tex_path = args.output / "NAMpy-reference.tex"
    tex_path.write_text(tex)
    print(f"Wrote {tex_path} ({len(topics)} topics)")
    if args.no_compile:
        return
    pdf = compile_manual(tex_path)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
