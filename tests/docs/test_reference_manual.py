"""The reference manual is an Rd-style PDF of the public help topics."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def load_manual():
    path = ROOT / "docs/scripts/build_reference_manual.py"
    spec = importlib.util.spec_from_file_location("build_reference_manual", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inline_markup_matches_reference_roles():
    manual = load_manual()
    rendered = manual.latex_inline("``s_x``")
    assert rendered == r"\texttt{s\_\allowbreak{}x}"
    quoted = manual.latex_inline("``formula='y ~ s(x, bs=\"tp\")'``")
    assert quoted == r"\texttt{formula='y \textasciitilde{} s(x, bs=" + '"tp"' + r")'}"
    assert quoted.startswith(r"\texttt{") and quoted.endswith("}")
    assert manual.latex_inline("`mgcv::tw`") == r"\texttt{mgcv:\allowbreak{}:\allowbreak{}tw}"
    heading = manual.latex_heading("Adaptive spline (``ad``)")
    assert r"\texttt{ad}" in heading
    assert "``" not in heading
    assert manual.latex_inline(":doc:`bs <bs>`") == r"\texttt{bs}"
    assert r"\allowbreak{}" in manual.latex_inline("``a,b``")
    assert manual.latex_inline(r":math:`\mu`") == r"$\mu$"
    assert "Statistical modeling guide" in manual.latex_inline(
        ":ref:`Statistical modeling guide <howto-user-guide-statistical-gams>`"
    )
    assert r"a \& b" == manual.latex_inline("a & b")
    assert manual.latex_inline("NAMpy's formula") == "NAMpy's formula"
    assert manual.latex_inline("requires {'trials': trials}") == r"requires \{`trials': trials\}"
    assert manual.latex_inline('requires {"trials": trials}') == r"requires \{``trials'': trials\}"


def _topics(page, found):
    if page.is_topic:
        found.append(page)
        return
    for item in page.items:
        if isinstance(item, type(page)):
            _topics(item, found)


def test_reference_pdf_is_compiled_into_its_output_directory(tmp_path, monkeypatch):
    manual = load_manual()
    tex = tmp_path / "NAMpy-reference.tex"
    tex.write_text("source")
    calls = []

    def run(command, cwd, check):
        calls.append((command, Path(cwd), check))
        (Path(cwd) / "NAMpy-reference.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(manual.shutil, "which", lambda name: "/usr/bin/latexmk")
    monkeypatch.setattr(manual.subprocess, "run", run)
    pdf = manual.compile_manual(tex)
    command, cwd, check = calls[0]
    assert check is True
    assert cwd == tmp_path.resolve()
    assert f"-outdir={tmp_path.resolve()}" in command
    assert pdf == tmp_path.resolve() / "NAMpy-reference.pdf"
    assert pdf.is_file()


def test_clean_checkout_materializes_reference_fragments(tmp_path, monkeypatch):
    manual = load_manual()
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(manual, "DOCS", docs)
    manual.ensure_generated_fragments()
    for name in (
        "GAMRegressor.inc",
        "SurvivalGAM.inc",
        "GAM_core_parameters.inc",
        "neural_fit_contract.inc",
    ):
        assert (docs / "_generated" / "reference" / name).is_file()


def test_predict_topic_has_usage_arguments_value_and_example():
    manual = load_manual()
    found = []
    for page in manual.build_manual():
        _topics(page, found)
    predict = next(
        page
        for page in found
        if page.path.name == "predict.rst"
        and "statistical_reference/methods" in page.path.as_posix()
    )
    text = manual.render_page(predict, 2, "Prediction, inference and diagnostics")
    assert r"\Rdsection{Usage}" in text
    assert r"\Rdsection{Arguments}" in text
    assert r"\Rdsection{Value}" in text
    assert r"\Rdsection{Examples}" in text
    assert r"\Rdarg{" in text
    assert r"\begin{Rdpy}" in text
    assert r"\Rdcaption" in text
    assert r"statistical\_\allowbreak{}reference" in text
    assert "Runnable reference example" not in text
    assert "Prediction, inference and diagnostics" in text
    assert "nampy reference" not in text
    assert "Download the complete example" not in text
    assert "example-result" not in text
    assert len(found) > 300


def test_argument_rows_keep_type_and_default_on_their_own_line():
    manual = load_manual()
    table = manual.Table(
        ["Parameter", "Type / input shape", "Default", "Description"],
        [["``family``", "``str | None``", "``None``", "Response family."]],
    )
    rendered = "\n".join(manual.render_blocks([table], topic=True))
    assert r"\Rdarg{family}{Response family.}" in rendered
    assert r"default\hspace{0.35em}\texttt{None}" in rendered
    assert "default: " not in rendered


def test_dense_reference_topics_render_as_tables_and_subsections():
    manual = load_manual()
    topics = []
    for page in manual.build_manual():
        _topics(page, topics)
    survival = next(page for page in topics if page.path.name == "SurvivalGAM.rst")
    rendered = manual.render_page(survival, 2, "Statistical estimators")
    operations = rendered.split(r"\Rdsection{Predictions and supported operations}", 1)[1]
    assert r"\begin{longtable}" in operations
    assert r"\textbf{Operation} & \textbf{Result}" in operations
    assert r"\Rdsubsection{Diagnostics and evaluation}" in operations
    assert r"\Rdsubsection{Effects and uncertainty}" in operations
    assert r"\Rdext{https://nampy.readthedocs.io/en/latest/howto/survival.html}{survival}" in operations
    for name in ("CoxGAM.rst", "NeuralEnsemble.rst", "penalties.rst", "gam_smooths.rst"):
        topic = next(page for page in topics if page.path.name == name)
        assert r"\begin{longtable}" in manual.render_page(topic, 2)
    optimizer = next(page for page in topics if page.path.name == "optimizer_endpoint_diagnostics.rst")
    rendered = manual.render_page(optimizer, 2)
    assert r"\Rdarg{" in rendered.split(r"\Rdsection{Outputs}", 1)[1]


def test_bare_doc_path_becomes_a_phrase_link():
    manual = load_manual()
    manual.DOC_PAGES["statistical_reference/methods/vcov"] = (
        "vcov",
        "ref-statistical_reference-methods-vcov-rst",
    )
    rendered = manual.latex_inline(":doc:`/statistical_reference/methods/vcov`")
    assert rendered == r"\Rdlink{ref-statistical_reference-methods-vcov-rst}{vcov}"
    external = manual.latex_inline(":doc:`/user_guide/calibration`")
    assert external == (
        r"\Rdext{https://nampy.readthedocs.io/en/latest/user_guide/calibration.html}"
        "{calibration}"
    )


def test_manual_opens_with_a_short_contents():
    manual = load_manual()
    tex = manual.render_manual([], "0.3.0", "November 2, 2026")
    assert r"\documentclass[11pt,a4paper]{article}" in tex
    assert "margin=2.8cm" in tex
    assert r"\setcounter{tocdepth}{1}" in tex
    assert r"\hrule height 0.35pt" not in tex
    assert "breaklines=true" in tex
    assert "breaksymbolleft={},breaksymbolright={}" in tex
    body = tex.split(r"\begin{document}", 1)[1]
    assert r"\Rdmeta{Author}" in body
    assert "Ananyapam De" in body
    assert body.index(r"\tableofcontents") < body.index(r"\clearpage")
