"""Compact fitting-control groups with one combined example per algorithm area."""

DOMAINS = {
    "epsilon": "> 0",
    "maxit": "Positive integer",
    "irls_reg": ">= 0",
    "rank_tol": "[0,1]; invalid values warn and reset to sqrt(machine epsilon)",
    "scale_est": "'fletcher', 'pearson', 'deviance'",
    "mgcv_tol": "> 0",
    "mgcv_half": "Nonnegative integer",
    "efs_lspmax": "Real log-penalty upper bound",
    "efs_tol": "> 0; nonpositive resets to 0.1",
    "edge_correct": "bool or nonnegative float",
    "scam_max_half": "Positive integer",
    "scam_devtol_fit": "> 0",
    "scam_steptol_fit": "> 0",
    "scam_b_notexp": "> 0",
    "scam_threshold_notexp": "> 0",
    "nthreads": "Positive integer (rounded)",
    "ncv_threads": "Positive integer (rounded)",
}
GROUP_NOTES = {
    "coefficient_fitting": "IRLS/coefficient updates consume these tolerances. A positive fixed model scale overrides scale_est. rank_tol has the reset policy shown in the table.",
    "smoothing_optimization": "Only the chosen outer optimizer consumes its nested mapping; other routes ignore it. The example uses outer Newton. NLM, optim and EFS options remain independently configurable.",
    "shape_fitting": "These settings apply to the SCAM transformed-coefficient route. Ordinary unconstrained bases do not use them.",
    "execution": "nthreads and ncv_threads are compatibility metadata and do not activate parallel native kernels. trace controls available diagnostics; keep_data records the requested retention policy.",
    "basis_penalty": "The compiler consumes these options before coefficient fitting. Linked terms must have compatible basis setup; changing penalty scaling changes the smoothing-parameter coordinates.",
}
EXAMPLE_OPTIONS = {
    "coefficient_fitting": "epsilon=1e-7, maxit=200, irls_reg=0.0",
    "smoothing_optimization": "mgcv_tol=1e-7, newton={'conv_tol': 1e-6}",
    "shape_fitting": "scam_devtol_fit=1e-7, scam_steptol_fit=1e-7",
    "execution": "trace=False, nthreads=1, ncv_threads=1",
    "basis_penalty": "id_links_bases=True, scale_penalty=True",
}


def generate(root):
    from dataclasses import fields

    from feature_reference import CONTROL_MEANINGS, heading, table, tree, write
    from public_reference import literal
    from reference_examples import BASE, example_rst
    from reference_organization import CONTROL_GROUPS, control_anchor
    from reference_schema import NESTED_CONTROLS

    from nampy.gam import gam_control
    from nampy.gam.control import _KEY_ALIASES

    control = gam_control()
    field_map = {f.name: f for f in fields(control)}
    names = {name for _, names in CONTROL_GROUPS.values() for name in names}
    if names != set(field_map):
        raise ValueError("Control groups differ from GAMControl fields")
    for slug, (title, names) in CONTROL_GROUPS.items():
        text = heading(title) + GROUP_NOTES[slug] + "\n\n"
        text += "Create one validated ``GAMControl`` with ``gam_control(...)`` and pass it as ``control=``. Unknown keys raise ``TypeError``; invalid values raise ``ValueError`` except the documented reset policies.\n\n"
        rows = []
        for name in names:
            value = getattr(control, name)
            domain = DOMAINS.get(
                name,
                "Mapping; see nested options below"
                if isinstance(value, dict)
                else "True or False",
            )
            default = "resolved below" if isinstance(value, dict) else literal(value)
            rows.append(
                (
                    f"``{name}``",
                    f"``{field_map[name].type}``, ``{default}``",
                    domain,
                    CONTROL_MEANINGS[name],
                )
            )
        rendered = table(["Setting", "Type / default", "Valid values", "Effect"], rows)
        for name in names:
            rendered = rendered.replace(
                f"   * - ``{name}``",
                f"   * - .. _{control_anchor(name)}:\n\n       ``{name}``",
            )
            write(
                root,
                f"statistical_reference/controls/{name}.rst",
                ":orphan:\n\n"
                + heading(name)
                + f"See :ref:`{name} <{control_anchor(name)}>` for its type, default, valid values and fitting role.\n",
            )
        text += rendered
        aliases = [
            (alias, target) for alias, target in _KEY_ALIASES.items() if target in names
        ]
        if aliases:
            text += (
                "Accepted aliases: "
                + ", ".join(f"``{a}`` → ``{b}``" for a, b in aliases)
                + ".\n\n"
            )
        for name in names:
            value = getattr(control, name)
            if isinstance(value, dict):
                text += heading(name + " nested options", "-") + table(
                    ["Option", "Type", "Resolved default", "Effect"],
                    [
                        (
                            f"``{key}``",
                            type(v).__name__,
                            f"``{literal(v)}``",
                            NESTED_CONTROLS[key],
                        )
                        for key, v in value.items()
                    ],
                )
        basis = "mpi" if slug == "shape_fitting" else "cr"
        code = BASE + "from nampy.gam import gam_control\n\n"
        code += f"control = gam_control({EXAMPLE_OPTIONS[slug]})\n"
        code += f'model = GAM(formula="y ~ s(x, bs=\'{basis}\', k=6)", control=control, optimize_smoothing={slug == "smoothing_optimization"}, smoothing_method="reml").fit(data=data)\n'
        code += "result = model.predict(data)[:3]\nprint(result)\n"
        # Downloadable and tested, without scalar-setting output inventories.
        example_rst(root, f"statistical_reference/controls/{slug}", code)
        source = f"/_examples/reference/statistical_reference/controls/{slug}.py"
        text += (
            heading("Combined example", "-")
            + f".. literalinclude:: {source}\n   :language: python\n\n:download:`Download this example <{source}>`.\n\n"
        )
        write(root, f"statistical_reference/controls/{slug}.rst", text)
    write(
        root,
        "statistical_reference/controls/index.rst",
        tree(
            "Fitting controls",
            [f"{title} <{slug}>" for slug, (title, _) in CONTROL_GROUPS.items()],
            "Choose settings by their consuming algorithm. Each group has one configuration table and one complete example.",
        ),
    )
