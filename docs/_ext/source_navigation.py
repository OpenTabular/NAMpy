"""Browse Sphinx's generated source pages alongside the authored document tree."""

import html

GROUP_NAMES = {
    "gam": "Statistical engine",
    "neural": "Neural engine",
    "models": "Estimator interfaces",
    "plotting": "Plotting",
}


def module_tree(modules):
    tree = {}
    for module, entry in sorted(modules.items()):
        if not entry:
            continue
        children = tree
        parts = module.split(".")
        for index, part in enumerate(parts):
            node = children.setdefault(
                part,
                {"name": ".".join(parts[: index + 1]), "source": False, "children": {}},
            )
            children = node["children"]
        node["source"] = True
    return tree


def source_page(name):
    return "_modules/" + name.replace(".", "/")


def source_menu(tree, pagename, uri, *, article=False):
    def render(nodes):
        items = []
        for short, node in nodes.items():
            name = node["name"]
            target = source_page(name)
            active = pagename == target
            ancestor = active or pagename.startswith(target + "/")
            label = name if article else GROUP_NAMES.get(short, short)
            text = html.escape(label)
            anchor = "package-" + name.replace(".", "-")
            if node["source"]:
                text = (
                    f'<a href="{html.escape(uri(pagename, target), quote=True)}"'
                    + (' class="current" aria-current="page"' if active else "")
                    + f">{text}</a>"
                )
            if node["children"]:
                children = render(node["children"])
                if article:
                    text = f'<span id="{anchor}">{text}</span><ul>{children}</ul>'
                else:
                    text = (
                        "<details" + (" open" if ancestor else "") + ">"
                        f"<summary>{text}</summary><ul>{children}</ul></details>"
                    )
            items.append(f"<li>{text}</li>")
        return "".join(items)

    roots = tree.get("nampy", {}).get("children", tree)
    return render(roots)


def page_context(app, pagename, templatename, context, doctree):
    tree = module_tree(getattr(app.env, "_viewcode_modules", {}))
    if not tree:
        return
    uri = app.builder.get_relative_uri
    context["module_navigation"] = source_menu(tree, pagename, uri)
    context["module_index_uri"] = uri(pagename, "_modules/index")
    context["module_index_active"] = pagename == "_modules/index"
    if pagename == "_modules/index":
        context["title"] = "Module code"
        context["body"] = (
            "<h1>Module code</h1>"
            "<p>Browse the implementation by package. Source pages link back to "
            "the corresponding API documentation through their [docs] links.</p>"
            '<ul class="module-source-index">'
            + source_menu(tree, pagename, uri, article=True)
            + "</ul>"
        )


def setup(app):
    app.connect("html-page-context", page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
