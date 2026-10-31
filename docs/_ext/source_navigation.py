"""Keep detailed source browsing on its Development landing page."""

import html


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


def source_menu(tree, pagename, uri):
    def render(nodes):
        items = []
        for node in nodes.values():
            name = node["name"]
            target = source_page(name)
            active = pagename == target
            text = html.escape(name)
            anchor = "package-" + name.replace(".", "-")
            if node["source"]:
                text = (
                    f'<a href="{html.escape(uri(pagename, target), quote=True)}"'
                    + (' class="current" aria-current="page"' if active else "")
                    + f">{text}</a>"
                )
            if node["children"]:
                children = render(node["children"])
                text = f'<span id="{anchor}">{text}</span><ul>{children}</ul>'
            items.append(f"<li>{text}</li>")
        return "".join(items)

    roots = tree.get("nampy", {}).get("children", tree)
    return render(roots)


def page_context(app, pagename, templatename, context, doctree):
    if pagename != "development/source_code":
        return
    tree = module_tree(getattr(app.env, "_viewcode_modules", {}))
    if not tree:
        return
    context["body"] += (
        '<ul class="module-source-index">'
        + source_menu(tree, pagename, app.builder.get_relative_uri)
        + "</ul>"
    )


def setup(app):
    app.connect("html-page-context", page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
