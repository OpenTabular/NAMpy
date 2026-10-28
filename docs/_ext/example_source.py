"""Keep executable teaching examples free of test-only assertion statements."""

import ast


def is_assertion(node):
    return isinstance(node, ast.Assert) or (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr.startswith("assert_")
        and isinstance(node.value.func.value, ast.Attribute)
        and node.value.func.value.attr == "testing"
    )


def without_assertions(source, *, validation_report=False):
    tree = ast.parse(source)
    assertions = [node for node in ast.walk(tree) if is_assertion(node)]
    if not assertions:
        return source
    lines = source.splitlines(keepends=True)
    keep_suite = set()
    for parent in ast.walk(tree):
        if isinstance(parent, ast.Module):
            continue
        for _, value in ast.iter_fields(parent):
            if (
                isinstance(value, list)
                and value
                and all(is_assertion(n) for n in value)
            ):
                keep_suite.add(value[0].lineno)
    for node in sorted(
        assertions, key=lambda n: (n.lineno, n.col_offset), reverse=True
    ):
        indent = lines[node.lineno - 1][: node.col_offset]
        if indent.strip():
            raise ValueError(
                "Put assertions on their own lines before exporting an example"
            )
        if validation_report and isinstance(node, ast.Assert):
            condition = ast.unparse(node.test)
            message = (
                ast.unparse(node.msg)
                if node.msg
                else repr("Validation failed: " + condition)
            )
            replacement = [
                f"{indent}if not ({condition}):\n{indent}    raise ValueError({message})\n"
            ]
        else:
            replacement = [indent + "pass\n"] if node.lineno in keep_suite else []
        lines[node.lineno - 1 : node.end_lineno] = replacement
    cleaned = "".join(lines)
    ast.parse(cleaned)
    return cleaned
