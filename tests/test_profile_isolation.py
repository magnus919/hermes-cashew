# tests/test_profile_isolation.py
# Phase 2 Plan 02-03 Decision Point 5: CONF-04 enforcement via grep-walk.
# Any file under plugins/ that matches a forbidden HOME-related pattern fails this test.

from __future__ import annotations

import pathlib
import re

# Each pattern is the literal string we forbid; matched as a regex against source
# with comments and docstrings stripped. String-literal content inside `os.environ["HOME"]`
# must survive stripping (it's part of the forbidden pattern itself), but standalone
# docstrings that *describe* a forbidden call must not. Implementation: strip COMMENT
# tokens unconditionally, and strip STRING tokens ONLY when they form a standalone
# expression statement (i.e., a docstring) — string literals used as arguments or
# subscripts (e.g., `os.environ["HOME"]`) are preserved by AST-guided stripping.
FORBIDDEN_PATTERNS = [
    (r"\bPath\.home\s*\(", "Path.home()"),
    (r"\bpathlib\.Path\.home\s*\(", "pathlib.Path.home()"),
    (r"\bexpanduser\s*\(", "expanduser(...)"),
    (r"\bos\.path\.expanduser\s*\(", "os.path.expanduser(...)"),
    (r"os\.environ\[\s*['\"]HOME['\"]\s*\]", 'os.environ["HOME"]'),
    (r"os\.environ\.get\s*\(\s*['\"]HOME['\"]", 'os.environ.get("HOME"...)'),
    (r"os\.getenv\s*\(\s*['\"]HOME['\"]", 'os.getenv("HOME"...)'),
]


def _strip_comments_and_docstrings(source: str) -> str:
    """Strip # line comments and standalone string-expression statements (docstrings).

    Preserves string literals used as arguments or subscripts (e.g., `os.environ["HOME"]`)
    because those are part of the patterns we're trying to CATCH. Only module/class/function
    docstrings (`Expr(Constant(str))` nodes) are stripped — using AST to identify them.

    Returns the source with those ranges blanked out (line numbers preserved for
    reporting). Falls back to raw source if tokenize/ast can't parse.
    """
    import ast
    import io
    import tokenize

    # Step 1: collect docstring byte-offset ranges via AST walk.
    docstring_ranges: list[tuple[int, int, int, int]] = []  # (start_line, start_col, end_line, end_col)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    def _is_docstring(node: ast.AST) -> ast.Constant | None:
        """If node.body[0] is an Expr(Constant(str)), return that Constant."""
        body = getattr(node, "body", None)
        if not body:
            return None
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            const = _is_docstring(node)
            if const is not None:
                docstring_ranges.append(
                    (const.lineno, const.col_offset, const.end_lineno or const.lineno, const.end_col_offset or 0)
                )

    # Step 2: build line-indexed list, blank out docstring ranges and comments.
    lines = source.splitlines(keepends=False)
    out_lines = list(lines)

    def _blank_range(sl: int, sc: int, el: int, ec: int) -> None:
        # 1-indexed line numbers from AST; convert to 0-indexed.
        sl0, el0 = sl - 1, el - 1
        if sl0 == el0:
            line = out_lines[sl0]
            out_lines[sl0] = line[:sc] + " " * (ec - sc) + line[ec:]
        else:
            out_lines[sl0] = out_lines[sl0][:sc] + " " * max(0, len(out_lines[sl0]) - sc)
            for i in range(sl0 + 1, el0):
                out_lines[i] = " " * len(out_lines[i])
            out_lines[el0] = " " * ec + out_lines[el0][ec:]

    for sl, sc, el, ec in docstring_ranges:
        _blank_range(sl, sc, el, ec)

    # Step 3: strip line comments via tokenize (after docstring blanking).
    blanked_source = "\n".join(out_lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(blanked_source).readline))
    except tokenize.TokenizeError:
        return blanked_source
    out_lines_v2 = blanked_source.splitlines(keepends=False)
    for tok_type, _tok_str, (sr, sc), (er, ec), _line in tokens:
        if tok_type == tokenize.COMMENT:
            sr0, er0 = sr - 1, er - 1
            if sr0 == er0 and 0 <= sr0 < len(out_lines_v2):
                line = out_lines_v2[sr0]
                out_lines_v2[sr0] = line[:sc] + " " * (ec - sc) + line[ec:]
    return "\n".join(out_lines_v2)


# Backwards-compat alias kept in case any outside code refers to the original name.
_strip_comments_and_strings = _strip_comments_and_docstrings


def test_no_home_related_patterns_under_plugins():
    """CONF-04: every path scopes under hermes_home — no Path.home, expanduser, or HOME env reads."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    plugins_dir = repo_root / "plugins"
    assert plugins_dir.is_dir(), f"expected plugins/ at {plugins_dir}"

    violations: list[str] = []
    for py_file in sorted(plugins_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary file masquerading as .py? skip.
        cleaned = _strip_comments_and_strings(source)
        for pattern, label in FORBIDDEN_PATTERNS:
            if re.search(pattern, cleaned):
                # Find the actual source line(s) that triggered for a useful failure message.
                for lineno, line in enumerate(source.splitlines(), start=1):
                    if re.search(pattern, line):
                        violations.append(
                            f"{py_file.relative_to(repo_root)}:{lineno}: forbidden pattern {label!r}\n"
                            f"    {line.rstrip()}"
                        )

    assert not violations, (
        "Profile-isolation violations detected (CONF-04). "
        "Every path must derive from kwargs['hermes_home']; never from $HOME or ~.\n\n"
        + "\n\n".join(violations)
        + "\n\nFix: replace with `pathlib.Path(hermes_home) / ...` per CLAUDE.md ### Profile Isolation."
    )
