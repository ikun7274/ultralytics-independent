"""Contract: no non-ASCII codepoint may reach a LOGGER message from ``ultralytics_ooo`` on Windows.

WHY THIS TESTED A SOURCE SCAN INSTEAD OF CAPTURING THE LOG. Upstream's Windows log formatter ends
with ``emojis(formatted)``, i.e. ``s.encode().decode("ascii", "ignore")``
(``ultralytics/utils/__init__.py``: ``PrefixFormatter.format``), so the deletion happens at FORMAT
time -- after our message exists and after every handler we could install. ``record.getMessage()``
therefore returns the INTACT string, and that is exactly what ``_capture_messages`` in
``test_ooo_branches.py`` observes: a capture-based test cannot see this defect no matter how it is
written. Measured with ``_perf_review/ooo7/log_ascii_probe.py``::

    source           '[resume_extend] 自动修补 checkpoint 元数据: 已完成 4 轮 -> 续训至 20 轮'
    getMessage()     '[resume_extend] 自动修补 checkpoint 元数据: 已完成 4 轮 -> 续训至 20 轮'
    formatter output '[resume_extend]  checkpoint :  4  ->  20 '      <- payload silently gone
    upstream's own   'ping: 0.0±0.0 ms' -> 'ping: 0.00.0 ms'         <- two numbers GLUED together

Three shapes of damage, all from one deleted codepoint: a glued number, a dangling separator
(``mAP/精度`` -> ``mAP/``), and a vanished clause. So the invariant has to be enforced on the source
literals, which is what this module does.

SCOPE. Every string literal in the package EXCEPT docstrings (never logged) and ``raise``
statements (exceptions are printed by Python's traceback machinery, which does not pass through the
ultralytics Formatter -- asserted by the probe's last case). The literal check is deliberately wider
than "literals lexically inside a ``LOGGER`` call": a message assembled by a helper
(``_describe_ims_cap``) is invisible to that narrower rule, and that is the observation-point trap
this package has already been bitten by once.

NON-VACUITY. ``test_the_scanner_flags_a_planted_character`` runs the same scanner over synthetic
sources, one clean (must report nothing) and three dirty (must report exactly one each, including the
helper-built case). ``test_the_scan_really_looked_at_the_package`` asserts the scanner really saw the
package's LOGGER calls -- otherwise an empty scan would pass for the same reason a broken one does.
"""

from __future__ import annotations

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "ultralytics_ooo"

# Coverage floors for the whole-package scan. Measured 2026-09-19: 52 LOGGER.* calls and 156 string
# literals inside them. A floor (not equality) so ordinary edits do not fail this test, but high
# enough that a scanner that silently stopped matching would.
MIN_LOGGER_CALLS = 40
MIN_LOGGER_LITERALS = 120


def _docstring_ids(tree: ast.AST) -> set[int]:
    """ids of the ``Constant`` nodes that are module/class/function docstrings."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _constants_under(tree: ast.AST, kinds: tuple[type, ...]) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, kinds):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    out.add(id(sub))
    return out


def _is_logger_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "LOGGER"


def scan(source: str) -> list[tuple[int, str, str]]:
    """Return one ``(lineno, kind, preview)`` per Windows-unsafe string literal in ``source``.

    ``kind`` is ``"logger"`` for a literal inside a ``LOGGER.*`` call and ``"runtime"`` for any other
    non-docstring literal -- the second kind is what catches a message built by a helper.
    """
    tree = ast.parse(source)
    docs = _docstring_ids(tree)
    in_raise = _constants_under(tree, (ast.Raise,))
    in_logger = _constants_under(tree, (ast.Call,))
    logger_lits = {
        id(sub)
        for node in ast.walk(tree)
        if _is_logger_call(node)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant)
    }
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docs or id(node) in in_raise:
            continue
        bad = sorted({c for c in node.value if ord(c) > 127})
        if not bad:
            continue
        kind = "logger" if id(node) in logger_lits else "runtime"
        cps = " ".join(f"U+{ord(c):04X} {c!r}" for c in bad)
        hits.append((node.lineno, kind, f"{cps} in {node.value.strip()[:60]!r}"))
    del in_logger  # kept only to document that every Call was walked; the precise set is logger_lits
    return hits


def _scan_package() -> tuple[list[str], int, int, int]:
    """Scan the package: (problems, logger calls, logger literals, raise-only literals)."""
    problems: list[str] = []
    calls = lits = raise_only = 0
    for path in sorted(PKG.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        rel = path.relative_to(PKG.parent).as_posix()
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if _is_logger_call(node):
                calls += 1
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        lits += 1
        docs = _docstring_ids(tree)
        in_raise = _constants_under(tree, (ast.Raise,))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docs:
                continue
            has_bad = any(ord(c) > 127 for c in node.value)
            if id(node) in in_raise:
                raise_only += int(has_bad)
                continue
        for lineno, kind, preview in scan(src):
            problems.append(f"{rel}:{lineno} [{kind}] {preview}")
    return problems, calls, lits, raise_only


def test_no_non_ascii_can_reach_a_logger_message():
    """The invariant. A failure here means the message would be silently mangled on Windows."""
    problems, _calls, _lits, _raise_only = _scan_package()
    assert not problems, (
        "these string literals contain non-ASCII characters that upstream's Windows formatter DELETES "
        "(`emojis()` = encode().decode('ascii', 'ignore')), turning a number into a glued number, a "
        "separator into a dangling '/', or a whole clause into nothing:\n  "
        + "\n  ".join(problems)
        + "\n\nFix the literal (English text), do NOT relax this test. Exceptions (`raise`) and "
        "docstrings are exempt on purpose -- see this module's docstring."
    )


def test_the_scan_really_looked_at_the_package():
    """Guard against a scanner that matches nothing: 'found no problem' must not mean 'looked at nothing'."""
    problems, calls, lits, raise_only = _scan_package()
    assert not problems, problems
    assert calls >= MIN_LOGGER_CALLS, f"only {calls} LOGGER.* calls found; the scanner is not seeing the package"
    assert lits >= MIN_LOGGER_LITERALS, f"only {lits} literals inside LOGGER calls; walk is broken"
    assert raise_only >= 1, (
        "no non-ASCII literal inside a `raise` was found, so the exception carve-out is untested and "
        "could be silently swallowing real hits"
    )


def test_the_scanner_flags_a_planted_character():
    """In-file injection check: the scanner must be able to fail, on all three planting sites."""
    assert scan('LOGGER.info("all ascii here")\n') == []
    assert scan('raise ValueError("中文 is fine in an exception")\n') == []
    assert scan('def f():\n    """doc 口径"""\n') == []

    logged = scan('LOGGER.info("read 721.7 MB/s, mAP/精度 unaffected")\n')
    assert len(logged) == 1 and logged[0][1] == "logger", logged

    # the case the narrow "literals inside a LOGGER call" rule would MISS: a helper-built message
    helper = scan('def _describe():\n    return "whole-image cache x 精度 frames"\n\n\nLOGGER.info(_describe())\n')
    assert len(helper) == 1 and helper[0][1] == "runtime", helper

    # an exception does not license a logged message
    mixed = scan('raise ValueError("x")\nLOGGER.warning("y 精度")\n')
    assert len(mixed) == 1 and mixed[0][1] == "logger", mixed
