"""AST fingerprint of a code tree, ignoring comments and docstrings.

Usage:
    python ast_fingerprint.py <root> <out.json> [--compare <other.json>]

The fingerprint hashes the parsed AST of every ``*.py`` file after docstring
constants are stripped, so comment/docstring translation does not change it.
Any change in executable code does.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys


def strip_docstrings(node):
    for child in ast.walk(node):
        if isinstance(child, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = child.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                child.body = body[1:] or [ast.Pass()]
    return node


def blank_strings(node):
    """Replace every string constant by '' (keeps structure, ignores wording).

    This lets us verify that translating comments, docstrings and user-facing
    messages did not touch any executable logic.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            child.value = ""
    return node


def fp(path: str) -> str:
    src = open(path, "r", encoding="utf-8").read()
    tree = blank_strings(strip_docstrings(ast.parse(src)))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def walk(root: str):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for name in sorted(filenames):
            if name.endswith(".py"):
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root).replace("\\", "/")
                out[rel] = fp(p)
    return out


def main() -> int:
    root, out_path = sys.argv[1], sys.argv[2]
    data = walk(root)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"wrote {len(data)} fingerprints to {out_path}")
    if "--compare" in sys.argv:
        other = json.load(open(sys.argv[sys.argv.index("--compare") + 1], encoding="utf-8"))
        keys = sorted(set(data) | set(other))
        same = diff = 0
        for k in keys:
            if data.get(k) == other.get(k):
                same += 1
            else:
                diff += 1
                print(f"CHANGED {k} ({other.get(k, 'missing')[:8]} -> {data.get(k, 'missing')[:8]})")
        print(f"identical={same} changed={diff}")
        return 0 if diff == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
