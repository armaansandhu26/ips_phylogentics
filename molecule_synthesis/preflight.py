"""Check the molecule-synthesis environment before starting a run."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

from .upstream import (
    RGFN_COMMIT,
    configure_runtime_environment,
    get_rgfn_commit,
    resolve_rgfn_root,
    validate_rgfn_root,
)


REQUIRED_IMPORTS = (
    "torch",
    "gin",
    "rdkit",
    "dgl",
    "torch_geometric",
    "wandb",
    "pandas",
)


def main(argv: list[str] | None = None) -> int:
    configure_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--strict-commit", action="store_true")
    args = parser.parse_args(argv)

    root = resolve_rgfn_root(args.rgfn_root)
    errors: list[str] = []
    try:
        validate_rgfn_root(root)
    except FileNotFoundError as exc:
        errors.append(str(exc))

    version = sys.version_info
    if (version.major, version.minor) != (3, 11):
        errors.append(f"Python 3.11 is required by upstream RGFN; found {platform.python_version()}")

    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # importing DGL can expose binary compatibility errors
            errors.append(f"Cannot import {module}: {type(exc).__name__}: {exc}")

    commit = get_rgfn_commit(root) if root.exists() else None
    if args.strict_commit and commit != RGFN_COMMIT:
        errors.append(f"RGFN commit is {commit or 'unknown'}, expected {RGFN_COMMIT}")

    print(f"python={platform.python_version()}")
    print(f"rgfn_root={root}")
    print(f"rgfn_commit={commit or 'unknown'}")
    if errors:
        print("preflight=FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("preflight=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
