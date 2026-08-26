"""Download and validate optional assets before submitting long jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import REPO_ROOT
from .upstream import configure_runtime_environment, resolve_rgfn_root, validate_rgfn_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--seh", action="store_true", help="Fetch the public sEH proxy weights")
    args = parser.parse_args(argv)
    if not args.seh:
        parser.error("select at least one asset, currently: --seh")

    configure_runtime_environment()
    rgfn_root = resolve_rgfn_root(args.rgfn_root)
    validate_rgfn_root(rgfn_root)
    for path in (str(REPO_ROOT), str(rgfn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    if args.seh:
        from rgfn.gfns.reaction_gfn.proxies.seh_proxy import load_original_model

        model = load_original_model(cache=True)
        n_parameters = sum(parameter.numel() for parameter in model.parameters())
        cache_path = (
            rgfn_root
            / "rgfn/gfns/reaction_gfn/proxies/cache/bengio2021flow_proxy.pkl.gz"
        )
        if not cache_path.is_file():
            raise RuntimeError(f"sEH proxy load returned without creating {cache_path}")
        print(f"SEH_PROXY={cache_path}")
        print(f"SEH_PROXY_BYTES={cache_path.stat().st_size}")
        print(f"SEH_PROXY_PARAMETERS={n_parameters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
