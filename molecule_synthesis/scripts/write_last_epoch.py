#!/usr/bin/env python3
"""Write train/checkpoints/last_epoch.txt from a last_gfn.pt checkpoint.

Safe on login nodes: does not import rgfn, dgl, or openbabel.
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from pathlib import Path


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("builtins", "collections", "typing", "_codecs")):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Dummy
        if module.startswith("numpy"):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Dummy
        return _Dummy


def checkpoint_epoch(path: Path) -> int:
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(name for name in zf.namelist() if name.endswith("data.pkl"))
        with zf.open(pkl_name) as handle:
            unpickler = _CheckpointUnpickler(handle)
            unpickler.persistent_load = lambda _pid: _Dummy()
            data = unpickler.load()

    if not isinstance(data, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(data)!r}")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or "epoch" not in metrics:
        raise KeyError("Checkpoint metrics.epoch missing")
    return int(metrics["epoch"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to last_gfn.pt")
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    epoch = checkpoint_epoch(ckpt)
    out = ckpt.with_name("last_epoch.txt")
    out.write_text(f"{epoch}\n", encoding="utf-8")
    print(f"checkpoint_epoch {epoch}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
