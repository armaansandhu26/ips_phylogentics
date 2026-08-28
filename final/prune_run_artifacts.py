"""Prune heavy regenerable artifacts under ``final/runs/``.

Keeps one final checkpoint per run directory (highest epoch). Optionally removes
other gitignored bulk (metrics jsonl, training resume state, eval caches).

Run before committing paper runs::

    python -m final.prune_run_artifacts --dry-run
    python -m final.prune_run_artifacts --apply --bulk
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from final.paths import RUNS_DIR

CHECKPOINT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^checkpoint_epoch(?P<epoch>\d+)\.pt$"), "checkpoint_epoch"),
    (re.compile(r"^learned_reverse_epoch(?P<epoch>\d+)\.pt$"), "learned_reverse_epoch"),
    (re.compile(r"^checkpoint_(?P<epoch>\d+)\.pt$"), "checkpoint"),
)

BULK_RELATIVE_PATHS = (
    "training_state.json",
    "trainer_state.pt",
    "learned_reverse_state.pt",
    "metrics.jsonl",
    "metrics_detailed.jsonl",
)

BULK_GLOBS = (
    "eval_dumps/*.pending",
    "tb_log",
    "backup",
    "wandb",
    "raw_samples",
    "plots/**/cache",
)


@dataclass
class PrunePlan:
    keep: list[Path] = field(default_factory=list)
    delete: list[Path] = field(default_factory=list)


def _epoch_from_name(name: str) -> tuple[str, int] | None:
    for pattern, prefix in CHECKPOINT_PATTERNS:
        match = pattern.match(name)
        if match:
            return prefix, int(match.group("epoch"))
    return None


def plan_checkpoints(runs_dir: Path) -> PrunePlan:
    plan = PrunePlan()
    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = {}

    for path in sorted(runs_dir.rglob("*.pt")):
        parsed = _epoch_from_name(path.name)
        if parsed is None:
            continue
        prefix, epoch = parsed
        key = (path.parent, prefix)
        groups.setdefault(key, []).append((epoch, path))

    for entries in groups.values():
        entries.sort(key=lambda item: item[0])
        *older, (_, newest) = entries
        plan.keep.append(newest)
        plan.delete.extend(path for _, path in older)

    return plan


def plan_bulk(runs_dir: Path) -> PrunePlan:
    plan = PrunePlan()

    for run_root in runs_dir.rglob("*"):
        if not run_root.is_dir():
            continue
        for relative in BULK_RELATIVE_PATHS:
            path = run_root / relative
            if path.is_file():
                plan.delete.append(path)
        for pattern in BULK_GLOBS:
            for path in run_root.glob(pattern):
                if path.is_file():
                    plan.delete.append(path)
                elif path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_file():
                            plan.delete.append(child)

    deduped = sorted(set(plan.delete))
    plan.delete = deduped
    return plan


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f} KB"
    return f"{num_bytes} B"


def _total_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _apply_deletes(paths: list[Path]) -> None:
    for path in sorted(paths):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            continue

    # Remove empty cache / tb_log / backup dirs left behind.
    for path in sorted({p.parent for p in paths}, key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue
        if path.name not in {"cache", "tb_log", "backup", "wandb", "raw_samples", "eval_dumps"}:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
        except OSError:
            continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help=f"Runs root (default: {RUNS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without deleting files (default)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files listed in the plan",
    )
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="Also remove gitignored bulk (jsonl, training_state, caches, ...)",
    )
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        parser.error("use either --dry-run or --apply, not both")

    dry_run = not args.apply
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        parser.error(f"runs dir not found: {runs_dir}")

    checkpoint_plan = plan_checkpoints(runs_dir)
    bulk_plan = plan_bulk(runs_dir) if args.bulk else PrunePlan()

    delete_paths = sorted(set(checkpoint_plan.delete + bulk_plan.delete))
    keep_paths = sorted(set(checkpoint_plan.keep))

    print(f"runs dir: {runs_dir}")
    print(f"mode: {'dry-run' if dry_run else 'apply'}")
    print(f"checkpoints kept: {len(keep_paths)} ({_format_bytes(_total_size(keep_paths))})")
    print(
        "checkpoints deleted: "
        f"{len(checkpoint_plan.delete)} ({_format_bytes(_total_size(checkpoint_plan.delete))})"
    )
    if args.bulk:
        print(
            "bulk deleted: "
            f"{len(bulk_plan.delete)} ({_format_bytes(_total_size(bulk_plan.delete))})"
        )
    print(f"total deleted: {len(delete_paths)} ({_format_bytes(_total_size(delete_paths))})")

    if keep_paths:
        print("\nKeeping final checkpoints:")
        for path in keep_paths:
            rel = path.relative_to(runs_dir)
            print(f"  keep  {rel}")

    if delete_paths:
        print("\nDeleting:")
        for path in delete_paths[:40]:
            rel = path.relative_to(runs_dir)
            print(f"  delete  {rel}")
        if len(delete_paths) > 40:
            print(f"  ... and {len(delete_paths) - 40} more")

    if dry_run:
        print("\nRe-run with --apply to delete files.")
        return 0

    _apply_deletes(delete_paths)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
