#!/usr/bin/env python3
"""Evaluate GFlowNet marginal log-likelihood (MLL) on DS1 for paper P1 reconciliation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYLOGFN_PAPER_MLL = -7108.95
MRBAYES_MLL = -7108.42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="GFlowNet run directory containing checkpoints/ and config.yaml.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "dataset/benchmark_datasets/DS1.pickle",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: latest checkpoints/checkpoint_*.pt).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Import src from here (default: <run-dir>/backup if present, else repo src).",
    )
    parser.add_argument(
        "--traj-size",
        type=int,
        default=1024,
        help="Trajectories per MLL estimate (PhyloGFN default: 1024).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="Independent MLL replicates (PhyloGFN final eval uses 5).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: <run-dir>/mll_eval.json).",
    )
    return parser.parse_args()


def configure_imports(source_root: Path) -> None:
    source_root = source_root.resolve()
    if (source_root / "src").is_dir():
        import_root = source_root
    elif source_root.name == "src":
        import_root = source_root.parent
    else:
        import_root = source_root
    if not (import_root / "src").is_dir():
        raise FileNotFoundError(f"missing src package under {import_root}")
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        return checkpoint
    candidates = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    return candidates[-1]


def resolve_config(run_dir: Path) -> Path:
    for name in ("config.yaml", "resolved_config.yaml"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing config under {run_dir}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    config_path = resolve_config(run_dir)
    source_root = args.source_root
    if source_root is None:
        backup = run_dir / "backup"
        source_root = backup if backup.is_dir() else REPO_ROOT
    configure_imports(source_root)

    from src.configs.defaults import get_cfg_defaults
    from src.env import build_env
    from src.gfn.build import build_gfn
    from src.gfn.gfn_evaluator import GFNEvaluator
    from src.gfn.rollout_worker_phylo import RolloutWorker
    from src.utils.utils import correct_cfg_data, load_sequences

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device(args.device)
    sequences = load_sequences(str(args.dataset))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(config_path))
    cfg.AMP = False
    cfg.LOGGING.ENABLE_TENSORBOARD = False
    cfg = correct_cfg_data(sequences, 1, cfg)

    env = build_env(cfg, sequences)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    generator.load(str(checkpoint))
    generator.eval()

    rollout_worker = RolloutWorker(env)
    evaluator = GFNEvaluator(
        cfg.GFN.MODEL.EVALUATION,
        rollout_worker,
        generator,
        verbose=True,
    )

    mlls = [
        evaluator.evaluate_marginal_likelihood(args.traj_size)
        for _ in range(args.replicates)
    ]
    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset": str(args.dataset),
        "source_root": str(source_root),
        "traj_size": args.traj_size,
        "replicates": args.replicates,
        "mll_values": mlls,
        "mll_mean": float(np.mean(mlls)),
        "mll_std": float(np.std(mlls)),
        "phylogfn_paper_reference_mll": PHYLOGFN_PAPER_MLL,
        "mrbayes_reference_mll": MRBAYES_MLL,
        "delta_vs_phylogfn_paper": float(np.mean(mlls) - PHYLOGFN_PAPER_MLL),
        "delta_vs_mrbayes": float(np.mean(mlls) - MRBAYES_MLL),
    }

    output = args.output or (run_dir / "mll_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
