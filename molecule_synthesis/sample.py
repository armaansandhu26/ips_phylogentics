"""Sample terminal molecules from a trained run and write comparison metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import REPO_ROOT
from .evaluation import exact_distribution_metrics
from .upstream import configure_runtime_environment, resolve_rgfn_root, validate_rgfn_root


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--target-json", default=None)
    parser.add_argument("--mode-threshold", type=float, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--max-modes", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--scaffold-thresholds", default="")
    return parser


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _write_sampling_progress(
    progress_path: Path,
    *,
    n_sampled: int,
    n_requested: int,
    batch_idx: int,
    n_batches: int,
    n_unique: int,
    mean_proxy: float | None,
) -> None:
    progress = {
        "n_sampled": n_sampled,
        "n_requested": n_requested,
        "progress_fraction": n_sampled / n_requested if n_requested else 0.0,
        "batch": batch_idx,
        "n_batches": n_batches,
        "n_unique": n_unique,
        "mean_proxy": mean_proxy,
    }
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> Path:
    configure_runtime_environment()
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("n-samples and batch-size must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    rgfn_root = resolve_rgfn_root(args.rgfn_root or manifest.get("rgfn_root"))
    validate_rgfn_root(rgfn_root)
    cfg = Path(manifest["config"]).resolve()
    checkpoint = run_dir / "train" / "checkpoints" / "last_gfn.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    for path in (str(REPO_ROOT), str(rgfn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import gin
    import torch
    from tqdm import tqdm

    import rgfn  # noqa: F401
    from rgfn.shared.samplers.random_sampler import RandomSampler
    from rgfn.trainer.trainer import Trainer
    from rgfn.utils.helpers import seed_everything

    from molecule_synthesis import objectives  # noqa: F401
    from molecule_synthesis import optimizers  # noqa: F401
    from molecule_synthesis import minichem  # noqa: F401

    bindings = list(manifest["bindings"])
    if args.device is not None:
        bindings.append(f"Trainer.device={json.dumps(args.device)}")

    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples_path = sample_dir / "samples.jsonl"
    progress_path = sample_dir / "progress.json"

    rows: list[dict] = []
    unique_smiles: set[str] = set()
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    trainer = None
    with _working_directory(rgfn_root):
        gin.clear_config()
        seed_everything(int(manifest["seed"]))
        gin.parse_config_files_and_bindings([str(cfg)], bindings=bindings)
        # RGFN's one-hot action embedding caches a reference to its weight tensor.
        # Once populated, PyTorch registers that cache as a parameter and writes it
        # to the checkpoint.  A fresh model has an empty cache, so strict loading
        # through Trainer(resume_path=...) rejects the derived ``._cache`` key.
        # Sampling only needs the model weights, not optimizer/replay state.
        trainer = Trainer(resume_path=None)
        checkpoint_dict = torch.load(checkpoint, map_location=trainer.device)
        model_state = {
            key: value
            for key, value in checkpoint_dict["model"].items()
            if not key.endswith("._cache")
        }
        trainer.objective.load_state_dict(model_state)
        trainer.objective.eval()
        try:
            training_sampler = trainer.train_forward_sampler
            sampler = RandomSampler(
                policy=trainer.objective.forward_policy,
                env=training_sampler.env,
                reward=training_sampler.reward,
            )
            with samples_path.open("w", encoding="utf-8") as samples_handle:
                batch_idx = 0
                for trajectories in tqdm(
                    sampler.get_trajectories_iterator(args.n_samples, args.batch_size),
                    total=n_batches,
                    desc="Sampling batches",
                    unit="batch",
                ):
                    states = trajectories.get_last_states_flat()
                    reward_output = trajectories.get_reward_outputs()
                    with torch.no_grad():
                        actions = trajectories.get_actions_flat()
                        flat_log_pf = trainer.objective.forward_policy.compute_action_log_probs(
                            states=trajectories.get_non_last_states_flat(),
                            action_spaces=trajectories.get_forward_action_spaces_flat(),
                            actions=actions,
                        )
                        flat_log_pb = trainer.objective.backward_policy.compute_action_log_probs(
                            states=trajectories.get_non_source_states_flat(),
                            action_spaces=trajectories.get_backward_action_spaces_flat(),
                            actions=actions,
                        ).to(flat_log_pf.device)
                        trajectory_index = trajectories.get_index_flat().to(flat_log_pf.device)
                        log_pf = torch.zeros(
                            len(trajectories), dtype=flat_log_pf.dtype, device=flat_log_pf.device
                        )
                        log_pb = torch.zeros_like(log_pf)
                        log_pf.scatter_add_(0, trajectory_index, flat_log_pf)
                        log_pb.scatter_add_(0, trajectory_index, flat_log_pb)
                        log_importance_weight = (
                            reward_output.log_reward.to(flat_log_pf.device) + log_pb - log_pf
                        )
                    batch_rows: list[dict] = []
                    for state, log_reward, reward, proxy, path_log_pf, path_log_pb, log_weight in zip(
                        states,
                        reward_output.log_reward.detach().cpu().tolist(),
                        reward_output.reward.detach().cpu().tolist(),
                        reward_output.proxy.detach().cpu().tolist(),
                        log_pf.detach().cpu().tolist(),
                        log_pb.detach().cpu().tolist(),
                        log_importance_weight.detach().cpu().tolist(),
                    ):
                        molecule = getattr(state, "molecule", None)
                        batch_rows.append(
                            {
                                "smiles": getattr(molecule, "smiles", None),
                                "terminal_state": type(state).__name__,
                                "log_reward": float(log_reward),
                                "reward": float(reward),
                                "proxy": float(proxy),
                                "log_pf": float(path_log_pf),
                                "log_pb": float(path_log_pb),
                                "log_importance_weight": float(log_weight),
                            }
                        )
                    for row in batch_rows:
                        samples_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        if row["smiles"] is not None:
                            unique_smiles.add(row["smiles"])
                    samples_handle.flush()
                    rows.extend(batch_rows)
                    batch_idx += 1
                    running_mean_proxy = _mean([row["proxy"] for row in rows])
                    _write_sampling_progress(
                        progress_path,
                        n_sampled=len(rows),
                        n_requested=args.n_samples,
                        batch_idx=batch_idx,
                        n_batches=n_batches,
                        n_unique=len(unique_smiles),
                        mean_proxy=running_mean_proxy,
                    )
                    proxy_text = (
                        f"{running_mean_proxy:.3f}"
                        if running_mean_proxy is not None
                        else "n/a"
                    )
                    tqdm.write(
                        f"sampled {len(rows)}/{args.n_samples} "
                        f"({100 * len(rows) / args.n_samples:.1f}%) | "
                        f"unique={len(unique_smiles)} | "
                        f"mean_proxy={proxy_text}"
                    )
        finally:
            if trainer is not None:
                trainer.close()
            gin.clear_config()

    valid = [row for row in rows if row["smiles"] is not None]
    unique = {row["smiles"] for row in valid}
    summary = {
        "schema_version": 2,
        "method": manifest["method"],
        "seed": manifest["seed"],
        "n_requested": args.n_samples,
        "n_sampled": len(rows),
        "n_valid": len(valid),
        "valid_fraction": len(valid) / len(rows) if rows else 0.0,
        "n_unique": len(unique),
        "unique_fraction": len(unique) / len(valid) if valid else 0.0,
        "mean_log_reward": _mean([row["log_reward"] for row in rows]),
        "mean_proxy": _mean([row["proxy"] for row in rows]),
        "checkpoint": str(checkpoint),
    }
    summary["training_checkpoint_metrics"] = {
        key: value
        for key, value in checkpoint_dict.get("metrics", {}).items()
        if isinstance(value, (int, float))
    }
    for key in (
        "ips_duplicate_fraction",
        "ips_unique_outcomes",
        "ips_clipped_fraction",
        "ips_ess_fraction",
        "reverse_loss",
    ):
        if key in summary["training_checkpoint_metrics"]:
            summary[f"train_final_{key}"] = summary["training_checkpoint_metrics"][key]
    log_weights = [row["log_importance_weight"] for row in rows]
    if log_weights:
        max_log_weight = max(log_weights)
        stable_weights = [math.exp(value - max_log_weight) for value in log_weights]
        weight_sum = sum(stable_weights)
        weight_square_sum = sum(value * value for value in stable_weights)
        importance_ess = weight_sum * weight_sum / max(weight_square_sum, 1e-300)
        summary.update(
            importance_ess=importance_ess,
            importance_ess_fraction=importance_ess / len(stable_weights),
            log_importance_weight_mean=_mean(log_weights),
            log_importance_weight_std=statistics.pstdev(log_weights),
        )
    if args.target_json is not None:
        target_path = Path(args.target_json).expanduser().resolve()
        with target_path.open(encoding="utf-8") as handle:
            target = json.load(handle)
        summary.update(exact_distribution_metrics(rows, target))
        summary["target_json"] = str(target_path)
    if args.mode_threshold is not None:
        from .chemistry_evaluation import molecular_discovery_metrics

        scaffold_thresholds = tuple(
            float(value) for value in args.scaffold_thresholds.split(",") if value.strip()
        )
        discovery_metrics, modes = molecular_discovery_metrics(
            rows,
            mode_threshold=args.mode_threshold,
            similarity_threshold=args.similarity_threshold,
            max_modes=args.max_modes,
            top_k=args.top_k,
            scaffold_thresholds=scaffold_thresholds,
        )
        summary.update(discovery_metrics)
        modes_path = sample_dir / "modes.jsonl"
        with modes_path.open("w", encoding="utf-8") as handle:
            for mode in modes:
                handle.write(json.dumps(mode, sort_keys=True) + "\n")
        summary["modes_path"] = str(modes_path)
    summary_path = sample_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"SAMPLE_SUMMARY={summary_path}")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
