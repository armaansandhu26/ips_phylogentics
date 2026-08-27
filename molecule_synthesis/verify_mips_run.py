"""Verify the configuration and basic health of a completed paper MIPS run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_BINDINGS = {
    "Trainer.n_iterations=4000",
    "Trainer.train_forward_n_trajectories=100",
    "Trainer.train_replay_n_trajectories=0",
    "ReactionEnv.max_num_reactions=4",
    "train/forward/RandomSampler.policy=%forward_policy",
    "MIPSOptimizer.forward_lr=0.0001",
    "MIPSOptimizer.reverse_lr=0.001",
    "MIPSOptimizer.reverse_train_epochs=4",
    "MIPSGRPOObjective.advantage_normalization=\"running\"",
    "MIPSGRPOObjective.running_scale_decay=0.9",
    "MIPSGRPOObjective.exploration_rate=0.0",
    "MIPSGRPOObjective.separate_reverse_updates=True",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify(suite_dir: Path, seed: int) -> tuple[Path, dict]:
    suite_path = suite_dir / "suite.json"
    if not suite_path.is_file():
        raise RuntimeError(f"Missing completed suite manifest: {suite_path}")
    suite = _load(suite_path)
    try:
        run_dir = Path(suite["runs"]["mips_grpo"][str(seed)])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"No completed MIPS-GRPO seed {seed} in {suite_path}") from exc

    manifest = _load(run_dir / "manifest.json")
    summary = _load(run_dir / "samples" / "summary.json")
    bindings = set(manifest.get("bindings", ()))
    missing = sorted(EXPECTED_BINDINGS - bindings)
    errors = []
    if manifest.get("method") != "mips_grpo" or int(manifest.get("seed", -1)) != seed:
        errors.append("run manifest method/seed does not match requested MIPS run")
    if missing:
        errors.append("missing frozen MIPS bindings: " + ", ".join(missing))
    if any("ExploratoryPolicy" in binding for binding in bindings):
        errors.append("MIPS run used an exploratory sampler")
    if int(summary.get("n_requested", -1)) != 100_000:
        errors.append("final-checkpoint evaluation did not request 100,000 samples")
    if int(summary.get("n_sampled", -1)) != 100_000:
        errors.append("final-checkpoint evaluation did not produce 100,000 samples")
    if float(summary.get("valid_fraction", 0.0)) < 0.99:
        errors.append("fewer than 99% of final samples are valid")
    if int(summary.get("n_unique", 0)) <= 1:
        errors.append("final policy collapsed to one or zero unique molecules")

    numeric_keys = (
        "mean_proxy",
        "importance_ess_fraction",
        "log_importance_weight_mean",
        "log_importance_weight_std",
        "train_final_reverse_loss",
    )
    for key in numeric_keys:
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite diagnostic: {key}")
    if errors:
        raise RuntimeError("MIPS_HEALTH=FAILED\n- " + "\n- ".join(errors))
    return run_dir, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir", default="molecule_synthesis/runs/seh_paper_main"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_dir, summary = verify(Path(args.suite_dir).expanduser().resolve(), args.seed)
    print("MIPS_HEALTH=PASS")
    print(f"RUN_DIR={run_dir}")
    for key in (
        "n_sampled",
        "n_unique",
        "n_modes",
        "mean_proxy",
        "importance_ess_fraction",
        "train_final_reverse_loss",
    ):
        print(f"{key}={summary.get(key, 'not_available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
