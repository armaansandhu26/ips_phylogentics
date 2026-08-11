"""Plot training curves from history JSON or stdout log."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UPDATE_RE = re.compile(
    r"update\s+(\d+)\s+return=([\d.]+)\s+max=([\d.]+).*?"
    r"loss=([-\d.]+)\s+entropy=([\d.]+)"
)
EVAL_RE = re.compile(
    r"eval@\s*(\d+)\s+hit=(\d+)/(\d+)\s+R²=([\d.]+)\s+logR²=([\d.]+)\s+"
    r"log_slope=([-\d.]+)\s+mean_ret=([\d.]+)"
)


@dataclass
class UpdateRow:
    step: int
    mean_return: float
    max_return: float
    loss: float
    entropy: float
    mean_ess: float | None = None


@dataclass
class EvalRow:
    step: int
    trajectories_hit: int
    r2: float
    mean_return: float
    log_r2: float | None = None
    log_slope: float | None = None


def parse_training_log(text: str) -> tuple[list[UpdateRow], list[EvalRow]]:
    updates = [
        UpdateRow(int(m[0]), float(m[1]), float(m[2]), float(m[3]), float(m[4]))
        for m in UPDATE_RE.findall(text)
    ]
    evals = [
        EvalRow(
            int(m[0]),
            int(m[1]),
            float(m[3]),
            float(m[6]),
            log_r2=float(m[4]),
            log_slope=float(m[5]),
        )
        for m in EVAL_RE.findall(text)
    ]
    return updates, evals


def history_to_series(history: list[dict]) -> tuple[list[UpdateRow], list[EvalRow]]:
    updates: list[UpdateRow] = []
    evals: list[EvalRow] = []
    for row in history:
        step = int(row["step"])
        updates.append(
            UpdateRow(
                step=step,
                mean_return=float(row["mean_return"]),
                max_return=float(row["max_return"]),
                loss=float(row["loss"]),
                entropy=float(row["entropy"]),
                mean_ess=float(row["mean_ess"]) if "mean_ess" in row else None,
            )
        )
        if "eval_r2" in row:
            evals.append(
                EvalRow(
                    step=step,
                    trajectories_hit=int(row["eval_traj_hit"]),
                    r2=float(row["eval_r2"]),
                    mean_return=float(row["eval_mean_return"]),
                    log_r2=float(row["eval_log_r2"]) if "eval_log_r2" in row else None,
                    log_slope=float(row["eval_log_slope"]) if "eval_log_slope" in row else None,
                )
            )
    return updates, evals


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training(
    updates: list[UpdateRow],
    evals: list[EvalRow],
    *,
    out_path: Path,
    title: str,
    smooth_window: int = 50,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    if updates:
        steps = [u.step for u in updates]
        returns = [u.mean_return for u in updates]
        axes[0, 0].plot(steps, returns, color="#74b9ff", linewidth=0.8, alpha=0.45, label="batch mean")
        if smooth_window > 1 and len(returns) >= smooth_window:
            smooth = _rolling_mean(returns, smooth_window)
            smooth_steps = steps[smooth_window - 1 :]
            axes[0, 0].plot(
                smooth_steps,
                smooth,
                color="#0984e3",
                linewidth=2.0,
                label=f"{smooth_window}-update mean",
            )
        axes[0, 0].set_ylabel("Mean return (batch)")
        axes[0, 0].set_title("Training return")
        axes[0, 0].legend(loc="lower right", fontsize=8)
        axes[0, 0].grid(alpha=0.25)

        ax_ent = axes[0, 1]
        ax_ent.plot(steps, [u.entropy for u in updates], color="#6c5ce7", linewidth=1.0, alpha=0.85)
        ax_ent.set_ylabel("Policy entropy", color="#6c5ce7")
        ax_ent.set_title("Entropy & IPS ESS")
        ax_ent.grid(alpha=0.25)
        ess_vals = [u.mean_ess for u in updates if u.mean_ess is not None]
        if ess_vals:
            ax_ess = ax_ent.twinx()
            ess_steps = [u.step for u in updates if u.mean_ess is not None]
            ax_ess.plot(ess_steps, ess_vals, color="#fdcb6e", linewidth=1.2, alpha=0.9, label="ESS")
            ax_ess.set_ylabel("IPS ESS", color="#fdcb6e")
            ax_ess.tick_params(axis="y", labelcolor="#fdcb6e")

    if evals:
        esteps = [e.step for e in evals]
        axes[1, 0].plot(esteps, [e.r2 for e in evals], "o-", color="#e17055", markersize=4, linewidth=1.2, label="density R²")
        if any(e.log_r2 is not None for e in evals):
            axes[1, 0].plot(
                [e.step for e in evals if e.log_r2 is not None],
                [e.log_r2 for e in evals if e.log_r2 is not None],
                "s--",
                color="#00b894",
                markersize=4,
                linewidth=1.2,
                label="log-log R²",
            )
        axes[1, 0].set_ylim(0, 1.02)
        axes[1, 0].set_ylabel("R²")
        axes[1, 0].set_title("Eval sampling fit")
        axes[1, 0].legend(loc="lower right", fontsize=8)
        axes[1, 0].grid(alpha=0.25)

        max_traj = max((e.trajectories_hit for e in evals), default=96)
        # Infer total from log if available; fallback for coverage ylim
        cov_ylim = max(96, int(max_traj * 1.05))
        axes[1, 1].plot(
            esteps,
            [e.trajectories_hit for e in evals],
            "o-",
            color="#00b894",
            markersize=4,
            linewidth=1.2,
        )
        axes[1, 1].set_ylim(0, cov_ylim)
        axes[1, 1].set_ylabel("Unique trajectories hit")
        axes[1, 1].set_title("Eval coverage")
        axes[1, 1].grid(alpha=0.25)

    for ax in axes[1, :]:
        ax.set_xlabel("Update")
    if updates:
        axes[0, 0].set_xlabel("Update")
        axes[0, 1].set_xlabel("Update")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_from_history(
    history_path: Path,
    out_path: Path,
    *,
    title: str,
    smooth_window: int = 50,
) -> tuple[list[UpdateRow], list[EvalRow]]:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    updates, evals = history_to_series(history)
    if not updates and not evals:
        raise ValueError(f"No metrics in {history_path}")
    plot_training(updates, evals, out_path=out_path, title=title, smooth_window=smooth_window)
    return updates, evals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=None, help="history.json from a run folder")
    parser.add_argument("--log", type=Path, default=None, help="train.log (fallback)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--title", type=str, default="IPS-GRPO v2 training")
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    if args.history is not None:
        out = args.out or args.history.parent / "training_curves.png"
        updates, evals = plot_training_from_history(
            args.history, out, title=args.title, smooth_window=args.smooth_window
        )
    elif args.log is not None:
        text = args.log.read_text(encoding="utf-8")
        updates, evals = parse_training_log(text)
        if not updates and not evals:
            raise SystemExit(f"No training lines found in {args.log}")
        out = args.out or args.log.with_suffix(".png")
        plot_training(updates, evals, out_path=out, title=args.title, smooth_window=args.smooth_window)
    else:
        raise SystemExit("Provide --history or --log")

    if updates:
        print(f"Plotted {len(updates)} updates ({updates[0].step}–{updates[-1].step})")
    if evals:
        best = max(evals, key=lambda e: e.r2)
        print(
            f"Evals: {len(evals)} — best R²={best.r2:.4f} @ step {best.step} "
            f"({best.trajectories_hit} hit)"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
