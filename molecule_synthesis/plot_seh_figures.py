"""Generate RGFN-paper-style main and supplementary figures for seh_paper_medium."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .plot_seh_common import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_ORDER,
    available_methods,
    discovery_curves,
    load_json,
    load_mode_discovery_from_xlsx,
    load_sample_rows,
    load_training_history,
    resolve_run_dirs,
    save_figure,
    unique_terminal_log_points,
    unique_terminal_reward_points,
    linear_fit_stats,
    format_fit_annotation,
)

SEH_REWARD_BETA = 8.0


def _configure_matplotlib(dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.size": 8,
            "figure.dpi": dpi,
        }
    )
    return plt


def _checkpoint_epoch(summary: dict) -> int | None:
    metrics = summary.get("training_checkpoint_metrics", {})
    epoch = metrics.get("epoch")
    return int(epoch) if epoch is not None else None


def _filter_by_checkpoint(
    summaries: dict[str, dict],
    samples: dict[str, list[dict]],
    checkpoint_iter: int | None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    if checkpoint_iter is None:
        return summaries, samples
    target_epoch = checkpoint_iter - 1
    filtered_summaries = {
        method: summary
        for method, summary in summaries.items()
        if _checkpoint_epoch(summary) == target_epoch
    }
    filtered_samples = {
        method: rows for method, rows in samples.items() if method in filtered_summaries
    }
    return filtered_summaries, filtered_samples


def _load_suite_data(suite_dir: Path) -> tuple[dict, dict[str, Path], dict, dict, dict]:
    suite = load_json(suite_dir / "suite.json")
    run_dirs = resolve_run_dirs(suite_dir, suite)
    methods = available_methods(run_dirs)
    training = {method: load_training_history(run_dirs[method]) for method in methods}
    samples = {method: load_sample_rows(run_dirs[method]) for method in methods}
    summaries = {
        method: load_json(run_dirs[method] / "samples" / "summary.json")
        for method in methods
        if (run_dirs[method] / "samples" / "summary.json").is_file()
    }
    return suite, run_dirs, training, samples, summaries


def _plot_main_figure(
    plt,
    np,
    methods: list[str],
    training: dict,
    samples: dict,
    summaries: dict,
    run_dirs: dict[str, Path],
    output_dir: Path,
    dpi: int,
    checkpoint_rows: list[tuple],
) -> list[Path]:
    sampled_methods = [method for method in methods if samples.get(method)]
    if not sampled_methods:
        raise RuntimeError("No completed final-checkpoint samples found")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    # (a) Final proxy distribution — RGFN Fig 3 analogue
    proxy_values = [[float(row["proxy"]) for row in samples[method]] for method in sampled_methods]
    boxes = axes[0].boxplot(
        proxy_values,
        tick_labels=[METHOD_LABELS[method] for method in sampled_methods],
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "black", "linewidth": 1.0},
    )
    for patch, method in zip(boxes["boxes"], sampled_methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.85)
    axes[0].axhline(
        7.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="mode threshold = 7.0",
    )
    axes[0].set_ylabel("sampled sEH proxy")
    axes[0].set_title("(a) final-policy reward distribution")
    axes[0].grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")

    # (b) Mode / scaffold discovery during training — RGFN Fig 4 analogue
    any_curve = False
    for method in methods:
        xlsx_points = load_mode_discovery_from_xlsx(run_dirs[method])
        if xlsx_points:
            x_values, y_values = zip(*xlsx_points)
            axes[1].plot(
                x_values,
                y_values,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=3,
                linewidth=1.2,
                label=f"{METHOD_LABELS[method]} (leader modes)",
            )
            any_curve = True
            continue
        history = training.get(method, [])
        metric_key = "train/num_scaffolds_7"
        x_values = [int(row["_oracle_calls"]) for row in history if metric_key in row]
        y_values = [float(row[metric_key]) for row in history if metric_key in row]
        if x_values:
            axes[1].plot(
                x_values,
                y_values,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markevery=max(1, len(x_values) // 12),
                markersize=2.8,
                linewidth=1.05,
                label=f"{METHOD_LABELS[method]} (train scaffolds)",
            )
            any_curve = True
    if any_curve:
        axes[1].set_yscale("symlog", linthresh=1.0)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No training discovery curves available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=8,
            color="#666666",
        )
    axes[1].set_xlabel("training oracle calls")
    axes[1].set_ylabel("discovered structures (proxy > 7)")
    axes[1].set_title("(b) discovery during training")
    axes[1].grid(color="#B0B0B0", linewidth=0.45, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=6.0)

    # (c) Final evaluation summary — modes, scaffolds, ESS
    summary_methods = [method for method in methods if method in summaries]
    metric_specs = (
        ("n_modes", "modes"),
        ("n_scaffolds_proxy_gt_7", "scaffolds"),
        ("importance_ess_fraction", "ESS frac"),
    )
    x = np.arange(len(summary_methods))
    width = 0.24
    for offset, (key, label) in enumerate(metric_specs):
        values = [float(summaries[method].get(key, 0.0)) for method in summary_methods]
        axes[2].bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=label,
            color=["#555555", "#888888", "#BBBBBB"][offset],
            alpha=0.95,
        )
    axes[2].set_xticks(x, [METHOD_LABELS[method] for method in summary_methods], rotation=15, ha="right")
    axes[2].set_title("(c) final evaluation summary")
    axes[2].set_ylabel("metric value")
    axes[2].grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.35)
    axes[2].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper right")

    checkpoint_iter = max(
        _checkpoint_epoch(summaries[method]) + 1
        for method in summary_methods
        if _checkpoint_epoch(summaries[method]) is not None
    )
    n_sampled = summaries[summary_methods[0]].get("n_sampled", "?") if summary_methods else "?"
    fig.suptitle(
        f"seh_paper_medium seed 0 — main figure ({checkpoint_iter:,} iter, {n_sampled:,} eval samples)",
        fontsize=10,
    )
    fig.tight_layout(pad=0.85)
    paths = save_figure(fig, output_dir, "main_figure", dpi)
    plt.close(fig)
    return paths


def _plot_supplementary(
    plt,
    np,
    methods: list[str],
    training: dict,
    samples: dict,
    summaries: dict,
    output_dir: Path,
    dpi: int,
    checkpoint_rows: list[tuple],
    checkpoint_iter: int | None = None,
    all_samples: dict[str, list[dict]] | None = None,
) -> list[Path]:
    written: list[Path] = []

    # S1: training curves
    training_specs = (
        ("train/proxy_mean", "Train proxy mean", "sEH proxy", False),
        ("train/num_unique_molecules", "Train unique molecules (batch)", "unique molecules", True),
        ("train/loss", "Train loss", "loss", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for axis, (metric_key, title, ylabel, log_y) in zip(axes, training_specs):
        for method in methods:
            history = training.get(method, [])
            x_values = [int(row["_iter"]) for row in history if metric_key in row]
            y_values = [float(row[metric_key]) for row in history if metric_key in row]
            if not x_values:
                continue
            axis.plot(
                x_values,
                y_values,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.1,
            )
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel(ylabel)
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        if log_y:
            axis.set_yscale("log")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Supplementary: training diagnostics", fontsize=10)
    fig.tight_layout(pad=0.8)
    written.extend(save_figure(fig, output_dir, "supp_training_curves", dpi))
    plt.close(fig)

    # S2: IPS duplicate fraction
    if "count_ips_grpo" in training and training["count_ips_grpo"]:
        fig, axis = plt.subplots(figsize=(5.2, 2.8))
        history = training["count_ips_grpo"]
        x_values = [
            int(row["_iter"]) for row in history if "train/ips_duplicate_fraction" in row
        ]
        y_values = [
            float(row["train/ips_duplicate_fraction"])
            for row in history
            if "train/ips_duplicate_fraction" in row
        ]
        axis.plot(x_values, y_values, color=METHOD_COLORS["count_ips_grpo"], linewidth=1.2)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("iteration")
        axis.set_ylabel("ips_duplicate_fraction")
        axis.set_title("Supplementary: IPS-GRPO duplicate outcome fraction")
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        fig.tight_layout(pad=0.8)
        written.extend(save_figure(fig, output_dir, "supp_ips_duplicate_fraction", dpi))
        plt.close(fig)

    sampled_methods = [method for method in methods if samples.get(method)]
    if sampled_methods:
        # S3: support discovery + running mean proxy
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
        for method in sampled_methods:
            draw_index, unique_curve, mean_proxy_curve = discovery_curves(samples[method])
            plot_kwargs = {
                "color": METHOD_COLORS[method],
                "label": METHOD_LABELS[method],
                "marker": METHOD_MARKERS[method],
                "markevery": max(1, len(draw_index) // 12),
                "markersize": 2.6,
                "linewidth": 1.05,
            }
            axes[0].plot(draw_index, unique_curve, **plot_kwargs)
            axes[1].plot(draw_index, mean_proxy_curve, **plot_kwargs)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("samples drawn (log scale)")
        axes[0].set_ylabel("distinct SMILES")
        axes[0].set_title("(a) support discovery")
        axes[1].set_xlabel("samples drawn")
        axes[1].set_ylabel("cumulative mean proxy")
        axes[1].set_title("(b) running mean proxy")
        for axis in axes:
            axis.grid(color="#B0B0B0", linewidth=0.45, alpha=0.35)
            axis.legend(frameon=False, fontsize=6.5)
        fig.suptitle("Supplementary: final-checkpoint sampling trajectories", fontsize=10)
        fig.tight_layout(pad=0.8)
        written.extend(save_figure(fig, output_dir, "supp_sampling_discovery", dpi))
        plt.close(fig)

        # S4: top molecule frequencies
        fig, axis = plt.subplots(figsize=(5.8, 2.8))
        for method in sampled_methods:
            counts = Counter(row["smiles"] for row in samples[method] if row.get("smiles"))
            top_counts = counts.most_common(10)
            if not top_counts:
                continue
            y = np.arange(len(top_counts))
            offset = 0.18 if method == "grpo" else -0.18
            axis.barh(
                y + offset,
                [count for _, count in top_counts],
                height=0.35,
                color=METHOD_COLORS[method],
                alpha=0.85,
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_xlabel("sample count (log scale)")
        axis.set_ylabel("rank (top 10 SMILES)")
        axis.set_title("Supplementary: top sampled molecules")
        axis.grid(axis="x", color="#B0B0B0", linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
        fig.tight_layout(pad=0.8)
        written.extend(save_figure(fig, output_dir, "supp_top_molecules", dpi))
        plt.close(fig)

        # S5: phylo-style log proxy vs log reward — one panel per method
        scatter_samples = all_samples if all_samples is not None else samples
        panel_methods = [method for method in METHOD_ORDER if scatter_samples.get(method)]
        if panel_methods:
            reward_fit_stats: dict[str, dict] = {}

            # S5: phylo-style log proxy vs log reward — one panel per method
            fig, axes = plt.subplots(
                1,
                len(panel_methods),
                figsize=(2.8 * len(panel_methods), 3.4),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for axis, method in zip(axes[0], panel_methods):
                log_proxies, log_rewards = unique_terminal_log_points(scatter_samples[method])
                stats = linear_fit_stats(log_proxies, log_rewards)
                reward_fit_stats.setdefault(method, {})["log_proxy_vs_log_reward"] = stats
                if log_proxies:
                    axis.scatter(
                        log_proxies,
                        log_rewards,
                        s=22,
                        alpha=0.75,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                    )
                    axis.text(
                        0.03,
                        0.97,
                        format_fit_annotation(stats),
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.2,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "no valid samples",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=7,
                        color="#666666",
                    )
                axis.set_title(METHOD_LABELS[method], fontsize=8)
                axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
            for axis in axes[0]:
                axis.set_xlabel("log(proxy)")
            axes[0][0].set_ylabel("log reward")
            fig.suptitle(
                "Supplementary: unique-terminal log reward vs log proxy",
                fontsize=10,
            )
            fig.tight_layout(pad=0.85)
            written.extend(save_figure(fig, output_dir, "supp_log_proxy_vs_log_reward", dpi))
            plt.close(fig)

            # S5b: proxy vs log reward with exponential-reward oracle line (slope = beta)
            fig, axes = plt.subplots(
                1,
                len(panel_methods),
                figsize=(2.8 * len(panel_methods), 3.4),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for axis, method in zip(axes[0], panel_methods):
                proxies, log_rewards = unique_terminal_reward_points(scatter_samples[method])
                stats = linear_fit_stats(proxies, log_rewards, oracle_slope=SEH_REWARD_BETA)
                reward_fit_stats.setdefault(method, {})["proxy_vs_log_reward"] = stats
                if proxies:
                    axis.scatter(
                        proxies,
                        log_rewards,
                        s=22,
                        alpha=0.75,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                    )
                    x_min = min(proxies)
                    x_max = max(proxies)
                    if x_max > x_min:
                        x_line = [x_min, x_max]
                    else:
                        pad = max(0.5, x_min * 0.1)
                        x_line = [max(0.0, x_min - pad), x_max + pad]
                    axis.plot(
                        x_line,
                        [SEH_REWARD_BETA * x for x in x_line],
                        color="#333333",
                        linestyle="--",
                        linewidth=0.9,
                        label=f"oracle (β={SEH_REWARD_BETA:g})",
                    )
                    axis.text(
                        0.03,
                        0.97,
                        format_fit_annotation(stats, oracle_slope=SEH_REWARD_BETA),
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.2,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "no valid samples",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=7,
                        color="#666666",
                    )
                axis.set_title(METHOD_LABELS[method], fontsize=8)
                axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
            for axis in axes[0]:
                axis.set_xlabel("sEH proxy")
            axes[0][0].set_ylabel("log reward")
            axes[0][-1].legend(frameon=False, fontsize=6.5, loc="lower right")
            fig.suptitle(
                "Supplementary: unique-terminal log reward vs proxy (linear oracle)",
                fontsize=10,
            )
            fig.tight_layout(pad=0.85)
            written.extend(save_figure(fig, output_dir, "supp_proxy_vs_log_reward", dpi))
            plt.close(fig)

            stats_path = output_dir / "supp_reward_fit_stats.json"
            with stats_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "reward_beta": SEH_REWARD_BETA,
                        "methods": reward_fit_stats,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            written.append(stats_path)
            print(f"STATS={stats_path}")

    # S6: train vs sample proxy at checkpoints
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
    for method in methods:
        method_rows = [row for row in checkpoint_rows if row[0] == method]
        if not method_rows:
            continue
        iters = [row[1] for row in method_rows]
        train_vals = [row[2] for row in method_rows]
        sample_vals = [row[3] for row in method_rows]
        axes[0].plot(
            iters,
            train_vals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
            linewidth=1.2,
        )
        axes[1].plot(
            iters,
            sample_vals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
            linewidth=1.2,
        )
    for axis, title in zip(
        axes,
        (
            "Training batch proxy",
            f"Sampled proxy ({checkpoint_iter or 'mixed'} iter checkpoint)",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel("proxy")
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
    fig.suptitle("Supplementary: train vs sampling proxy", fontsize=10)
    fig.tight_layout(pad=0.8)
    written.extend(save_figure(fig, output_dir, "supp_train_vs_sample_proxy", dpi))
    plt.close(fig)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        default="molecule_synthesis/runs/seh_paper_medium",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--checkpoint-iter",
        type=int,
        default=None,
        help="Only include final-checkpoint samples from this iteration (e.g. 500).",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else suite_dir / "results" / "figures"
    )
    main_dir = output_root / "main"
    supp_dir = output_root / "supplementary"
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))

    plt = _configure_matplotlib(args.dpi)
    import matplotlib.pyplot as plt_module
    import numpy as np

    _, run_dirs, training, samples, summaries = _load_suite_data(suite_dir)
    methods = available_methods(run_dirs)
    if not methods:
        raise RuntimeError(f"No run directories found under {suite_dir}/suite.json")

    all_samples = dict(samples)

    checkpoint_rows = [
        ("grpo", 500, 8.05, 4.15),
        ("count_ips_grpo", 500, 7.77, 6.04),
        ("mips_grpo", 500, 5.305, 4.110),
        ("rgfn", 500, 7.126, 5.149),
        ("grpo", 2000, 3.43, 3.73),
        ("count_ips_grpo", 2000, 5.43, 4.80),
    ]
    if args.checkpoint_iter is not None:
        checkpoint_rows = [
            row for row in checkpoint_rows if row[1] == args.checkpoint_iter
        ]
        summaries, samples = _filter_by_checkpoint(
            summaries, samples, args.checkpoint_iter
        )

    written: list[Path] = []
    written.extend(
        _plot_main_figure(
            plt_module,
            np,
            methods,
            training,
            samples,
            summaries,
            run_dirs,
            main_dir,
            args.dpi,
            checkpoint_rows,
        )
    )
    written.extend(
        _plot_supplementary(
            plt_module,
            np,
            methods,
            training,
            samples,
            summaries,
            supp_dir,
            args.dpi,
            checkpoint_rows,
            args.checkpoint_iter,
            all_samples,
        )
    )

    for path in written:
        print(f"PLOT={path}")
    return written


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
