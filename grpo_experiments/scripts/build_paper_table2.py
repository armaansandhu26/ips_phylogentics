#!/usr/bin/env python3
"""Build paper Table 2: matched-transform Pearson correlations for MIPS-GRPO vs GFlowNet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "final/paper/manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final/paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Paper comparison manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for table2.csv, table2.md, table2.tex, table2.json.",
    )
    return parser.parse_args()


def resolve_repo_path(path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics_path = resolve_repo_path(entry.get("metrics"))
    if metrics_path is not None and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    fallback = entry.get("metrics_fallback")
    if fallback is not None:
        return dict(fallback)
    raise FileNotFoundError(
        f"missing metrics for {entry['method_label']} ({entry['taxa']} taxa): "
        f"{entry.get('metrics')!r}"
    )


def pearson_linear(metrics: dict[str, Any]) -> float | None:
    for key in (
        "probability_vs_reward_pearson_vs_ideal",
        "model_probability_vs_reward_pearson_vs_ideal",
    ):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def pearson_loglog(metrics: dict[str, Any]) -> float | None:
    for key in (
        "log_probability_vs_log_reward_pearson_vs_ideal",
        "log_model_probability_vs_log_reward_pearson_vs_ideal",
    ):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def format_float(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "—"
    if value != value:  # NaN
        return "—"
    return f"{value:.{digits}f}"


def build_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reward_shifts = manifest.get("reward_shifts", {})
    for entry in manifest["comparisons"]:
        metrics = load_metrics(entry)
        taxa = int(entry["taxa"])
        rows.append(
            {
                "taxa": taxa,
                "method_id": entry["method_id"],
                "method_label": entry["method_label"],
                "reward": f"R(x) = {reward_shifts.get(str(taxa), '?')} + log L(x)",
                "pearson_linear": pearson_linear(metrics),
                "pearson_loglog": pearson_loglog(metrics),
                "ess_fraction": metrics.get("importance_ess_fraction"),
                "unique_signatures": metrics.get("unique_observed_signatures"),
                "samples": metrics.get("samples"),
                "metrics_path": str(resolve_repo_path(entry.get("metrics")) or ""),
                "note": metrics.get("note", entry.get("note", "")),
            }
        )
    rows.sort(key=lambda row: (row["taxa"], row["method_id"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "taxa",
        "method_id",
        "method_label",
        "reward",
        "pearson_linear",
        "pearson_loglog",
        "ess_fraction",
        "unique_signatures",
        "samples",
        "metrics_path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Table 2 — Matched-transform Pearson correlations",
        "",
        "Pathwise terminal probability vs reward on 1M self-sampled trajectories.",
        "Both methods report **linear** \(P(x)\) vs \(R(x)\) and **log-log** "
        "\(\\log P(x)\) vs \\(\\log R(x)\) under the same transform within each column.",
        "",
        "| Taxa | Method | Linear \(r\) | Log-log \(r\) | ESS | Unique sig. / 1M |",
        "|-----:|--------|------------:|-------------:|----:|-----------------:|",
    ]
    for row in rows:
        ess = row["ess_fraction"]
        ess_text = format_float(float(ess), digits=3) if ess is not None else "—"
        unique = row["unique_signatures"]
        unique_text = f"{int(unique):,}" if unique is not None else "—"
        lines.append(
            f"| {row['taxa']} | {row['method_label']} | "
            f"{format_float(row['pearson_linear'])} | "
            f"{format_float(row['pearson_loglog'])} | "
            f"{ess_text} | {unique_text} |"
        )
    lines.extend(
        [
            "",
            "## Caption draft",
            "",
            "Pearson correlation between pathwise implied terminal probability and "
            "terminal reward on 1M forward samples. Linear and log-log correlations "
            "are reported for both MIPS-GRPO and GFlowNet under matched transforms. "
            "At 27 taxa, linear axes compress dynamic range; log-log panels are the "
            "primary visual comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "% Paper Table 2 — auto-generated by build_paper_table2.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Matched-transform Pearson correlations between pathwise terminal "
        "probability and reward (1M self-sampled trajectories).}",
        "\\label{tab:matched-transform-pearson}",
        "\\begin{tabular}{rlrrrr}",
        "\\toprule",
        "Taxa & Method & Linear $r$ & Log-log $r$ & ESS & Unique / 1M \\\\",
        "\\midrule",
    ]
    current_taxa: int | None = None
    for row in rows:
        if current_taxa is not None and row["taxa"] != current_taxa:
            lines.append("\\addlinespace")
        current_taxa = row["taxa"]
        ess = row["ess_fraction"]
        ess_text = format_float(float(ess), digits=3) if ess is not None else "---"
        unique = row["unique_signatures"]
        unique_text = f"{int(unique):,}" if unique is not None else "---"
        method = row["method_label"].replace("MIPS-GRPO", "MIPS-GRPO")
        lines.append(
            f"{row['taxa']} & {method} & "
            f"{format_float(row['pearson_linear'])} & "
            f"{format_float(row['pearson_loglog'])} & "
            f"{ess_text} & {unique_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = build_rows(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "table2.json"
    csv_path = args.output_dir / "table2.csv"
    md_path = args.output_dir / "table2.md"
    tex_path = args.output_dir / "table2.tex"

    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    write_latex(tex_path, rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
