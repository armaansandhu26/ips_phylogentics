"""Dependency-light distribution metrics for enumerable terminal spaces."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, median


def _kl(left: list[float], right: list[float]) -> float:
    return sum(p * math.log(p / q) for p, q in zip(left, right) if p > 0 and q > 0)


def exact_distribution_metrics(rows: list[dict], target: dict) -> dict[str, float | int]:
    target_rows = target["outcomes"]
    target_probability = {row["smiles"]: float(row["target_probability"]) for row in target_rows}
    observed = Counter(row["smiles"] for row in rows if row.get("smiles") is not None)
    n = sum(observed.values())
    if n == 0:
        raise ValueError("Cannot evaluate an empty sample")

    out_of_support = sum(count for smiles, count in observed.items() if smiles not in target_probability)
    keys = sorted(target_probability)
    empirical = [observed[key] / n for key in keys] + [out_of_support / n]
    expected = [target_probability[key] for key in keys] + [0.0]
    l1 = sum(abs(p - q) for p, q in zip(empirical, expected))
    midpoint = [(p + q) / 2 for p, q in zip(empirical, expected)]
    js = 0.5 * _kl(empirical, midpoint) + 0.5 * _kl(expected, midpoint)

    covered = [key for key in keys if observed[key] > 0]
    target_mass_covered = sum(target_probability[key] for key in covered)
    log_expected = [math.log(target_probability[key]) for key in covered]
    log_empirical = [math.log(observed[key] / n) for key in covered]
    if len(covered) >= 2:
        mean_x = fmean(log_expected)
        mean_y = fmean(log_empirical)
        var_x = sum((value - mean_x) ** 2 for value in log_expected)
        var_y = sum((value - mean_y) ** 2 for value in log_empirical)
        covariance = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(log_expected, log_empirical)
        )
        calibration_slope = covariance / var_x if var_x > 0 else 0.0
        log_probability_correlation = (
            covariance / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
        )
    else:
        calibration_slope = 0.0
        log_probability_correlation = 0.0

    proxy_values = [float(row["proxy"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    unique_rewards = {
        str(row["smiles"]): float(row["reward"])
        for row in rows
        if row.get("smiles") is not None
    }
    sorted_unique_rewards = sorted(unique_rewards.values(), reverse=True)
    target_high_reward = {row["smiles"] for row in target_rows if float(row["qed"]) >= 0.6}
    observed_high_reward = target_high_reward.intersection(observed)
    return {
        "target_n_outcomes": len(keys),
        "sample_support_coverage": len(covered) / len(keys) if keys else 0.0,
        "target_mass_covered": target_mass_covered,
        "out_of_support_fraction": out_of_support / n,
        "l1_to_reward_target": l1,
        "tv_to_reward_target": 0.5 * l1,
        "js_to_reward_target": js,
        "log_probability_correlation": log_probability_correlation,
        "log_probability_calibration_slope": calibration_slope,
        "median_proxy": median(proxy_values),
        "median_reward": median(rewards),
        # Discovery metrics must deduplicate molecules. Otherwise one repeated
        # high-reward mode can make a collapsed sampler look artificially good.
        "top_10_unique_mean_reward": fmean(sorted_unique_rewards[:10]),
        "top_100_unique_mean_reward": fmean(sorted_unique_rewards[:100]),
        "high_reward_outcomes_total": len(target_high_reward),
        "high_reward_outcomes_found": len(observed_high_reward),
    }
