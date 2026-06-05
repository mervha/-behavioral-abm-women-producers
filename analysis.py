"""
Post-hoc statistical analysis: Welch t-tests vs. baseline, Cohen's d,
and 1000-replication bootstrap confidence intervals.

Author: [BLINDED for peer review]
"""

import argparse
import csv
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def bootstrap_ci(arr, B=1000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(arr)
    boots = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(B)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results_policies_mid.csv")
    parser.add_argument("--output", default="data/analysis_summary.csv")
    parser.add_argument("--metric", default="mean_welfare",
                        choices=["mean_welfare", "survival_rate"])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rng = np.random.default_rng(42)

    results = []
    for mode in df["exit_mode"].unique():
        sub = df[df["exit_mode"] == mode]
        baseline = sub[sub["scenario"] == "baseline"][args.metric].values
        b_mean = float(np.mean(baseline))
        b_lo, b_hi = bootstrap_ci(baseline, rng=rng)

        for scenario in sub["scenario"].unique():
            if scenario == "baseline":
                continue
            arr = sub[sub["scenario"] == scenario][args.metric].values
            mean = float(np.mean(arr))
            lo, hi = bootstrap_ci(arr, rng=rng)
            t, p = stats.ttest_ind(arr, baseline, equal_var=False)
            d = cohens_d(arr, baseline)
            marginal_pct = 100.0 * (mean - b_mean) / b_mean if b_mean else 0.0

            results.append({
                "exit_mode": mode,
                "scenario": scenario,
                "mean": round(mean, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "baseline_mean": round(b_mean, 4),
                "marginal_pct": round(marginal_pct, 2),
                "welch_t": round(float(t), 3),
                "p_value": round(float(p), 4),
                "cohens_d": round(d, 3),
            })

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
