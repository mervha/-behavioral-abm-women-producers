"""
Baseline runner: no policy interventions, endogenous and exogenous exit modes.
Author: [BLINDED for peer review]
"""

import argparse
import csv
import os
from model import ProducerSupplierModel, load_params, load_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json")
    parser.add_argument("--seeds", default="seeds_used.txt")
    parser.add_argument("--output", default="data/results_baseline.csv")
    parser.add_argument("--replications", type=int, default=200)
    args = parser.parse_args()

    params = load_params(args.params)
    seeds = load_seeds(args.seeds)[: args.replications]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = []
    for mode in ("endogenous", "exogenous"):
        for rep, seed in enumerate(seeds):
            model = ProducerSupplierModel(params, seed=seed, policies={}, exit_mode=mode)
            out = model.run()
            rows.append({
                "scenario": "baseline",
                "exit_mode": mode,
                "replication": rep,
                "seed": seed,
                "survival_rate": round(out["survival_rate"], 4),
                "mean_welfare": round(out["mean_welfare"], 4),
                "survival_new": round(out["by_class"].get("new", {}).get("survival", 0), 4),
                "survival_mid": round(out["by_class"].get("mid", {}).get("survival", 0), 4),
                "survival_exp": round(out["by_class"].get("experienced", {}).get("survival", 0), 4),
            })
        print(f"  [{mode}] 200 replications done.")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
