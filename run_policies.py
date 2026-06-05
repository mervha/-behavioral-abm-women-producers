"""
Policy scenarios runner. Runs four single-policy scenarios plus the
cumulative policy package, under endogenous and exogenous exit modes.

Author: [BLINDED for peer review]
"""

import argparse
import copy
import csv
import os
from model import ProducerSupplierModel, load_params, load_seeds


SCENARIOS = {
    "baseline":          {},
    "supply_quota":      {"supply_quota": True},
    "producer_signal":   {"producer_signal": True},
    "cooperative":       {"cooperative": True},
    "supplier_training": {"supplier_training": True},
    "cumulative_policy": {"supply_quota": True, "producer_signal": True,
                          "cooperative": True, "supplier_training": True},
}


def apply_calibration(params, calibration):
    if calibration == "mid":
        return params
    cal = params["calibrations"][calibration]
    out = copy.deepcopy(params)
    out["discrimination"]["mu_M"] = cal["mu_M"]
    out["supply_side"]["lambda_m"] = cal["lambda_m"]
    out["supply_side"]["omega_alpha"] = cal["omega_alpha"]
    out["supply_side"]["gamma"] = cal["gamma"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json")
    parser.add_argument("--seeds", default="seeds_used.txt")
    parser.add_argument("--calibration", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--output", default=None)
    parser.add_argument("--replications", type=int, default=200)
    args = parser.parse_args()

    out_path = args.output or f"data/results_policies_{args.calibration}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_params = load_params(args.params)
    params = apply_calibration(base_params, args.calibration)
    seeds = load_seeds(args.seeds)[: args.replications]

    rows = []
    for scenario, policies in SCENARIOS.items():
        for mode in ("endogenous", "exogenous"):
            for rep, seed in enumerate(seeds):
                model = ProducerSupplierModel(params, seed=seed,
                                              policies=policies, exit_mode=mode)
                out = model.run()
                rows.append({
                    "scenario": scenario,
                    "calibration": args.calibration,
                    "exit_mode": mode,
                    "replication": rep,
                    "seed": seed,
                    "survival_rate": round(out["survival_rate"], 4),
                    "mean_welfare": round(out["mean_welfare"], 4),
                })
            print(f"  [{scenario} | {mode}] done.")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
