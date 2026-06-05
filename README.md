# -behavioral-abm-women-producers
Agent-based behavioral model for micro-scale women producers — replication materials
# Behavioral Agent-Based Model for Micro-scale Women Producers

Replication materials for an agent-based behavioral economics model 
of micro-scale women producers operating at the intersection of a 
male-concentrated supply corridor and a female-concentrated demand side.

## Author

[BLINDED for peer review]

## Overview

The model simulates 30 heterogeneous women producers and 30 suppliers 
over a 120-month horizon on a Watts–Strogatz small-world network. 
It evaluates four policy interventions (Supply Quota, Producer Signal, 
Cooperative Structure, Supplier Training) and their cumulative effect 
under two exit modes (endogenous, exogenous), across three discrimination 
calibrations (low, mid, high).

## Requirements

- Python 3.12
- Mesa 3.5
- NetworkX, NumPy, Pandas, SciPy

Install dependencies:

    pip install -r requirements.txt

## Repository structure

    .
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── model.py                 # Main ABM (Mesa Model and Agent classes)
    ├── run_baseline.py          # Baseline (no-intervention) runner
    ├── run_policies.py          # Four policy scenarios + cumulative
    ├── analysis.py              # Welch t-tests, Cohen's d, bootstrap CIs
    ├── params.json              # All parameters from Table 1
    ├── seeds_used.txt           # Random seeds for 200 replications
    └── data/
        ├── results_baseline.csv
        ├── results_policies.csv
        ├── sensitivity.csv
        └── analysis_summary.csv

## Reproducing the main results

    # Baseline (200 replications)
    python run_baseline.py

    # All policy scenarios (mid calibration)
    python run_policies.py --calibration mid

    # Sensitivity analysis
    python run_policies.py --calibration low
    python run_policies.py --calibration high

    # Statistical analysis (Welch t-tests, Cohen's d, bootstrap CIs)
    python analysis.py --input data/results_policies.csv

## Parameters

All model parameters and their justifications are documented in 
`params.json` and in Table 2 of the manuscript.

## Output files

- `data/results_baseline.csv`: survival rate, cumulative welfare, exit timing per replication
- `data/results_policies.csv`: per-scenario aggregates across 6 scenarios × 2 exit modes × 200 replications
- `data/sensitivity.csv`: marginal welfare effects under low / mid / high discrimination calibrations
- `data/analysis_summary.csv`: Welch t-test, Cohen's d, and bootstrap CI summary

## Citation

[BLINDED — citation information will be added upon acceptance.]

## License

MIT
