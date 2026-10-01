# -*- coding: utf-8 -*-
"""Reproduce Section 4.1 estimation Monte Carlo and Figures 2--3."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config

from simulation_core import (
    SPATIAL_DESIGNS, ERROR_DISTRIBUTIONS,
    FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT,
    simulate_estimation_dgp, run_skater_path, summarize_estimation_replication,
)


def one_rep(design: str, error: str, rep: int, start_seed: int):
    d_idx = SPATIAL_DESIGNS.index(design)
    e_idx = ERROR_DISTRIBUTIONS.index(error)
    seed = int(start_seed + 100_000 * d_idx + 10_000 * e_idx + rep)
    data = simulate_estimation_dgp(spatial_design=design, error_distribution=error, seed=seed)
    result = run_skater_path(
        data, fve_threshold=FVE_DEFAULT, n_components=None,
        k_neighbors=K_NEIGHBORS_DEFAULT, min_size=MIN_SIZE_DEFAULT,
        K_max=None,
    )
    row = summarize_estimation_replication(data, result)
    row.update({"spatial_design": design, "error_distribution": error,
                "rep": rep + 1, "seed": seed})
    return row


def make_table1(raw: pd.DataFrame) -> pd.DataFrame:
    return (raw.groupby(["spatial_design", "error_distribution"], as_index=False)
            .agg(mean_K=("selected_K", "mean"), mean_ARI=("ARI", "mean"), mean_JI=("JI", "mean")))


def _method_boxplot(raw: pd.DataFrame, prefix: str, ylabel: str, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    designs = ["uniform", "non-uniform"]
    metrics = ["RMSE", "MAE"]
    methods = ["Global", "SKATER", "Oracle"]
    for r, design in enumerate(designs):
        # Pool the three disturbance distributions, matching the paper's compact method comparison.
        sub = raw[raw["spatial_design"] == design]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            cols = [f"{prefix}_{metric}_{m}" for m in methods]
            ax.boxplot([sub[x].dropna().to_numpy() for x in cols], tick_labels=methods, showfliers=False)
            ax.set_ylabel(metric)
            ax.set_title(f"({chr(97 + r*2+c)}) {metric} of {design} design")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=3000)
    ap.add_argument("--output-dir", type=Path, default=Path("results/estimation"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(d, e, r) for d in SPATIAL_DESIGNS for e in ERROR_DISTRIBUTIONS for r in range(args.reps)]
    if args.jobs == 1:
        rows = [one_rep(d, e, r, args.start_seed) for d, e, r in tasks]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=args.jobs, batch_size=1)(
                delayed(one_rep)(d, e, r, args.start_seed) for d, e, r in tasks
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "estimation_raw.csv", index=False)
    table1 = make_table1(raw)
    table1.to_csv(args.output_dir / "table1_group_recovery.csv", index=False)
    _method_boxplot(raw, "Beta", "Functional coefficient", args.output_dir / "figure2_beta_errors.png")
    _method_boxplot(raw, "Gamma", "Scalar coefficients", args.output_dir / "figure3_gamma_errors.png")
    print(table1.to_string(index=False))


if __name__ == "__main__":
    main()