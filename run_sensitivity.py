# -*- coding: utf-8 -*-
"""Reproduce Section 4.1.4 one-factor sensitivity analysis and Figures 8--10."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config

from simulation_core import (
    SPATIAL_DESIGNS, SENS_FVE_GRID, SENS_KNN_GRID, SENS_MIN_SIZE_GRID,
    FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT,
    simulate_estimation_dgp, run_skater_path, partition_to_labels, pairwise_jaccard,
)
from sklearn.metrics import adjusted_rand_score


def scenarios():
    out = []
    for x in SENS_FVE_GRID:
        out.append(("FVE", float(x), float(x), K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT))
    for x in SENS_KNN_GRID:
        out.append(("KNN", int(x), FVE_DEFAULT, int(x), MIN_SIZE_DEFAULT))
    for x in SENS_MIN_SIZE_GRID:
        out.append(("MIN_SIZE", int(x), FVE_DEFAULT, K_NEIGHBORS_DEFAULT, int(x)))
    return out


def one_rep(design, scen_idx, scen, rep, start_seed):
    typ, value, fve, knn, min_size = scen
    d_idx = SPATIAL_DESIGNS.index(design)
    seed = int(start_seed + 100_000*d_idx + 1_000*scen_idx + rep)
    data = simulate_estimation_dgp(spatial_design=design, error_distribution="normal_N025", seed=seed)
    result = run_skater_path(data, fve_threshold=fve, k_neighbors=knn, min_size=min_size, K_max=None)
    true = data["true_group"]
    pred = partition_to_labels(result["best_record"]["partition"], len(true))
    return {
        "spatial_design": design, "sensitivity_type": typ, "parameter_value": value,
        "rep": rep+1, "seed": seed, "selected_K": result["best_K"],
        "ARI": float(adjusted_rand_score(true, pred)), "JI": pairwise_jaccard(true, pred),
        "M_fpca": int(result["best_record"]["M_fpca"]),
        "used_k": int(result["used_k_neighbors"]),
    }


def plot_one(summary, typ, path):
    sub = summary[summary["sensitivity_type"] == typ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for design, g in sub.groupby("spatial_design"):
        g = g.sort_values("parameter_value")
        axes[0].plot(g["parameter_value"], g["ARI"], marker="o", label=design)
        axes[1].plot(g["parameter_value"], g["JI"], marker="o", label=design)
    for ax, y in zip(axes, ["ARI", "JI"]):
        ax.set_ylabel(y); ax.set_ylim(0, 1.05); ax.grid(alpha=0.2); ax.legend(frameon=False)
    xlabel = {"FVE":"FPCA cumulative FVE threshold", "KNN":"k in kNN adjacency graph", "MIN_SIZE":"Minimum group size"}[typ]
    axes[0].set_xlabel(xlabel); axes[1].set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=6000)
    ap.add_argument("--output-dir", type=Path, default=Path("results/sensitivity"))
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    scens = scenarios()
    tasks = [(d, i, s, r) for d in SPATIAL_DESIGNS for i,s in enumerate(scens) for r in range(args.reps)]
    if args.jobs == 1:
        rows = [one_rep(d,i,s,r,args.start_seed) for d,i,s,r in tasks]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=args.jobs, batch_size=1)(delayed(one_rep)(d,i,s,r,args.start_seed) for d,i,s,r in tasks)
    raw = pd.DataFrame(rows); raw.to_csv(args.output_dir / "sensitivity_raw.csv", index=False)
    summary = (raw.groupby(["spatial_design","sensitivity_type","parameter_value"], as_index=False)
               .agg(ARI=("ARI","mean"), JI=("JI","mean"), mean_K=("selected_K","mean"), mean_M=("M_fpca","mean")))
    summary.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    plot_one(summary, "FVE", args.output_dir / "figure8_fve.png")
    plot_one(summary, "KNN", args.output_dir / "figure9_knn.png")
    plot_one(summary, "MIN_SIZE", args.output_dir / "figure10_min_size.png")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()