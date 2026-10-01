# -*- coding: utf-8 -*-
"""Run the county-level corn-yield application from transparent CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from simulation_core import (
    load_empirical_csvs, run_skater_path, partition_to_labels,
    recover_beta_curve, joint_homogeneity_test,
)


def _paper_output_order(record):
    """Return raw group indices in the display order used by the manuscript.

    Regression-group labels are mathematically arbitrary. The internal SKATER
    implementation uses a deterministic canonical ordering; for the six-group
    empirical solution we relabel the same partition to the manuscript's display
    order so group sizes/gamma values can be compared line-by-line with the paper.
    """
    sizes = [len(g) for g in record["partition"]]
    if len(sizes) == 6 and sizes == [35, 29, 66, 27, 15, 30]:
        return [0, 4, 5, 1, 3, 2]
    return list(range(len(sizes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/empirical"))
    ap.add_argument("--bootstrap", type=int, default=0, help="Set to 1000 to reproduce the paper joint-homogeneity test.")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_empirical_csvs(
        args.data_dir / "corn_panel.csv",
        args.data_dir / "temperature_curves.csv",
        args.data_dir / "county_nodes.csv",
        args.data_dir / "distance_matrix.csv",
    )
    result = run_skater_path(data, fve_threshold=.95, k_neighbors=5, min_size=10, rho_edge=.5, K_max=None)
    rec = result["best_record"]
    raw_labels = partition_to_labels(rec["partition"], len(data["county_ids"]))
    order = _paper_output_order(rec)
    raw_to_paper = {raw: paper for paper, raw in enumerate(order)}
    labels = np.array([raw_to_paper[int(x)] for x in raw_labels], dtype=int)

    county = pd.DataFrame({
        "CountyI": data["county_ids"], "State": data["state_names"],
        "County": data["county_names"], "estimated_group": labels + 1,
    })
    county.to_csv(args.output_dir / "county_group_membership.csv", index=False)

    bic = pd.DataFrame([{k: r[k] for k in ["K", "SSR", "TSS", "R2", "BIC", "M_fpca", "q"]} for r in result["records"]])
    bic.to_csv(args.output_dir / "bic_path.csv", index=False)

    M = rec["M_fpca"]
    basis = result["prepared_data"]["fpca_basis"]
    gamma, beta = [], []
    for paper_g, raw_g in enumerate(order, start=1):
        fit = rec["group_results"][raw_g]
        th = fit["theta"]
        gamma.append({"group": paper_g, "size": len(rec["partition"][raw_g]), "gamma_avgPRCP": float(th[M])})
        beta.append(recover_beta_curve(th, basis))
    pd.DataFrame(gamma).to_csv(args.output_dir / "group_gamma_coefficients.csv", index=False)
    pd.DataFrame(beta, columns=[f"day_{d:03d}" for d in range(1, 366)]).to_csv(args.output_dir / "group_beta_curves.csv", index=False)

    print(f"FVE=0.95 retained M={M}; edge mode={result['edge_weight_mode']}; selected K={result['best_K']}")
    if args.bootstrap > 0:
        test = joint_homogeneity_test(
            data, B=args.bootstrap, random_state=args.seed,
            fve_threshold=.95, k_neighbors=5, min_size=10, rho_edge=.5, K_max=None,
        )
        pd.DataFrame([{k: v for k, v in test.items() if k not in {"observed_result", "failures"}}]).to_csv(
            args.output_dir / "joint_homogeneity_test.csv", index=False
        )
        print("bootstrap p-value =", test["p_value"])


if __name__ == "__main__":
    main()