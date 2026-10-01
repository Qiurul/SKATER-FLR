# Simulation code for SKATER–FLR

This repository reproduces the simulation study and empirical application in the manuscript
*SKATER–FLR: Functional Linear Regression with Spatially Contiguous Coefficient Groups*.
It contains the data-generating mechanisms, the SKATER–FLR estimation procedure, sensitivity experiments, the joint homogeneity test, and the real-data inputs used in the paper.

## Simulation design

- `n = 200`, `T = 50`, functional grid size `R = 100`, Fourier dimension `L = 10`, and `p = 3` scalar covariates.
- The true number of spatial groups in the estimation experiment is `K0 = 4`.
- The default FPCA threshold is `FVE = 0.95`.
- Spatial adjacency is constructed from coordinates using a symmetrized geographic `k = 5` nearest-neighbour graph.
- The minimum admissible group size is `n_min = 10`.
- Uniform and non-uniform spatial designs use group probabilities `(0.25, 0.25, 0.25, 0.25)` and `(0.35, 0.275, 0.225, 0.15)`, respectively.
- Estimation and sensitivity results are based on 100 Monte Carlo replications.
- The joint size/power experiment uses `c in {1.0, 1.1, 1.2, 1.3}`, 200 Monte Carlo replications, and 1000 bootstrap replications.

## Repository files

| File / folder | Purpose |
|---|---|
| `simulation_core.py` | Core functions for data generation, FPCA, within transformation, geographic kNN adjacency, MST construction, recursive pruning, BIC selection, coefficient estimation, evaluation, and bootstrap inference. |
| `run_estimation.py` | Runs the Monte Carlo estimation experiment. |
| `run_example.py` | Produces representative simulation figures. |
| `run_sensitivity.py` | Runs the FVE, kNN, and minimum-group-size sensitivity experiments. |
| `run_joint_test.py` | Runs the joint beta/gamma homogeneity size and power experiment. |
| `run_empirical.py` | Runs the county-level corn-yield empirical application. |
| `dataset/` | Real-data inputs used in the empirical application. |

## Software requirements

Install the required Python packages with

```bash
pip install -r requirements.txt
```

Run the individual parts of the replication with

```bash
python run_estimation.py
python run_example.py
python run_sensitivity.py
python run_joint_test.py
python run_empirical.py
```

## Real data

The empirical application uses a balanced panel of 202 counties observed from 1999 to 2008.

- **Y**: annual county corn yield in `dataset/corn_panel.csv` (`Yield`).
- **X**: centered 365-day daily mean-temperature trajectories in `dataset/temperature_curves.csv` (`day_001` to `day_365`).
- **Z**: standardized annual mean daily precipitation in `dataset/corn_panel.csv` (`avgPRCP_std`).
- **Spatial coordinates**: county longitude and latitude in `dataset/county_nodes.csv`.
- **Distance matrix**: pairwise geographic distances in `dataset/distance_matrix.csv`.
- `dataset/source_regdat_1999_2008_balanced202.csv` retains the compact annual panel used to construct the empirical inputs.

The spatial adjacency matrix `A` is generated from the county coordinates using the geographic k-nearest-neighbour rule. Thus, `county_nodes.csv` is the source spatial input for constructing `A`; the distance matrix is retained for reproducibility and checking.
