# Simulation code for SKATER–FLR

This repository provides the code and real-data inputs used for the manuscript
*SKATER–FLR: Functional Linear Regression with Spatially Contiguous Coefficient Groups*.

It contains the simulation data-generating mechanisms, the SKATER–FLR estimation procedure, sensitivity experiments, the joint homogeneity test, and the empirical application.

## Simulation design

- `n = 200`, `T = 50`, functional grid size `R = 100`.
- Fourier basis dimension `L = 10`; scalar controls `p = 3`.
- Four spatial groups are used in the estimation study.
- The default FPCA threshold is `FVE = 0.95`.
- The geographic adjacency matrix `A` is constructed from spatial coordinates using a symmetrized `k = 5` nearest-neighbour graph.
- The minimum admissible group size is `n_min = 10`.
- Monte Carlo estimation and sensitivity experiments use 100 replications.
- The joint size/power experiment uses `c in {1.0, 1.1, 1.2, 1.3}`, 200 Monte Carlo replications, and 1000 bootstrap replications.

## Repository files

| File / folder | Purpose |
|---|---|
| `run_all.py` | Unified smoke/full entry point. |
| `simulation/simulation_core.py` | Data generation, FPCA, spatial kNN graph, MST construction, recursive pruning, BIC selection, estimation metrics, and bootstrap test. |
| `simulation/run_estimation.py` | Estimation Monte Carlo experiment. |
| `simulation/run_sensitivity.py` | FVE, kNN and minimum-group-size sensitivity experiments. |
| `simulation/run_joint_test.py` | Joint beta/gamma homogeneity size and power experiment. |
| `simulation/run_example.py` | Representative simulation figures. |
| `empirical/run_empirical.py` | Real-data SKATER–FLR analysis. |
| `empirical/data/` | Real-data inputs, including response/covariates, daily temperature curves, spatial coordinates and pairwise distances. |
| `validate_implementation.py` | Fast deterministic implementation checks. |

## Software requirements

Install the Python dependencies with

```bash
pip install -r requirements.txt
```

## Real data

The empirical application uses 202 counties observed from 1999 to 2008.

- **Y**: annual county corn yield, stored in `empirical/data/corn_panel.csv`.
- **X**: centered 365-day daily mean-temperature trajectory, stored in `empirical/data/temperature_curves.csv`.
- **Z**: standardized annual mean daily precipitation, stored in `empirical/data/corn_panel.csv`.
- **Spatial coordinates**: longitude and latitude for the 202 county nodes, stored in `empirical/data/county_nodes.csv`.
- **Distance matrix**: pairwise geographic distances, stored in `empirical/data/distance_matrix.csv`.

The adjacency matrix `A` used by SKATER–FLR is not an externally supplied fixed matrix: it is generated from the county spatial coordinates by the geographic k-nearest-neighbour rule used in the analysis. The coordinate file is therefore included explicitly in the repository.

Run the empirical analysis with

```bash
python -m empirical.run_empirical
```
