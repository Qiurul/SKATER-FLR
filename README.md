# SKATER–FLR reproducibility code

Code and data for **“SKATER–FLR: Functional Linear Regression with Spatially Contiguous Coefficient Groups.”**

The repository separates the simulation study from the unchanged empirical application. The cleaned simulation scripts under `simulation/` are the recommended replication entry points.

## Repository structure

```text
SKATER-FLR/
├── README.md
├── requirements.txt
├── run_all.py
├── validate_implementation.py
├── simulation/
│   ├── simulation_core.py
│   ├── run_estimation.py
│   ├── run_sensitivity.py
│   ├── run_joint_test.py
│   └── run_example.py
├── empirical/
│   ├── run_empirical.py
│   └── data/
│       ├── README.md
│       ├── corn_panel.csv
│       ├── temperature_curves.csv
│       ├── county_nodes.csv
│       ├── distance_matrix.csv
│       └── source_regdat_1999_2008_balanced202.csv
└── outputs/
```

## Simulation design

The estimation simulation uses `n=200`, `T=50`, a functional grid of `R=100`, Fourier dimension `L=10`, `p=3` scalar covariates, and `K0=4` spatial groups. The default estimator uses FPCA FVE `0.95`, a symmetrized geographic `k=5` nearest-neighbour graph, and minimum group size `n_min=10`.

The simulated objects are generated directly in code, so no fixed simulation input CSV is required:

- `Y_it`: simulated scalar response, shape `(n,T)`;
- `X_it(s)`: simulated functional predictor, shape `(n,T,R)`;
- `Z_it`: simulated three-dimensional scalar covariates, shape `(n,T,3)`;
- coordinates: simulated spatial locations, shape `(n,2)`.

The sensitivity grids are FVE `{0.80, 0.85, 0.90, 0.95}`, kNN `{4,5,6,7,8}`, and minimum group size `{5,10,15,20,25}`.

The joint size/power experiment uses `c={1.0,1.1,1.2,1.3}`, `N_MC=200`, and `B=1000`. Under the two-group alternative, both coefficient types vary: `beta_2=c beta_1` and `gamma_2=c gamma_1`.

## Real-data Y, X and Z

The empirical application is kept unchanged. It uses 202 counties over 1999–2008.

| Symbol | Repository representation | Meaning |
|---|---|---|
| **Y** | `empirical/data/corn_panel.csv` → `Yield` | Annual county corn yield. |
| **X** | `empirical/data/temperature_curves.csv` → `day_001` … `day_365` | Centered daily mean-temperature trajectory. |
| **Z** | `empirical/data/corn_panel.csv` → `avgPRCP_std` | Standardized annual mean daily precipitation. |

See `empirical/data/README.md` for the row alignment and file definitions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the simulation study

```bash
python -m simulation.run_estimation --reps 100 --jobs 8
python -m simulation.run_sensitivity --reps 100 --jobs 8
python -m simulation.run_joint_test --reps 200 --B 1000 --jobs 50
python -m simulation.run_example
```

## Reproduce the empirical application

```bash
python -m empirical.run_empirical
python -m empirical.run_empirical --bootstrap 1000
```

## Quick validation

```bash
python validate_implementation.py
```

The full joint-test simulation is computationally expensive because each bootstrap replicate repeats FPCA, graph/MST construction, recursive pruning, BIC selection, and refitting.
