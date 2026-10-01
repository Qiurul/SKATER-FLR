# Dataset for the empirical application

This folder contains the real-data inputs used in the manuscript's county-level corn-yield application. The empirical specification is unchanged.

The panel contains 202 counties observed annually from 1999 to 2008, for 2,020 county-year observations.

| Model object | File / columns | Shape | Meaning |
|---|---|---:|---|
| **Y** | `corn_panel.csv` → `Yield` | `(202, 10)` | Annual county-level corn yield. |
| **X** | `temperature_curves.csv` → `day_001` … `day_365` | `(202, 10, 365)` | Centered 365-day daily mean-temperature trajectory. |
| **Z** | `corn_panel.csv` → `avgPRCP_std` | `(202, 10, 1)` | Standardized annual mean daily precipitation. |
| Spatial coordinates | `county_nodes.csv` → `longitude`, `latitude` | `(202, 2)` | Coordinates used to construct the geographic kNN adjacency graph `A`. |
| Pairwise distances | `distance_matrix.csv` | `(202, 202)` | Geographic distance matrix retained for reproducibility and checking. |

`source_regdat_1999_2008_balanced202.csv` is the compact annual panel retained for traceability. `Area` is not included in the fitted regression.

Rows across the empirical files are aligned by `CountyI` and `Year` where applicable.