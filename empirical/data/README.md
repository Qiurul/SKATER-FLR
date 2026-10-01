# Real-data inputs

This folder contains the **unchanged empirical-analysis inputs used for the manuscript**. The purpose of this export is only to make the model inputs explicit and easy to reproduce; the empirical specification itself is not revised here.

The application uses a balanced panel of **202 counties observed from 1999 to 2008**, giving **2,020 county-year observations**.

## Y, X and Z

| Symbol | File / columns | Model shape | Meaning |
|---|---|---:|---|
| **Y** | `corn_panel.csv` → `Yield` | `(202, 10)` | Annual county-level corn yield. |
| **X** | `temperature_curves.csv` → `day_001` … `day_365` | `(202, 10, 365)` | Centered 365-day daily mean-temperature trajectory. |
| **Z** | `corn_panel.csv` → `avgPRCP_std` | `(202, 10, 1)` | Standardized annual mean daily precipitation (`avgPRCP`). |
| coordinates | `county_nodes.csv` | `(202, 2)` | County spatial locations. |
| distances | `distance_matrix.csv` | `(202, 202)` | Geographic distance matrix used to form the kNN adjacency graph. |

`source_regdat_1999_2008_balanced202.csv` is the compact annual panel retained for traceability. It contains `Year`, `State`, `County`, `CountyI`, `Yield`, `avgPRCP`, and `Area`. `Area` is not included in the fitted regression.

The annual panel does not contain the 365 daily temperatures required for the functional predictor, so X is stored separately in `temperature_curves.csv`. Rows are aligned by `CountyI` and `Year`.
