"""Fast deterministic checks for the cleaned GitHub package."""
from pathlib import Path
import numpy as np
from simulation.simulation_core import (
    simulate_estimation_dgp, simulate_test_dgp, run_skater_path,
    joint_homogeneity_test, load_empirical_csvs,
)

ROOT = Path(__file__).resolve().parent

data = simulate_estimation_dgp(n=48, T_panel=20, grid=40, seed=1234)
fit = run_skater_path(data, fve_threshold=.90, k_neighbors=4, min_size=8, K_max=3)
assert fit["records"][0]["K"] == 1
assert np.isfinite([r["BIC"] for r in fit["records"]]).all()

h0 = simulate_test_dgp(c=1.0, n=48, T_panel=20, grid=40, seed=4321)
test = joint_homogeneity_test(h0, B=2, random_state=99, fve_threshold=.90,
                              k_neighbors=4, min_size=8, K_max=3)
assert test["B_valid"] + test["B_failed"] == 2

D = ROOT / "empirical" / "data"
emp = load_empirical_csvs(D/"corn_panel.csv", D/"temperature_curves.csv",
                          D/"county_nodes.csv", D/"distance_matrix.csv")
assert emp["Y_mat"].shape == (202, 10)
assert emp["X_curves"].shape == (202, 10, 365)
assert emp["Z_std"].shape == (202, 10, 1)
assert emp["coords"].shape == (202, 2)
assert emp["distance_matrix"].shape == (202, 202)

print("Validation passed")
print("simulation selected K:", fit["best_K"])
print("empirical Y/X/Z:", emp["Y_mat"].shape, emp["X_curves"].shape, emp["Z_std"].shape)
