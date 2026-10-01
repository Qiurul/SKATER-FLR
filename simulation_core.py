# -*- coding: utf-8 -*-
"""Core reproducibility code for SKATER–FLR.

This module is aligned to the simulation and testing specifications stated in
Sections 2--4 of the accompanying manuscript:

    SKATER–FLR: Functional Linear Regression with Spatially Contiguous
    Coefficient Groups.

It contains:
- paper data-generating mechanisms for estimation and size/power studies;
- pooled FPCA selected by FVE;
- within transformation for individual fixed effects;
- symmetrized geographic kNN adjacency;
- coefficient-based or covariate-based MST edge weights;
- regression-driven recursive tree pruning and standard within-BIC;
- Global / SKATER / Oracle evaluation helpers;
- full-pipeline residual bootstrap for joint beta/gamma homogeneity.

The implementation is deterministic for fixed random seeds.
"""
from __future__ import annotations

import os
for _var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# Paper defaults
# -----------------------------------------------------------------------------
N = 200
T = 50
GRID = 100
L = 10
P = 3
K_TRUE = 4

FVE_DEFAULT = 0.95
K_NEIGHBORS_DEFAULT = 5
MIN_SIZE_DEFAULT = 10
RIDGE_PRELIM_DEFAULT = 0.001
RHO_EDGE_DEFAULT = 0.5

SPATIAL_DESIGNS = ("uniform", "non-uniform")
ERROR_DISTRIBUTIONS = ("normal_N025", "t8_025", "uniform_025")
SENS_FVE_GRID = (0.80, 0.85, 0.90, 0.95)
SENS_KNN_GRID = (4, 5, 6, 7, 8)
SENS_MIN_SIZE_GRID = (5, 10, 15, 20, 25)
TEST_C_VALUES = (1.0, 1.1, 1.2, 1.3)
TEST_ALPHAS = (0.01, 0.05, 0.10)

PI_UNIFORM = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
PI_NONUNIFORM = np.array([0.35, 0.275, 0.225, 0.15], dtype=float)


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------
def trapezoid_weights(s_grid: np.ndarray) -> np.ndarray:
    """Quadrature weights for integral over the supplied one-dimensional grid."""
    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or len(s) < 2:
        raise ValueError("s_grid must be a one-dimensional grid with at least 2 points.")
    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("s_grid must be strictly increasing.")
    w = np.empty_like(s)
    w[0] = ds[0] / 2.0
    w[-1] = ds[-1] / 2.0
    if len(s) > 2:
        w[1:-1] = (ds[:-1] + ds[1:]) / 2.0
    return w


def make_fourier_basis(s_grid: np.ndarray, L_dim: int = L) -> np.ndarray:
    """Paper Fourier basis: sqrt(2) sin/cos pairs on [0,1].

    Column ordering is phi_1=sin(2*pi*s), phi_2=cos(2*pi*s), ... .
    Returned shape is (R, L_dim).
    """
    if L_dim % 2 != 0:
        raise ValueError("L_dim must be even for paired sine/cosine basis.")
    s = np.asarray(s_grid, dtype=float)
    cols: List[np.ndarray] = []
    for k in range(1, L_dim // 2 + 1):
        cols.append(np.sqrt(2.0) * np.sin(2.0 * np.pi * k * s))
        cols.append(np.sqrt(2.0) * np.cos(2.0 * np.pi * k * s))
    return np.column_stack(cols)


def make_covariance_matrix(dim: int, rho: float) -> np.ndarray:
    idx = np.arange(dim)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def draw_errors(rng: np.random.Generator, shape: Tuple[int, ...], distribution: str) -> np.ndarray:
    """Three paper disturbances, all with variance 0.25."""
    d = str(distribution).lower().replace("-", "_")
    if d in {"normal_n025", "normal025", "n025"}:
        return rng.normal(0.0, 0.5, size=shape)
    if d in {"t8_025", "t8", "scaled_t8"}:
        # Var(t8)=8/(8-2)=4/3; (sqrt(3)/4)^2 * 4/3 = 1/4.
        return (np.sqrt(3.0) / 4.0) * rng.standard_t(df=8, size=shape)
    if d in {"uniform_025", "uniform", "u025"}:
        a = np.sqrt(3.0) / 2.0
        return rng.uniform(-a, a, size=shape)
    # Kept only for auditing old working notebooks; not a paper default.
    if d in {"normal_n01", "normal01", "n01"}:
        return rng.normal(0.0, 1.0, size=shape)
    raise ValueError(f"Unknown error distribution: {distribution}")


# -----------------------------------------------------------------------------
# Paper DGP
# -----------------------------------------------------------------------------
def _quadrant_box(g: int) -> Tuple[float, float, float, float]:
    # Q1 top-left, Q2 top-right, Q3 bottom-right, Q4 bottom-left.
    boxes = (
        (0.0, 0.5, 0.5, 1.0),
        (0.5, 1.0, 0.5, 1.0),
        (0.5, 1.0, 0.0, 0.5),
        (0.0, 0.5, 0.0, 0.5),
    )
    return boxes[int(g)]


def generate_spatial_structure(
    n: int,
    rng: np.random.Generator,
    spatial_design: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw group labels iid categorical(pi), then coordinates uniformly in Q_g."""
    if spatial_design == "uniform":
        pi = PI_UNIFORM
    elif spatial_design == "non-uniform":
        pi = PI_NONUNIFORM
    else:
        raise ValueError("spatial_design must be 'uniform' or 'non-uniform'.")

    labels = rng.choice(4, size=int(n), p=pi)
    coords = np.empty((int(n), 2), dtype=float)
    for i, g in enumerate(labels):
        x0, x1, y0, y1 = _quadrant_box(int(g))
        coords[i, 0] = rng.uniform(x0, x1)
        coords[i, 1] = rng.uniform(y0, y1)
    counts = np.bincount(labels, minlength=4)
    return coords, labels.astype(int), counts.astype(int)


def estimation_beta_coefficients() -> np.ndarray:
    """Section 4.1.1 coefficient vectors b_1,...,b_4, shape (4,10)."""
    return np.array([
        [1.00, 0.25, 0.00, -0.25, 0.00, 0.30, 0.35, 0.08, 0.85, -0.15],
        [-0.45, 0.00, 0.00, 0.55, 0.10, 0.90, -0.15, 0.00, 0.75, 0.00],
        [0.60, 0.15, -0.80, 0.00, 0.00, 0.20, 0.00, -0.35, -0.20, 0.00],
        [0.20, 0.00, 0.65, 0.20, 0.00, -0.75, 0.00, -0.35, 0.30, 0.10],
    ], dtype=float)


def estimation_gamma_coefficients() -> np.ndarray:
    return np.array([
        [0.70, -0.40, 0.25],
        [-0.50, 0.60, 0.15],
        [0.20, 0.30, -0.70],
        [-0.30, -0.20, 0.55],
    ], dtype=float)


def test_b0() -> np.ndarray:
    return np.array([1.00, -0.45, 0.60, 0.20, 0.25, 0.00, 0.15, 0.00, 0.00, 0.00], dtype=float)


def test_gamma0() -> np.ndarray:
    return np.array([0.70, -0.40, 0.25], dtype=float)


def _generate_common_covariates(
    n: int,
    T_panel: int,
    grid: int,
    L_dim: int,
    p: int,
    spatial_design: str,
    error_distribution: str,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed))
    coords, quadrant, counts = generate_spatial_structure(n, rng, spatial_design)
    s_grid = np.linspace(0.0, 1.0, int(grid))
    weights = trapezoid_weights(s_grid)
    Phi = make_fourier_basis(s_grid, L_dim)

    alpha = rng.normal(0.0, 0.6, size=int(n))
    alpha = alpha - alpha.mean()

    Sigma_x = make_covariance_matrix(L_dim, rho=0.5)
    xi = rng.multivariate_normal(np.zeros(L_dim), Sigma_x, size=int(n) * int(T_panel))
    xi = xi.reshape(int(n), int(T_panel), int(L_dim))
    X_curves = (xi.reshape(int(n) * int(T_panel), int(L_dim)) @ Phi.T).reshape(
        int(n), int(T_panel), int(grid)
    )

    Sigma_z = make_covariance_matrix(p, rho=0.3)
    Z = rng.multivariate_normal(np.zeros(p), Sigma_z, size=int(n) * int(T_panel))
    Z = Z.reshape(int(n), int(T_panel), int(p))

    eps = draw_errors(rng, (int(n), int(T_panel)), error_distribution)
    return {
        "rng": rng,
        "coords": coords,
        "quadrant": quadrant,
        "quadrant_counts": counts,
        "s_grid": s_grid,
        "l2_weights": weights,
        "Phi": Phi,
        "alpha": alpha,
        "xi": xi,
        "X_curves": X_curves,
        "Z": Z,
        "eps_mat": eps,
        "Sigma_x": Sigma_x,
        "Sigma_z": Sigma_z,
    }


def simulate_estimation_dgp(
    n: int = N,
    T_panel: int = T,
    grid: int = GRID,
    L_dim: int = L,
    p: int = P,
    spatial_design: str = "uniform",
    error_distribution: str = "normal_N025",
    seed: int = 2026,
) -> Dict[str, object]:
    if L_dim != 10 or p != 3:
        raise ValueError("The paper estimation DGP uses L=10 and p=3.")
    base = _generate_common_covariates(
        n, T_panel, grid, L_dim, p, spatial_design, error_distribution, seed
    )
    true_group = np.asarray(base["quadrant"], dtype=int)
    b_true = estimation_beta_coefficients()
    gamma_true = estimation_gamma_coefficients()
    beta_curves = b_true @ np.asarray(base["Phi"]).T

    X = np.asarray(base["X_curves"])
    Z = np.asarray(base["Z"])
    w = np.asarray(base["l2_weights"])
    alpha = np.asarray(base["alpha"])
    eps = np.asarray(base["eps_mat"])

    Y = np.empty((n, T_panel), dtype=float)
    for i in range(n):
        g = int(true_group[i])
        functional = np.asarray(base["xi"])[i] @ b_true[g]
        scalar = Z[i] @ gamma_true[g]
        Y[i] = alpha[i] + functional + scalar + eps[i]

    return {
        **base,
        "true_group": true_group,
        "true_group_counts": np.bincount(true_group, minlength=4),
        "b_true": b_true,
        "beta_curves": beta_curves,
        "gamma_true": gamma_true,
        "Y_mat": Y,
        "Z_std": Z,  # common internal key; simulation Z is generated directly as stated in paper.
        "settings": {
            "experiment": "estimation",
            "n": int(n), "T": int(T_panel), "grid": int(grid), "L": int(L_dim), "p": int(p),
            "K_true": 4, "spatial_design": spatial_design,
            "error_distribution": error_distribution, "seed": int(seed),
        },
    }


def simulate_test_dgp(
    c: float = 1.0,
    n: int = N,
    T_panel: int = T,
    grid: int = GRID,
    L_dim: int = L,
    p: int = P,
    spatial_design: str = "uniform",
    error_distribution: str = "normal_N025",
    seed: int = 7000,
) -> Dict[str, object]:
    """Section 4.2 DGP: upper half vs lower half, beta_2=c beta_0 and gamma_2=c gamma_0."""
    c = float(c)
    if c <= 0 or not np.isfinite(c):
        raise ValueError("c must be positive and finite.")
    base = _generate_common_covariates(
        n, T_panel, grid, L_dim, p, spatial_design, error_distribution, seed
    )
    quadrant = np.asarray(base["quadrant"], dtype=int)
    # Q1/Q2 are upper half, Q3/Q4 are lower half.
    true_group = np.isin(quadrant, [2, 3]).astype(int)

    b0 = test_b0()
    g0 = test_gamma0()
    b_true = np.vstack([b0, c * b0])
    gamma_true = np.vstack([g0, c * g0])
    beta_curves = b_true @ np.asarray(base["Phi"]).T

    X = np.asarray(base["X_curves"])
    Z = np.asarray(base["Z"])
    w = np.asarray(base["l2_weights"])
    alpha = np.asarray(base["alpha"])
    eps = np.asarray(base["eps_mat"])

    Y = np.empty((n, T_panel), dtype=float)
    for i in range(n):
        g = int(true_group[i])
        functional = np.asarray(base["xi"])[i] @ b_true[g]
        scalar = Z[i] @ gamma_true[g]
        Y[i] = alpha[i] + functional + scalar + eps[i]

    return {
        **base,
        "true_group": true_group,
        "true_group_counts": np.bincount(true_group, minlength=2),
        "b_true": b_true,
        "beta_curves": beta_curves,
        "gamma_true": gamma_true,
        "Y_mat": Y,
        "Z_std": Z,
        "settings": {
            "experiment": "size" if np.isclose(c, 1.0) else "power",
            "h0_true": bool(np.isclose(c, 1.0)),
            "c": c,
            "n": int(n), "T": int(T_panel), "grid": int(grid), "L": int(L_dim), "p": int(p),
            "K_true": 1 if np.isclose(c, 1.0) else 2,
            "spatial_design": spatial_design,
            "error_distribution": error_distribution,
            "seed": int(seed),
        },
    }


# -----------------------------------------------------------------------------
# FPCA and within-regression sufficient statistics
# -----------------------------------------------------------------------------
def compute_fpca(
    X_curves: np.ndarray,
    fve_threshold: float = FVE_DEFAULT,
    n_components: Optional[int] = None,
) -> Dict[str, np.ndarray | int]:
    """Pooled FPCA under the discrete mean inner product on the observation grid.

    The numerical inner product is mean_r f(s_r)g(s_r), matching the convention
    used to generate the manuscript's reported results on equally spaced grids.
    """
    X_curves = np.asarray(X_curves, dtype=float)
    if X_curves.ndim != 3:
        raise ValueError("X_curves must have shape (n,T,R).")
    n, T_panel, R = X_curves.shape
    X = X_curves.reshape(n * T_panel, R)
    mean_curve = X.mean(axis=0)
    Xc = X - mean_curve

    gram = Xc.T @ Xc
    eig_raw, eigvec = np.linalg.eigh(gram)
    order = np.argsort(eig_raw)[::-1]
    eig_raw = np.clip(eig_raw[order], 0.0, None)
    eigvec = eigvec[:, order]

    # mean(psi_m^2)=1 and score=<X-mu,psi>_grid.
    basis_all = eigvec.T * np.sqrt(R)
    scores_all = (Xc @ basis_all.T) / R
    eigvals_all = eig_raw / (n * T_panel * R)
    total = eigvals_all.sum()
    ratio = eigvals_all / total if total > 0 else np.zeros_like(eigvals_all)
    cum = np.cumsum(ratio)

    if n_components is None:
        M = int(np.searchsorted(cum, float(fve_threshold), side="left") + 1)
    else:
        M = int(n_components)
    M = max(1, min(M, R))

    return {
        "scores": scores_all[:, :M].reshape(n, T_panel, M),
        "basis": basis_all[:M],
        "mean_curve": mean_curve,
        "eigvals": eigvals_all[:M],
        "explained_ratio": ratio[:M],
        "cum_explained_ratio": cum[:M],
        "M": M,
        "s_grid": np.linspace(0.0, 1.0, R),
    }

def prepare_data(
    data: Dict[str, object],
    fve_threshold: float = FVE_DEFAULT,
    n_components: Optional[int] = None,
) -> Dict[str, object]:
    fpca = compute_fpca(
        np.asarray(data["X_curves"]),
        fve_threshold=fve_threshold,
        n_components=n_components,
    )
    out = dict(data)
    out["fpca_scores"] = fpca["scores"]
    out["fpca_basis"] = fpca["basis"]
    out["fpca_mean_curve"] = fpca["mean_curve"]
    out["fpca_eigvals"] = fpca["eigvals"]
    out["fpca_explained_ratio"] = fpca["explained_ratio"]
    out["fpca_cum_explained_ratio"] = fpca["cum_explained_ratio"]
    out["M_fpca"] = int(fpca["M"])
    return out


def _within(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return A - A.mean(axis=1, keepdims=True)


def build_within_stats(Y: np.ndarray, scores: np.ndarray, Z: np.ndarray) -> Dict[str, object]:
    Y = np.asarray(Y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    Z = np.asarray(Z, dtype=float)
    W = np.concatenate([scores, Z], axis=2)
    Yw = _within(Y)
    Ww = W - W.mean(axis=1, keepdims=True)
    n, T_panel, q = Ww.shape

    XtX = np.einsum("ntq,ntr->nqr", Ww, Ww)
    Xty = np.einsum("ntq,nt->nq", Ww, Yw)
    yty = np.einsum("nt,nt->n", Yw, Yw)
    return {"Yw": Yw, "W": W, "Ww": Ww, "XtX": XtX, "Xty": Xty, "yty": yty, "q": q}


def _aggregate_stats(nodes: Sequence[int], stats: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, float]:
    idx = np.asarray(nodes, dtype=int)
    return (
        np.asarray(stats["XtX"])[idx].sum(axis=0),
        np.asarray(stats["Xty"])[idx].sum(axis=0),
        float(np.asarray(stats["yty"])[idx].sum()),
    )


def group_full_rank(nodes: Sequence[int], stats: Dict[str, object]) -> bool:
    XtX, _, _ = _aggregate_stats(nodes, stats)
    q = int(stats["q"])
    return bool(np.linalg.matrix_rank(XtX) == q)


def fit_group(nodes: Sequence[int], stats: Dict[str, object]) -> Dict[str, object]:
    nodes = tuple(sorted(int(x) for x in nodes))
    XtX, Xty, yty = _aggregate_stats(nodes, stats)
    q = int(stats["q"])
    rank = int(np.linalg.matrix_rank(XtX))
    if rank < q:
        raise np.linalg.LinAlgError(f"Group design is rank deficient: rank={rank}, q={q}")
    theta = np.linalg.solve(XtX, Xty)
    ssr = float(yty - 2.0 * theta @ Xty + theta @ XtX @ theta)
    ssr = max(ssr, 0.0)
    return {"nodes": nodes, "theta": theta, "ssr": ssr, "tss": yty, "rank": rank}


def fit_partition(partition: Sequence[Sequence[int]], stats: Dict[str, object], cache: Optional[dict] = None) -> Dict[str, object]:
    if cache is None:
        cache = {}
    group_results = []
    total_ssr = 0.0
    total_tss = 0.0
    for group in normalize_partition(partition):
        if group not in cache:
            cache[group] = fit_group(group, stats)
        fit = cache[group]
        group_results.append(fit)
        total_ssr += float(fit["ssr"])
        total_tss += float(fit["tss"])
    return {"partition": normalize_partition(partition), "group_results": group_results,
            "total_ssr": total_ssr, "total_tss": total_tss}


# -----------------------------------------------------------------------------
# Graph, edge weights, MST
# -----------------------------------------------------------------------------
def build_spatial_adjacency(coords: np.ndarray, k_neighbors: int = K_NEIGHBORS_DEFAULT):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    k = min(int(k_neighbors), n - 1)
    while k < n:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        indices = nbrs.kneighbors(coords, return_distance=False)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in indices[i, 1:]:
                G.add_edge(int(i), int(j))
        if nx.is_connected(G):
            return G, k
        k += 1
    raise RuntimeError("Could not construct a connected kNN graph.")



def build_spatial_adjacency_from_distance(distance_matrix: np.ndarray, k_neighbors: int = K_NEIGHBORS_DEFAULT):
    """Symmetrized kNN graph from a precomputed geographic distance matrix."""
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("distance_matrix must be square.")
    k = min(max(1, int(k_neighbors)), n - 1)
    while k < n:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            order = np.argsort(D[i])
            neigh = [int(j) for j in order if int(j) != i][:k]
            for j in neigh:
                a, b = sorted((int(i), int(j)))
                G.add_edge(a, b, geo_distance=float(D[i, j]))
        if nx.is_connected(G):
            return G, k
        k += 1
    raise RuntimeError("Could not construct a connected kNN graph from distance_matrix.")


def preliminary_unit_coefficients(
    Y: np.ndarray,
    scores: np.ndarray,
    Z: np.ndarray,
    ridge: float = RIDGE_PRELIM_DEFAULT,
) -> Optional[np.ndarray]:
    stats = build_within_stats(Y, scores, Z)
    n, T_panel = np.asarray(Y).shape
    q = int(stats["q"])
    if T_panel - 1 < q:
        return None
    theta = np.empty((n, q), dtype=float)
    for i in range(n):
        XtX = np.asarray(stats["XtX"])[i]
        if np.linalg.matrix_rank(XtX) < q:
            return None
        Xty = np.asarray(stats["Xty"])[i]
        theta[i] = np.linalg.solve(XtX + float(ridge) * np.eye(q), Xty)
    return theta


def _standardize_flat(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    flat = A.reshape(-1, A.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd


def build_weighted_graph(
    adjacency: nx.Graph,
    prepared: Dict[str, object],
    ridge_prelim: float = RIDGE_PRELIM_DEFAULT,
    rho_edge: float = RHO_EDGE_DEFAULT,
) -> Tuple[nx.Graph, str, Optional[np.ndarray]]:
    Y = np.asarray(prepared["Y_mat"], dtype=float)
    scores = np.asarray(prepared["fpca_scores"], dtype=float)
    Z = np.asarray(prepared["Z_std"], dtype=float)
    theta = preliminary_unit_coefficients(Y, scores, Z, ridge=ridge_prelim)

    G = nx.Graph()
    G.add_nodes_from(adjacency.nodes())
    if theta is not None:
        mu = theta.mean(axis=0)
        sd = theta.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        theta_std = (theta - mu) / sd
        q = theta.shape[1]
        for i, j in sorted(adjacency.edges()):
            weight = np.linalg.norm(theta_std[i] - theta_std[j]) / np.sqrt(q)
            G.add_edge(int(i), int(j), weight=float(weight))
        return G, "coefficient", theta

    # Covariate fallback, Eq. (11), used when unit coefficients are not estimable.
    X = np.asarray(prepared["X_curves"], dtype=float)
    Xflat = X.reshape(-1, X.shape[-1])
    x_mu = Xflat.mean(axis=0)
    x_sd = Xflat.std(axis=0, ddof=0)
    x_sd[x_sd == 0] = 1.0
    Xs = (X - x_mu[None, None, :]) / x_sd[None, None, :]
    Zs = _standardize_flat(Z)
    for i, j in sorted(adjacency.edges()):
        # Paper Eq. (11), approximated by the discrete mean over time/grid.
        dX = float(np.sqrt(np.mean((Xs[i] - Xs[j]) ** 2)))
        dZ = float(np.sqrt(np.mean((Zs[i] - Zs[j]) ** 2)))
        weight = float(rho_edge) * dX + (1.0 - float(rho_edge)) * dZ
        G.add_edge(int(i), int(j), weight=weight)
    return G, "covariate", None


def build_mst(
    prepared: Dict[str, object],
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    ridge_prelim: float = RIDGE_PRELIM_DEFAULT,
    rho_edge: float = RHO_EDGE_DEFAULT,
):
    if prepared.get("distance_matrix") is not None:
        adjacency, used_k = build_spatial_adjacency_from_distance(np.asarray(prepared["distance_matrix"]), k_neighbors)
    else:
        adjacency, used_k = build_spatial_adjacency(np.asarray(prepared["coords"]), k_neighbors)
    weighted, mode, theta = build_weighted_graph(adjacency, prepared, ridge_prelim, rho_edge)
    mst = nx.minimum_spanning_tree(weighted, weight="weight", algorithm="kruskal")
    return adjacency, weighted, mst, used_k, mode, theta


# -----------------------------------------------------------------------------
# Recursive pruning and BIC
# -----------------------------------------------------------------------------
def normalize_partition(partition: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    groups = [tuple(sorted(int(x) for x in g)) for g in partition]
    groups.sort(key=lambda g: (g[0] if g else -1, len(g), g))
    return tuple(groups)


def partition_to_labels(partition: Sequence[Sequence[int]], n: int) -> np.ndarray:
    labels = np.full(int(n), -1, dtype=int)
    for g, nodes in enumerate(normalize_partition(partition)):
        labels[np.asarray(nodes, dtype=int)] = g
    if np.any(labels < 0):
        raise ValueError("Partition does not cover all nodes.")
    return labels


def compute_bic(ssr: float, n: int, T_panel: int, K: int, q: int) -> float:
    n_eff = int(n) * (int(T_panel) - 1)
    return float(n_eff * np.log(max(float(ssr), 1e-12) / n_eff) + int(K) * int(q) * np.log(n_eff))


def _component_containing(forest: nx.Graph, node: int) -> Tuple[int, ...]:
    return tuple(sorted(nx.node_connected_component(forest, int(node))))


def choose_best_cut(
    forest: nx.Graph,
    stats: Dict[str, object],
    min_size: int,
    cache: dict,
):
    components = [tuple(sorted(c)) for c in nx.connected_components(forest)]
    parent_ssr = {}
    for comp in components:
        if comp not in cache:
            cache[comp] = fit_group(comp, stats)
        parent_ssr[comp] = float(cache[comp]["ssr"])

    node_to_parent = {}
    for comp in components:
        for x in comp:
            node_to_parent[int(x)] = comp

    best = None
    for u, v, attrs in sorted(forest.edges(data=True), key=lambda e: (min(e[0],e[1]), max(e[0],e[1]))):
        parent = node_to_parent[int(u)]
        if len(parent) < 2 * int(min_size):
            continue
        weight_attrs = dict(attrs)
        forest.remove_edge(u, v)
        c1 = _component_containing(forest, u)
        c2 = _component_containing(forest, v)
        forest.add_edge(u, v, **weight_attrs)
        if len(c1) < int(min_size) or len(c2) < int(min_size):
            continue
        if not group_full_rank(c1, stats) or not group_full_rank(c2, stats):
            continue
        if c1 not in cache:
            cache[c1] = fit_group(c1, stats)
        if c2 not in cache:
            cache[c2] = fit_group(c2, stats)
        reduction = parent_ssr[parent] - float(cache[c1]["ssr"]) - float(cache[c2]["ssr"])
        candidate = (float(reduction), (int(u), int(v)), c1, c2)
        if best is None or candidate[0] > best[0] + 1e-12 or (
            abs(candidate[0] - best[0]) <= 1e-12 and tuple(sorted(candidate[1])) < tuple(sorted(best[1]))
        ):
            best = candidate
    return best


def run_skater_path(
    data: Dict[str, object],
    fve_threshold: float = FVE_DEFAULT,
    n_components: Optional[int] = None,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    min_size: int = MIN_SIZE_DEFAULT,
    ridge_prelim: float = RIDGE_PRELIM_DEFAULT,
    rho_edge: float = RHO_EDGE_DEFAULT,
    K_max: Optional[int] = None,
) -> Dict[str, object]:
    prepared = prepare_data(data, fve_threshold=fve_threshold, n_components=n_components)
    Y = np.asarray(prepared["Y_mat"], dtype=float)
    Z = np.asarray(prepared["Z_std"], dtype=float)
    scores = np.asarray(prepared["fpca_scores"], dtype=float)
    n, T_panel = Y.shape
    q = scores.shape[-1] + Z.shape[-1]
    if K_max is None:
        K_max = max(1, n // int(min_size))

    stats = build_within_stats(Y, scores, Z)
    adjacency, weighted, mst, used_k, weight_mode, theta_ind = build_mst(
        prepared, k_neighbors=k_neighbors, ridge_prelim=ridge_prelim, rho_edge=rho_edge
    )
    forest = mst.copy()
    cache: dict = {}
    records: List[Dict[str, object]] = []
    cut_edges: List[Tuple[int, int]] = []

    for K in range(1, int(K_max) + 1):
        partition = normalize_partition([tuple(sorted(c)) for c in nx.connected_components(forest)])
        fit = fit_partition(partition, stats, cache=cache)
        ssr = float(fit["total_ssr"])
        tss = float(fit["total_tss"])
        records.append({
            "K": K,
            "SSR": ssr,
            "TSS": tss,
            "R2": float(1.0 - ssr / tss) if tss > 0 else np.nan,
            "BIC": compute_bic(ssr, n, T_panel, K, q),
            "partition": partition,
            "group_results": fit["group_results"],
            "M_fpca": int(scores.shape[-1]),
            "q": int(q),
        })
        if K >= int(K_max):
            break
        best = choose_best_cut(forest, stats, min_size=min_size, cache=cache)
        if best is None:
            break
        _, edge, _, _ = best
        forest.remove_edge(*edge)
        cut_edges.append(tuple(edge))

    bic_values = np.array([r["BIC"] for r in records], dtype=float)
    min_bic = np.nanmin(bic_values)
    best_idx = int(np.where(np.isclose(bic_values, min_bic, rtol=0.0, atol=1e-10))[0][0])
    best_record = records[best_idx]
    return {
        "prepared_data": prepared,
        "records": records,
        "best_record": best_record,
        "best_K": int(best_record["K"]),
        "adjacency": adjacency,
        "weighted_graph": weighted,
        "mst": mst,
        "cut_edges": cut_edges,
        "used_k_neighbors": int(used_k),
        "edge_weight_mode": weight_mode,
        "theta_ind": theta_ind,
    }


def get_record(result: Dict[str, object], K: int) -> Dict[str, object]:
    for rec in result["records"]:
        if int(rec["K"]) == int(K):
            return rec
    raise KeyError(f"K={K} is not available on the pruning path.")


def alternative_record(result: Dict[str, object]) -> Dict[str, object]:
    candidates = [r for r in result["records"] if int(r["K"]) >= 2]
    if not candidates:
        raise RuntimeError("The pruning path contains no K>=2 alternative.")
    min_bic = min(float(r["BIC"]) for r in candidates)
    return next(r for r in candidates if np.isclose(float(r["BIC"]), min_bic, atol=1e-10, rtol=0.0))


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def pairwise_jaccard(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    n = len(true_labels)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    a = (true_labels[:, None] == true_labels[None, :])[upper]
    b = (pred_labels[:, None] == pred_labels[None, :])[upper]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)


def match_true_to_estimated(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[int, int]:
    tu = np.unique(true_labels)
    pu = np.unique(pred_labels)
    C = np.zeros((len(tu), len(pu)), dtype=int)
    for i, g in enumerate(tu):
        for j, h in enumerate(pu):
            C[i, j] = int(np.sum((true_labels == g) & (pred_labels == h)))
    rows, cols = linear_sum_assignment(-C)
    return {int(tu[r]): int(pu[c]) for r, c in zip(rows, cols)}


def recover_beta_curve(theta: np.ndarray, fpca_basis: np.ndarray) -> np.ndarray:
    M = fpca_basis.shape[0]
    return np.asarray(theta[:M]) @ np.asarray(fpca_basis)


def oracle_record(data: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
    prepared = result["prepared_data"]
    true = np.asarray(data["true_group"], dtype=int)
    partition = [tuple(np.where(true == g)[0]) for g in sorted(np.unique(true))]
    stats = build_within_stats(prepared["Y_mat"], prepared["fpca_scores"], prepared["Z_std"])
    fit = fit_partition(partition, stats, cache={})
    n, T_panel = np.asarray(prepared["Y_mat"]).shape
    q = int(prepared["fpca_scores"].shape[-1] + prepared["Z_std"].shape[-1])
    K = len(partition)
    return {
        "K": K, "SSR": fit["total_ssr"], "TSS": fit["total_tss"],
        "R2": 1.0 - fit["total_ssr"] / fit["total_tss"],
        "BIC": compute_bic(fit["total_ssr"], n, T_panel, K, q),
        "partition": normalize_partition(partition), "group_results": fit["group_results"],
        "M_fpca": int(prepared["fpca_scores"].shape[-1]), "q": q,
    }


def coefficient_errors(data: Dict[str, object], result: Dict[str, object], record: Dict[str, object], method: str) -> Dict[str, float]:
    true_labels = np.asarray(data["true_group"], dtype=int)
    beta_true = np.asarray(data["beta_curves"], dtype=float)
    gamma_true = np.asarray(data["gamma_true"], dtype=float)
    basis = np.asarray(result["prepared_data"]["fpca_basis"], dtype=float)
    M = basis.shape[0]
    K0 = len(np.unique(true_labels))

    beta_hat = np.full_like(beta_true, np.nan, dtype=float)
    gamma_hat = np.full_like(gamma_true, np.nan, dtype=float)
    if method == "global":
        theta = np.asarray(record["group_results"][0]["theta"])
        b = recover_beta_curve(theta, basis)
        g = theta[M:]
        beta_hat[:] = b
        gamma_hat[:] = g
    else:
        pred = partition_to_labels(record["partition"], len(true_labels))
        mapping = match_true_to_estimated(true_labels, pred)
        for tg in range(K0):
            if tg not in mapping:
                continue
            eg = mapping[tg]
            theta = np.asarray(record["group_results"][eg]["theta"])
            beta_hat[tg] = recover_beta_curve(theta, basis)
            gamma_hat[tg] = theta[M:]

    beta_diff = beta_hat - beta_true
    gamma_diff = gamma_hat - gamma_true
    return {
        "Beta_RMSE": float(np.sqrt(np.nanmean(beta_diff ** 2))),
        "Beta_MAE": float(np.nanmean(np.abs(beta_diff))),
        "Gamma_RMSE": float(np.sqrt(np.nanmean(gamma_diff ** 2))),
        "Gamma_MAE": float(np.nanmean(np.abs(gamma_diff))),
    }


def summarize_estimation_replication(data: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
    selected = result["best_record"]
    pooled = get_record(result, 1)
    oracle = oracle_record(data, result)
    true = np.asarray(data["true_group"], dtype=int)
    pred = partition_to_labels(selected["partition"], len(true))
    row = {
        "selected_K": int(selected["K"]),
        "ARI": float(adjusted_rand_score(true, pred)),
        "JI": pairwise_jaccard(true, pred),
        "M_fpca": int(selected["M_fpca"]),
        "FPCA_cumFVE": float(result["prepared_data"]["fpca_cum_explained_ratio"][-1]),
        "edge_weight_mode": result["edge_weight_mode"],
        "used_k_neighbors": int(result["used_k_neighbors"]),
    }
    for label, rec, method in [
        ("Global", pooled, "global"),
        ("SKATER", selected, "skater"),
        ("Oracle", oracle, "oracle"),
    ]:
        err = coefficient_errors(data, result, rec, method)
        for key, value in err.items():
            row[f"{key}_{label}"] = value
    return row


# -----------------------------------------------------------------------------
# Full-pipeline joint homogeneity bootstrap
# -----------------------------------------------------------------------------
def _null_components(data: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
    prepared = result["prepared_data"]
    rec0 = get_record(result, 1)
    theta0 = np.asarray(rec0["group_results"][0]["theta"], dtype=float)
    Y = np.asarray(prepared["Y_mat"], dtype=float)
    scores = np.asarray(prepared["fpca_scores"], dtype=float)
    Z = np.asarray(prepared["Z_std"], dtype=float)
    W = np.concatenate([scores, Z], axis=2)
    alpha0 = Y.mean(axis=1) - np.einsum("nq,q->n", W.mean(axis=1), theta0)
    fitted = alpha0[:, None] + np.einsum("ntq,q->nt", W, theta0)
    residual = Y - fitted
    residual_centered = residual - residual.mean()
    return {
        "theta0": theta0,
        "alpha0": alpha0,
        "residual_centered": residual_centered,
        "scores_original": scores,
        "Z_original": Z,
        "X_original": np.asarray(prepared["X_curves"], dtype=float),
        "SSR0": float(rec0["SSR"]),
    }


def joint_homogeneity_test(
    data: Dict[str, object],
    B: int = 1000,
    random_state: int = 2026,
    fve_threshold: float = FVE_DEFAULT,
    n_components: Optional[int] = None,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    min_size: int = MIN_SIZE_DEFAULT,
    ridge_prelim: float = RIDGE_PRELIM_DEFAULT,
    rho_edge: float = RHO_EDGE_DEFAULT,
    K_max: Optional[int] = None,
    return_bootstrap_statistics: bool = False,
) -> Dict[str, object]:
    """Paper Section 3 full-pipeline residual bootstrap.

    The alternative is selected by BIC over K>=2 within each original/bootstrap
    pruning path; it is not fixed to K=2.
    """
    observed = run_skater_path(
        data, fve_threshold=fve_threshold, n_components=n_components,
        k_neighbors=k_neighbors, min_size=min_size, ridge_prelim=ridge_prelim,
        rho_edge=rho_edge, K_max=K_max,
    )
    rec0 = get_record(observed, 1)
    recA = alternative_record(observed)
    SSR0 = float(rec0["SSR"])
    SSRA = float(recA["SSR"])
    T_obs = (SSR0 - SSRA) / max(SSRA, 1e-12)
    null = _null_components(data, observed)

    Y = np.asarray(data["Y_mat"], dtype=float)
    n, T_panel = Y.shape
    rng = np.random.default_rng(int(random_state))
    T_boot: List[float] = []
    K_boot: List[int] = []
    failures: List[str] = []

    for _ in range(int(B)):
        try:
            src = rng.integers(0, n, size=n)
            X_star = null["X_original"][src].copy()
            Z_star = null["Z_original"][src].copy()
            scores_gen = null["scores_original"][src]
            alpha_star = null["alpha0"][src]
            W_gen = np.concatenate([scores_gen, Z_star], axis=2)
            eps_star = rng.choice(null["residual_centered"].ravel(), size=(n, T_panel), replace=True)
            Y_star = alpha_star[:, None] + np.einsum("ntq,q->nt", W_gen, null["theta0"]) + eps_star
            boot_data = {
                "Y_mat": Y_star,
                "X_curves": X_star,
                "Z_std": Z_star,
                # Coordinates stay fixed: adjacency is geographic and not resampled.
                "coords": np.asarray(data["coords"], dtype=float).copy(),
            }
            boot_result = run_skater_path(
                boot_data, fve_threshold=fve_threshold, n_components=n_components,
                k_neighbors=k_neighbors, min_size=min_size, ridge_prelim=ridge_prelim,
                rho_edge=rho_edge, K_max=K_max,
            )
            b0 = get_record(boot_result, 1)
            bA = alternative_record(boot_result)
            s0 = float(b0["SSR"])
            sA = float(bA["SSR"])
            T_boot.append((s0 - sA) / max(sA, 1e-12))
            K_boot.append(int(bA["K"]))
        except Exception as exc:  # preserve failed-rep accounting instead of silently changing B
            failures.append(f"{type(exc).__name__}: {exc}")

    T_arr = np.asarray(T_boot, dtype=float)
    valid_B = len(T_arr)
    p_value = float(np.mean(T_arr >= T_obs)) if valid_B else np.nan
    out = {
        "T_obs": float(T_obs),
        "SSR0": SSR0,
        "SSRA": SSRA,
        "K_alt": int(recA["K"]),
        "B_requested": int(B),
        "B_valid": int(valid_B),
        "B_failed": int(len(failures)),
        "p_value": p_value,
        "reject_001": bool(p_value < 0.01) if np.isfinite(p_value) else False,