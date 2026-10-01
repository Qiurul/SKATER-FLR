# Paper alignment notes

The repository root treats the current manuscript as the source of truth. The cleaned scripts in this repository implement the final replication specification.

## Final simulation specification implemented here

### Estimation experiment (Section 4.1)

- `n = 200`, `T = 50`, functional grid `R = 100`.
- Fourier basis dimension `L = 10`; scalar controls `p = 3`.
- Four spatial groups: Q1 top-left, Q2 top-right, Q3 bottom-right, Q4 bottom-left.
- Uniform sampling probabilities `(0.25, 0.25, 0.25, 0.25)`.
- Non-uniform probabilities `(0.35, 0.275, 0.225, 0.15)`.
- Functional-score covariance: `(Sigma_X)_{lm} = 0.5^{|l-m|}`.
- Scalar-covariate covariance: `(Sigma_Z)_{jk} = 0.3^{|j-k|}`.
- FPCA FVE threshold `0.95`, geographic kNN `k = 5`, minimum group size `n_min = 10`.
- All three paper disturbance designs have variance `0.25`.
- Monte Carlo replications: `100`.

The group-specific Fourier coefficient vectors are:

```text
b1 = ( 1.00, 0.25,  0.00, -0.25, 0.00,  0.30,  0.35,  0.08,  0.85, -0.15)
b2 = (-0.45, 0.00,  0.00,  0.55, 0.10,  0.90, -0.15,  0.00,  0.75,  0.00)
b3 = ( 0.60, 0.15, -0.80,  0.00, 0.00,  0.20,  0.00, -0.35, -0.20,  0.00)
b4 = ( 0.20, 0.00,  0.65,  0.20, 0.00, -0.75,  0.00, -0.35,  0.30,  0.10)
```

and scalar coefficients are:

```text
gamma1 = ( 0.70, -0.40,  0.25)
gamma2 = (-0.50,  0.60,  0.15)
gamma3 = ( 0.20,  0.30, -0.70)
gamma4 = (-0.30, -0.20,  0.55)
```

### Sensitivity experiment (Section 4.1.4)

- FVE: `0.80, 0.85, 0.90, 0.95`.
- kNN: `4, 5, 6, 7, 8`.
- minimum group size: `5, 10, 15, 20, 25`.
- One factor is varied at a time; the other two remain at `FVE=0.95`, `k=5`, `n_min=10`.

### Joint size/power experiment (Section 4.2)

Null coefficients:

```text
b0 = (1.00, -0.45, 0.60, 0.20, 0.25, 0.00, 0.15, 0.00, 0.00, 0.00)
gamma0 = (0.70, -0.40, 0.25)
```

- Under `H0`, all units share `(beta0, gamma0)`.
- Under the alternative, G1 is the upper half (Q1 ∪ Q2) and G2 is the lower half (Q3 ∪ Q4).
- `(beta1, gamma1) = (beta0, gamma0)`.
- `(beta2, gamma2) = c * (beta0, gamma0)`.
- Paper c-grid: `1.0, 1.1, 1.2, 1.3`.
- `N_MC = 200`, `B = 1000` bootstrap replications.
- Size is reported at `alpha = 0.01, 0.05, 0.10`; power at `alpha = 0.05`.
- Each bootstrap replicate repeats FPCA, edge weighting, MST construction, pruning, BIC selection over the alternative path, and refitting.
