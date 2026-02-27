# Shrinkage Inputs for Markowitz
This project stabilizes portfolio inputs by shrinking sample expected returns and covariance.
`src/shrinkage.py` implements cleaning, sample stats, constant-correlation target, Ledoit-Wolf-style covariance shrinkage, and Jorion mean shrinkage.
`scripts/run_shrinkage.py` loads a returns CSV, ignores non-numeric columns (for example Date), runs shrinkage, and prints diagnostics.
`tests/test_shrinkage.py` checks bounded shrinkage weights, PSD/symmetric covariance, small-sample regularization, mean-shrink direction, and degenerate constant assets.

## Pipeline (Step-by-Step)
1. Read the returns matrix from CSV (`scripts/run_shrinkage.py`).
2. Keep only numeric return columns and drop rows with NaN/inf (`clean_returns`).
3. Compute sample mean and sample covariance (`sample_mean_cov`).
4. Build a constant-correlation covariance target (`constant_corr_target`).
5. Compute covariance shrinkage intensity and combine sample/target covariances (`lw_shrink_to_target`).
6. Compute Jorion mean shrinkage toward the GLS grand mean (`jorion_mean_shrink`).
7. Save outputs:
   - `.npz`: `mu_shrunk`, `Sigma_shrunk`
   - `.json`: summary diagnostics for reporting

## Run and Verify
- Run shrinkage:
  `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz`
- Run tests:
  `pytest -q`
- Expected: test suite passes and CLI prints `T`, `N`, shrinkage weights, and covariance norm diagnostics.

Run demo: `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz`
CLI prints `T`, `N`, `w_mu`, `w_cov`, `rho_bar`, `mu_g`, first 5 `mu_sample` vs `mu_shrunk`, and Frobenius norms.
Outputs: `out_shrinkage.npz` with `mu_shrunk` and `Sigma_shrunk`, plus `out_shrinkage.json` with summary diagnostics.
Run tests from project root: `pytest -q`
