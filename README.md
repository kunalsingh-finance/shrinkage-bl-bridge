# Shrinkage Inputs for Markowitz
This project stabilizes portfolio inputs by shrinking sample expected returns and covariance.
`src/shrinkage.py` implements cleaning, sample stats, constant-correlation target, Ledoit-Wolf-style covariance shrinkage, and Jorion mean shrinkage.
`scripts/run_shrinkage.py` loads a returns CSV, ignores non-numeric columns (for example Date), runs shrinkage, and prints diagnostics.
`tests/test_shrinkage.py` checks bounded shrinkage weights, PSD/symmetric covariance, small-sample regularization, mean-shrink direction, and degenerate constant assets.
Run demo: `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz`
CLI prints `T`, `N`, `w_mu`, `w_cov`, `rho_bar`, `mu_g`, first 5 `mu_sample` vs `mu_shrunk`, and Frobenius norms.
Outputs: `out_shrinkage.npz` with `mu_shrunk` and `Sigma_shrunk`, plus `out_shrinkage.json` with summary diagnostics.
Run tests from project root: `pytest -q`
