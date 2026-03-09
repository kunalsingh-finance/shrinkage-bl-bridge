# Shrinkage Inputs for Markowitz
This project stabilizes portfolio inputs by shrinking sample expected returns and covariance.
`src/shrinkage.py` implements cleaning, sample stats, constant-correlation target, Ledoit-Wolf-style covariance shrinkage, and Jorion mean shrinkage.
It now also exposes `bl_shrinkage_inputs(...)` that returns exactly:
- `mu_js_table` with columns `ETF`, `mu_js` (shape `N x 2`)
- `Sigma_shrunk` (shape `N x N`)
`src/bl_integration.py` exposes:
- `build_bl_inputs_from_returns(...)` -> `{tickers, mu, Sigma}` for direct BL optimizer calls
- `to_bl_inputs(...)` for validating/aliging externally supplied shrinkage outputs

## What We Are Doing
Quant forecasts can be noisy.  
We shrink both expected returns and covariance before BL uses them.
Project default policy is `jorion` mean-target mode (`mu_0 = mu_g * 1`), set in `src/shrinkage.py`.

Data flow:
`Quant returns -> Shrinkage -> BL inputs`

BL inputs from this repo are:
- `tickers`
- `mu`
- `Sigma`
`scripts/run_shrinkage.py` loads a returns CSV, ignores non-numeric columns (for example Date), runs shrinkage, and prints diagnostics.
`tests/test_shrinkage.py` checks bounded shrinkage weights, PSD/symmetric covariance, small-sample regularization, mean-shrink direction, and degenerate constant assets.

## Pipeline (Step-by-Step)
1. Read the returns matrix from CSV (`scripts/run_shrinkage.py`).
2. Keep only numeric return columns and drop rows with NaN/inf (`clean_returns`).
3. Compute sample mean and sample covariance (`sample_mean_cov`).
4. Build a constant-correlation covariance target (`constant_corr_target`).
5. Compute covariance shrinkage intensity and combine sample/target covariances (`lw_shrink_to_target`).
6. Compute James-Stein mean shrinkage with selectable target:
   - `jorion` (default): shrink toward GLS grand mean `mu_g` (`mu_0 = mu_g * 1`)
   - `benchmark`/`basic`: shrink toward user benchmark `mu_0 * 1` (or inferred trailing market average)
7. Save outputs:
   - `.npz`: `mu_shrunk`/`mu_js`, `asset_names`, `Sigma_shrunk`
   - `_mu_js.csv`: `ETF,mu_js`
   - `.json`: summary diagnostics for reporting

## Run and Verify
- Run shrinkage:
  `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz`
- Run shrinkage with basic benchmark target:
  `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz --mean-target-mode benchmark --benchmark-mu0 0.001`
- Run tests:
  `pytest -q`
- Expected: test suite passes and CLI prints `T`, `N`, shrinkage weights, and covariance norm diagnostics.

Run demo: `python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz`
CLI prints `T`, `N`, `w_mu`, `w_cov`, `rho_bar`, `mu_g`, target mode/level, first 5 `mu_sample` vs `mu_shrunk`, and Frobenius norms.
Outputs: `out_shrinkage.npz` with `mu_shrunk` and `Sigma_shrunk`, plus `out_shrinkage.json` with summary diagnostics.
Run tests from project root: `pytest -q`
BL bridge example:
`from src.bl_integration import build_bl_inputs_from_returns`
`bl_inputs = build_bl_inputs_from_returns(df_returns)`
`mu = bl_inputs["mu"]`
`Sigma = bl_inputs["Sigma"]`

Export BL-ready files from CSV:
`python scripts/run_bl_bridge.py --in returns.csv --out out_bl_inputs.npz`
