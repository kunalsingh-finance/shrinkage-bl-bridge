# Shrinkage Progress Update (March 8, 2026)

- Implemented James-Stein mean shrinkage with selectable targets:
  - `jorion` (default): `mu_0 = mu_g * 1`
  - `benchmark` / `basic`: `mu_0 = benchmark_mu0 * 1` (or inferred trailing market average)
- Pipeline default is now explicitly hard-set via `DEFAULT_MEAN_TARGET_MODE = "jorion"` in `src/shrinkage.py`, and used by CLI/BL entrypoints.

## Run Results on `returns.csv`

- Common dimensions: `T=160`, `N=8`
- Covariance shrinkage: `w_cov=1.000000` in both modes

- `jorion` mode:
  - `w_mu=0.729177`
  - `mu_g=-0.000210`
  - `mu_target_scalar=-0.000210`

- `benchmark` mode (trailing lookback `6`):
  - `w_mu=0.162839`
  - `mu_target_scalar=0.001734`
  - `benchmark_lookback_used=6`

## Allocation Impact (Proxy BL Weights)

- Using proxy allocation `w ∝ Sigma^-1 * mu` (sum-to-1 normalization) for side-by-side comparison:
  - `||w_jorion - w_benchmark||_2 = 1.8398`
  - `max_i |w_jorion_i - w_benchmark_i| = 1.4184`
- Interpretation: mean-target choice materially changes implied allocations on this dataset.

## Artifacts

- `out_shrinkage_jorion.*` and `out_shrinkage_benchmark.*` (mode-specific shrinkage outputs)
- `out_mode_compare.json` (summary metrics)
- `out_mode_compare_weights.csv` (ETF-level proxy weight comparison)
- `notebooks/shrinkage_mode_comparison.ipynb` (plot-ready comparison notebook)
