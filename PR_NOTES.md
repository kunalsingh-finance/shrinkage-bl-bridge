# Shrinkage PR Notes

## What this PR does

This PR adds a clean shrinkage module that can be plugged into the BL model.

Pipeline now is:

1. Read quant return data.
2. Shrink expected returns (`mu_hat -> mu_js`).
3. Shrink covariance matrix (`Sigma -> Sigma_shrunk`).
4. Pass both outputs into BL.

## Why we are doing this

Raw quant inputs can be noisy.  
Shrinkage gives more stable inputs so BL is less sensitive to estimation error.

## Main files

- `src/shrinkage.py`
  - Core formulas and shrinkage calculations.
  - BL-facing function: `bl_shrinkage_inputs(...)`.
- `src/bl_integration.py`
  - Converts shrinkage outputs into BL-ready arrays:
    - `tickers`
    - `mu`
    - `Sigma`
- `scripts/run_shrinkage.py`
  - CLI that runs shrinkage and exports diagnostics.
- `scripts/run_bl_bridge.py`
  - CLI that exports final BL-ready files from returns CSV.

## Output contract

Shrinkage output used by BL:

- `mu_js_table` with columns:
  - `ETF`
  - `mu_js`
- `Sigma_shrunk` with shape `N x N`

BL bridge output:

- `tickers` (list of ETFs)
- `mu` (shape `N`)
- `Sigma` (shape `N x N`)

## Quick run commands

```bash
python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz
python scripts/run_bl_bridge.py --in returns.csv --out out_bl_inputs.npz
pytest -q
```

## Validation done

- Shape checks
- Finite-value checks
- Covariance symmetry + PSD checks
- Existing shrinkage tests
- BL integration contract tests
