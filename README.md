# Shrinkage-to-Black-Litterman Input Bridge

This project extends a shrinkage input-estimation workflow into Black-Litterman-ready portfolio inputs. It converts cleaned return data into stabilized expected returns and covariance matrices, then exposes outputs that can be passed directly into downstream Black-Litterman optimization workflows.

The project is educational and does not represent financial advice, a live portfolio recommendation, or production investment software.

## What This Project Does

- Cleans numeric return data from CSV input
- Computes sample mean and covariance estimates
- Builds a constant-correlation covariance target
- Applies covariance shrinkage to improve stability
- Applies Jorion-style mean shrinkage
- Exports Black-Litterman-ready `mu` and `Sigma` inputs
- Provides validation helpers for externally supplied shrinkage outputs
- Includes tests for shrinkage behavior and input alignment

## Why This Project Matters

Black-Litterman workflows still depend on the quality of their inputs. Raw sample means and covariances can be unstable, especially with small samples or many assets. This project demonstrates a bridge between statistical shrinkage and downstream portfolio construction.

The focus is input quality, not claiming that any final portfolio is optimal.

## Data Flow

```text
Asset returns -> Shrinkage estimates -> BL-ready inputs
```

The exported Black-Litterman input dictionary includes:

- `tickers`
- `mu`
- `Sigma`

## Core Modules

- `src/shrinkage.py`: cleaning, sample stats, covariance shrinkage, and mean shrinkage
- `src/bl_integration.py`: Black-Litterman input builders and validation helpers
- `scripts/run_shrinkage.py`: shrinkage CLI runner
- `scripts/run_bl_bridge.py`: BL-ready export CLI runner
- `tests/test_shrinkage.py`: shrinkage and covariance tests

## How to Run

Use Python 3.11 or newer. Create a virtual environment, then install the pinned
numerical dependencies and test runner:

```bash
python -m venv .venv
# Activate .venv using your platform's normal command.
python -m pip install -r requirements-dev.txt
```

The included `data/sample_returns.csv` contains six synthetic monthly observations
for an offline smoke demonstration. Means and covariance retain the input's
per-observation units; they are not annualized or historical performance evidence.

```bash
python scripts/run_shrinkage.py --in data/sample_returns.csv --out outputs/sample.npz
python scripts/run_bl_bridge.py --in data/sample_returns.csv --out outputs/bridge.npz
python -m pytest -q
```

GitHub Actions runs the numerical suite and CLI artifact readback on Linux and Windows.

Run the shrinkage workflow:

```bash
python scripts/run_shrinkage.py --in returns.csv --out out_shrinkage.npz
```

Export Black-Litterman-ready files:

```bash
python scripts/run_bl_bridge.py --in returns.csv --out out_bl_inputs.npz
```

Run tests:

```bash
pytest -q
```

## Outputs

The workflow writes:

- `.npz` files containing shrunk return and covariance inputs
- `_mu_js.csv` with ETF-level shrunk mean estimates
- `.json` summary diagnostics
- BL-ready input dictionaries containing `tickers`, `mu`, and `Sigma`

## Skills Demonstrated

- Portfolio input estimation
- Covariance shrinkage
- Expected return shrinkage
- Black-Litterman input preparation
- Numerical validation and testing
- Python workflow modularization

## Limitations

- Shrinkage improves input stability but does not create reliable return forecasts.
- The covariance target is intentionally transparent rather than highly customized.
- This project prepares inputs for downstream optimizers; it is not a full investment management system.
- Results are educational analytics, not investment advice.
