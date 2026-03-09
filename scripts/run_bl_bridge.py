"""Build BL-ready inputs from a returns CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bl_integration import build_bl_inputs_from_returns
from src.shrinkage import DEFAULT_MEAN_TARGET_MODE


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Quant -> Shrinkage bridge and export BL-ready inputs."
    )
    p.add_argument("--in", dest="input_csv", required=True, help="Returns CSV path.")
    p.add_argument("--out", dest="output_npz", required=True, help="Output .npz path.")
    p.add_argument(
        "--mean-target-mode",
        dest="mean_target_mode",
        choices=["jorion", "benchmark", "basic"],
        default=DEFAULT_MEAN_TARGET_MODE,
        help="Mean shrink target: 'jorion' (mu_0=mu_g) or 'benchmark'/'basic' (mu_0=user benchmark).",
    )
    p.add_argument(
        "--benchmark-mu0",
        dest="benchmark_mu0",
        type=float,
        default=None,
        help=(
            "Scalar benchmark target for benchmark/basic mode. "
            "If omitted, uses trailing equal-weight market average."
        ),
    )
    p.add_argument(
        "--benchmark-lookback",
        dest="benchmark_lookback",
        type=int,
        default=6,
        help=(
            "Trailing periods for inferred benchmark target when --benchmark-mu0 is not set "
            "(benchmark/basic mode only)."
        ),
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    in_path = Path(args.input_csv)
    out_npz = Path(args.output_npz)

    if not in_path.exists():
        raise SystemExit(f"Input CSV does not exist: {in_path}")

    out_mu_csv = out_npz.with_name(f"{out_npz.stem}_mu.csv")
    out_sigma_csv = out_npz.with_name(f"{out_npz.stem}_sigma.csv")

    df = pd.read_csv(in_path)
    bl_inputs = build_bl_inputs_from_returns(
        df,
        mean_target_mode=args.mean_target_mode,
        benchmark_mu0=args.benchmark_mu0,
        benchmark_lookback=args.benchmark_lookback,
    )
    tickers = bl_inputs["tickers"]
    mu = bl_inputs["mu"]
    sigma = bl_inputs["Sigma"]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        tickers=np.array(tickers, dtype=object),
        mu=mu,
        Sigma=sigma,
    )

    pd.DataFrame({"ETF": tickers, "mu": mu}).to_csv(out_mu_csv, index=False)
    pd.DataFrame(sigma, index=tickers, columns=tickers).to_csv(out_sigma_csv)

    print(f"N={len(tickers)}")
    print(f"mu shape={mu.shape}, Sigma shape={sigma.shape}")
    print(f"saved: {out_npz}")
    print(f"saved: {out_mu_csv}")
    print(f"saved: {out_sigma_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
