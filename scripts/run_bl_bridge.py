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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Quant -> Shrinkage bridge and export BL-ready inputs."
    )
    p.add_argument("--in", dest="input_csv", required=True, help="Returns CSV path.")
    p.add_argument("--out", dest="output_npz", required=True, help="Output .npz path.")
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
    bl_inputs = build_bl_inputs_from_returns(df)
    tickers = bl_inputs["tickers"]
    mu = bl_inputs["mu"]
    sigma = bl_inputs["Sigma"]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        tickers=np.array(tickers, dtype=str),
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
