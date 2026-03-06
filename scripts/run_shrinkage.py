"""Run shrinkage estimators on a returns CSV and save outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shrinkage import shrink_inputs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shrink expected returns and covariance from CSV.")
    p.add_argument("--in", dest="input_csv", required=True, help="Returns CSV path.")
    p.add_argument("--out", dest="output_npz", required=True, help="Output .npz path.")
    return p


def main() -> int:
    args = _parser().parse_args()
    in_path = Path(args.input_csv)
    out_npz = Path(args.output_npz)
    out_json = out_npz.with_suffix(".json")
    out_mu_csv = out_npz.with_name(f"{out_npz.stem}_mu_js.csv")

    if not in_path.exists():
        raise SystemExit(f"Input CSV does not exist: {in_path}")

    # Keep CSV loading here so non-numeric columns can be filtered inside shrink_inputs.
    df = pd.read_csv(in_path)
    out = shrink_inputs(df)

    mu_sample = out["mu_sample"]
    mu_shrunk = out["mu_shrunk"]
    mu_js_table = out["mu_js_table"]
    Sigma_sample = out["Sigma_sample"]
    Sigma_shrunk = out["Sigma_shrunk"]
    Sigma_target = out["Sigma_target_cc"]

    print(f"T={out['T']}, N={out['N']}")
    print(
        f"w_mu={out['w_mu']:.6f}, w_cov={out['w_cov']:.6f}, "
        f"rho_bar={out['rho_bar']:.6f}, mu_g={out['mu_g']:.6f}"
    )
    print("first 5 mu_sample vs mu_shrunk")
    for i, name in enumerate(out["asset_names"][:5]):
        print(f"  {name}: {mu_sample[i]:.6e} -> {mu_shrunk[i]:.6e}")

    # Diagnostic norms: how far sample covariance sits from shrunk and target matrices.
    norm_sample_shrunk = float(np.linalg.norm(Sigma_sample - Sigma_shrunk, ord="fro"))
    norm_sample_target = float(np.linalg.norm(Sigma_sample - Sigma_target, ord="fro"))
    print(f"||Sigma_sample - Sigma_shrunk||_F = {norm_sample_shrunk:.6e}")
    print(f"||Sigma_sample - Sigma_target_cc||_F = {norm_sample_target:.6e}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        mu_shrunk=mu_shrunk,
        mu_js=mu_shrunk,
        asset_names=np.array(out["asset_names"], dtype=object),
        Sigma_shrunk=Sigma_shrunk,
    )
    mu_js_table.to_csv(out_mu_csv, index=False)

    # JSON summary is intentionally lightweight for reporting / handoff.
    summary = {
        "T": int(out["T"]),
        "N": int(out["N"]),
        "w_mu": float(out["w_mu"]),
        "w_cov": float(out["w_cov"]),
        "rho_bar": float(out["rho_bar"]),
        "mu_g": float(out["mu_g"]),
        "norm_sample_minus_shrunk": norm_sample_shrunk,
        "norm_sample_minus_target_cc": norm_sample_target,
        "asset_names": out["asset_names"],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {out_npz}")
    print(f"saved: {out_mu_csv}")
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
