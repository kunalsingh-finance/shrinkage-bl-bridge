"""Exercise the documented offline CLI and read back its saved contract."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_documented_sample_cli(tmp_path):
    output = tmp_path / "result.npz"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_shrinkage.py"),
         "--in", str(ROOT / "data/sample_returns.csv"), "--out", str(output)],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    with np.load(output, allow_pickle=False) as saved:
        covariance = saved["Sigma_shrunk"]
        assert saved["asset_names"].tolist() == ["SPY", "AGG", "GLD"]
        assert saved["mu_shrunk"].shape == (3,)
        assert covariance.shape == (3, 3)
        assert np.isfinite(covariance).all()
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["T"] == 6
    assert summary["asset_names"] == ["SPY", "AGG", "GLD"]


def test_documented_bridge_cli(tmp_path):
    output = tmp_path / "bridge.npz"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_bl_bridge.py"),
         "--in", str(ROOT / "data/sample_returns.csv"), "--out", str(output)],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    with np.load(output, allow_pickle=False) as saved:
        assert saved["tickers"].tolist() == ["SPY", "AGG", "GLD"]
        assert saved["mu"].shape == (3,)
        assert saved["Sigma"].shape == (3, 3)
        assert np.isfinite(saved["mu"]).all()
    assert output.with_name("bridge_mu.csv").exists()
    assert output.with_name("bridge_sigma.csv").exists()
