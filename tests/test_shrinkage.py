"""Tests for shrinkage estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.shrinkage import bl_shrinkage_inputs, shrink_inputs


def test_w_mu_and_w_cov_bounded() -> None:
    rng = np.random.default_rng(123)
    R = rng.normal(0.001, 0.02, size=(100, 12))
    out = shrink_inputs(R)
    assert 0.0 <= out["w_mu"] <= 1.0
    assert 0.0 <= out["w_cov"] <= 1.0


def test_sigma_shrunk_symmetric_and_psd() -> None:
    rng = np.random.default_rng(7)
    R = rng.normal(0.0, 0.01, size=(80, 10))
    out = shrink_inputs(R)
    Sigma = out["Sigma_shrunk"]
    assert np.allclose(Sigma, Sigma.T, atol=1e-12)
    min_eig = float(np.min(np.linalg.eigvalsh(Sigma)))
    assert min_eig >= -1e-10


def test_small_t_large_n_yields_positive_cov_shrinkage() -> None:
    rng = np.random.default_rng(99)
    R = rng.normal(0.0, 0.02, size=(10, 30))
    out = shrink_inputs(R)
    assert out["w_cov"] > 0.0


def test_mu_shrunk_moves_toward_grand_mean_if_shrinking_active() -> None:
    rng = np.random.default_rng(202)
    R = rng.normal(0.0005, 0.015, size=(90, 8))
    out = shrink_inputs(R)
    if out["w_mu"] > 0:
        one = np.ones(out["N"])
        d_sample = float(np.linalg.norm(out["mu_sample"] - out["mu_g"] * one))
        d_shrunk = float(np.linalg.norm(out["mu_shrunk"] - out["mu_g"] * one))
        assert d_shrunk <= d_sample + 1e-12


def test_constant_asset_does_not_crash() -> None:
    rng = np.random.default_rng(404)
    R = rng.normal(0.0, 0.01, size=(60, 4))
    const = np.full((60, 1), 0.001)
    R = np.hstack([R, const])
    df = pd.DataFrame(R, columns=["a", "b", "c", "d", "const_asset"])
    df.insert(0, "Date", pd.date_range("2024-01-01", periods=60, freq="D"))

    out = shrink_inputs(df)
    idx = out["asset_names"].index("const_asset")
    Sigma = out["Sigma_shrunk"]
    assert np.isclose(Sigma[idx, idx], 0.0, atol=1e-12)
    assert np.allclose(Sigma[idx, :], 0.0, atol=1e-12)
    assert np.allclose(Sigma[:, idx], 0.0, atol=1e-12)


def test_bl_shrinkage_outputs_have_expected_contract() -> None:
    rng = np.random.default_rng(77)
    df = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(50, 3)),
        columns=["SPY", "QQQ", "IWM"],
    )

    mu_js_table, Sigma_shrunk = bl_shrinkage_inputs(df)

    assert list(mu_js_table.columns) == ["ETF", "mu_js"]
    assert mu_js_table.shape == (3, 2)
    assert mu_js_table["ETF"].tolist() == ["SPY", "QQQ", "IWM"]
    assert np.all(np.isfinite(mu_js_table["mu_js"].to_numpy(dtype=float)))

    assert Sigma_shrunk.shape == (3, 3)
    assert np.allclose(Sigma_shrunk, Sigma_shrunk.T, atol=1e-12)


def test_benchmark_mode_uses_explicit_mu0_target() -> None:
    rng = np.random.default_rng(1234)
    R = rng.normal(0.001, 0.02, size=(90, 6))
    mu0 = 0.0025
    out = shrink_inputs(R, mean_target_mode="benchmark", benchmark_mu0=mu0)

    assert out["mean_target_mode"] == "benchmark"
    assert np.isclose(out["mu_target_scalar"], mu0, atol=1e-15)
    assert out["benchmark_lookback_used"] is None

    if out["w_mu"] > 0:
        target = np.full(out["N"], mu0)
        d_sample = float(np.linalg.norm(out["mu_sample"] - target))
        d_shrunk = float(np.linalg.norm(out["mu_shrunk"] - target))
        assert d_shrunk <= d_sample + 1e-12


def test_benchmark_mode_infers_trailing_market_average_by_default() -> None:
    # Last 6 row means are: 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 -> average 0.055
    R = np.array(
        [
            [0.0, 0.01, 0.02],
            [0.01, 0.02, 0.03],
            [0.02, 0.03, 0.04],
            [0.01, 0.02, 0.03],
            [0.02, 0.03, 0.04],
            [0.03, 0.04, 0.05],
            [0.04, 0.05, 0.06],
            [0.05, 0.06, 0.07],
            [0.06, 0.07, 0.08],
            [0.07, 0.08, 0.09],
        ],
        dtype=float,
    )
    out = shrink_inputs(R, mean_target_mode="benchmark")
    assert out["mean_target_mode"] == "benchmark"
    assert out["benchmark_lookback_used"] == 6
    assert np.isclose(out["mu_target_scalar"], 0.055, atol=1e-12)
