"""Tests for BL integration bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bl_integration import build_bl_inputs_from_returns, to_bl_inputs


def test_build_bl_inputs_contract() -> None:
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        rng.normal(0.0, 0.015, size=(120, 4)),
        columns=["SPY", "QQQ", "IWM", "EFA"],
    )
    out = build_bl_inputs_from_returns(df)

    assert set(out.keys()) == {"tickers", "mu", "Sigma"}
    assert out["tickers"] == ["SPY", "QQQ", "IWM", "EFA"]
    assert out["mu"].shape == (4,)
    assert out["Sigma"].shape == (4, 4)
    assert np.allclose(out["Sigma"], out["Sigma"].T, atol=1e-12)


def test_to_bl_inputs_rejects_bad_columns() -> None:
    bad_table = pd.DataFrame({"ticker": ["SPY"], "mu": [0.01]})
    Sigma = np.array([[0.1]])
    with pytest.raises(ValueError):
        to_bl_inputs(bad_table, Sigma)


def test_to_bl_inputs_rejects_duplicate_tickers() -> None:
    table = pd.DataFrame({"ETF": ["SPY", "SPY"], "mu_js": [0.01, 0.02]})
    Sigma = np.eye(2)
    with pytest.raises(ValueError):
        to_bl_inputs(table, Sigma)


def test_build_bl_inputs_supports_benchmark_mode() -> None:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        rng.normal(0.0, 0.015, size=(120, 4)),
        columns=["SPY", "QQQ", "IWM", "EFA"],
    )
    out = build_bl_inputs_from_returns(df, mean_target_mode="benchmark", benchmark_mu0=0.001)
    assert out["mu"].shape == (4,)
    assert out["Sigma"].shape == (4, 4)
    assert np.allclose(out["Sigma"], out["Sigma"].T, atol=1e-12)
