"""Bridge shrinkage outputs into BL-ready inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.shrinkage import bl_shrinkage_inputs

EPS = 1e-12


def to_bl_inputs(mu_js_table: pd.DataFrame, Sigma_shrunk: np.ndarray) -> dict[str, Any]:
    """
    Convert shrinkage outputs into BL-ready aligned arrays.

    Returns:
      - tickers: list[str]
      - mu: np.ndarray shape (N,)
      - Sigma: np.ndarray shape (N, N)
    """
    if not isinstance(mu_js_table, pd.DataFrame):
        raise TypeError("mu_js_table must be a pandas DataFrame.")
    if list(mu_js_table.columns) != ["ETF", "mu_js"]:
        raise ValueError("mu_js_table must have columns ['ETF', 'mu_js'].")

    tickers = [str(x) for x in mu_js_table["ETF"].tolist()]
    if len(tickers) == 0:
        raise ValueError("mu_js_table is empty.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("ETF ticker names must be unique.")

    mu = mu_js_table["mu_js"].to_numpy(dtype=float).reshape(-1)
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu_js contains non-finite values.")

    Sigma = np.asarray(Sigma_shrunk, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma_shrunk must be square, got shape {Sigma.shape}.")
    if Sigma.shape[0] != mu.shape[0]:
        raise ValueError(
            f"Shape mismatch: len(mu)={mu.shape[0]} but Sigma is {Sigma.shape}."
        )
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma_shrunk contains non-finite values.")

    Sigma = 0.5 * (Sigma + Sigma.T)
    min_eig = float(np.min(np.linalg.eigvalsh(Sigma)))
    if min_eig < -EPS:
        raise ValueError(f"Sigma_shrunk must be PSD; min eigenvalue={min_eig:.3e}.")

    return {"tickers": tickers, "mu": mu, "Sigma": Sigma}


def build_bl_inputs_from_returns(R: Any) -> dict[str, Any]:
    """
    End-to-end bridge:
      Quant returns data -> shrinkage -> BL-ready arrays.
    """
    mu_js_table, Sigma_shrunk = bl_shrinkage_inputs(R)
    return to_bl_inputs(mu_js_table, Sigma_shrunk)
