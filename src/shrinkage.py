"""Shrinkage estimators for expected returns and covariance inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12
ZERO_VAR_TOL = 1e-14
PSD_TOL = -1e-10


def _as_2d_float_array(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array with shape (T, N), got ndim={arr.ndim}.")
    return arr


def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _enforce_psd(m: np.ndarray, zero_var_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Force symmetry and project to PSD if eigenvalues are materially negative.
    Assets with near-zero variance are pinned to exactly zero variance/covariance.
    """
    out = _symmetrize(np.asarray(m, dtype=float))

    if zero_var_mask is not None and np.any(zero_var_mask):
        out[zero_var_mask, :] = 0.0
        out[:, zero_var_mask] = 0.0
        diag = np.diag(out).copy()
        diag[zero_var_mask] = 0.0
        np.fill_diagonal(out, diag)

    eigvals, eigvecs = np.linalg.eigh(out)
    if float(np.min(eigvals)) < PSD_TOL:
        eigvals = np.clip(eigvals, 0.0, None)
        out = eigvecs @ np.diag(eigvals) @ eigvecs.T
        out = _symmetrize(out)

    if zero_var_mask is not None and np.any(zero_var_mask):
        out[zero_var_mask, :] = 0.0
        out[:, zero_var_mask] = 0.0
        diag = np.diag(out).copy()
        diag[zero_var_mask] = 0.0
        np.fill_diagonal(out, diag)

    return out


def clean_returns(R: Any) -> tuple[np.ndarray, list[str]]:
    """
    Clean returns and extract asset names.

    - Accepts `numpy.ndarray` or `pandas.DataFrame`
    - DataFrame path keeps numeric columns only (Date/text columns are ignored)
    - Drops rows containing any non-finite value
    """
    if isinstance(R, pd.DataFrame):
        numeric = R.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("DataFrame input contains no numeric return columns.")
        arr = numeric.to_numpy(dtype=float)
        asset_names = [str(c) for c in numeric.columns]
    else:
        arr = _as_2d_float_array(np.asarray(R, dtype=float))
        asset_names = [f"asset_{i}" for i in range(arr.shape[1])]

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Returns matrix must have at least one row and one column.")

    keep = np.all(np.isfinite(arr), axis=1)
    arr = arr[keep]
    if arr.shape[0] < 2:
        raise ValueError(
            "Need at least 2 complete observations after dropping rows with NaN/inf."
        )

    return arr, asset_names


def sample_mean_cov(R_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample mean, sample covariance (ddof=1), and demeaned returns."""
    R_np = _as_2d_float_array(R_np)
    if not np.all(np.isfinite(R_np)):
        raise ValueError("R_np contains non-finite values; run clean_returns first.")
    T, _ = R_np.shape
    if T < 2:
        raise ValueError("At least two observations are required for covariance (ddof=1).")

    mu_hat = np.mean(R_np, axis=0)
    X = R_np - mu_hat
    S = (X.T @ X) / (T - 1)
    S = _symmetrize(S)
    return mu_hat, S, X


def constant_corr_target(S: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build constant-correlation covariance target from sample covariance matrix S.

    Off-diagonal correlations with zero-volatility assets are treated as 0.
    """
    S = _as_2d_float_array(S)
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got shape {S.shape}.")

    S = _symmetrize(S)
    N = S.shape[0]
    var = np.clip(np.diag(S), 0.0, None)
    vol = np.sqrt(var)

    # Build a robust sample correlation matrix from S.
    C = np.zeros((N, N), dtype=float)
    np.fill_diagonal(C, 1.0)

    for i in range(N):
        for j in range(i + 1, N):
            denom = vol[i] * vol[j]
            # If either asset has (near) zero volatility, define correlation as 0.
            if denom > EPS:
                cij = float(np.clip(S[i, j] / denom, -1.0, 1.0))
            else:
                cij = 0.0
            C[i, j] = cij
            C[j, i] = cij

    if N < 2:
        rho_bar = 0.0
    else:
        iu = np.triu_indices(N, k=1)
        rho_bar = float(np.mean(C[iu])) if iu[0].size else 0.0

    # Constant-correlation target: keep each variance, replace all pairwise
    # correlations by the global average off-diagonal correlation.
    C_cc = np.full((N, N), rho_bar, dtype=float)
    np.fill_diagonal(C_cc, 1.0)
    Sigma_cc = np.outer(vol, vol) * C_cc
    Sigma_cc = _symmetrize(Sigma_cc)
    return Sigma_cc, rho_bar


def lw_shrink_to_target(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Shrink sample covariance S toward target F using a Ledoit-Wolf-style intensity.

    phi = (1/T) * sum_t ||x_t x_t' - S||_F^2
    gamma = ||S - F||_F^2
    w_cov = clip(phi / gamma, 0, 1), or 1 when gamma is effectively zero.
    """
    X = _as_2d_float_array(X)
    T, N = X.shape
    if T < 2:
        raise ValueError("Need at least 2 observations for covariance shrinkage.")

    F = _as_2d_float_array(F)
    if F.shape != (N, N):
        raise ValueError(f"Target F must have shape {(N, N)}, got {F.shape}.")
    F = _symmetrize(F)

    S = (X.T @ X) / (T - 1)
    S = _symmetrize(S)

    # Estimate phi = average squared noise around S.
    phi_acc = 0.0
    for t in range(T):
        d = np.outer(X[t], X[t]) - S
        phi_acc += float(np.sum(d * d))
    phi = phi_acc / T

    # gamma is the squared distance between sample covariance and target.
    gamma = float(np.sum((S - F) ** 2))
    w_cov = 1.0 if gamma <= EPS else float(np.clip(phi / gamma, 0.0, 1.0))

    # Convex combination between sample covariance and target covariance.
    Sigma_shr = (1.0 - w_cov) * S + w_cov * F
    zero_var_mask = np.diag(S) <= ZERO_VAR_TOL
    Sigma_shr = _enforce_psd(Sigma_shr, zero_var_mask=zero_var_mask)
    return Sigma_shr, w_cov, S


def jorion_mean_shrink(
    mu_hat: np.ndarray, Sigma_shr: np.ndarray, T: int
) -> tuple[np.ndarray, float, float]:
    """Shrink sample means toward the GLS grand mean using Jorion's formula."""
    mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
    Sigma_shr = _as_2d_float_array(Sigma_shr)

    N = mu_hat.shape[0]
    if Sigma_shr.shape != (N, N):
        raise ValueError(f"Sigma_shr shape mismatch: expected {(N, N)}, got {Sigma_shr.shape}.")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    Sigma_inv = np.linalg.pinv(_symmetrize(Sigma_shr))
    one = np.ones(N, dtype=float)

    den = float(one @ Sigma_inv @ one)
    if abs(den) <= EPS:
        # Fallback: if GLS denominator is unstable, use simple average mean.
        mu_g = float(np.mean(mu_hat))
    else:
        mu_g = float((one @ Sigma_inv @ mu_hat) / den)

    d = mu_hat - mu_g * one
    quad = max(float(d @ Sigma_inv @ d), 0.0)
    # Larger cross-sectional dispersion (quad) -> less mean shrinkage.
    w_mu = float((N + 2) / (N + 2 + T * quad))
    w_mu = float(np.clip(w_mu, 0.0, 1.0))
    mu_shr = (1.0 - w_mu) * mu_hat + w_mu * (mu_g * one)
    return mu_shr, w_mu, mu_g


def shrink_inputs(R: Any) -> dict[str, Any]:
    """Produce shrunk mean/covariance inputs and diagnostics for Markowitz."""
    R_np, asset_names = clean_returns(R)
    T, N = R_np.shape

    # 1) Estimate sample moments.
    mu_hat, S, X = sample_mean_cov(R_np)
    zero_var_mask = np.diag(S) <= ZERO_VAR_TOL
    S = _enforce_psd(S, zero_var_mask=zero_var_mask)

    # 2) Build a structured covariance target (constant correlation).
    Sigma_cc, rho_bar = constant_corr_target(S)
    Sigma_cc = _enforce_psd(Sigma_cc, zero_var_mask=zero_var_mask)

    # 3) Shrink covariance toward target with data-driven intensity.
    Sigma_shr, w_cov, S_from_lw = lw_shrink_to_target(X, Sigma_cc)
    S_from_lw = _enforce_psd(S_from_lw, zero_var_mask=zero_var_mask)
    Sigma_shr = _enforce_psd(Sigma_shr, zero_var_mask=zero_var_mask)

    # 4) Shrink means toward the GLS grand mean.
    mu_shr, w_mu, mu_g = jorion_mean_shrink(mu_hat, Sigma_shr, T=T)

    return {
        "T": T,
        "N": N,
        "asset_names": asset_names,
        "mu_sample": mu_hat,
        "Sigma_sample": S_from_lw,
        "rho_bar": rho_bar,
        "Sigma_target_cc": Sigma_cc,
        "w_cov": w_cov,
        "Sigma_shrunk": Sigma_shr,
        "mu_g": mu_g,
        "w_mu": w_mu,
        "mu_shrunk": mu_shr,
    }
