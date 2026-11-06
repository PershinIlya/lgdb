"""Meta-level axes as latent factors of the text x topic position matrix.

Rows are texts, columns are bipolar topics, values are the text's position
(share of its claims in camp A, 0..1). Most cells are empty because a text
touches few topics, so the factorization only fits observed cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def _standardize(x: np.ndarray):
    mask = ~np.isnan(x)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    z = (x - mu) / sd
    return np.where(mask, z, 0.0), mask


def fit_masked_mf(x: np.ndarray, k: int, lam: float = 1.0, iters: int = 200,
                  tol: float = 1e-6, seed: int = 0):
    """Alternating least squares on observed cells only. Returns (U, V)."""
    z, mask = _standardize(x)
    n, m = z.shape
    if k == 0:
        return np.zeros((n, 0)), np.zeros((m, 0))
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=0.1, size=(n, k))
    v = rng.normal(scale=0.1, size=(m, k))
    eye = lam * np.eye(k)
    prev = np.inf
    for _ in range(iters):
        for i in range(n):
            o = mask[i]
            if o.any():
                vo = v[o]
                u[i] = np.linalg.solve(vo.T @ vo + eye, vo.T @ z[i, o])
        for j in range(m):
            o = mask[:, j]
            if o.any():
                uo = u[o]
                v[j] = np.linalg.solve(uo.T @ uo + eye, uo.T @ z[o, j])
        loss = np.sum((mask * (z - u @ v.T)) ** 2)
        if abs(prev - loss) < tol * max(1.0, prev):
            break
        prev = loss
    return u, v


def heldout_error(x: np.ndarray, k: int, frac: float = 0.1, repeats: int = 5,
                  lam: float = 1.0, seed: int = 0):
    """RMSE on randomly held-out observed cells, standardized scale."""
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(x))
    errs = []
    for r in range(repeats):
        take = rng.choice(len(obs), size=max(1, int(frac * len(obs))), replace=False)
        test = obs[take]
        train = x.copy()
        train[test[:, 0], test[:, 1]] = np.nan
        z_full, _ = _standardize(train)  # scale from training cells only
        mu = np.nanmean(train, axis=0)
        sd = np.nanstd(train, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        u, v = fit_masked_mf(train, k, lam=lam, seed=seed + r)
        pred = (u @ v.T) if k else np.zeros_like(z_full)
        truth = (x[test[:, 0], test[:, 1]] - mu[test[:, 1]]) / sd[test[:, 1]]
        errs.append(np.sqrt(np.mean((pred[test[:, 0], test[:, 1]] - truth) ** 2)))
    return float(np.mean(errs)), float(np.std(errs, ddof=1) / np.sqrt(len(errs))) if len(errs) > 1 else 0.0


def choose_k(x: np.ndarray, k_max: int = 8, **kw):
    """Smallest k whose held-out error is within 1 standard error of the minimum.
    k = 0 (column means only) is a candidate, so 'no latent structure' can win."""
    table = {k: heldout_error(x, k, **kw) for k in range(0, k_max + 1)}
    best = min(table, key=lambda k: table[k][0])
    limit = table[best][0] + table[best][1]
    chosen = min(k for k in table if table[k][0] <= limit)
    return chosen, table


def varimax(phi: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-6):
    p, k = phi.shape
    if k < 2:
        return phi, np.eye(k)
    r = np.eye(k)
    d = 0.0
    for _ in range(q):
        lam = phi @ r
        u, s, vh = np.linalg.svd(
            phi.T @ (lam ** 3 - (gamma / p) * lam @ np.diag(np.diag(lam.T @ lam))))
        r = u @ vh
        d_new = np.sum(s)
        if d_new < d * (1 + tol):
            break
        d = d_new
    return phi @ r, r


def tucker(a: np.ndarray, b: np.ndarray) -> float:
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den else 0.0


def loadings(x: np.ndarray, k: int, lam: float = 1.0, seed: int = 0) -> np.ndarray:
    _, v = fit_masked_mf(x, k, lam=lam, seed=seed)
    if k >= 2:
        v, _ = varimax(v)
    return v


@dataclass
class FactorResult:
    k: int
    cv_table: dict
    loadings: np.ndarray          # topics x k
    stability: np.ndarray         # share of bootstraps with |congruence| >= threshold
    stable: np.ndarray            # bool per factor


def analyze(x: np.ndarray, k_max: int = 8, n_boot: int = 1000, congruence_min: float = 0.85,
            stable_share: float = 0.8, lam: float = 1.0, seed: int = 0) -> FactorResult:
    k, table = choose_k(x, k_max=k_max, lam=lam, seed=seed)
    if k == 0:
        return FactorResult(0, table, np.zeros((x.shape[1], 0)), np.array([]), np.array([], dtype=bool))
    ref = loadings(x, k, lam=lam, seed=seed)
    rng = np.random.default_rng(seed)
    hits = np.zeros(k)
    n = x.shape[0]
    for b in range(n_boot):
        rows = rng.integers(0, n, size=n)
        vb = loadings(x[rows], k, lam=lam, seed=seed + b + 1)
        c = np.array([[abs(tucker(ref[:, i], vb[:, j])) for j in range(k)] for i in range(k)])
        ri, ci = linear_sum_assignment(-c)
        for i, j in zip(ri, ci):
            if c[i, j] >= congruence_min:
                hits[i] += 1
    share = hits / n_boot
    return FactorResult(k, table, ref, share, share >= stable_share)
