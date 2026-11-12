"""Piecewise-exponential first-reply hazard with a position covariate (step 4).

log h_i(t) = alpha_k + beta * x_i(t)  for t in time bin k,

fitted as a Poisson GLM on (bin x covariate-bucket) aggregated events and exposure. M0 has the
bin effects only (the non-parametric time decay); M1 adds beta * x. The covariate may change across
bins (time-varying position, proxy d) by passing one bucket per (parent, bin).

"Time decay" is summarised as the slope of alpha_k on log(bin midpoint) over a fixed range (weighted
by events). Flattening = slope(M1) - slope(M0) > 0.
"""
from __future__ import annotations

import numpy as np


def aggregate(t: np.ndarray, d: np.ndarray, edges: np.ndarray, bucket, w: np.ndarray | None = None,
              n_buckets: int | None = None) -> dict:
    """bucket: int array (static, one per parent) or callable k -> int array (bucket in bin k)."""
    w = np.ones_like(t) if w is None else w
    K = len(edges) - 1
    B = n_buckets or (int(bucket.max()) + 1 if not callable(bucket) else None)
    ev = np.zeros((K, B))
    ex = np.zeros((K, B))
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        bk = bucket(k) if callable(bucket) else bucket
        e = np.clip(t - a, 0.0, b - a)
        inb = d & (t >= a) & (t < b)
        ex[k] = np.bincount(bk, weights=w * e, minlength=B)[:B]
        ev[k] = np.bincount(bk, weights=w * inb, minlength=B)[:B]
    return {"events": ev, "exposure": ex}


def fit(tab: dict, xval: np.ndarray | None, iters: int = 100) -> dict:
    """Poisson GLM: events[k, j] ~ exposure[k, j] * exp(alpha_k + beta * xval[j])."""
    ev, ex = tab["events"], tab["exposure"]
    K = ev.shape[0]
    ok_bin = ex.sum(1) > 0
    alpha = np.full(K, np.nan)
    with np.errstate(divide="ignore"):
        alpha[ok_bin] = np.log(np.maximum(ev[ok_bin].sum(1), 1e-12) / ex[ok_bin].sum(1))
    if xval is None:
        ll = _ll(ev, ex, alpha, 0.0, np.zeros(ev.shape[1]))
        return {"alpha": alpha, "beta": None, "se_beta": None, "loglik": ll}
    beta = 0.0
    for _ in range(iters):
        r = np.exp(beta * xval)[None, :]
        # profile alpha_k given beta
        denom = (ex * r).sum(1)
        alpha = np.where(denom > 0, np.log(np.maximum(ev.sum(1), 1e-12) / np.maximum(denom, 1e-300)), np.nan)
        mu = np.exp(np.nan_to_num(alpha, nan=-np.inf))[:, None] * ex * r
        g = ((ev - mu) * xval[None, :]).sum()
        # profiled Hessian (Schur complement over the alpha_k)
        s1 = (mu * xval[None, :]).sum(1)
        s0 = mu.sum(1)
        h = (mu * xval[None, :] ** 2).sum() - np.sum(np.where(s0 > 0, s1 ** 2 / np.maximum(s0, 1e-300), 0))
        step = g / h if h > 0 else 0.0
        beta += step
        if abs(step) < 1e-10:
            break
    se = 1 / np.sqrt(h) if h > 0 else np.nan
    return {"alpha": alpha, "beta": float(beta), "se_beta": float(se),
            "loglik": _ll(ev, ex, alpha, beta, xval)}


def _ll(ev, ex, alpha, beta, xval) -> float:
    a = np.nan_to_num(alpha, nan=-np.inf)[:, None]
    lin = a + beta * xval[None, :]
    mu = np.exp(lin) * ex
    with np.errstate(invalid="ignore", divide="ignore"):
        term = np.where(ev > 0, ev * lin, 0.0)
    return float(np.nansum(term) - mu.sum())


def heldout_ll(tab: dict, model: dict, xval: np.ndarray | None) -> float:
    """Log-likelihood of held-out aggregated data under fitted alpha (and beta), per unit weight
    of exposure-bearing parents is handled by the caller."""
    beta = model["beta"] or 0.0
    xv = xval if xval is not None else np.zeros(tab["events"].shape[1])
    return _ll(tab["events"], tab["exposure"], model["alpha"], beta, xv)


def decay_slope(alpha: np.ndarray, edges: np.ndarray, events: np.ndarray, t_lo: float, t_hi: float) -> float:
    """Events-weighted slope of alpha_k (log hazard) on log10(bin midpoint), bins within [t_lo, t_hi]."""
    mid = np.sqrt(np.maximum(edges[:-1], 1.0) * edges[1:])
    ev = events.sum(1)
    sel = (mid >= t_lo) & (mid <= t_hi) & np.isfinite(alpha) & (ev >= 5)
    if sel.sum() < 3:
        return float("nan")
    x, y, wt = np.log10(mid[sel]), alpha[sel] / np.log(10), ev[sel]
    xm, ym = np.average(x, weights=wt), np.average(y, weights=wt)
    return float(np.sum(wt * (x - xm) * (y - ym)) / np.sum(wt * (x - xm) ** 2))


class Prepared:
    """Per-parent, per-bin exposure / event / bucket arrays, so that aggregation under new weights
    (bootstrap) and thread-clustered scores are cheap. K x N arrays."""

    def __init__(self, t: np.ndarray, d: np.ndarray, edges: np.ndarray, bucket, n_buckets: int):
        K = len(edges) - 1
        self.K, self.B, self.edges = K, n_buckets, edges
        self.E = np.empty((K, len(t)), dtype=np.float32)
        self.D = np.zeros((K, len(t)), dtype=bool)
        self.Bk = np.empty((K, len(t)), dtype=np.int32)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            self.E[k] = np.clip(t - a, 0.0, b - a)
            self.D[k] = d & (t >= a) & (t < b)
            self.Bk[k] = bucket(k) if callable(bucket) else bucket

    def table(self, w: np.ndarray) -> dict:
        ev = np.zeros((self.K, self.B))
        ex = np.zeros((self.K, self.B))
        for k in range(self.K):
            ex[k] = np.bincount(self.Bk[k], weights=w * self.E[k], minlength=self.B)[:self.B]
            ev[k] = np.bincount(self.Bk[k], weights=w * self.D[k], minlength=self.B)[:self.B]
        return {"events": ev, "exposure": ex}

    def cluster_se_beta(self, model: dict, xval: np.ndarray, w: np.ndarray, thread_idx: np.ndarray) -> float:
        """Thread-clustered SE of beta (alpha_k profiled out): sandwich on per-thread efficient scores
        sum_i sum_k w_i (x_ik - c_k)(d_ik - mu_ik), c_k = weighted mean of x among expected events in bin k."""
        beta = model["beta"]
        G = int(thread_idx.max()) + 1
        eff = np.zeros(G)
        info = 0.0
        for k in range(self.K):
            a = model["alpha"][k]
            if not np.isfinite(a):
                continue
            x = xval[self.Bk[k]]
            mu = np.exp(a + beta * x) * self.E[k]
            wm = w * mu
            s = wm.sum()
            if s <= 0:
                continue
            ck = (wm * x).sum() / s
            info += (wm * (x - ck) ** 2).sum()
            eff += np.bincount(thread_idx, weights=w * (x - ck) * (self.D[k] - mu), minlength=G)
        return float(np.sqrt((eff ** 2).sum()) / info) if info > 0 else float("nan")


def fit_multi(prep: "Prepared", X: np.ndarray, w: np.ndarray, thread_idx: np.ndarray, iters: int = 60) -> dict:
    """First-reply PEM with several static covariates: log h_i(t) = alpha_k + x_i' beta.
    alpha_k profiled out; Newton on beta; thread-clustered sandwich covariance from per-thread
    efficient scores sum_i sum_k w_i (x_i - c_k)(d_ik - mu_ik), c_k = weighted mean of x among expected
    events in bin k."""
    N, p = X.shape
    beta = np.zeros(p)
    E = prep.E.astype(np.float64)
    D = prep.D
    G = int(thread_idx.max()) + 1
    for it in range(iters):
        eta = X @ beta
        ex = np.exp(eta)
        score = np.zeros(p)
        H = np.zeros((p, p))
        for k in range(prep.K):
            wm0 = w * ex * E[k]
            s0 = wm0.sum()
            dk = (w * D[k]).sum()
            if s0 <= 0 or dk <= 0:
                continue
            a = np.log(dk / s0)
            mu = np.exp(a) * wm0
            m1 = mu @ X
            score += X.T @ (w * D[k]) - m1
            H += (X * mu[:, None]).T @ X - np.outer(m1, m1) / mu.sum()
        step = np.linalg.solve(H, score)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = X @ beta
    ex = np.exp(eta)
    eff = np.zeros((G, p))
    H = np.zeros((p, p))
    alpha = np.full(prep.K, np.nan)
    for k in range(prep.K):
        wm0 = w * ex * E[k]
        s0 = wm0.sum()
        dk = (w * D[k]).sum()
        if s0 <= 0 or dk <= 0:
            continue
        alpha[k] = np.log(dk / s0)
        mu_unw = np.exp(alpha[k]) * ex * E[k]
        mu = w * mu_unw
        ck = (mu @ X) / mu.sum()
        H += (X * mu[:, None]).T @ X - np.outer(mu @ X, mu @ X) / mu.sum()
        r = w * (D[k] - mu_unw)
        for j in range(p):
            eff[:, j] += np.bincount(thread_idx, weights=r * (X[:, j] - ck[j]), minlength=G)
    Hinv = np.linalg.inv(H)
    V = Hinv @ (eff.T @ eff) @ Hinv
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "alpha": alpha, "n": N, "threads": G,
            "events": float((w * D.sum(0)).sum())}
