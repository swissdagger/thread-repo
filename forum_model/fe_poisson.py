"""Poisson regression with group (thread) fixed effects via the conditional likelihood.

Given thread totals N_g, counts within a thread are multinomial with probabilities
p_i = exp(x_i b + o_i) / sum_{j in g} exp(x_j b + o_j): the thread intercepts (thread exposure,
feed rank, popularity) cancel. This is the conditional-logit form of parent choice.
Standard errors are cluster-robust by thread (sandwich on per-thread scores).
"""
from __future__ import annotations

import numpy as np


def _group_sum(g: np.ndarray, v: np.ndarray, G: int) -> np.ndarray:
    if v.ndim == 1:
        return np.bincount(g, weights=v, minlength=G)
    return np.stack([np.bincount(g, weights=v[:, k], minlength=G) for k in range(v.shape[1])], axis=1)


def fit_fe_poisson(g: np.ndarray, y: np.ndarray, X: np.ndarray, offset: np.ndarray, iters: int = 50) -> dict:
    # keep threads with >= 1 event and >= 2 candidate parents
    G0 = g.max() + 1
    N = np.bincount(g, weights=y, minlength=G0)
    n = np.bincount(g, minlength=G0)
    keep = (N[g] > 0) & (n[g] >= 2)
    g, y, X, o = g[keep], y[keep], X[keep], offset[keep]
    _, g = np.unique(g, return_inverse=True)
    G = g.max() + 1
    N = np.bincount(g, weights=y, minlength=G)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + o
        mx = np.full(G, -np.inf)
        np.maximum.at(mx, g, eta)
        e = np.exp(eta - mx[g])
        S = np.bincount(g, weights=e, minlength=G)
        p = e / S[g]
        r = y - N[g] * p
        grad = X.T @ r
        xbar = _group_sum(g, p[:, None] * X, G)
        H = X.T @ (X * (N[g] * p)[:, None]) - (xbar * N[:, None]).T @ xbar
        step = np.linalg.solve(H, grad)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = X @ b + o
    mx = np.full(G, -np.inf)
    np.maximum.at(mx, g, eta)
    e = np.exp(eta - mx[g])
    S = np.bincount(g, weights=e, minlength=G)
    p = e / S[g]
    sc = _group_sum(g, X * (y - N[g] * p)[:, None], G)
    Hinv = np.linalg.inv(X.T @ (X * (N[g] * p)[:, None]) - (_group_sum(g, p[:, None] * X, G) * N[:, None]).T
                         @ _group_sum(g, p[:, None] * X, G))
    V = Hinv @ (sc.T @ sc) @ Hinv
    return {"coef": b, "se": np.sqrt(np.diag(V)), "n_parents": int(len(y)), "n_threads": int(G),
            "n_events": float(y.sum())}


def fit_pooled_poisson(g: np.ndarray, y: np.ndarray, X: np.ndarray, offset: np.ndarray, iters: int = 50) -> dict:
    """Poisson GLM with a common intercept (no thread effects); thread-clustered SEs."""
    Xi = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(Xi.shape[1])
    b[0] = np.log(max(y.mean(), 1e-9)) - np.mean(offset)
    for _ in range(iters):
        mu = np.exp(Xi @ b + offset)
        grad = Xi.T @ (y - mu)
        H = Xi.T @ (Xi * mu[:, None])
        step = np.linalg.solve(H, grad)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = np.exp(Xi @ b + offset)
    _, gi = np.unique(g, return_inverse=True)
    sc = _group_sum(gi, Xi * (y - mu)[:, None], gi.max() + 1)
    Hinv = np.linalg.inv(Xi.T @ (Xi * mu[:, None]))
    V = Hinv @ (sc.T @ sc) @ Hinv
    return {"coef": b[1:], "se": np.sqrt(np.diag(V))[1:], "n_parents": int(len(y)), "n_events": float(y.sum())}
