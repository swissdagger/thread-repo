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


def fit_fe_poisson(g: np.ndarray, y: np.ndarray, X: np.ndarray, offset: np.ndarray, iters: int = 50,
                   cluster: np.ndarray | None = None) -> dict:
    """g: fixed-effect groups; cluster: clustering units for the sandwich (default g), e.g. threads
    when g is thread x time-bin cells."""
    cluster = g if cluster is None else cluster
    # keep groups with >= 1 event and >= 2 candidates
    G0 = g.max() + 1
    N = np.bincount(g, weights=y, minlength=G0)
    n = np.bincount(g, minlength=G0)
    keep = (N[g] > 0) & (n[g] >= 2)
    g, y, X, o, cluster = g[keep], y[keep], X[keep], offset[keep], cluster[keep]
    _, cluster = np.unique(cluster, return_inverse=True)
    _, g = np.unique(g, return_inverse=True)
    G = g.max() + 1
    N = np.bincount(g, weights=y, minlength=G)
    # drop columns with no within-group variation (absorbed by the fixed effects) or linearly dependent
    # on earlier columns after demeaning; their coefficients are reported as NaN
    p_all = X.shape[1]
    n_g = np.bincount(g, minlength=G).astype(float)
    Xw = X - _group_sum(g, X, G)[g] / n_g[g][:, None]
    ss = (Xw ** 2).sum(0)
    cols = []
    for j in np.argsort(-ss):
        if ss[j] <= 1e-10 * max(len(y), 1):
            continue
        cand = sorted(cols + [j])
        if np.linalg.matrix_rank(Xw[:, cand].T @ Xw[:, cand], tol=1e-8 * len(y)) == len(cand):
            cols = cand
    cols = sorted(cols)
    X = X[:, cols]
    b = np.zeros(X.shape[1])

    def probs(b):
        eta = X @ b + o
        mx = np.full(G, -np.inf)
        np.maximum.at(mx, g, eta)
        e = np.exp(eta - mx[g])
        S = np.bincount(g, weights=e, minlength=G)
        logp = eta - mx[g] - np.log(S[g])
        return np.exp(logp), float(np.sum(y * logp))

    p, ll = probs(b)
    for _ in range(iters):
        r = y - N[g] * p
        grad = X.T @ r
        xbar = _group_sum(g, p[:, None] * X, G)
        H = X.T @ (X * (N[g] * p)[:, None]) - (xbar * N[:, None]).T @ xbar
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # step-halving on the conditional log-likelihood (full Newton steps can overshoot)
        lam = 1.0
        while True:
            p_new, ll_new = probs(b + lam * step)
            if ll_new >= ll - 1e-10 * abs(ll) or lam < 1e-8:
                break
            lam /= 2
        b, p, ll = b + lam * step, p_new, ll_new
        if np.max(np.abs(lam * step)) < 1e-9:
            break
    eta = X @ b + o
    mx = np.full(G, -np.inf)
    np.maximum.at(mx, g, eta)
    e = np.exp(eta - mx[g])
    S = np.bincount(g, weights=e, minlength=G)
    p = e / S[g]
    sc = _group_sum(cluster, X * (y - N[g] * p)[:, None], int(cluster.max()) + 1)
    Hinv = np.linalg.inv(X.T @ (X * (N[g] * p)[:, None]) - (_group_sum(g, p[:, None] * X, G) * N[:, None]).T
                         @ _group_sum(g, p[:, None] * X, G))
    V = Hinv @ (sc.T @ sc) @ Hinv
    coef, se = np.full(p_all, np.nan), np.full(p_all, np.nan)
    coef[cols], se[cols] = b, np.sqrt(np.diag(V))
    return {"coef": coef, "se": se, "dropped": [j for j in range(p_all) if j not in cols],
            "n_parents": int(len(y)), "n_threads": int(G),
            "n_clusters": int(cluster.max()) + 1, "n_events": float(y.sum())}


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
