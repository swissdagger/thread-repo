"""Estimator for the two-clock intensity model (docs/fit1_brief.md).

Stage A: comment and root streams, cells x thread fixed effects (conditional Poisson, FEPoissonBoot).
  * comment cells: each comment x comment-age bins (0, then BINS log-spaced edges 10 s .. 3 d), further split
    at the comment's own reply events (so "direct replies so far" is exact within a cell) and at thread age
    30 min (the early-position indicator); y = comment-stream events ending the cell, offset log exposure;
  * root cells: each root x thread-age bins, split at its own top-level events (n_top exact);
  * covariates: clock terms at the cell's geometric-midpoint age; hour of day at the cell's arithmetic
    midpoint (absolute UTC time), shared by both streams within a population;
  * groups: comment cells by thread (theta_T), root cells by thread (b theta_T, a separate fixed effect).
Stage B: fitness. Given Stage A's coefficients, thread totals N_c, N_R and expected counts E_c, E_R
  (without intercept) give alpha_c, alpha_R, r, b by maximum likelihood with z ~ Gamma(r, 1/r), integrated
  per thread on an adaptive log-z grid.
Stage C: reply-back stream, rho g_back(u) exp(beta_back' x) by direct maximum likelihood.
Bootstrap: estimating-function (one-step) thread bootstrap. Thread weights w ~ Multinomial; for each
  stage theta* = theta_hat + I^{-1} sum_T (w_T - 1) s_T (Hu & Kalbfleisch 2000), with Stage B evaluated at
  the replicate's Stage A coefficients (first-stage uncertainty propagated).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from .fe_poisson import FEPoissonBoot, _group_sum

H_DEFAULT = 3 * 86400.0
N_BINS = 40
EDGES = np.r_[0.0, np.logspace(1, np.log10(H_DEFAULT), N_BINS)]   # 40 bins: [0,10 s] + 39 log bins
HOURS = [f"hour_{h}" for h in range(1, 24)]
C_NAMES = ["k1_log_u", "k2_log_u_sq", "a1_log_tau", "a2_log_tau_sq", "dep2", "dep3p", "op", "logn", "pos_early"]


def _cells(unit_idx, start_abs, s0, U, ev_unit, ev_t, ev_counts, ev_any, extra=None):
    """Generic cell builder. Breakpoints: EDGES clipped to U, event times, optional extra split.
    Returns per-cell arrays: unit, a, b, y (counted events ending at b), n (any-events with time <= a)."""
    nU = len(U)
    e = np.minimum(EDGES[None, :], U[:, None])
    bu = [np.repeat(np.arange(nU), e.shape[1]), np.repeat(np.arange(nU), 1)]
    bt = [e.ravel(), U]
    by = [np.zeros(e.size), np.zeros(nU)]
    ba = [np.zeros(e.size), np.zeros(nU)]
    if extra is not None:
        m = (extra > 0) & (extra < U)
        bu.append(np.arange(nU)[m]); bt.append(extra[m]); by.append(np.zeros(m.sum())); ba.append(np.zeros(m.sum()))
    tt = np.clip(ev_t, 0.5, None)
    keep = tt <= U[ev_unit]
    bu.append(ev_unit[keep]); bt.append(tt[keep]); by.append(ev_counts[keep]); ba.append(ev_any[keep])
    df = pl.DataFrame({"unit": np.concatenate(bu), "t": np.concatenate(bt), "y": np.concatenate(by),
                       "anyev": np.concatenate(ba)})
    df = df.group_by("unit", "t").agg(pl.col("y").sum(), pl.col("anyev").sum()).sort("unit", "t")
    df = df.with_columns(a=pl.col("t").shift(1).over("unit"), n=pl.col("anyev").cum_sum().shift(1).over("unit"))
    df = df.filter(pl.col("a").is_not_null() & (pl.col("t") > pl.col("a")))
    return {"unit": df["unit"].to_numpy(), "a": df["a"].to_numpy(), "b": df["t"].to_numpy(),
            "y": df["y"].to_numpy().astype(float), "n": df["n"].fill_null(0).to_numpy().astype(float)}


def _gm(a, b):
    return np.where(a > 0, np.sqrt(np.maximum(a, 1e-9) * b), b / 2)


def _lx(age):
    return np.log(np.maximum(age, 1e-3) / 60.0)


def hour_dummies(abs_t):
    h = (np.mod(abs_t, 86400.0) // 3600).astype(np.int64)
    return np.column_stack([(h == k).astype(float) for k in range(1, 24)])


EV_OFFSET = np.log(1e-9)  # event rows: the event's log-intensity enters exactly; their exposure is negligible


def _event_order(unit, t):
    """0-based index of each event among its unit's events in time order (= events strictly before it)."""
    o = np.lexsort((t, unit))
    rank = np.empty(len(unit), np.int64)
    us = unit[o]
    start = np.r_[True, us[1:] != us[:-1]] if len(us) else np.zeros(0, bool)
    gs = np.maximum.accumulate(np.where(start, np.arange(len(us)), 0)) if len(us) else np.zeros(0, int)
    rank[o] = np.arange(len(us)) - gs
    return rank


def _comment_X(u_age, s, depth, op, n, pos, abs_t, tau_for_pos):
    return (np.column_stack([_lx(u_age), _lx(u_age) ** 2, _lx(s + u_age), _lx(s + u_age) ** 2, (depth == 2).astype(float),
                             (depth >= 3).astype(float), op.astype(float), np.log1p(n),
                             np.log(np.maximum(pos, 1)) * (tau_for_pos <= 1800.0)]),
            hour_dummies(abs_t))


def build(S: dict, post_types: list[str] | None = None) -> dict:
    """Design for Stage A. Exposure cells carry y = 0 (covariates: clocks at the geometric-midpoint age, hour
    and the early-position indicator at the arithmetic midpoint); every event is its own row with y = 1,
    covariates at the exact event time (n = events strictly before it) and a negligible exposure offset.
    This is the point-process likelihood with the integral of the intensity taken by the midpoint rule."""
    roots = S["roots"]
    tids = roots["thread_id"].to_list()
    tmap = {t: i for i, t in enumerate(tids)}
    units = S["units"]
    uid = {k: i for i, k in enumerate(units["id"].to_list())}
    ec = S["events_c"]
    ev_unit = np.array([uid[x] for x in ec["unit"].to_list()], dtype=np.int64)
    ev_u = np.clip(ec["u"].to_numpy(), 0.5, None)
    is_c = (ec["stream"] == "comment").to_numpy()
    s = units["s"].to_numpy()
    U = units["U"].to_numpy()
    T0u = units["T0"].to_numpy()
    depth_u = units["depth"].to_numpy()
    op_u = units["op_flag"].to_numpy()
    pos_u = units["pos"].to_numpy()
    cc = _cells(np.arange(units.height), T0u + s, s, U, ev_unit, ev_u, np.zeros(len(ev_unit)),
                np.ones(len(ev_unit)), extra=1800.0 - s)
    i = cc["unit"]
    gm = _gm(cc["a"], cc["b"])
    mid = (cc["a"] + cc["b"]) / 2
    Xe, He = _comment_X(gm, s[i], depth_u[i], op_u[i], cc["n"], pos_u[i], T0u[i] + s[i] + mid, s[i] + mid)
    # event rows (comment-stream events only; reply-backs only raise n)
    order = _event_order(ev_unit, ev_u)
    k = np.nonzero(is_c & (ev_u <= U[ev_unit]))[0]
    ku = ev_unit[k]
    Xv, Hv = _comment_X(ev_u[k], s[ku], depth_u[ku], op_u[ku], order[k].astype(float), pos_u[ku],
                        T0u[ku] + s[ku] + ev_u[k], s[ku] + ev_u[k])
    thr_u = np.array([tmap[t] for t in units["thread_id"].to_list()], dtype=np.int64)
    Xc = np.vstack([Xe, Xv])
    Hc = np.vstack([He, Hv])
    yc = np.r_[np.zeros(len(i)), np.ones(len(k))]
    oc = np.r_[np.log(cc["b"] - cc["a"]), np.full(len(k), EV_OFFSET)]
    thr_c = np.r_[thr_u[i], thr_u[ku]]
    R = build_root(S, post_types, tmap)
    return {"Xc": Xc, "Hc": Hc, "yc": yc, "oc": oc, "thr_c": thr_c, **R, "tids": tids, "rows_c": len(yc)}


def build_root(S: dict, post_types: list[str] | None = None, tmap: dict | None = None) -> dict:
    """Root-stream exposure cells (y = 0) and event rows (y = 1, exact time). With several post types: a root
    clock per type, gamma shared, and n_top x 1[Ask HN] (coefficient = -kappa)."""
    roots = S["roots"]
    tids = roots["thread_id"].to_list()
    tmap = tmap or {t: i for i, t in enumerate(tids)}
    er = S["events_r"]
    rt_unit = np.array([tmap[t] for t in er["thread_id"].to_list()], dtype=np.int64)
    rt_t = np.clip(er["tau"].to_numpy(), 0.5, None)
    UR = roots["U"].to_numpy()
    T0 = roots["T0"].to_numpy()
    ptype_all = roots["post_type"].to_numpy()
    rc = _cells(np.arange(len(tids)), T0, np.zeros(len(tids)), UR, rt_unit, rt_t,
                np.zeros(len(rt_unit)), np.ones(len(rt_unit)))
    keep = rt_t <= UR[rt_unit]
    rt_unit, rt_t = rt_unit[keep], rt_t[keep]
    nev = _event_order(rt_unit, rt_t).astype(float)
    j = np.r_[rc["unit"], rt_unit]
    age = np.r_[_gm(rc["a"], rc["b"]), rt_t]
    n = np.r_[rc["n"], nev]
    absmid = np.r_[T0[rc["unit"]] + (rc["a"] + rc["b"]) / 2, T0[rt_unit] + rt_t]
    ptype = ptype_all[j]
    types = post_types or sorted(set(roots["post_type"].to_list()))
    Xr_cols, Xr_names = [], []
    for ty in types:
        ind = (ptype == ty).astype(float) if len(types) > 1 else np.ones(len(j))
        Xr_cols += [ind * _lx(age), ind * _lx(age) ** 2]
        Xr_names += [f"aR1_{ty}", f"aR2_{ty}"] if len(types) > 1 else ["aR1", "aR2"]
    Xr_cols.append(n)
    Xr_names.append("gamma")
    if len(types) > 1:
        Xr_cols.append(n * (ptype == "ask_hn"))
        Xr_names.append("minus_kappa")  # coefficient = -kappa
    multi = len(types) > 1
    Xr = np.column_stack(Xr_cols)
    Hr = hour_dummies(absmid)
    yr = np.r_[np.zeros(len(rc["unit"])), np.ones(len(rt_unit))]
    orr = np.r_[np.log(rc["b"] - rc["a"]), np.full(len(rt_unit), EV_OFFSET)]
    out = {"Xr": Xr, "Hr": Hr, "yr": yr, "or": orr, "thr_r": j,
           "r_names": Xr_names, "nT": len(tids), "rows_r": len(yr)}
    if multi:  # per-type root intercepts (used by the random-effects joint fit instead of a single alpha_R)
        out["R_type"] = np.column_stack([(ptype == ty).astype(float) for ty in types])
        out["R_type_names"] = [f"alpha_R_{ty}" for ty in types]
    return out


def root_only_design(S: dict, post_types: list[str]) -> dict:
    """Root-stream-only design for the random-effects joint fit (no comment stream)."""
    R = build_root(S, post_types)
    pc, ph = len(C_NAMES), 23
    return {"Xc": np.zeros((0, 0)), "Hc": np.zeros((0, ph)), "yc": np.zeros(0), "oc": np.zeros(0),
            "thr_c": np.zeros(0, np.int64), **R}


def stage_a(D: dict, model: str = "full") -> dict:
    """Joint conditional Poisson. model: 'full' (comment + root streams, shared hours), 'B1' (comment stream:
    comment kernel + thread fitness only), 'root_only' (root stream with hours; used for the joint HN fit)."""
    nT = D["nT"]
    pc, pr = (D["Xc"].shape[1] if "Xc" in D else 0), D["Xr"].shape[1]
    if model == "full":
        Xc = np.column_stack([D["Xc"], np.zeros((len(D["yc"]), pr)), D["Hc"]])
        Xr = np.column_stack([np.zeros((len(D["yr"]), pc)), D["Xr"], D["Hr"]])
        X = np.vstack([Xc, Xr])
        y = np.r_[D["yc"], D["yr"]]
        o = np.r_[D["oc"], D["or"]]
        g = np.r_[D["thr_c"], nT + D["thr_r"]]
        cl = np.r_[D["thr_c"], D["thr_r"]]
        names = C_NAMES + D["r_names"] + HOURS
    elif model == "B1":
        X, y, o, g, cl = D["Xc"][:, :2], D["yc"], D["oc"], D["thr_c"], D["thr_c"]
        names = C_NAMES[:2]
    elif model == "root_only":
        X = np.column_stack([D["Xr"], D["Hr"]])
        y, o, g, cl = D["yr"], D["or"], D["thr_r"], D["thr_r"]
        names = D["r_names"] + HOURS
    else:
        raise ValueError(model)
    prep = FEPoissonBoot(g, y, X, o, cl)
    coef, se = prep.fit(se=True)
    return {"prep": prep, "coef": coef, "se": se, "names": names, "model": model}


def fe_scores(A: dict, C: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster score contributions (C x p_kept) and information matrix at the estimate."""
    prep = A["prep"]
    b = np.nan_to_num(A["coef"][prep.cols])
    w = np.ones(len(prep.y))
    p, _ = prep._probs(b, w)
    X, g, G, N = prep.X, prep.g, prep.G, prep.N
    r = prep.y - N[g] * p
    C = int(prep.cl.max()) + 1 if C is None else C
    sc = _group_sum(prep.cl, X * r[:, None], C)
    xbar = _group_sum(g, p[:, None] * X, G)
    I = X.T @ (X * (N[g] * p)[:, None]) - (xbar * N[:, None]).T @ xbar
    return sc, I


def expected_counts(D: dict, A: dict, coef: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-thread expected counts without intercept and fitness: E_c (comment stream), E_R (root stream)."""
    c = A["coef"] if coef is None else coef
    c = np.nan_to_num(c)
    pc, pr = D["Xc"].shape[1], D["Xr"].shape[1]
    bc, br, bh = c[:pc], c[pc:pc + pr], c[pc + pr:]
    ec = np.exp(D["Xc"] @ bc + D["Hc"] @ bh + D["oc"])
    er = np.exp(D["Xr"] @ br + D["Hr"] @ bh + D["or"])
    return (np.bincount(D["thr_c"], weights=ec, minlength=D["nT"]),
            np.bincount(D["thr_r"], weights=er, minlength=D["nT"]))


# ---------------------------------------------------------------- Stage B: fitness
GRID = np.linspace(-7.0, 7.0, 57)


def thread_loglik(th, Nc, Ec, NR, ER):
    """Per-thread log-likelihood of totals (up to terms free of the parameters), z ~ Gamma(r, 1/r)."""
    ac, aR, lr, b = th
    r = np.exp(lr)
    A_ = np.exp(ac) * Ec
    B_ = np.exp(aR) * ER
    # mode of the log-z integrand by Newton (vectorised), then a grid of +-7 curvature sd around it
    c0 = r + Nc + b * NR
    m0 = np.log((r + Nc + NR) / (r + A_ + B_))
    for _ in range(8):
        e1, eb = np.exp(m0), np.exp(b * m0)
        g1 = c0 - (r + A_) * e1 - b * B_ * eb
        h1 = -(r + A_) * e1 - b * b * B_ * eb
        m0 = m0 - np.clip(g1 / h1, -3, 3)
    sd = 1.0 / np.sqrt((r + A_) * np.exp(m0) + b * b * B_ * np.exp(b * m0))
    lz = m0[:, None] + sd[:, None] * GRID[None, :]
    z = np.exp(lz)
    f = (r * np.log(r) - gammaln(r) + r * lz - r * z                       # Gamma density in log z (incl. Jacobian z)
         + Nc[:, None] * lz - A_[:, None] * z + b * NR[:, None] * lz - B_[:, None] * np.exp(b * lz))
    m = f.max(axis=1)
    dl = sd * (GRID[1] - GRID[0])
    return m + np.log(np.sum(np.exp(f - m[:, None]), axis=1) * dl) + Nc * ac + NR * aR


def stage_b(Nc, Ec, NR, ER, th0=None, w=None):
    w = np.ones(len(Nc)) if w is None else w
    th0 = np.array([np.log(max(Nc.sum(), 1) / max(Ec.sum(), 1e-12)), np.log(max(NR.sum(), 1) / max(ER.sum(), 1e-12)),
                    0.0, 1.0]) if th0 is None else th0
    f = lambda th: -float(np.sum(w * thread_loglik(th, Nc, Ec, NR, ER)))
    r = minimize(f, th0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    r = minimize(f, r.x, method="BFGS")
    return r.x


def num_scores(fun, th, eps=1e-4):
    """Per-unit gradient of a vector-valued log-likelihood fun(th) -> (n,) by central differences."""
    out = []
    for k in range(len(th)):
        e = np.zeros(len(th)); e[k] = eps
        out.append((fun(th + e) - fun(th - e)) / (2 * eps))
    return np.column_stack(out)


def num_info(fun_total, th, eps=1e-3):
    k = len(th)
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = eps; ej[j] = eps
            v = (fun_total(th + ei + ej) - fun_total(th + ei - ej) - fun_total(th - ei + ej) + fun_total(th - ei - ej)) / (4 * eps * eps)
            Hm[i, j] = Hm[j, i] = v
    return -Hm


# ---------------------------------------------------------------- Stage C: reply-back stream
def rb_data(S: dict) -> dict:
    units = S["units"].filter(pl.col("parent_author").is_not_null())
    roots = S["roots"]
    tmap = {t: i for i, t in enumerate(roots["thread_id"].to_list())}
    uid = {k: i for i, k in enumerate(units["id"].to_list())}
    ev = S["events_c"].filter(pl.col("stream") == "reply_back").filter(pl.col("unit").is_in(units["id"].implode()))
    depth = units["depth"].to_numpy()
    return {"U": units["U"].to_numpy(), "x": np.column_stack([(depth == 1), (depth >= 3)]).astype(float),
            "thr": np.array([tmap[t] for t in units["thread_id"].to_list()], dtype=np.int64),
            "ev_unit": np.array([uid[x] for x in ev["unit"].to_list()], dtype=np.int64),
            "ev_u": np.clip(ev["u"].to_numpy(), 0.5, None), "nT": len(tmap)}


def rb_unit_loglik(th, R):
    """Per-unit log-likelihood: theta = (log rho, m = log median minutes, log sigma, beta_root, beta_dep3p)."""
    lrho, m, ls, b1, b2 = th
    s = np.exp(ls)
    mu = np.log(np.exp(m) * 60.0)
    lin = lrho + R["x"] @ np.array([b1, b2])
    comp = -np.exp(lin) * norm.cdf((np.log(R["U"]) - mu) / s)
    lu = np.log(R["ev_u"])
    le = lin[R["ev_unit"]] - lu - np.log(s) - 0.5 * np.log(2 * np.pi) - (lu - mu) ** 2 / (2 * s * s)
    return comp + np.bincount(R["ev_unit"], weights=le, minlength=len(R["U"]))


def stage_c(R, th0=None):
    th0 = np.array([np.log(max(len(R["ev_u"]), 1) / len(R["U"])), np.log(60.0), np.log(1.5), 0.0, 0.0]) if th0 is None else th0
    f = lambda th: -float(rb_unit_loglik(th, R).sum())
    r = minimize(f, th0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 6000})
    r = minimize(f, r.x, method="BFGS")
    return r.x


# ---------------------------------------------------------------- full fit + one-step bootstrap
def fit_all(S: dict, B: int, rng: np.random.Generator, arms=("RE", "FE"), with_b1: bool = True) -> dict:
    """Arm 'FE' (the brief's estimator): thread fixed effects for the comment and root streams, then fitness
    (alpha_c, alpha_R, r, b) by maximum likelihood on thread totals. Arm 'RE': the same model with the gamma
    fitness integrated out (marginal likelihood, Newton). Reply-back stream (no fitness) and B1 (comment kernel +
    thread fixed effects) are shared. Intervals: one-step thread bootstrap with shared multinomial weights."""
    D = build(S)
    out = {"design": {"rows_c": D["rows_c"], "rows_r": D["rows_r"], "threads": D["nT"]}}
    W = rng.multinomial(D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)  # B x T
    if "FE" in arms:
        A = stage_a(D, "full")
        reps = onestep_reps(A, D["nT"], W)
        res = {"names": A["names"], "coef": A["coef"], "se_sandwich": A["se"], "reps": reps}
        Nc = np.bincount(D["thr_c"], weights=D["yc"], minlength=D["nT"])
        NR = np.bincount(D["thr_r"], weights=D["yr"], minlength=D["nT"])
        Ec, ER = expected_counts(D, A)
        th = stage_b(Nc, Ec, NR, ER)
        I2inv = np.linalg.inv(num_info(lambda t: float(np.sum(thread_loglik(t, Nc, Ec, NR, ER))), th))
        reps2 = np.empty((B, 4))
        for bi in range(B):
            Ec_b, ER_b = expected_counts(D, A, reps[bi])
            s2 = num_scores(lambda t: thread_loglik(t, Nc, Ec_b, NR, ER_b), th)
            reps2[bi] = th + I2inv @ (W[bi] @ s2)
        res["fitness"] = {"names": ["alpha_c", "alpha_R", "log_r", "b"], "coef": th, "reps": reps2}
        out["FE"] = res
    if "RE" in arms:
        M = REModel(D)
        st = np.zeros(M.p + 2)
        ec_rows, er_rows = D["yc"] == 0, D["yr"] == 0
        st[M.p - 2] = np.log(max(D["yc"].sum(), 1) / np.exp(D["oc"][ec_rows]).sum())
        st[M.p - 1] = np.log(max(D["yr"].sum(), 1) / np.exp(D["or"][er_rows]).sum())
        st[M.p], st[M.p + 1] = np.log(5.0), 1.0
        th = M.fit(st, iters=60)
        I = M.info(th)
        sc = M.scores(th)
        Iinv = np.linalg.inv(I)
        reps = th[None, :] + ((W - 1.0) @ sc) @ Iinv.T
        se = np.sqrt(np.clip(np.diag(Iinv @ (sc.T @ sc) @ Iinv), 0, None))
        out["RE"] = {"names": M.names + ["log_r", "b"], "coef": th, "reps": reps, "se_sandwich": se,
                     "n_iter": M.n_iter}
    R = rb_data(S)
    thc = stage_c(R)
    sc3 = np.zeros((D["nT"], 5))
    ss = num_scores(lambda t: rb_unit_loglik(t, R), thc)
    for k in range(5):
        sc3[:, k] = np.bincount(R["thr"], weights=ss[:, k], minlength=D["nT"])
    I3 = num_info(lambda t: float(rb_unit_loglik(t, R).sum()), thc)
    reps3 = thc[None, :] + ((W - 1.0) @ sc3) @ np.linalg.inv(I3).T
    out["reply_back"] = {"names": ["log_rho", "back_m", "log_back_sigma", "beta_back_root", "beta_back_dep3p"],
                         "coef": thc, "reps": reps3}
    if with_b1:
        A1 = stage_a(D, "B1")
        out["B1"] = {"names": A1["names"], "coef": A1["coef"], "se_sandwich": A1["se"],
                     "reps": onestep_reps(A1, D["nT"], W)}
    return out


# ---------------------------------------------------------------- random-effects (marginal) estimator
def frailty_terms(Nc, NR, A, B, r, b, moments2: bool = False):
    """Per thread: log of int Gamma(z; r, 1/r) z^(Nc + b NR) exp(-z A - z^b B) dz on a log-z grid around the mode,
    posterior E[z], E[z^b] and (moments2) Var(z), Var(z^b), Cov(z, z^b)."""
    c0 = r + Nc + b * NR
    m0 = np.log((r + Nc + NR) / (r + A + B))
    for _ in range(12):
        e1, eb = np.exp(np.clip(m0, -40, 40)), np.exp(np.clip(b * m0, -40, 40))
        g1 = c0 - (r + A) * e1 - b * B * eb
        h1 = -(r + A) * e1 - b * b * B * eb - 1e-12
        m0 = np.clip(m0 - np.clip(g1 / h1, -3, 3), -35, 35)
    sd = 1.0 / np.sqrt((r + A) * np.exp(m0) + b * b * B * np.exp(np.clip(b * m0, -40, 40)) + 1e-12)
    sd = np.minimum(sd, 5.0)
    lz = np.clip(m0[:, None] + sd[:, None] * GRID[None, :], -40, 40)
    z = np.exp(lz)
    zb = np.exp(np.clip(b * lz, -60, 60))
    f = (r * np.log(r) - gammaln(r) + r * lz - r * z + Nc[:, None] * lz - A[:, None] * z
         + b * NR[:, None] * lz - B[:, None] * zb)
    mx = f.max(axis=1)
    wts = np.exp(f - mx[:, None])
    sw = wts.sum(axis=1)
    logI = mx + np.log(sw * sd * (GRID[1] - GRID[0]))
    Ez = (wts * z).sum(axis=1) / sw
    Ezb = (wts * zb).sum(axis=1) / sw
    if not moments2:
        return logI, Ez, Ezb
    Vz = (wts * z * z).sum(axis=1) / sw - Ez ** 2
    Vzb = (wts * zb * zb).sum(axis=1) / sw - Ezb ** 2
    Czb = (wts * z * zb).sum(axis=1) / sw - Ez * Ezb
    return logI, Ez, Ezb, Vz, Vzb, Czb


class REModel:
    """Two-stream model with gamma thread fitness integrated out (marginal likelihood).
    theta = [linear parameters (comment covariates, root covariates, shared hours, alpha_c, alpha_R), log r, b].
    Newton-Raphson with the analytic Hessian for the linear block and finite differences for (log r, b)."""

    def __init__(self, D: dict):
        nc, nr = len(D["yc"]), len(D["yr"])
        pc = D["Xc"].shape[1] if nc else 0
        pr, ph = D["Xr"].shape[1], D["Hr"].shape[1]
        rint = D.get("R_type", np.ones((nr, 1)))
        rint_names = D.get("R_type_names", ["alpha_R"])
        ci = 1 if nc else 0
        self.names = (C_NAMES[:pc] if nc else []) + D["r_names"] + HOURS + (["alpha_c"] if nc else []) + rint_names
        X = np.zeros((nc + nr, pc + pr + ph + ci + rint.shape[1]))
        if nc:
            X[:nc, :pc] = D["Xc"]
            X[:nc, pc + pr:pc + pr + ph] = D["Hc"]
            X[:nc, pc + pr + ph] = 1.0
        X[nc:, pc:pc + pr] = D["Xr"]
        X[nc:, pc + pr:pc + pr + ph] = D["Hr"]
        X[nc:, pc + pr + ph + ci:] = rint
        self.n_alpha = ci + rint.shape[1]
        self.X, self.o = X, np.r_[D["oc"], D["or"]]
        self.y = np.r_[D["yc"], D["yr"]]
        self.thr = np.r_[D["thr_c"], D["thr_r"]]
        self.is_c = np.r_[np.ones(nc, bool), np.zeros(nr, bool)]
        self.nT = D["nT"]
        self.Nc = np.bincount(D["thr_c"], weights=D["yc"], minlength=self.nT)
        self.NR = np.bincount(D["thr_r"], weights=D["yr"], minlength=self.nT)
        self.p = X.shape[1]
        self.nc = nc

    def _parts(self, th, moments2=False):
        beta, lr, b = th[:self.p], th[self.p], th[self.p + 1]
        eta = self.X @ beta
        mu = np.exp(np.clip(eta + self.o, -700, 300))
        A = np.bincount(self.thr[:self.nc], weights=mu[:self.nc], minlength=self.nT)
        B = np.bincount(self.thr[self.nc:], weights=mu[self.nc:], minlength=self.nT)
        ft = frailty_terms(self.Nc, self.NR, A, B, np.exp(lr), b, moments2)
        ev = np.bincount(self.thr, weights=self.y * eta, minlength=self.nT)
        return eta, mu, A, B, ft, ev

    def loglik_threads(self, th):
        _, _, _, _, ft, ev = self._parts(th)
        return ft[0] + ev

    def loglik(self, th, w=None):
        lt = self.loglik_threads(th)
        return float(lt.sum() if w is None else w @ lt)

    def scores(self, th):
        """Per-thread score contributions (nT x (p + 2))."""
        eta, mu, A, B, (logI, Ez, Ezb), ev = self._parts(th)
        m = np.where(self.is_c, Ez[self.thr], Ezb[self.thr])
        gl = _group_sum(self.thr, self.X * (self.y - m * mu)[:, None], self.nT)
        lr, b = th[self.p], th[self.p + 1]
        e = 1e-5
        g_lr = (frailty_terms(self.Nc, self.NR, A, B, np.exp(lr + e), b)[0]
                - frailty_terms(self.Nc, self.NR, A, B, np.exp(lr - e), b)[0]) / (2 * e)
        g_b = (frailty_terms(self.Nc, self.NR, A, B, np.exp(lr), b + e)[0]
               - frailty_terms(self.Nc, self.NR, A, B, np.exp(lr), b - e)[0]) / (2 * e)
        return np.column_stack([gl, g_lr, g_b])

    def grad_hess(self, th, w=None):
        p = self.p
        w = np.ones(self.nT) if w is None else w
        wr = w[self.thr]
        eta, mu, A, B, (logI, Ez, Ezb, Vz, Vzb, Czb), ev = self._parts(th, moments2=True)
        m = np.where(self.is_c, Ez[self.thr], Ezb[self.thr])
        g = np.zeros(p + 2)
        g[:p] = self.X.T @ ((self.y - m * mu) * wr)
        Hm = np.zeros((p + 2, p + 2))
        # linear block: -X' diag(m mu) X + sum_T [Vz gA gA' + Vzb gB gB' + Czb (gA gB' + gB gA')]
        wv = m * mu * wr
        Vz, Vzb, Czb = Vz * w, Vzb * w, Czb * w
        for c0 in range(0, len(wv), 500000):
            Xs = self.X[c0:c0 + 500000]
            Hm[:p, :p] -= Xs.T @ (Xs * wv[c0:c0 + 500000, None])
        gA = _group_sum(self.thr[:self.nc], self.X[:self.nc] * mu[:self.nc, None], self.nT)
        gB = _group_sum(self.thr[self.nc:], self.X[self.nc:] * mu[self.nc:, None], self.nT)
        Hm[:p, :p] += (gA * Vz[:, None]).T @ gA + (gB * Vzb[:, None]).T @ gB + (gA * Czb[:, None]).T @ gB \
            + (gB * Czb[:, None]).T @ gA
        # (log r, b) block and cross terms by finite differences of the frailty moments at fixed A, B
        lr, b = th[p], th[p + 1]
        e = 1e-4

        def F(lr_, b_):
            return frailty_terms(self.Nc, self.NR, A, B, np.exp(lr_), b_)
        base = F(lr, b)
        for k, (dl, db) in enumerate([(e, 0.0), (0.0, e)]):
            up, dn = F(lr + dl, b + db), F(lr - dl, b - db)
            g[p + k] = float(w @ (up[0] - dn[0]) / (2 * e))
            dEz = w * (up[1] - dn[1]) / (2 * e)
            dEzb = w * (up[2] - dn[2]) / (2 * e)
            Hm[:p, p + k] = Hm[p + k, :p] = -(gA.T @ dEz + gB.T @ dEzb)
        for k1, (a1, b1) in enumerate([(e, 0.0), (0.0, e)]):
            for k2, (a2, b2) in enumerate([(e, 0.0), (0.0, e)]):
                if k2 < k1:
                    continue
                f = lambda dl, db: float(w @ F(lr + dl, b + db)[0])
                v = (f(a1 + a2, b1 + b2) - f(a1 - a2, b1 - b2) - f(-a1 + a2, -b1 + b2) + f(-a1 - a2, -b1 - b2)) / (4 * e * e)
                Hm[p + k1, p + k2] = Hm[p + k2, p + k1] = v
        return g, Hm

    def pooled_start(self, th0, iters=30):
        """Concave warm start: Poisson GLM of all linear terms without fitness (z = 1)."""
        beta = th0[:self.p].copy()
        for _ in range(iters):
            mu = np.exp(np.clip(self.X @ beta + self.o, -700, 300))
            g = self.X.T @ (self.y - mu)
            Hm = np.zeros((self.p, self.p))
            for c0 in range(0, len(mu), 500000):
                Xs = self.X[c0:c0 + 500000]
                Hm += Xs.T @ (Xs * mu[c0:c0 + 500000, None])
            step = np.linalg.solve(Hm + 1e-9 * np.eye(self.p), g)
            step *= min(1.0, 3.0 / max(np.abs(step).max(), 1e-12))
            beta = beta + step
            if np.abs(step).max() < 1e-8:
                break
        out = th0.copy()
        out[:self.p] = beta
        return out

    def fit_weighted(self, th0, w, iters=40, tol=1e-7):
        """Refit under thread weights, starting from th0 (full-refit bootstrap)."""
        return self.fit(th0, iters, tol, w=w, warm=False)

    def fit(self, th0, iters=40, tol=1e-7, w=None, warm=True):
        th = self.pooled_start(th0) if warm else th0.copy()
        ll = self.loglik(th, w)
        for it in range(iters):
            g, Hm = self.grad_hess(th, w)
            try:
                step = -np.linalg.solve(Hm, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(Hm, g, rcond=None)[0]
            if g @ step < 0:  # not an ascent direction: fall back to a scaled gradient step
                step = g / (np.abs(np.diag(Hm)).max() + 1e-9)
            # step limits: (log r, b) at most 0.5 per iteration, linear terms at most 2
            sc_ = max(1.0, np.abs(step[self.p:]).max() / 0.5, np.abs(step[:self.p]).max() / 2.0)
            step = step / sc_
            lam = 1.0
            while True:
                cand = th + lam * step
                cand[self.p + 1] = max(cand[self.p + 1], 0.05)
                lc = self.loglik(cand, w)
                if np.isfinite(lc) and lc >= ll - 1e-9 * abs(ll):
                    break
                lam /= 2
                if lam < 1e-6:
                    break
            th, ll = cand, lc
            if np.max(np.abs(lam * step)) < tol:
                break
        self.n_iter = it + 1
        return th

    def info(self, th):
        _, Hm = self.grad_hess(th)
        return -Hm


def fit_re(D: dict, start: np.ndarray):
    M = REModel(D)
    th = M.fit(start)
    return M, th


def onestep_reps(A: dict, nT: int, W: np.ndarray) -> np.ndarray:
    """One-step thread bootstrap replicates for a Stage A (fixed-effects) fit."""
    sc, I = fe_scores(A, nT)
    kept = A["prep"].cols
    reps = np.full((W.shape[0], len(A["coef"])), np.nan)
    reps[:, kept] = A["coef"][kept][None, :] + ((W - 1.0) @ sc) @ np.linalg.inv(I).T
    return reps
