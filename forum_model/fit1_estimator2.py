"""Estimator v2 for the two-clock model (after the Stage 1 review; docs/results_register.md Rev-1, 5, 7, 8).

  * Rev-1: thread fitness integrated out (marginal likelihood); the only estimator from now on.
  * Rev-7: exposure is hour-exact. Each exposure cell (comment- or thread-age bin, split at the unit's own
    events and, for comments, at thread age 30 min) is further split at UTC hour boundaries into sub-rows.
    Non-hour covariates stay at the cell (clocks at the geometric-midpoint age, early-position indicator at the
    arithmetic midpoint); the hour effect is exact. Hours are stored as an index, not as dense columns.
  * Rev-8 (optional): bins below 10 s ([0, 1 s] and four log-spaced bins from 1 to 10 s).
  * Rev-9 (optional, quad=True): exposure integral by 2-point Gauss-Legendre quadrature in log age on every
    exposure piece wider than 0.1 in log age; narrower pieces keep the one-point rule. A piece starting at age 0
    is integrated by composite 2-point Gauss-Legendre in log age over [1 ms, b] in sub-pieces of log width <= 2,
    plus [0, 1 ms] at one point. Each node is its own exposure row (same thread, hour and non-clock covariates).
  * Rev-5: multi-start over b. b is profiled on {0.5, 1, 2, 3.5, 5} (all other parameters re-maximised, up to
    15 Newton iterations each), Newton starts from the best point, cap 100 iterations; convergence is reported.
Events enter as their own cells with covariates at the exact event time (Rev-2).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .fe_poisson import FEPoissonBoot
from .fit1_estimator import (C_NAMES, HOURS, _event_order, _gm, _lx, frailty_terms, num_info, num_scores,
                             onestep_reps, rb_data, rb_unit_loglik, stage_c)

H_DEFAULT = 3 * 86400.0
BASE_EDGES = np.r_[0.0, np.logspace(1, np.log10(H_DEFAULT), 40)]
REV8_EDGES = np.r_[0.0, np.logspace(0, 1, 5)[:-1], BASE_EDGES[1:]]
EV_LOGEXP = np.log(1e-9)
B_GRID = [0.5, 1.0, 2.0, 3.5, 5.0]


QUAD_N, QUAD_DELTA, QUAD_W0, QUAD_FLOOR = 2, 0.1, 2.0, 1e-3


def quad_nodes(a: np.ndarray, b: np.ndarray, quad: bool):
    """Exposure nodes for pieces [a, b] (seconds): (piece index, node age, log weight). Without quad: one node per
    piece at the geometric midpoint (arithmetic for a = 0) with weight b - a, as before Rev-9."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if not quad:
        return np.arange(n), _gm(a, b), np.log(b - a)
    xg, wg = np.polynomial.legendre.leggauss(QUAD_N)
    lw = np.where(a > 0, np.log(b / np.maximum(a, 1e-300)), np.inf)
    narrow = (a > 0) & (lw <= QUAD_DELTA)
    wide = (a > 0) & ~narrow
    zero = a <= 0
    idx, age, logw = [np.nonzero(narrow)[0]], [_gm(a[narrow], b[narrow])], [np.log(b[narrow] - a[narrow])]

    def gl(ii, lo, hi):  # GL nodes in log age on [lo, hi] for pieces ii
        la, lb = np.log(lo), np.log(hi)
        half, mid = (lb - la) / 2, (lb + la) / 2
        x = mid[:, None] + half[:, None] * xg[None, :]
        idx.append(np.repeat(ii, QUAD_N))
        age.append(np.exp(x).ravel())
        logw.append((np.log(half)[:, None] + np.log(wg)[None, :] + x).ravel())
    wi = np.nonzero(wide)[0]
    gl(wi, a[wi], b[wi])
    zi = np.nonzero(zero)[0]
    tiny = zi[b[zi] <= QUAD_FLOOR]
    idx.append(tiny); age.append(b[tiny] / 2); logw.append(np.log(b[tiny]))
    zi = zi[b[zi] > QUAD_FLOOR]
    if len(zi):
        idx.append(zi); age.append(np.full(len(zi), QUAD_FLOOR / 2)); logw.append(np.full(len(zi), np.log(QUAD_FLOOR)))
        k = np.ceil((np.log(b[zi]) - np.log(QUAD_FLOOR)) / QUAD_W0).astype(int)
        for kk in np.unique(k):
            ii = zi[k == kk]
            e = np.exp(np.linspace(np.log(QUAD_FLOOR), np.log(b[ii]), kk + 1, axis=1))  # (len(ii), kk+1)
            for j in range(kk):
                gl(ii, e[:, j], e[:, j + 1])
    idx = np.concatenate(idx)
    o = np.argsort(idx, kind="stable")
    return idx[o], np.concatenate(age)[o], np.concatenate(logw)[o]


def edges_for(rev8: bool) -> np.ndarray:
    return REV8_EDGES if rev8 else BASE_EDGES


def _cells(U, ev_unit, ev_t, edges, extra=None):
    nU = len(U)
    e = np.minimum(edges[None, :], U[:, None])
    bu = [np.repeat(np.arange(nU), e.shape[1]), np.arange(nU)]
    bt = [e.ravel(), U]
    ba = [np.zeros(e.size), np.zeros(nU)]
    if extra is not None:
        m = (extra > 0) & (extra < U)
        bu.append(np.arange(nU)[m]); bt.append(extra[m]); ba.append(np.zeros(m.sum()))
    keep = ev_t <= U[ev_unit]
    bu.append(ev_unit[keep]); bt.append(ev_t[keep]); ba.append(np.ones(keep.sum()))
    df = pl.DataFrame({"unit": np.concatenate(bu), "t": np.concatenate(bt), "anyev": np.concatenate(ba)})
    df = df.group_by("unit", "t").agg(pl.col("anyev").sum()).sort("unit", "t")
    df = df.with_columns(a=pl.col("t").shift(1).over("unit"), n=pl.col("anyev").cum_sum().shift(1).over("unit"))
    df = df.filter(pl.col("a").is_not_null() & (pl.col("t") > pl.col("a")))
    return df["unit"].to_numpy(), df["a"].to_numpy(), df["t"].to_numpy(), df["n"].fill_null(0).to_numpy().astype(float)


def hour_split(abs_a, abs_b):
    """Split intervals [abs_a, abs_b] at UTC hour boundaries: (interval index, hour 0-23, seconds)."""
    ha = np.floor(abs_a / 3600.0).astype(np.int64)
    hb = np.floor((abs_b - 1e-6) / 3600.0).astype(np.int64)
    k = np.maximum(hb - ha + 1, 1)
    idx = np.repeat(np.arange(len(abs_a)), k)
    off = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    hs = (ha[idx] + off) * 3600.0
    lo = np.maximum(abs_a[idx], hs)
    hi = np.minimum(abs_b[idx], hs + 3600.0)
    sec = hi - lo
    ok = sec > 0
    return idx[ok], (np.mod(ha[idx] + off, 24))[ok].astype(np.int8), sec[ok]


def _hour(abs_t):
    return (np.mod(abs_t, 86400.0) // 3600).astype(np.int8)


def _cX(u_age, s, depth, op, n, pos, tau_pos):
    return np.column_stack([_lx(u_age), _lx(u_age) ** 2, _lx(s + u_age), _lx(s + u_age) ** 2, (depth == 2),
                            (depth >= 3), op, np.log1p(n), np.log(np.maximum(pos, 1)) * (tau_pos <= 1800.0)]
                           ).astype(np.float64)


def build_comment(S: dict, tmap: dict, edges: np.ndarray, quad: bool = False) -> dict:
    units = S["units"]
    uid = {k: i for i, k in enumerate(units["id"].to_list())}
    ec = S["events_c"]
    ev_unit = np.array([uid[x] for x in ec["unit"].to_list()], dtype=np.int64)
    ev_u = np.clip(ec["u"].to_numpy(), 0.5, None)
    is_c = (ec["stream"] == "comment").to_numpy()
    s, U, T0 = units["s"].to_numpy(), units["U"].to_numpy(), units["T0"].to_numpy()
    depth, op, pos = units["depth"].to_numpy(), units["op_flag"].to_numpy().astype(float), units["pos"].to_numpy()
    thr_u = np.array([tmap[t] for t in units["thread_id"].to_list()], dtype=np.int64)
    i, a, b, n = _cells(U, ev_unit, ev_u, edges, extra=1800.0 - s)
    # Rev-7: split every exposure cell at UTC hour boundaries; each piece carries its own covariates
    sc, sh, ss = hour_split(T0[i] + s[i] + a, T0[i] + s[i] + b)
    cnt = np.bincount(sc, minlength=len(i))
    k_in = np.arange(len(sc)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pa = np.where(k_in == 0, a[sc], np.ceil((T0[i][sc] + s[i][sc] + a[sc]) / 3600.0 + k_in - 1) * 3600.0 - T0[i][sc] - s[i][sc])
    pa = np.maximum(pa, a[sc])
    pb = np.minimum(pa + ss, b[sc])
    i, a, b, n = i[sc], pa, pb, n[sc]
    okp = b > a
    i, a, b, n, sh = i[okp], a[okp], b[okp], n[okp], sh[okp]
    qi, qage, qlogw = quad_nodes(a, b, quad)  # Rev-9 (identity without quad)
    tpos = s[i] + (a + b) / 2
    i, n, sh, tpos = i[qi], n[qi], sh[qi], tpos[qi]
    Xe = _cX(qage, s[i], depth[i], op[i], n, pos[i], tpos)
    order = _event_order(ev_unit, ev_u)
    k = np.nonzero(is_c & (ev_u <= U[ev_unit]))[0]
    ku = ev_unit[k]
    Xv = _cX(ev_u[k], s[ku], depth[ku], op[ku], order[k].astype(float), pos[ku], s[ku] + ev_u[k])
    ne = len(i)
    return {"X": np.vstack([Xe, Xv]).astype(np.float32), "thr": np.r_[thr_u[i], thr_u[ku]],
            "y": np.r_[np.zeros(ne), np.ones(len(k))], "logw": np.r_[qlogw, np.full(len(k), EV_LOGEXP)],
            "sub_cell": np.arange(ne + len(k)),
            "sub_hour": np.r_[sh, _hour(T0[ku] + s[ku] + ev_u[k])],
            "sub_logexp": np.r_[qlogw, np.full(len(k), EV_LOGEXP)].astype(np.float32)}


def build_root(S: dict, tmap: dict, edges: np.ndarray, post_types: list[str], quad: bool = False) -> dict:
    roots = S["roots"]
    er = S["events_r"]
    rt_unit = np.array([tmap[t] for t in er["thread_id"].to_list()], dtype=np.int64)
    rt_t = np.clip(er["tau"].to_numpy(), 0.5, None)
    UR, T0 = roots["U"].to_numpy(), roots["T0"].to_numpy()
    ptype_all = roots["post_type"].to_numpy()
    j, a, b, n = _cells(UR, rt_unit, rt_t, edges)
    sc, sh_e, ss = hour_split(T0[j] + a, T0[j] + b)
    cnt = np.bincount(sc, minlength=len(j))
    k_in = np.arange(len(sc)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pa = np.where(k_in == 0, a[sc], np.ceil((T0[j][sc] + a[sc]) / 3600.0 + k_in - 1) * 3600.0 - T0[j][sc])
    pa = np.maximum(pa, a[sc])
    pb = np.minimum(pa + ss, b[sc])
    j, a, b, n = j[sc], pa, pb, n[sc]
    okp = b > a
    j, a, b, n, sh_e = j[okp], a[okp], b[okp], n[okp], sh_e[okp]
    qi, qage, qlogw = quad_nodes(a, b, quad)  # Rev-9 (identity without quad)
    j, n, sh_e = j[qi], n[qi], sh_e[qi]
    keep = rt_t <= UR[rt_unit]
    rt_unit, rt_t = rt_unit[keep], rt_t[keep]
    nev = _event_order(rt_unit, rt_t).astype(float)
    jj = np.r_[j, rt_unit]
    age = np.r_[qage, rt_t]
    nn = np.r_[n, nev]
    ptype = ptype_all[jj]
    multi = len(post_types) > 1
    cols, names = [], []
    for ty in post_types:
        ind = (ptype == ty).astype(float) if multi else np.ones(len(jj))
        cols += [ind * _lx(age), ind * _lx(age) ** 2]
        names += [f"aR1_{ty}", f"aR2_{ty}"] if multi else ["aR1", "aR2"]
    cols.append(nn)
    names.append("gamma")
    if multi:
        cols.append(nn * (ptype == "ask_hn"))
        names.append("minus_kappa")
    ne = len(j)
    out = {"X": np.column_stack(cols).astype(np.float32), "thr": jj, "y": np.r_[np.zeros(ne), np.ones(len(rt_unit))],
           "logw": np.r_[qlogw, np.full(len(rt_unit), EV_LOGEXP)],
           "sub_cell": np.arange(ne + len(rt_unit)),
           "sub_hour": np.r_[sh_e, _hour(T0[rt_unit] + rt_t)],
           "sub_logexp": np.r_[qlogw, np.full(len(rt_unit), EV_LOGEXP)].astype(np.float32), "names": names}
    if multi:
        out["R_type"] = np.column_stack([(ptype == ty) for ty in post_types]).astype(np.float32)
        out["R_type_names"] = [f"alpha_R_{ty}" for ty in post_types]
    return out


def build(S: dict, rev8: bool = False, post_types: list[str] | None = None, comment_stream: bool = True,
          quad: bool = False) -> dict:
    roots = S["roots"]
    tids = roots["thread_id"].to_list()
    tmap = {t: i for i, t in enumerate(tids)}
    types = post_types or sorted(set(roots["post_type"].to_list()))
    edges = edges_for(rev8)
    D = {"nT": len(tids), "tids": tids, "R": build_root(S, tmap, edges, types, quad)}
    if comment_stream:
        D["C"] = build_comment(S, tmap, edges, quad)
    return D


class REModel2:
    """Marginal likelihood with gamma fitness; hours as an exact sub-row index (Rev-7)."""

    def __init__(self, D: dict):
        C, R = D.get("C"), D["R"]
        pc = C["X"].shape[1] if C is not None else 0
        pr = R["X"].shape[1]
        rint = R.get("R_type", np.ones((len(R["y"]), 1), np.float32))
        rnames = R.get("R_type_names", ["alpha_R"])
        nc = len(C["y"]) if C is not None else 0
        nr = len(R["y"])
        ia = 1 if C is not None else 0
        Pr = pc + pr + ia + rint.shape[1]
        X = np.zeros((nc + nr, Pr), np.float32)
        if C is not None:
            X[:nc, :pc] = C["X"]
            X[:nc, pc + pr] = 1.0
        X[nc:, pc:pc + pr] = R["X"]
        X[nc:, pc + pr + ia:] = rint
        self.names = (C_NAMES if C is not None else []) + R["names"] + (["alpha_c"] if C is not None else []) + rnames \
            + HOURS + ["log_r", "b"]
        self.X, self.Pr, self.nT = X, Pr, D["nT"]
        self.p = Pr + 23  # linear parameters (rest + hours 1..23)
        self.y = np.r_[C["y"], R["y"]] if C is not None else R["y"]
        self.thr = np.r_[C["thr"], R["thr"]] if C is not None else R["thr"]
        self.is_c = np.r_[np.ones(nc, bool), np.zeros(nr, bool)]
        self.sub_cell = np.r_[C["sub_cell"], nc + R["sub_cell"]] if C is not None else R["sub_cell"]
        self.sub_hour = np.r_[C["sub_hour"], R["sub_hour"]].astype(np.int64) if C is not None else R["sub_hour"].astype(np.int64)
        self.sub_logexp = np.r_[C["sub_logexp"], R["sub_logexp"]] if C is not None else R["sub_logexp"]
        assert np.array_equal(self.sub_cell, np.arange(len(self.y))), "Rev-7 design: one hour per cell"
        self.sub_thr, self.sub_y, self.sub_is_c = self.thr, self.y, self.is_c  # identity sub_cell: alias, no copy
        self.nc = nc
        self.Nc = np.bincount(self.thr[self.is_c], weights=self.y[self.is_c], minlength=self.nT)
        self.NR = np.bincount(self.thr[~self.is_c], weights=self.y[~self.is_c], minlength=self.nT)

    # ---------------------------------------------------------------- core pieces
    def _mu(self, th):
        eta = self.X @ th[:self.Pr].astype(np.float64)
        bh = np.r_[0.0, th[self.Pr:self.p]]
        mu_sub = np.exp(np.clip(eta + bh[self.sub_hour] + self.sub_logexp, -700, 300))  # one hour per cell (Rev-7)
        return eta, bh, mu_sub

    def _parts(self, th, moments2=False, frailty=True):
        eta, bh, mu_sub = self._mu(th)
        A = np.bincount(self.sub_thr[self.sub_is_c], weights=mu_sub[self.sub_is_c], minlength=self.nT)
        B = np.bincount(self.sub_thr[~self.sub_is_c], weights=mu_sub[~self.sub_is_c], minlength=self.nT)
        ev = (np.bincount(self.thr, weights=self.y * eta, minlength=self.nT)
              + np.bincount(self.sub_thr, weights=self.sub_y * bh[self.sub_hour], minlength=self.nT))
        ft = frailty_terms(self.Nc, self.NR, A, B, np.exp(th[self.p]), th[self.p + 1], moments2) if frailty else None
        return eta, bh, mu_sub, A, B, ft, ev

    def loglik_threads(self, th):
        *_, ft, ev = self._parts(th)
        return ft[0] + ev

    def loglik(self, th, w=None):
        lt = self.loglik_threads(th)
        return float(lt.sum() if w is None else w @ lt)

    def _rest_groupsum(self, vals_cell, rows_mask=None):
        """sum over cells of X * vals by thread -> (nT x Pr)."""
        out = np.zeros((self.nT, self.Pr))
        m = slice(None) if rows_mask is None else rows_mask
        thr = self.thr[m]
        v = vals_cell[m]
        X = self.X[m]
        for k in range(self.Pr):
            out[:, k] = np.bincount(thr, weights=v * X[:, k], minlength=self.nT)
        return out

    def _hour_groupsum(self, vals_sub, mask=None):
        m = slice(None) if mask is None else mask
        key = self.sub_thr[m] * 24 + self.sub_hour[m]
        return np.bincount(key, weights=vals_sub[m], minlength=self.nT * 24).reshape(self.nT, 24)[:, 1:]

    def _lin_blocks(self, wsub_mu, cell_w):
        """Poisson information blocks for linear parameters with per-sub-row weights wsub_mu (= m mu w) and
        per-cell weights cell_w (= sum over the cell's sub-rows)."""
        Pr = self.Pr
        I = np.zeros((self.p, self.p))
        for c0 in range(0, len(cell_w), 400000):
            Xs = self.X[c0:c0 + 400000].astype(np.float64)
            I[:Pr, :Pr] += Xs.T @ (Xs * cell_w[c0:c0 + 400000, None])
        for k in range(Pr):
            I[:Pr, Pr:][k] = np.bincount(self.sub_hour, weights=wsub_mu * self.X[:, k], minlength=24)[1:]
        I[Pr:, :Pr] = I[:Pr, Pr:].T
        I[Pr:, Pr:] = np.diag(np.bincount(self.sub_hour, weights=wsub_mu, minlength=24)[1:])
        return I

    def _lin_grad(self, resid_cell, resid_sub):
        g = np.zeros(self.p)
        for c0 in range(0, len(resid_cell), 400000):
            g[:self.Pr] += self.X[c0:c0 + 400000].astype(np.float64).T @ resid_cell[c0:c0 + 400000]
        g[self.Pr:] = np.bincount(self.sub_hour, weights=resid_sub, minlength=24)[1:]
        return g

    # ---------------------------------------------------------------- derivatives
    def grad_hess(self, th, w=None, fixed_b=False):
        p = self.p
        w = np.ones(self.nT) if w is None else w
        eta, bh, mu_sub, A, B, (logI, Ez, Ezb, Vz, Vzb, Czb), ev = self._parts(th, moments2=True)
        m_sub = np.where(self.sub_is_c, Ez[self.sub_thr], Ezb[self.sub_thr])
        wsub = w[self.sub_thr]
        M_cell = m_sub * mu_sub * wsub
        resid_cell = self.y * w[self.thr] - M_cell
        resid_sub = (self.sub_y - m_sub * mu_sub) * wsub
        g = np.zeros(p + 2)
        g[:p] = self._lin_grad(resid_cell, resid_sub)
        Hm = np.zeros((p + 2, p + 2))
        Hm[:p, :p] = -self._lin_blocks(m_sub * mu_sub * wsub, M_cell)
        # frailty variance terms: per-thread gradients of A and B
        Mc_raw = mu_sub
        gA = np.hstack([self._rest_groupsum(Mc_raw, self.is_c), self._hour_groupsum(mu_sub, self.sub_is_c)])
        gB = np.hstack([self._rest_groupsum(Mc_raw, ~self.is_c), self._hour_groupsum(mu_sub, ~self.sub_is_c)])
        Vz_, Vzb_, Czb_ = Vz * w, Vzb * w, Czb * w
        Hm[:p, :p] += (gA * Vz_[:, None]).T @ gA + (gB * Vzb_[:, None]).T @ gB + (gA * Czb_[:, None]).T @ gB \
            + (gB * Czb_[:, None]).T @ gA
        lr, b = th[p], th[p + 1]
        e = 1e-4

        def F(lr_, b_):
            return frailty_terms(self.Nc, self.NR, A, B, np.exp(lr_), b_)
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

    def scores(self, th):
        p = self.p
        eta, bh, mu_sub, A, B, (logI, Ez, Ezb), ev = self._parts(th)
        m_sub = np.where(self.sub_is_c, Ez[self.sub_thr], Ezb[self.sub_thr])
        M_cell = m_sub * mu_sub
        rest = self._rest_groupsum(self.y - M_cell)
        hours = self._hour_groupsum(self.sub_y - m_sub * mu_sub)
        lr, b = th[p], th[p + 1]
        e = 1e-5
        g_lr = (frailty_terms(self.Nc, self.NR, A, B, np.exp(lr + e), b)[0]
                - frailty_terms(self.Nc, self.NR, A, B, np.exp(lr - e), b)[0]) / (2 * e)
        g_b = (frailty_terms(self.Nc, self.NR, A, B, np.exp(lr), b + e)[0]
               - frailty_terms(self.Nc, self.NR, A, B, np.exp(lr), b - e)[0]) / (2 * e)
        return np.column_stack([rest, hours, g_lr, g_b])

    # ---------------------------------------------------------------- fitting
    def start(self) -> np.ndarray:
        """Concave warm start: Poisson GLM of the linear terms without fitness (z = 1)."""
        th = np.zeros(self.p + 2)
        if self.nc:
            expo = np.bincount(self.sub_cell, weights=np.exp(self.sub_logexp.astype(np.float64)), minlength=len(self.y))
            th[self.names.index("alpha_c")] = np.log(max(self.Nc.sum(), 1) / expo[self.is_c & (self.y == 0)].sum())
        th[self.p], th[self.p + 1] = np.log(5.0), 1.0
        for _ in range(40):
            eta, bh, mu_sub = self._mu(th)
            M_cell = mu_sub
            g = self._lin_grad(self.y - M_cell, self.sub_y - mu_sub)
            I = self._lin_blocks(mu_sub, M_cell)
            step = np.linalg.solve(I + 1e-9 * np.eye(self.p), g)
            step *= min(1.0, 3.0 / max(np.abs(step).max(), 1e-12))
            th[:self.p] += step
            if np.abs(step).max() < 1e-8:
                break
        return th

    def newton(self, th0, w=None, iters=100, tol=1e-7, fix_b=False):
        th = th0.copy()
        ll = self.loglik(th, w)
        conv = False
        for it in range(iters):
            g, Hm = self.grad_hess(th, w)
            idx = np.arange(len(th)) if not fix_b else np.r_[np.arange(self.p + 1)]
            step = np.zeros(len(th))
            try:
                step[idx] = -np.linalg.solve(Hm[np.ix_(idx, idx)], g[idx])
            except np.linalg.LinAlgError:
                step[idx] = -np.linalg.lstsq(Hm[np.ix_(idx, idx)], g[idx], rcond=None)[0]
            if g[idx] @ step[idx] < 0:
                step[idx] = g[idx] / (np.abs(np.diag(Hm)[idx]).max() + 1e-9)
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
                conv = True
                break
        return th, {"iterations": it + 1, "converged": conv, "loglik": ll}

    def fit_multistart(self, profile_iters: int = 8, iters: int = 100) -> tuple[np.ndarray, dict]:
        """Rev-5: profile b on B_GRID (others re-maximised), then full Newton from the best profile point."""
        st = self.start()
        prof = []
        for bv in B_GRID:
            t0 = st.copy()
            t0[self.p + 1] = bv
            th_b, info_b = self.newton(t0, iters=profile_iters, fix_b=True)
            prof.append((info_b["loglik"], bv, th_b))
        best = max(prof, key=lambda x: x[0])
        th, info = self.newton(best[2], iters=iters)
        info.update({"profile_b": [p_[1] for p_ in prof], "profile_loglik": [p_[0] for p_ in prof],
                     "start_b": best[1]})
        return th, info

    def fit_weighted(self, th0, w, iters=60):
        return self.newton(th0, w=w, iters=iters)[0]

    def info(self, th):
        return -self.grad_hess(th)[1]


def fit_all(S: dict, B: int, rng: np.random.Generator, rev8: bool = False, with_b1: bool = True,
            quad: bool = False) -> dict:
    D = build(S, rev8=rev8, quad=quad)
    M = REModel2(D)
    if with_b1:  # keep only what B1 needs; the full design now lives in M
        D["C"]["X2"] = D["C"]["X"][:, :2].copy()
    D["C"].pop("X", None)
    D["R"].pop("X", None)
    th, conv = M.fit_multistart()
    I = M.info(th)
    sc = M.scores(th)
    Iinv = np.linalg.inv(I)
    W = rng.multinomial(D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)
    reps = th[None, :] + ((W - 1.0) @ sc) @ Iinv.T
    se = np.sqrt(np.clip(np.diag(Iinv @ (sc.T @ sc) @ Iinv), 0, None))
    out = {"design": {"cells": len(M.y), "sub_rows": len(M.sub_cell), "threads": D["nT"]},
           "RE": {"names": M.names, "coef": th, "reps": reps, "se_sandwich": se, "n_iter": conv["iterations"],
                  "converged": conv["converged"], "start_b": conv["start_b"], "profile_loglik": conv["profile_loglik"]}}
    R = rb_data(S)
    thc = stage_c(R)
    sc3 = np.zeros((D["nT"], 5))
    ss = num_scores(lambda t: rb_unit_loglik(t, R), thc)
    for k in range(5):
        sc3[:, k] = np.bincount(R["thr"], weights=ss[:, k], minlength=D["nT"])
    I3 = num_info(lambda t: float(rb_unit_loglik(t, R).sum()), thc)
    out["reply_back"] = {"names": ["log_rho", "back_m", "log_back_sigma", "beta_back_root", "beta_back_dep3p"],
                         "coef": thc, "reps": thc[None, :] + ((W - 1.0) @ sc3) @ np.linalg.inv(I3).T}
    if with_b1:
        C = D["C"]
        prep = FEPoissonBoot(C["thr"], C["y"], C["X2"].astype(np.float64), C["logw"], C["thr"])
        coef, se1 = prep.fit(se=True)
        A1 = {"prep": prep, "coef": coef, "se": se1}
        out["B1"] = {"names": C_NAMES[:2], "coef": coef, "se_sandwich": se1, "reps": onestep_reps(A1, D["nT"], W)}
    out["_M"] = M
    return out
