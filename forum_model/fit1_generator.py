"""Generator for the two-clock intensity model (docs/fit1_brief.md).

Given each thread's real root arrival time and post type, draw a thread fitness z = exp(theta) ~
Gamma(r, 1/r), then simulate three streams to thread age H (3 days):

  root stream     lambda_R(t) = z^b exp(alpha_R + aR(t) + c(h) + (gamma - kappa q) n_top(t))
  comment stream  lambda_c(t) = z   exp(alpha_c + k(u) + a(tau) + c(h) + beta' x_c(t))
  reply-back      lambda_back,c(t) = rho g_back(u) exp(beta_back' x_c)

with t thread age, u comment age, tau = s_c + u, h the UTC hour of the absolute time, and
x_c(t) = [depth 2, depth 3+, OP comment, log(1 + direct replies so far), log rank x 1[tau <= 30 min]].
Clock shapes are log-normal: f(x) = l1 x + l2 x^2 with x = log(age in minutes) (constants in alpha).

Simulation is exact for this model up to the time grid. Each comment's two streams depend only on its
own history (its own replies so far) and static covariates. Each thread's root stream depends only on its
own top-level count. So comments are simulated generation by generation, vectorised:
  * each stream's base rate is integrated on a per-unit grid (240 log-spaced points from 1 s to H, plus
    every UTC hour boundary and the 30-minute thread-age boundary), with the rate held at its value at the
    segment's geometric midpoint;
  * events of a state-dependent stream are drawn by time rescaling: the next event solves
    Lambda(t) = Lambda(t_now) + E / m(n), E ~ Exp(1), where m(n) is the current state multiplier
    (exp((gamma - kappa q) n) for the root stream, (1 + n)^beta_logn for a comment);
  * reply-backs are an inhomogeneous Poisson process drawn by inverse CDF, interleaved with the
    comment stream (each one raises n).

Authors (brief, Stage 3 author model): a reply-back is written by the author of the comment's parent.
Any other event, with probability pi_prev, is by a previous participant in the thread (chosen in
proportion to their items so far, the post counting as one item for the OP); otherwise it is by a new
author. Self-replies are never produced. A comment-stream reply never comes from the author of the
comment's parent (that would be a reply-back), and a top-level comment never comes from the OP.
"Previous participants" are those already generated with an earlier creation time. Because generations
are simulated in order, a later-generation item that is earlier in time is not yet visible when an author
is drawn. This affects authorship only, not rates, except through the OP-comment flag.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

H_DEFAULT = 3 * 86400.0
LOG_GRID = np.r_[0.0, np.logspace(0, np.log10(H_DEFAULT), 240)]
MAX_PER_THREAD = 20000


def clock_integral(lin: float, quad_: float, lo_s: float = 0.0, hi_s: float = H_DEFAULT) -> float:
    """Integral over age [lo_s, hi_s] seconds of exp(lin x + quad x^2), x = log(age / 60): used to set
    intercepts from target counts."""
    g = np.r_[max(lo_s, 1e-3), np.logspace(np.log10(max(lo_s, 1e-3) + 1e-3), np.log10(hi_s), 4000)]
    mid = np.sqrt(g[:-1] * g[1:])
    x = np.log(mid / 60.0)
    return float(np.sum(np.exp(lin * x + quad_ * x * x) * np.diff(g)))


def lognormal_coefs(median_min: float, sigma: float) -> tuple[float, float]:
    """log density of a log-normal in t (minutes), up to a constant, as l1 x + l2 x^2 with x = log t."""
    m = np.log(median_min)
    return m / sigma ** 2 - 1.0, -1.0 / (2 * sigma ** 2)


def _clock(lin: float, quad: float, age_s: np.ndarray) -> np.ndarray:
    x = np.log(np.maximum(age_s, 1e-3) / 60.0)
    return lin * x + quad * x * x


def _grids(start_abs: np.ndarray, span: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Per-row sorted grid on [0, span]: log grid, UTC hour boundaries, optional extra point."""
    n = len(start_abs)
    g = np.minimum(LOG_GRID[None, :], span[:, None])
    first = (3600.0 - np.mod(start_abs, 3600.0))
    hb = first[:, None] + 3600.0 * np.arange(74)[None, :]
    hb = np.minimum(hb, span[:, None])
    cols = [g, hb, span[:, None]]
    if extra is not None:
        cols.append(np.clip(extra, 0, span)[:, None])
    G = np.concatenate(cols, axis=1)
    G.sort(axis=1)
    return G


def _mid(G: np.ndarray) -> np.ndarray:
    lo, hi = G[:, :-1], G[:, 1:]
    return np.where(lo > 0, np.sqrt(np.maximum(lo, 1e-9) * hi), hi / 2)


def _invert(Lam: np.ndarray, G: np.ndarray, rate: np.ndarray, rows: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each (row, target), the time t with Lambda_row(t) = target (inf if beyond the row's total)."""
    L = Lam[rows]
    out = np.full(len(rows), np.inf)
    ok = target < L[:, -1]
    if not ok.any():
        return out
    r, tg = rows[ok], target[ok]
    Lr = Lam[r]
    lo = np.zeros(len(r), dtype=np.int64)
    hi = np.full(len(r), Lam.shape[1] - 1, dtype=np.int64)
    while True:
        act = hi - lo > 1
        if not act.any():
            break
        m = (lo + hi) // 2
        go = Lr[np.arange(len(r)), m] <= tg
        lo = np.where(act & go, m, lo)
        hi = np.where(act & ~go, m, hi)
    j = lo
    rt = rate[r, j]
    t = G[r, j] + np.where(rt > 0, (tg - Lr[np.arange(len(r)), j]) / np.maximum(rt, 1e-300), 0.0)
    out[ok] = np.minimum(t, G[r, j + 1])
    return out


def hour_of(abs_t: np.ndarray) -> np.ndarray:
    return (np.mod(abs_t, 86400.0) // 3600).astype(np.int64)


class Generator:
    def __init__(self, params: dict, H: float = H_DEFAULT, max_total: int = 3_000_000):
        self.max_total = max_total
        self.p = params
        self.H = H
        self.hour = np.asarray(params["hour"], dtype=float)  # 24 values, log c(h); hour 0 is 0

    # ---------------- root stream ----------------
    def _root_stream(self, T0, q, z, rng):
        p, H = self.p, self.H
        n = len(T0)
        G = _grids(T0, np.full(n, H))
        mid = _mid(G)
        lr = (p["alpha_R"] + b_pow(z, p["b"])[:, None] + _clock(*p["aR"], mid)
              + self.hour[hour_of(T0[:, None] + mid)])
        rate = np.exp(lr)
        Lam = np.concatenate([np.zeros((n, 1)), np.cumsum(rate * np.diff(G, axis=1), axis=1)], axis=1)
        slope = p["gamma"] - p.get("kappa", 0.0) * q
        t_now = np.zeros(n)
        L_now = np.zeros(n)
        cnt = np.zeros(n, dtype=np.int64)
        ev_t, ev_row = [], []
        active = np.arange(n)
        while len(active):
            E = rng.exponential(size=len(active))
            target = L_now[active] + E * np.exp(-slope[active] * cnt[active])
            t = _invert(Lam, G, rate, active, target)
            hit = np.isfinite(t)
            ev_t.append(t[hit])
            ev_row.append(active[hit])
            t_now[active[hit]] = t[hit]
            L_now[active[hit]] = target[hit]
            cnt[active[hit]] += 1
            active = active[hit & (cnt[active] < MAX_PER_THREAD)]
        return np.concatenate(ev_row) if ev_row else np.zeros(0, int), np.concatenate(ev_t) if ev_t else np.zeros(0)

    # ---------------- comment and reply-back streams ----------------
    def _comment_streams(self, C: dict, T0, z, rng):
        """C: arrays thread, s, depth, op, pos (static covariates) for one generation of comments.
        Returns (parent index, u, is_reply_back) of children."""
        p, H = self.p, self.H
        nC = len(C["thread"])
        span = H - C["s"]
        start = T0[C["thread"]] + C["s"]
        # reply-back stream: inhomogeneous Poisson, inverse CDF of the log-normal kernel
        bb = p["beta_back"]
        xb = bb[0] * (C["depth"] == 1) + bb[1] * (C["depth"] >= 3)
        m_s, sg = np.log(p["back_median_min"] * 60.0), p["back_sigma"]
        Gspan = norm.cdf((np.log(np.maximum(span, 1e-9)) - m_s) / sg)
        nb = rng.poisson(p["rho"] * np.exp(xb) * Gspan)
        rb_par = np.repeat(np.arange(nC), nb)
        rb_u = np.exp(m_s + sg * norm.ppf(rng.random(len(rb_par)) * Gspan[rb_par]))
        # comment stream base rate on per-comment grids, in chunks
        beta = p["beta"]
        static = (beta["dep2"] * (C["depth"] == 2) + beta["dep3p"] * (C["depth"] >= 3) + beta["op"] * C["op"])
        posx = beta["pos"] * np.log(np.maximum(C["pos"], 1))
        out_par, out_u, out_rb = [rb_par], [rb_u], [np.ones(len(rb_par), bool)]
        # sorted reply-back times per comment (padded)
        order = np.lexsort((rb_u, rb_par))
        rb_par_s, rb_u_s = rb_par[order], rb_u[order]
        first_idx = np.searchsorted(rb_par_s, np.arange(nC))
        last_idx = np.searchsorted(rb_par_s, np.arange(nC), side="right")
        CH = 40000
        for c0 in range(0, nC, CH):
            idx = np.arange(c0, min(nC, c0 + CH))
            G = _grids(start[idx], span[idx], extra=1800.0 - C["s"][idx])
            mid = _mid(G)
            tau = C["s"][idx][:, None] + mid
            lr = (p["alpha_c"] + np.log(z[C["thread"][idx]])[:, None] + _clock(*p["k"], mid) + _clock(*p["a"], tau)
                  + self.hour[hour_of(start[idx][:, None] + mid)] + static[idx][:, None]
                  + posx[idx][:, None] * (tau <= 1800.0))
            rate = np.exp(lr)
            Lam = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(rate * np.diff(G, axis=1), axis=1)], axis=1)
            k = len(idx)
            u_now = np.zeros(k)
            n = np.zeros(k)
            ptr = first_idx[idx].copy()
            end = last_idx[idx]
            active = np.arange(k)
            Lrow = np.zeros(k)
            while len(active):
                if len(rb_u_s):
                    nxt_rb = np.where(ptr[active] < end[active], rb_u_s[np.minimum(ptr[active], len(rb_u_s) - 1)], np.inf)
                else:
                    nxt_rb = np.full(len(active), np.inf)
                E = rng.exponential(size=len(active))
                target = Lrow[active] + E / np.power(1.0 + n[active], beta["logn"])
                t = _invert(Lam, G, rate, active, target)
                ev = np.isfinite(t) & (t < nxt_rb)
                rbh = ~ev & np.isfinite(nxt_rb)
                a_ev, a_rb = active[ev], active[rbh]
                out_par.append(idx[a_ev])
                out_u.append(t[ev])
                out_rb.append(np.zeros(len(a_ev), bool))
                u_now[a_ev] = t[ev]
                Lrow[a_ev] = target[ev]
                n[a_ev] += 1
                # a reply-back arrives first: move to it (memoryless restart), n increases
                if len(a_rb):
                    u_now[a_rb] = nxt_rb[rbh]
                    Lrow[a_rb] = _lam_at(Lam, G, rate, a_rb, u_now[a_rb])
                    n[a_rb] += 1
                    ptr[a_rb] += 1
                active = active[(ev | rbh) & (n[active] < MAX_PER_THREAD)]
        return np.concatenate(out_par), np.concatenate(out_u), np.concatenate(out_rb)

    # ---------------- full simulation ----------------
    def simulate(self, roots: pl.DataFrame, rng: np.random.Generator) -> tuple[pl.DataFrame, dict]:
        """roots: thread_id, T0 (epoch s), post_type, question (bool). Returns normalised items and diagnostics."""
        p, H = self.p, self.H
        T0 = roots["T0"].to_numpy().astype(float)
        q = roots["question"].to_numpy().astype(float)
        nT = len(T0)
        z = rng.gamma(p["r"], 1.0 / p["r"], nT)
        A = AuthorModel(nT, p["pi_prev"], rng)
        rows_thread, rows_s, rows_depth, rows_author, rows_parent, rows_id = [], [], [], [], [], []
        # root stream -> top-level comments
        tr, tt = self._root_stream(T0, q, z, rng)
        if len(tr) > self.max_total:
            raise RuntimeError(f"generator: more than {self.max_total} top-level comments; parameters explosive?")
        order = np.lexsort((tt, tr))
        tr, tt = tr[order], tt[order]
        pos = _rank_within(tr)
        auth = A.assign(tr, tt, forbid1=np.zeros(len(tr), np.int64), forbid2=np.full(len(tr), -1), fixed=None)
        next_id = 0
        ids = np.arange(next_id, next_id + len(tr))
        next_id += len(tr)
        gen = {"thread": tr, "s": tt, "depth": np.ones(len(tr), np.int64), "op": (auth == 0).astype(float),
               "pos": pos, "author": auth, "parent_author": np.zeros(len(tr), np.int64), "id": ids,
               "parent": np.full(len(tr), -1)}
        n_per_thread = np.bincount(tr, minlength=nT)
        capped = 0
        total = 0
        while len(gen["thread"]):
            total += len(gen["thread"])
            if total > self.max_total:
                raise RuntimeError(f"generator: more than {self.max_total} comments; parameters explosive?")
            for k_, v in [("thread", rows_thread), ("s", rows_s), ("depth", rows_depth), ("author", rows_author),
                          ("parent", rows_parent), ("id", rows_id)]:
                v.append(gen[k_])
            par, u, isrb = self._comment_streams(gen, T0, z, rng)
            th = gen["thread"][par]
            tchild = gen["s"][par] + u
            keep = tchild < H
            par, u, isrb, th, tchild = par[keep], u[keep], isrb[keep], th[keep], tchild[keep]
            # per-thread guard
            room = MAX_PER_THREAD - n_per_thread
            o = np.lexsort((tchild, th))
            par, u, isrb, th, tchild = par[o], u[o], isrb[o], th[o], tchild[o]
            rk = _rank_within(th)
            ok = rk <= room[th]
            capped += int((~ok).sum())
            par, u, isrb, th, tchild = par[ok], u[ok], isrb[ok], th[ok], tchild[ok]
            n_per_thread += np.bincount(th, minlength=nT)
            # authors: reply-backs fixed; others drawn
            fixed = np.where(isrb, gen["parent_author"][par], -1)
            auth = A.assign(th, tchild, forbid1=gen["author"][par], forbid2=gen["parent_author"][par], fixed=fixed)
            # sibling rank among children of the same parent (all streams)
            o2 = np.lexsort((tchild, gen["id"][par]))
            rank = np.empty(len(par), np.int64)
            rank[o2] = _rank_within(gen["id"][par][o2])
            ids = np.arange(next_id, next_id + len(par))
            next_id += len(par)
            gen = {"thread": th, "s": tchild, "depth": gen["depth"][par] + 1, "op": (auth == 0).astype(float),
                   "pos": rank, "author": auth, "parent_author": gen["author"][par], "id": ids,
                   "parent": gen["id"][par]}
        th = np.concatenate(rows_thread) if rows_thread else np.zeros(0, int)
        tid = roots["thread_id"].to_numpy()
        com = pl.DataFrame({
            "thread_id": tid[th], "id": [f"c{i}" for i in np.concatenate(rows_id)],
            "parent_id": np.where(np.concatenate(rows_parent) < 0, tid[th].astype(object),
                                  np.array([f"c{i}" for i in np.concatenate(rows_parent)], dtype=object)).astype(str),
            "is_root": False, "created": T0[th] + np.concatenate(rows_s),
            "author": [f"a{t}_{a}" for t, a in zip(th, np.concatenate(rows_author))],
            "depth": np.concatenate(rows_depth).astype(np.int32), "obs_end": T0[th] + H,
            "post_type": roots["post_type"].to_numpy()[th]})
        rt = pl.DataFrame({"thread_id": tid, "id": tid, "parent_id": None, "is_root": True, "created": T0,
                           "author": [f"a{t}_0" for t in range(nT)], "depth": np.zeros(nT, np.int32),
                           "obs_end": T0 + H, "post_type": roots["post_type"].to_numpy()},
                          schema_overrides={"parent_id": pl.String})
        items = pl.concat([rt, com], how="vertical_relaxed")
        return items, {"z": z, "capped_events": capped, "max_thread_size": int(n_per_thread.max()) if nT else 0}


def b_pow(z, b):
    return b * np.log(z)


def _lam_at(Lam, G, rate, rows, t):
    """Lambda_row(t) for given rows and times inside the rows' grids (vectorised bisection on G)."""
    Gr = G[rows]
    ar = np.arange(len(rows))
    lo = np.zeros(len(rows), dtype=np.int64)
    hi = np.full(len(rows), G.shape[1] - 1, dtype=np.int64)
    while True:
        act = hi - lo > 1
        if not act.any():
            break
        m = (lo + hi) // 2
        go = Gr[ar, m] <= t
        lo = np.where(act & go, m, lo)
        hi = np.where(act & ~go, m, hi)
    return Lam[rows, lo] + rate[rows, lo] * (t - G[rows, lo])


def _rank_within(key: np.ndarray) -> np.ndarray:
    """1-based rank within runs of equal consecutive keys (key must be grouped)."""
    if len(key) == 0:
        return np.zeros(0, np.int64)
    start = np.r_[True, key[1:] != key[:-1]]
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(key)), 0))
    return np.arange(len(key)) - grp_start + 1


class AuthorModel:
    """Per-thread participants: author ids (0 = OP) with item times; draws authors for new items."""

    def __init__(self, nT: int, pi_prev: float, rng):
        self.times = [np.zeros(1) for _ in range(nT)]          # OP's post at thread age 0
        self.auth = [np.zeros(1, np.int64) for _ in range(nT)]
        self.next = np.ones(nT, np.int64)
        self.pi = pi_prev
        self.rng = rng

    def assign(self, th, t, forbid1, forbid2, fixed):
        out = np.empty(len(th), np.int64)
        if len(th) == 0:
            return out
        u = self.rng.random(len(th))
        order = np.lexsort((t, th))
        bounds = np.flatnonzero(np.r_[True, th[order][1:] != th[order][:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            T = th[idx[0]]
            times, auth = self.times[T], self.auth[T]
            new_t, new_a = [], []
            for i in idx:
                if fixed is not None and fixed[i] >= 0:
                    au = int(fixed[i])
                else:
                    au = -1
                    if u[i] < self.pi:
                        m = times < t[i]
                        cand = auth[m]
                        if new_t:
                            nt = np.array(new_t)
                            cand = np.r_[cand, np.array(new_a)[nt < t[i]]]
                        cand = cand[(cand != forbid1[i]) & (cand != forbid2[i])]
                        if len(cand):
                            au = int(cand[self.rng.integers(len(cand))])  # proportional to items so far
                    if au < 0:
                        au = int(self.next[T])
                        self.next[T] += 1
                out[i] = au
                new_t.append(t[i])
                new_a.append(au)
            self.times[T] = np.r_[times, np.array(new_t)]
            self.auth[T] = np.r_[auth, np.array(new_a, np.int64)]
        return out
