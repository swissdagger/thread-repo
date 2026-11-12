"""Stage 2 of the first model fit (docs/fit1_brief.md): fit and compare.

Models, all fitted on the inner fit fold with the current estimator (Rev-1, 5, 7; Rev-8 bins only for Moltbook):
  full  the two-clock model: comment stream + root stream with gamma fitness integrated out, plus the reply-back
        stream;
  B1    single-clock: the full model without the thread clock a(.) in the comment stream (nested);
  B3    two-clock without the reply-back stream: reply-backs pooled into the comment stream;
  B2    Gomez parent choice: conditional logit over the candidates present at each reply, features
        log(1 + replies so far), log recency rank (1 = newest), root indicator.
Comparisons on the inner held-out fold (pre-stated in docs/results_register.md, F2-*):
  timing log-likelihood per event (full vs B1, full vs B3), parent-choice log-likelihood per reply (full vs B2),
  top-3 parent accuracy (full, B2) and the depth-tail error P(depth >= 10) from simulated held-out threads
  (full, B1, B3), plus the model-implied mu and mu0 from the same simulations.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import fit1_estimator2 as E2
from scipy.special import gammaln
from scipy.stats import norm

from .fit1_estimator import (C_NAMES, GRID, _lx, frailty_terms, num_info, num_scores, rb_data, rb_unit_loglik,
                             stage_c)
from .fit1_generator import Generator
from .hazard import SortedCensor, mean_cumulative

H3 = 3 * 86400.0
B1_KEEP = [i for i, n in enumerate(C_NAMES) if n not in ("a1_log_tau", "a2_log_tau_sq")]
RB_NAMES = ["log_rho", "back_m", "log_back_sigma", "beta_back_root", "beta_back_dep3p"]
EV_CLIP = 0.5  # seconds, as in the estimator


# ------------------------------------------------------------------------------------------ models
def pooled(S: dict) -> dict:
    S3 = dict(S)
    S3["events_c"] = S["events_c"].with_columns(stream=pl.lit("comment"))
    return S3


def design(S: dict, model: str, with_time: bool = False, rev8: bool = False, pooled_events: bool = False) -> dict:
    """pooled_events: every reply event (comment-stream and reply-back) is an event row of the comment stream, with
    covariates at the event time; exposure rows are unchanged (cells are split at all reply events either way)."""
    D = E2.build(pooled(S) if (model == "B3" or pooled_events) else S, rev8=rev8, with_time=with_time)
    if model == "B1":
        D["C"]["X"] = np.ascontiguousarray(D["C"]["X"][:, B1_KEEP])
        D["C"]["names"] = [C_NAMES[i] for i in B1_KEEP]
    return D


def fit_re(S: dict, model: str, W: np.ndarray, rev8: bool = False) -> dict:
    D = design(S, model, rev8=rev8)
    M = E2.REModel2(D)
    th, conv = M.fit_multistart()
    I, sc = M.info(th), M.scores(th)
    reps = th[None, :] + ((W - 1.0) @ sc) @ np.linalg.inv(I).T
    return {"names": M.names, "coef": th, "reps": reps, "n_iter": conv["iterations"], "converged": conv["converged"],
            "start_b": conv["start_b"], "loglik": conv["loglik"]}


def fit_reply_back(S: dict, W: np.ndarray) -> dict:
    R = rb_data(S)
    th = stage_c(R)
    ss = num_scores(lambda t: rb_unit_loglik(t, R), th)
    sc = np.zeros((R["nT"], 5))
    for k in range(5):
        sc[:, k] = np.bincount(R["thr"], weights=ss[:, k], minlength=R["nT"])
    I = num_info(lambda t: float(rb_unit_loglik(t, R).sum()), th)
    return {"names": RB_NAMES, "coef": th, "reps": th[None, :] + ((W - 1.0) @ sc) @ np.linalg.inv(I).T}


# ------------------------------------------------------------------------------------------ held-out timing
def heldout_timing(S: dict, fits: dict, rb: dict) -> dict:
    """Per held-out thread: UNMARKED timing log-likelihood of the same events for every model. Events: every reply
    to a non-root comment (comment-stream and reply-back events alike, as "a reply to c") and every root-stream event,
    within the 3-day horizon. Full and B1 score a reply to c at t with the pooled intensity z lambda_c(t) +
    lambda_back,c(t); B3 with its pooled comment stream. Thread fitness is integrated out on the log-z grid. (The
    marked likelihood, which also predicts whether a reply is a reply-back, is not comparable with B3's.)"""
    R = rb_data(S)
    rb_idx = {k: i for i, k in enumerate(S["units"].filter(pl.col("parent_author").is_not_null())["id"].to_list())}
    lrho, m_back, ls_back, b_root, b_d3 = rb["coef"]
    s_back, mu_back = np.exp(ls_back), np.log(np.exp(m_back) * 60.0)
    lin = lrho + R["x"] @ np.array([b_root, b_d3])
    lam_back_tot = np.bincount(R["thr"], weights=np.exp(lin) * norm.cdf((np.log(R["U"]) - mu_back) / s_back), minlength=R["nT"])
    # reply events in design order (pooled): rows of events_c with u <= U, in events_c order
    units = S["units"]
    uid = {k: i for i, k in enumerate(units["id"].to_list())}
    ec = S["events_c"]
    ev_unit = np.array([uid[x] for x in ec["unit"].to_list()], dtype=np.int64)
    ev_u = np.clip(ec["u"].to_numpy(), 0.5, None)
    keep = ev_u <= units["U"].to_numpy()[ev_unit]
    ev_unit, ev_u = ev_unit[keep], ev_u[keep]
    rbi = np.array([rb_idx.get(k, -1) for k in units["id"].to_list()], dtype=np.int64)[ev_unit]
    lu = np.log(ev_u)
    log_back = np.where(rbi >= 0, lin[np.maximum(rbi, 0)] - lu - ls_back - 0.5 * np.log(2 * np.pi)
                        - (lu - mu_back) ** 2 / (2 * s_back ** 2), -np.inf)
    out = {}
    for m in ("full", "B1", "B3"):
        M = E2.REModel2(design(S, m, pooled_events=True))
        th = fits[m]["coef"]
        assert M.names == fits[m]["names"], m
        eta, bh, mu = M._mu(th)
        ex = M.y == 0
        A = np.bincount(M.thr[ex & M.is_c], weights=mu[ex & M.is_c], minlength=M.nT)
        Bt = np.bincount(M.thr[ex & ~M.is_c], weights=mu[ex & ~M.is_c], minlength=M.nT)
        evc = (M.y == 1) & M.is_c
        evr = (M.y == 1) & ~M.is_c
        assert evc.sum() == len(ev_unit), (m, evc.sum(), len(ev_unit))
        la = (eta + bh[M.sub_hour])[evc]          # log comment-stream rate without z, at each reply
        thr_c = M.thr[evc]
        lb = log_back if m != "B3" else np.full(len(la), -np.inf)
        root_ev = np.bincount(M.thr[evr], weights=(eta + bh[M.sub_hour])[evr], minlength=M.nT)
        Nc = np.bincount(thr_c, minlength=M.nT).astype(float)
        NR = np.bincount(M.thr[evr], minlength=M.nT).astype(float)
        r, b = np.exp(th[M.p]), th[M.p + 1]
        # grid around the mode of the pure-fitness version (as in frailty_terms), then the exact integrand
        c0 = r + Nc + b * NR
        m0 = np.log((r + Nc + NR) / (r + A + Bt))
        for _ in range(12):
            e1, eb = np.exp(np.clip(m0, -40, 40)), np.exp(np.clip(b * m0, -40, 40))
            g1 = c0 - (r + A) * e1 - b * Bt * eb
            h1 = -(r + A) * e1 - b * b * Bt * eb - 1e-12
            m0 = np.clip(m0 - np.clip(g1 / h1, -3, 3), -35, 35)
        sd = np.minimum(1.0 / np.sqrt((r + A) * np.exp(m0) + b * b * Bt * np.exp(np.clip(b * m0, -40, 40)) + 1e-12), 5.0)
        lz = np.clip(m0[:, None] + sd[:, None] * GRID[None, :], -40, 40)
        z, zb = np.exp(lz), np.exp(np.clip(b * lz, -60, 60))
        f = r * np.log(r) - gammaln(r) + r * lz - r * z - A[:, None] * z + b * NR[:, None] * lz - Bt[:, None] * zb
        terms = np.logaddexp(lz[thr_c] + la[:, None], lb[:, None])   # log(z a_i + b_i)
        for g in range(len(GRID)):
            f[:, g] += np.bincount(thr_c, weights=terms[:, g], minlength=M.nT)
        mx = f.max(axis=1)
        logI = mx + np.log(np.exp(f - mx[:, None]).sum(axis=1) * sd * (GRID[1] - GRID[0]))
        out[m] = logI + root_ev - (lam_back_tot if m != "B3" else 0.0)
        if m == "B3":  # sanity: equals the B3 model's own marginal log-likelihood
            out["B3_check_max_abs_diff"] = float(np.abs(out[m] - M.loglik_threads(th)).max())
        out["n_events"] = Nc + NR
        del M, eta, mu
    return out


# ------------------------------------------------------------------------------------------ parent choice
def _threads(S: dict):
    """Per thread (in roots order): units sorted by creation and reply events sorted by time."""
    roots = S["roots"]
    tid = roots["thread_id"].to_list()
    tmap = {t: i for i, t in enumerate(tid)}
    units = S["units"].with_columns(ti=pl.col("thread_id").replace_strict(tmap, return_dtype=pl.Int64)).sort("ti", "created", "id")
    uidx = {k: i for i, k in enumerate(units["id"].to_list())}
    ec = S["events_c"].join(units.select(pl.col("id").alias("unit"), "ti", "created"), on="unit").with_columns(
        t=pl.col("created") + pl.col("u"), parent=pl.col("unit").replace_strict(uidx, return_dtype=pl.Int64),
        is_c=pl.col("stream") == "comment")
    er = S["events_r"].join(roots.select("thread_id", "T0"), on="thread_id").with_columns(
        ti=pl.col("thread_id").replace_strict(tmap, return_dtype=pl.Int64), t=pl.col("T0") + pl.col("tau"),
        parent=pl.lit(-1, pl.Int64), is_c=pl.lit(False))
    ev = pl.concat([ec.select("ti", "t", "parent", "is_c"), er.select("ti", "t", "parent", "is_c")]).sort("ti", "t", "parent")
    U = {k: units[k].to_numpy() for k in ("ti", "created", "s", "depth", "op_flag", "pos")}
    U["rb_ok"] = units["parent_author"].is_not_null().to_numpy()
    V = {k: ev[k].to_numpy() for k in ("ti", "t", "parent", "is_c")}
    ub = np.searchsorted(U["ti"], np.arange(len(tid) + 1))
    vb = np.searchsorted(V["ti"], np.arange(len(tid) + 1))
    T0 = roots["T0"].to_numpy()
    for i in range(len(tid)):
        if vb[i + 1] == vb[i]:
            continue
        u0 = ub[i]
        yield i, T0[i], {k: v[ub[i]:ub[i + 1]] for k, v in U.items()}, \
            {"t": V["t"][vb[i]:vb[i + 1]], "parent": V["parent"][vb[i]:vb[i + 1]] - np.where(V["parent"][vb[i]:vb[i + 1]] >= 0, u0, 0),
             "is_c": V["is_c"][vb[i]:vb[i + 1]]}


def _hour(t):
    return (np.mod(t, 86400.0) // 3600).astype(np.int64)


def _cand_block(T0, u, v, e0, e1, base_n, base_ntop):
    """Candidate structure for events e0..e1: mask (E x K), n replies so far (E x K), n_top (E)."""
    t = v["t"][e0:e1]
    par = v["parent"][e0:e1]
    K = len(u["created"])
    mask = u["created"][None, :] < t[:, None]
    has_p = par >= 0
    mask[np.nonzero(has_p)[0], par[has_p]] = True
    oh = np.zeros((e1 - e0, K))
    oh[np.nonzero(has_p)[0], par[has_p]] = 1.0
    n = base_n[None, :] + np.cumsum(oh, axis=0) - oh
    is_top = ~has_p
    ntop = base_ntop + np.cumsum(is_top) - is_top
    return t, par, mask, n, ntop, oh.sum(axis=0), int(is_top.sum())


def b2_features(u, t, mask, n, ntop):
    """Gomez features for comments (E x K x 3) and the root (E x 3)."""
    K = mask.shape[1]
    # recency rank among present candidates: 1 = newest; the root is the oldest
    present = mask.astype(np.int64)
    newer = np.cumsum(present[:, ::-1], axis=1)[:, ::-1]  # units sorted by creation: candidates at or after k
    rank_c = np.maximum(newer, 1)
    Fc = np.stack([np.log1p(n), np.log(rank_c), np.zeros_like(n)], axis=2)
    Fr = np.stack([np.log1p(ntop), np.log(present.sum(axis=1) + 1.0), np.ones(len(t))], axis=1)
    return Fc, Fr


def fit_b2(S: dict, iters: int = 30) -> dict:
    """Conditional logit (Gomez) by Newton on the fit fold."""
    beta = np.zeros(3)
    for it in range(iters):
        g, Hm, ll = np.zeros(3), np.zeros((3, 3)), 0.0
        for i, T0, u, v in _threads(S):
            K, E = len(u["created"]), len(v["t"])
            bn, bt = np.zeros(K), 0
            for e0 in range(0, E, 400):
                e1 = min(E, e0 + 400)
                t, par, mask, n, ntop, dn, dt = _cand_block(T0, u, v, e0, e1, bn, bt)
                bn, bt = bn + dn, bt + dt
                Fc, Fr = b2_features(u, t, mask, n, ntop)
                sc = np.where(mask, Fc @ beta, -np.inf)
                sr = Fr @ beta
                mx = np.maximum(sc.max(axis=1, initial=-np.inf), sr)
                pc, pr = np.exp(sc - mx[:, None]), np.exp(sr - mx)
                Z = pc.sum(axis=1) + pr
                pc, pr = pc / Z[:, None], pr / Z
                Ft = np.where((par >= 0)[:, None], Fc[np.arange(len(t)), np.maximum(par, 0)], Fr)
                ll += float(np.sum(np.log(np.where(par >= 0, pc[np.arange(len(t)), np.maximum(par, 0)], pr))))
                Ex = np.einsum("ek,ekj->ej", pc, Fc) + pr[:, None] * Fr
                g += (Ft - Ex).sum(axis=0)
                Exx = np.einsum("ek,eki,ekj->ij", pc, Fc, Fc) + np.einsum("e,ei,ej->ij", pr, Fr, Fr)
                Hm -= Exx - Ex.T @ Ex
        step = np.linalg.solve(Hm - 1e-9 * np.eye(3), -g)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return {"names": ["b2_log_popularity", "b2_log_recency_rank", "b2_root"], "coef": beta, "loglik": ll,
            "n_iter": it + 1}


def parent_choice(S: dict, fit: dict, rb: dict, b2: dict) -> pl.DataFrame:
    """Per held-out thread: sum of log P(true parent) and top-3 hits for the full model and B2.
    Full model: P(c) = E[lambda_c | history] / E[sum lambda | history], with lambda_c = comment-stream rate (x fitness
    z) + reply-back rate, root rate x z^b; E[z], E[z^b] are the posterior means given the thread's comment-stream
    and root-stream events before the reply and their expected counts so far (exposure rows with midpoint
    before the reply)."""
    g = dict(zip(fit["names"], fit["coef"]))
    bc = np.array([g[n] for n in C_NAMES])
    hour = np.r_[0.0, [g[f"hour_{h}"] for h in range(1, 24)]]
    r, b = np.exp(g["log_r"]), g["b"]
    lrho, m_back, ls_back, bb_root, bb_d3 = rb["coef"]
    mu_back, s_back = np.log(np.exp(m_back) * 60.0), np.exp(ls_back)
    # expected counts so far (A: comment stream, B: root stream) per event, from the held-out design
    D = design(S, "full", with_time=True)
    M = E2.REModel2(D)
    _, _, mu = M._mu(fit["coef"])
    tm = np.r_[D["C"]["abs_mid"], D["R"]["abs_mid"]]
    expo = M.y == 0
    del D
    cum = {}
    for key, msk in (("A", expo & M.is_c), ("B", expo & ~M.is_c)):
        thr, tt, mm = M.thr[msk], tm[msk], mu[msk]
        o = np.lexsort((tt, thr))
        cum[key] = (thr[o], tt[o], np.cumsum(mm[o]))
    del M, mu, tm

    def so_far(key, ti, t):
        thr, tt, cs = cum[key]
        lo, hi = np.searchsorted(thr, ti), np.searchsorted(thr, ti, side="right")
        k = lo + np.searchsorted(tt[lo:hi], t, side="left")
        base = cs[lo - 1] if lo > 0 else 0.0
        return np.where(k > lo, cs[np.maximum(k - 1, 0)] - base, 0.0)
    rows = []
    for i, T0, u, v in _threads(S):
        K, E = len(u["created"]), len(v["t"])
        bn, bt, nc_before = np.zeros(K), 0, 0
        ll_m = ll_b = 0.0
        top3_m = top3_b = n4 = top3_m4 = top3_b4 = 0
        for e0 in range(0, E, 400):
            e1 = min(E, e0 + 400)
            t, par, mask, n, ntop, dn, dt = _cand_block(T0, u, v, e0, e1, bn, bt)
            bn, bt = bn + dn, bt + dt
            isc = v["is_c"][e0:e1]
            Nc_t = nc_before + np.cumsum(isc) - isc
            nc_before += int(isc.sum())
            post = frailty_terms(Nc_t.astype(float), ntop.astype(float), so_far("A", i, t), so_far("B", i, t), r, b)
            Ez, Ezb = post[1], post[2]
            uage = np.maximum(t[:, None] - u["created"][None, :], EV_CLIP)
            tau = np.maximum(t - T0, EV_CLIP)
            lu, lt = _lx(uage), _lx(tau)[:, None] + 0 * uage
            ltc = _lx(np.maximum(u["s"][None, :] + uage, EV_CLIP))
            X = np.stack([lu, lu ** 2, ltc, ltc ** 2, np.broadcast_to(u["depth"] == 2, uage.shape),
                          np.broadcast_to(u["depth"] >= 3, uage.shape), np.broadcast_to(u["op_flag"], uage.shape),
                          np.log1p(n), np.log(np.maximum(u["pos"], 1))[None, :] * ((t - T0)[:, None] <= 1800.0)], axis=2)
            eta_c = X @ bc + g["alpha_c"] + hour[_hour(t)][:, None]
            lam = Ez[:, None] * np.exp(eta_c)
            lfl = -np.log(uage) - ls_back - 0.5 * np.log(2 * np.pi) - (np.log(uage) - mu_back) ** 2 / (2 * s_back ** 2)
            xb = lrho + bb_root * (u["depth"] == 1) + bb_d3 * (u["depth"] >= 3)
            lam = lam + np.where(u["rb_ok"][None, :], np.exp(xb[None, :] + lfl), 0.0)
            lam = np.where(mask, lam, 0.0)
            eta_r = g["aR1"] * _lx(tau) + g["aR2"] * _lx(tau) ** 2 + g["gamma"] * ntop + g["alpha_R"] + hour[_hour(t)]
            lam_r = Ezb * np.exp(eta_r)
            Z = lam.sum(axis=1) + lam_r
            ar = np.arange(len(t))
            lt_true = np.where(par >= 0, lam[ar, np.maximum(par, 0)], lam_r)
            ll_m += float(np.sum(np.log(lt_true / Z)))
            rank_m = 1 + (lam > lt_true[:, None]).sum(axis=1) + (lam_r > lt_true)
            # B2
            Fc, Fr = b2_features(u, t, mask, n, ntop)
            sc = np.where(mask, Fc @ b2["coef"], -np.inf)
            sr = Fr @ b2["coef"]
            mx = np.maximum(sc.max(axis=1, initial=-np.inf), sr)
            pc, pr = np.exp(sc - mx[:, None]), np.exp(sr - mx)
            st = np.where(par >= 0, pc[ar, np.maximum(par, 0)], pr)
            ll_b += float(np.sum(np.log(st / (pc.sum(axis=1) + pr))))
            rank_b = 1 + (pc > st[:, None]).sum(axis=1) + (pr > st)
            ncand = mask.sum(axis=1) + 1
            top3_m += int((rank_m <= 3).sum())
            top3_b += int((rank_b <= 3).sum())
            big = ncand >= 4
            n4 += int(big.sum())
            top3_m4 += int((rank_m[big] <= 3).sum())
            top3_b4 += int((rank_b[big] <= 3).sum())
        rows.append({"ti": i, "events": E, "ll_full": ll_m, "ll_b2": ll_b, "top3_full": top3_m, "top3_b2": top3_b,
                     "events_4plus": n4, "top3_full_4plus": top3_m4, "top3_b2_4plus": top3_b4})
    return pl.DataFrame(rows)


# ------------------------------------------------------------------------------------------ simulation metrics
def gen_params(fit: dict, rb: dict, model: str) -> dict:
    g = dict(zip(fit["names"], fit["coef"]))
    return {"alpha_c": g["alpha_c"], "alpha_R": g["alpha_R"], "r": float(np.exp(g["log_r"])), "b": g["b"],
            "k": (g["k1_log_u"], g["k2_log_u_sq"]), "a": (g.get("a1_log_tau", 0.0), g.get("a2_log_tau_sq", 0.0)),
            "aR": (g["aR1"], g["aR2"]), "hour": np.r_[0.0, [g[f"hour_{h}"] for h in range(1, 24)]],
            "beta": {"dep2": g["dep2"], "dep3p": g["dep3p"], "op": g["op"], "logn": g["logn"], "pos": g["pos_early"]},
            "gamma": g["gamma"], "kappa": 0.0, "rho": 0.0 if model == "B3" else float(np.exp(rb["coef"][0])),
            "back_median_min": float(np.exp(rb["coef"][1])), "back_sigma": float(np.exp(rb["coef"][2])),
            "beta_back": [float(rb["coef"][3]), float(rb["coef"][4])], "pi_prev": 0.2}  # authors unused here


def thread_stats(items: pl.DataFrame, roots: pl.DataFrame, observed: bool) -> dict:
    """Per thread (roots order): comments within 3 d and those at depth >= 10; arrays for mu (non-root parents) and
    mu0 (roots) with censoring at thread age 3 d (and the crawl, for observed data)."""
    tid = roots["thread_id"].to_list()
    tmap = {t: i for i, t in enumerate(tid)}
    it = items.filter(pl.col("thread_id").is_in(tid)).join(roots.select("thread_id", "T0"), on="thread_id")
    com = it.filter(~pl.col("is_root") & (pl.col("created") - pl.col("T0") < H3) & (pl.col("created") >= pl.col("T0")))
    com = com.with_columns(ti=pl.col("thread_id").replace_strict(tmap, return_dtype=pl.Int64))
    nT = len(tid)
    n_com = np.bincount(com["ti"].to_numpy(), minlength=nT)
    n_deep = np.bincount(com.filter(pl.col("depth") >= 10)["ti"].to_numpy(), minlength=nT)
    end = (pl.min_horizontal(pl.col("T0") + H3, pl.col("obs_end")) if observed else pl.col("T0") + H3)
    par = com.with_columns(c=end - pl.col("created")).filter(pl.col("c") > 0)
    pa = it.select(pl.col("id").alias("parent_id"), pauthor="author", p_root="is_root", pcreated="created")
    kids = com.join(pa, on="parent_id", how="inner").with_columns(delay=pl.col("created") - pl.col("pcreated"))
    if observed:  # non-self replies, as in the audit (bots are already dropped)
        kids = kids.filter(pl.col("author") != pl.col("pauthor"))
    k_nr = kids.filter(~pl.col("p_root")).join(par.select(pl.col("id").alias("parent_id"), "c"), on="parent_id")
    k_nr = k_nr.filter(pl.col("delay") <= pl.col("c"))
    rts = roots.join(it.filter(pl.col("is_root")).select("thread_id", "obs_end", op="author"), on="thread_id", how="left")
    c0 = np.minimum(H3, (rts["obs_end"] - rts["T0"]).to_numpy()) if observed else np.full(nT, H3)
    k_r = kids.filter(pl.col("p_root"))
    if observed:  # non-OP top-level comments
        k_r = k_r.join(rts.select("thread_id", "op"), on="thread_id").filter(pl.col("author") != pl.col("op"))
    k_r = k_r.with_columns(ti=pl.col("thread_id").replace_strict(tmap, return_dtype=pl.Int64))
    return {"n_com": n_com, "n_deep": n_deep,
            "mu": {"c": par["c"].to_numpy(), "pt": par["ti"].to_numpy(), "r": k_nr["delay"].to_numpy(),
                   "rt": k_nr["ti"].to_numpy()},
            "mu0": {"c": c0, "pt": np.arange(nT), "r": k_r["delay"].to_numpy(), "rt": k_r["ti"].to_numpy()}}


def simulate_stats(params: dict, roots: pl.DataFrame, reps: int, seed: list[int]) -> tuple[list[dict], dict]:
    out, diag = [], {"capped_events": 0, "max_thread_size": 0, "failed": 0}
    gen = Generator(params)
    for k in range(reps):
        rng = np.random.default_rng(seed + [k])
        try:
            items, d = gen.simulate(roots, rng)
        except RuntimeError:
            diag["failed"] += 1
            continue
        diag["capped_events"] += int(d["capped_events"])
        diag["max_thread_size"] = max(diag["max_thread_size"], int(d["max_thread_size"]))
        out.append(thread_stats(items, roots, observed=False))
    return out, diag


def mcf(st: dict, w: np.ndarray) -> float:
    return mean_cumulative(SortedCensor(st["c"]), w[st["pt"]], st["r"], w[st["rt"]], H3)


def boot_weights(nT: int, B: int, seed: list[int]) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(nT, np.full(nT, 1.0 / nT), size=B).astype(float)


def interval(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)
