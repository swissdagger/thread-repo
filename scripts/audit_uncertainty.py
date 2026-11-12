"""Audit: definitions-consistent mu and thread-clustered bootstrap intervals for Steps 2-3b.

Usage: python scripts/audit_uncertainty.py [mu] [weibull] [step2] [early_late] [lognormal] [table]

  mu         mu = censoring-adjusted mean offspring (MCF, horizons 3 d and 7 d), p_reply (Kaplan-Meier,
             3 d), median first-reply delay (KM, among replied by 3 d); raw and per-author; B = 200
  weibull    Weibull first-reply fit (mu_first, shape, half-life) on the capped fit fold; B = 100,
             warm-started (non-root units)
  step2      Step 2 headline numbers (bots removed; full and size-matched populations; raw and
             per-author); B = 200
  early_late Step 3b early/late shares (1-day horizon) and human/Moltbook ratios; B = 200
  lognormal  held-out log-likelihood difference, log-normal minus Weibull; B = 200 (held-out fold)
  table      the mu comparison table (outputs/audit/mu_definitions.csv)

All bootstraps resample THREADS with replacement (multinomial counts), seed from partitions.json.
Step 4 intervals and clustered SEs are produced by scripts/step4_position.py itself.
Populations, parents and events: forum_model.units (HN dead items excluded since check 6).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model import structure as S  # noqa: E402
from forum_model.units import (build_units, cap_fit, hn_population, load, moltbook_population,  # noqa: E402
                               reddit_population)

P = load_partitions()
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED = P["seed"]
DAY = 86400.0


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def populations(units=("nonroot", "root_live", "depth1", "depth2plus")):
    """(dataset, community, unit, U, it) over the step-3 populations; HN dead excluded (audit default)."""
    for sub in P["reddit"]["subreddits"]:
        it, f = reddit_population(sub)
        for u in units:
            if u in f:
                yield "reddit", sub, u, build_units(it, f[u], f"reddit:{sub}")
        del it
        gc.collect()
    hn = load("hn", None, ["body_status"])
    for comm in ["ask_hn", "link", "show_hn"]:
        it, f = hn_population(comm, hn_items=hn)
        for u in units:
            if u in f:
                yield "hn", comm, u, build_units(it, f[u], f"hn:{comm}")
    del hn
    gc.collect()
    it, f = moltbook_population("complete")
    for u in units:
        if u in f:
            yield "moltbook_crawl", "moltbook", u, build_units(it, f[u], "moltbook_crawl")


# ------------------------------------------------------------------------------------------
# 1. mu (MCF), p_reply, median delay
# ------------------------------------------------------------------------------------------

def run_mu(B: int = 200):
    rows = []
    rng = np.random.default_rng(SEED)
    for ds, comm, unit, U in populations():
        par, rep = U["parents"], U["replies"]
        _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
        G = int(tix.max()) + 1
        pid = {k: i for i, k in enumerate(par["id"].to_list())}
        rpar = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
        c, t, d = par["c"].to_numpy(), par["t"].to_numpy(), par["d"].to_numpy()
        r = rep["delay"].to_numpy()
        sc = H.SortedCensor(c)
        km = H.FirstReplyKM(t, d, 3 * DAY)
        # precompute risk-set indices for the two MCF horizons
        prec = {}
        for hz in (3 * DAY, 7 * DAY):
            m = r <= hz
            prec[hz] = (np.nonzero(m)[0], np.searchsorted(sc.cs, r[m], side="left"))

        def stats(w):
            cw = np.r_[0.0, np.cumsum(w[sc.order])]
            out = []
            for hz in (3 * DAY, 7 * DAY):
                idx, k = prec[hz]
                Y = cw[-1] - cw[k]
                wr = w[rpar[idx]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    out.append(float(np.nansum(np.where(Y > 0, wr / Y, 0.0))))
            p, med = km.p_reply_and_median(w)
            return out + [p, med]

        for wlab in ["raw", "per_author"]:
            w0 = par["w_pa"].to_numpy() if wlab == "per_author" else np.ones(par.height)
            est = stats(w0)
            reps = np.array([stats(w0 * rng.multinomial(G, np.full(G, 1.0 / G))[tix]) for _ in range(B)])
            row = {"dataset": ds, "community": comm, "unit": unit, "weighting": wlab, "parents": par.height,
                   "threads": G, "bootstrap_replicates": B}
            for j, name in enumerate(["mu_mcf_3d", "mu_mcf_7d", "p_reply_3d", "median_delay_s"]):
                lo, hi = ci(reps[:, j])
                row.update({name: est[j], f"{name}_ci_lo": lo, f"{name}_ci_hi": hi})
            rows.append(row)
            print(f"[mu] {comm} {unit} {wlab}: mu3d={est[0]:.4f} [{row['mu_mcf_3d_ci_lo']:.4f},{row['mu_mcf_3d_ci_hi']:.4f}] "
                  f"p={est[2]:.3f} med={est[3]:.0f}s", flush=True)
    pl.DataFrame(rows).write_csv(OUT / "mu_bootstrap.csv")


# ------------------------------------------------------------------------------------------
# 2. Weibull first-reply bootstrap (non-root)
# ------------------------------------------------------------------------------------------

def run_weibull(B: int = 100):
    rows = []
    rng = np.random.default_rng(SEED + 7)
    for ds, comm, unit, U in populations(units=("nonroot", "depth1", "depth2plus")):
        fit = cap_fit(U["parents"])
        t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
        _, tix = np.unique(fit["thread_id"].to_numpy(), return_inverse=True)
        G = int(tix.max()) + 1
        f0 = H.fit_first_reply("weibull", t, d)
        reps = []
        for b in range(B):
            w = rng.multinomial(G, np.full(G, 1.0 / G))[tix].astype(float)
            fb = H.fit_first_reply_from("weibull", t, d, w, f0.theta)
            reps.append((fb.mu, float(np.exp(fb.theta[0])), H.half_life("weibull", fb.theta) / 3600))
        R = np.array(reps)
        est = (f0.mu, float(np.exp(f0.theta[0])), H.half_life("weibull", f0.theta) / 3600)
        row = {"dataset": ds, "community": comm, "unit": unit, "n_fit": fit.height, "threads": G, "bootstrap_replicates": B}
        for j, name in enumerate(["mu_first_weibull", "weibull_shape", "half_life_h"]):
            lo, hi = ci(R[:, j])
            row.update({name: est[j], f"{name}_ci_lo": lo, f"{name}_ci_hi": hi})
        rows.append(row)
        print(f"[weibull] {comm} {unit}: mu={est[0]:.4f} [{row['mu_first_weibull_ci_lo']:.4f},"
              f"{row['mu_first_weibull_ci_hi']:.4f}] shape={est[1]:.3f}", flush=True)
        pl.DataFrame(rows).write_csv(OUT / "weibull_bootstrap.csv")


# ------------------------------------------------------------------------------------------
# 3. Step 2 headline numbers
# ------------------------------------------------------------------------------------------

def _wq(values: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile for non-negative integer values via bincount."""
    if w.sum() <= 0:
        return np.nan
    h = np.bincount(values, weights=w)
    cdf = np.cumsum(h) / h.sum()
    return float(np.searchsorted(cdf, q, side="left"))


def _sarle_w(x: np.ndarray, w: np.ndarray) -> float:
    n = w.sum()
    if n < 4:
        return np.nan
    m = np.average(x, weights=w)
    d = x - m
    s2 = np.average(d ** 2, weights=w)
    if s2 <= 0:
        return np.nan
    g = np.average(d ** 3, weights=w) / s2 ** 1.5 * np.sqrt(n * (n - 1)) / (n - 2)
    k = np.average(d ** 4, weights=w) / s2 ** 2 - 3
    k = ((n + 1) * k + 6) * (n - 1) / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def step2_tables(items: pl.DataFrame, live_expr: pl.Expr, X: set[str]):
    """Per-thread and per-(thread, author) arrays for the calibration split (bots removed)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2", ROOT / "scripts" / "step2_structure.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    roots = s2.roots_table(items, live_expr).filter(pl.col("split") == "calibration")
    com = items.filter(~pl.col("is_root") & (pl.col("split") == "calibration")).join(
        roots.select("thread_id"), on="thread_id", how="semi")
    R = S.retained(com, X)
    R, roots = S.with_filtered_children(R, roots)
    t = S.thread_table(R, roots)
    t = t.with_row_index("tix")
    tix_map = t.select("thread_id", "tix")
    Rt = R.join(tix_map, on="thread_id", how="inner")
    # reciprocity pairs per thread
    Xl = list(X)
    e = Rt.filter(pl.col("author").is_not_null() & pl.col("parent_author").is_not_null()
                  & (pl.col("author") != pl.col("parent_author")) & ~pl.col("parent_author").is_in(Xl)).select(
        "tix", lo=pl.min_horizontal("author", "parent_author"), hi=pl.max_horizontal("author", "parent_author"),
        fwd=pl.col("author") < pl.col("parent_author"))
    pairs = e.group_by("tix", "lo", "hi").agg(f=pl.col("fwd").any(), r=(~pl.col("fwd")).any()).group_by("tix").agg(
        n_pairs=pl.len(), n_recip=(pl.col("f") & pl.col("r")).sum())
    t = t.join(pairs, on="tix", how="left").with_columns(pl.col("n_pairs", "n_recip").fill_null(0))
    per_thread = Rt.group_by("tix").agg(
        n_c=pl.len(), kids_c=pl.col("kids").sum(), with_reply=(pl.col("kids") > 0).sum(),
        n_d1=(pl.col("depth") == 1).sum(), kids_d1=pl.col("kids").filter(pl.col("depth") == 1).sum(),
        n_d2=(pl.col("depth") == 2).sum(), kids_d2=pl.col("kids").filter(pl.col("depth") == 2).sum())
    t = t.join(per_thread, on="tix", how="left").with_columns(
        pl.col("n_c", "kids_c", "with_reply", "n_d1", "kids_d1", "n_d2", "kids_d2").fill_null(0)).sort("tix")
    ta = Rt.filter(pl.col("author").is_not_null()).group_by("tix", "author").agg(
        n=pl.len(), kids=pl.col("kids").sum(), with_reply=(pl.col("kids") > 0).sum())
    ta = ta.with_columns(author_total=pl.col("n").sum().over("author"))
    return t, ta


def step2_stats(t: pl.DataFrame, ta: pl.DataFrame, mt: np.ndarray, X: set[str]) -> dict:
    """All headline numbers under thread multipliers mt (per thread, aligned with t.tix)."""
    out = {}
    size = t["size"].to_numpy().astype(int)
    md = t["max_depth"].to_numpy().astype(int)
    wa = t["w_author"].to_numpy()
    has = size > 0
    for lab, w in [("raw", mt), ("per_author", mt * wa)]:
        W = w.sum()
        out[f"size_mean_{lab}"] = float((w * size).sum() / W)
        out[f"size_q50_{lab}"] = _wq(size, w, 0.5)
        out[f"size_q99_{lab}"] = _wq(size, w, 0.99)
        out[f"size_share_ge20_{lab}"] = float((w * (size >= 20)).sum() / W)
        out[f"maxdepth_q90_{lab}"] = _wq(md[has], w[has], 0.9)
        out[f"maxdepth_q99_{lab}"] = _wq(md[has], w[has], 0.99)
    kids_root = t["kids"].to_numpy()
    live = t["live"].to_numpy()
    out["root_fanout_all"] = float((mt * kids_root).sum() / mt.sum())
    out["root_fanout_live"] = float((mt * kids_root * live).sum() / (mt * live).sum()) if (mt * live).sum() > 0 else np.nan
    n_c, kc, wr = t["n_c"].to_numpy(), t["kids_c"].to_numpy(), t["with_reply"].to_numpy()
    out["mean_replies_nonroot"] = float((mt * kc).sum() / (mt * n_c).sum())
    out["share_with_reply"] = float((mt * wr).sum() / (mt * n_c).sum())
    out["fanout_d1"] = float((mt * t["kids_d1"].to_numpy()).sum() / max((mt * t["n_d1"].to_numpy()).sum(), 1e-12))
    out["fanout_d2"] = float((mt * t["kids_d2"].to_numpy()).sum() / max((mt * t["n_d2"].to_numpy()).sum(), 1e-12))
    # per-author comment-level: each comment weighted 1 / (its author's comments in the FULL sample);
    # weights are held fixed across replicates (a smooth ratio estimator; the point estimate equals the
    # mean over authors of author means). Recomputing author means inside each replicate biases the
    # bootstrap, because single-thread authors drop out of replicates more often than prolific ones.
    wta = mt[ta["tix"].to_numpy()]
    inv_n = 1.0 / ta["author_total"].to_numpy()
    num_k = (wta * inv_n * ta["kids"].to_numpy()).sum()
    num_r = (wta * inv_n * ta["with_reply"].to_numpy()).sum()
    den = (wta * inv_n * ta["n"].to_numpy()).sum()
    out["mean_replies_nonroot_per_author"] = float(num_k / den)
    out["share_with_reply_per_author"] = float(num_r / den)
    # OP return (threads with >= 1 non-OP comment; OP not excluded)
    ra = t["root_author"].to_list()
    op_ok = np.array([(a is None) or (a not in X) for a in ra])
    elig = (t["nonop_comments"].to_numpy() > 0) & op_ok
    ret = t["op_comments"].to_numpy() > 0
    comp = t["complete_authorship"].to_numpy()
    for lab, m in [("complete", elig & comp), ("all", elig)]:
        out[f"op_return_{lab}"] = float((mt * (m & ret)).sum() / max((mt * m).sum(), 1e-12))
    pairs, rec = t["n_pairs"].to_numpy(), t["n_recip"].to_numpy()
    out["reciprocity_thread_all"] = float((mt * rec).sum() / max((mt * pairs).sum(), 1e-12))
    out["reciprocity_thread_complete"] = float((mt * rec * comp).sum() / max((mt * pairs * comp).sum(), 1e-12))
    big = size >= 20
    dc = t["distinct_commenters"].to_numpy()
    out["bimod_bc_log_distinct"] = _sarle_w(np.log10(np.maximum(dc[big], 1)), mt[big])
    return out


def run_step2(B: int = 200):
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2", ROOT / "scripts" / "step2_structure.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    rng = np.random.default_rng(SEED + 11)
    rows = []
    jobs = []
    for sub in P["reddit"]["subreddits"]:
        jobs.append(("reddit", sub, lambda sub=sub: s2.load("reddit", ["removed_by_category"], [sub]),
                     pl.col("removed_by_category").is_null(), {"full": None, "lt100": 100}))
    jobs += [("hn", c, (lambda c=c: _hn_items(s2, c)), pl.col("body_status") == "ok", {"full": None, "lt100": 100})
             for c in ["ask_hn", "link", "show_hn"]]
    jobs.append(("moltbook_crawl", "moltbook", lambda: _molt_items(s2), pl.lit(True), {"complete": "complete"}))
    for ds, comm, loader, live, pops in jobs:
        items = loader()
        X = s2.exclusion_sets(items.filter(~pl.col("is_root")), ["base"])["base"]
        if ds == "moltbook_crawl":
            items = items.filter(pl.col("thread_id").is_in(
                items.filter(pl.col("is_root") & (pl.col("thread_size") >= pl.col("claimed_comments")))["thread_id"].implode()))
        t, ta = step2_tables(items, live, X)
        del items
        gc.collect()
        for pop, cap in pops.items():
            mask = np.ones(t.height, dtype=float) if cap in (None, "complete") else (t["size_full"].to_numpy() < cap).astype(float)
            est = step2_stats(t, ta, mask, X)
            G = t.height
            reps = [step2_stats(t, ta, mask * rng.multinomial(G, np.full(G, 1.0 / G)).astype(float), X) for _ in range(B)]
            for k, v in est.items():
                lo, hi = ci([r[k] for r in reps])
                rows.append({"dataset": ds, "community": comm, "population": pop, "metric": k, "estimate": v,
                             "ci_lo": lo, "ci_hi": hi, "bootstrap_replicates": B})
            print(f"[step2] {comm} {pop}: done ({len(est)} metrics)", flush=True)
        pl.DataFrame(rows).write_csv(OUT / "step2_bootstrap.csv")


def _hn_items(s2, comm):
    items = s2.load("hn", ["body_status"])
    dead_roots = items.filter(pl.col("is_root") & (pl.col("body_status") == "deleted"))["thread_id"]
    items = items.filter((pl.col("community") == comm) & ~pl.col("thread_id").is_in(dead_roots.implode()))
    return items


def _molt_items(s2):
    items = s2.load("moltbook_crawl", ["claimed_comments"]).with_columns(community=pl.lit("moltbook"))
    spam = pl.read_parquet(ROOT / "data" / "processed" / "moltbook_crawl" / "spam_flags.parquet").filter(pl.col("spam"))
    return items.filter(~pl.col("thread_id").is_in(spam["thread_id"].implode()))


# ------------------------------------------------------------------------------------------
# 4. early / late shares, 5. log-normal held-out difference
# ------------------------------------------------------------------------------------------

def run_early_late(B: int = 200):
    rng = np.random.default_rng(SEED + 13)
    rows, reps_all = [], {}
    for ds, comm, unit, U in populations(units=("nonroot",)):
        par = U["parents"].filter(pl.col("c") >= DAY)
        na = par.filter(pl.col("author").is_not_null()).group_by("author").agg(_n=pl.len())
        par = par.join(na, on="author", how="left").with_columns(
            wpa=pl.when(pl.col("author").is_null()).then(0.0).otherwise(1.0 / pl.col("_n")))
        _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
        G = int(tix.max()) + 1
        t, d = par["t"].to_numpy(), par["d"].to_numpy() & (par["t"].to_numpy() < DAY)
        win = [d & (t < 600), d & (t >= 600) & (t < 3600), d & (t >= 3600) & (t < DAY)]
        for wlab, w0 in [("raw", np.ones(par.height)), ("per_author", par["wpa"].to_numpy())]:
            def stat(w):
                W = w.sum()
                return [float((w * x).sum() / W) for x in win]
            est = stat(w0)
            R = np.array([stat(w0 * rng.multinomial(G, np.full(G, 1.0 / G))[tix]) for _ in range(B)])
            reps_all[(comm, wlab)] = R
            for j, name in enumerate(["lt_10min", "10_60min", "1h_1d"]):
                lo, hi = ci(R[:, j])
                rows.append({"community": comm, "weighting": wlab, "window": name, "per_comment_share": est[j],
                             "ci_lo": lo, "ci_hi": hi})
        print(f"[early_late] {comm} done", flush=True)
    # ratios human / Moltbook from independent replicates
    for (comm, wlab), R in reps_all.items():
        if comm == "moltbook":
            continue
        M = reps_all[("moltbook", wlab)]
        for j, name in enumerate(["lt_10min", "10_60min", "1h_1d"]):
            est = [r for r in rows if r["community"] == comm and r["weighting"] == wlab and r["window"] == name][0]
            em = [r for r in rows if r["community"] == "moltbook" and r["weighting"] == wlab and r["window"] == name][0]
            lo, hi = ci(R[:, j] / M[:, j])
            rows.append({"community": comm, "weighting": wlab, "window": f"ratio_vs_moltbook_{name}",
                         "per_comment_share": est["per_comment_share"] / em["per_comment_share"], "ci_lo": lo, "ci_hi": hi})
    pl.DataFrame(rows).write_csv(OUT / "early_late_bootstrap.csv")


def run_lognormal(B: int = 200):
    """Held-out LL difference per comment (log-normal - Weibull), parameters as fitted in step 3/3b."""
    rng = np.random.default_rng(SEED + 17)
    rows = []
    comp = pl.concat([pl.read_csv(ROOT / "outputs" / ds / "step3b_competitors.csv", infer_schema_length=None)
                      for ds in ["reddit", "hn", "moltbook_crawl"]], how="diagonal_relaxed")
    fits = pl.concat([pl.read_csv(ROOT / "outputs" / ds / "step3_fits.csv", infer_schema_length=None)
                      for ds in ["reddit", "hn", "moltbook_crawl"]], how="diagonal_relaxed")
    for ds, comm, unit, U in populations(units=("nonroot",)):
        held = U["parents"].filter(pl.col("fold") == "heldout")
        t, d = np.maximum(held["t"].to_numpy(), H.TMIN), held["d"].to_numpy()
        ln = comp.filter((pl.col("community") == comm) & (pl.col("unit") == unit) & (pl.col("family") == "lognormal")).row(0, named=True)
        wb = fits.filter((pl.col("community") == comm) & (pl.col("unit") == unit) & (pl.col("fit_type") == "first_reply")
                         & (pl.col("family") == "weibull")).row(0, named=True)
        th_ln = np.array([np.log(ln["lognormal_median_h"] * 3600), np.log(ln["lognormal_sigma"])])
        th_wb = np.array([np.log(wb["weibull_shape"]), np.log(wb["weibull_scale_h"] * 3600)])
        lg1, G1 = H.FAMILIES["lognormal"](th_ln, t)
        lg2, G2 = H.FAMILIES["weibull"](th_wb, t)
        diff = (d * (np.log(ln["mu"]) + lg1) - ln["mu"] * G1) - (d * (np.log(wb["mu"]) + lg2) - wb["mu"] * G2)
        _, tix = np.unique(held["thread_id"].to_numpy(), return_inverse=True)
        Gn = int(tix.max()) + 1
        reps = [float((diff * (m := rng.multinomial(Gn, np.full(Gn, 1.0 / Gn))[tix])).sum() / m.sum()) for _ in range(B)]
        lo, hi = ci(reps)
        rows.append({"dataset": ds, "community": comm, "unit": unit, "heldout_parents": held.height,
                     "ll_diff_lognormal_minus_weibull": float(diff.mean()), "ci_lo": lo, "ci_hi": hi})
        print(f"[lognormal] {comm}: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    pl.DataFrame(rows).write_csv(OUT / "lognormal_heldout_bootstrap.csv")


# ------------------------------------------------------------------------------------------
# 6. the mu comparison table
# ------------------------------------------------------------------------------------------

def run_table():
    mu = pl.read_csv(OUT / "mu_bootstrap.csv")
    wb = pl.read_csv(OUT / "weibull_bootstrap.csv") if (OUT / "weibull_bootstrap.csv").exists() else None
    s3 = pl.concat([pl.read_csv(ROOT / "outputs" / ds / "step3_summary.csv", infer_schema_length=None)
                    for ds in ["reddit", "hn", "moltbook_crawl"]], how="diagonal_relaxed")
    s2 = pl.read_csv(ROOT / "outputs" / "combined" / "step2_key_numbers.csv", infer_schema_length=None).filter(
        (pl.col("variant") == "base") & pl.col("population").is_in(["full", "complete"]))
    raw = mu.filter(pl.col("weighting") == "raw").drop("weighting")
    pa = mu.filter(pl.col("weighting") == "per_author").select(
        "community", "unit", mu_mcf_3d_per_author="mu_mcf_3d", mu_pa_ci_lo="mu_mcf_3d_ci_lo", mu_pa_ci_hi="mu_mcf_3d_ci_hi")
    t = raw.join(pa, on=["community", "unit"], how="left").join(
        s3.select("community", "unit", step3_mu_first_best="mu_kernel_first_reply", step3_mu_first_weibull="mu_kernel_weibull",
                  step3_best_family="best_family_heldout", step3_mean_observed="mean_replies_observed",
                  step3_mu_all_replies="mu_all_replies_weibull"),
        on=["community", "unit"], how="left")
    st2 = s2.select("community", step2_mean_replies_nonroot="mean_replies_nonroot", step2_root_fanout_live="root_fanout_live",
                    step2_share_with_reply="share_with_reply")
    t = t.join(st2, on="community", how="left")
    if wb is not None:
        t = t.join(wb.select("community", "unit", "mu_first_weibull", "mu_first_weibull_ci_lo", "mu_first_weibull_ci_hi",
                             "weibull_shape", "weibull_shape_ci_lo", "weibull_shape_ci_hi", "half_life_h",
                             "half_life_h_ci_lo", "half_life_h_ci_hi"), on=["community", "unit"], how="left")
    t.write_csv(OUT / "mu_definitions.csv")
    print(t)


if __name__ == "__main__":
    which = sys.argv[1:] or ["mu", "early_late", "lognormal", "step2", "weibull", "table"]
    for w in which:
        {"mu": run_mu, "weibull": run_weibull, "step2": run_step2, "early_late": run_early_late,
         "lognormal": run_lognormal, "table": run_table}[w]()
