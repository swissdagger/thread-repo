"""Step 4 — the position check (Reddit and HN).

Usage: python scripts/step4_position.py [a] [b] [c] [d] [figures]

Does the time decay of the reply hazard flatten once page position (among siblings) is in the
model, under each position proxy, reported separately:
  (a) Reddit: rank among siblings by FINAL score (look-ahead-affected; scores grow with age and
      include votes caused by the replies being predicted)
  (b) Reddit and HN: early thread only — comments in the thread's first 30 minutes, position =
      arrival rank among siblings, censored at 30 minutes of thread age (main identification)
  (c) contest-mode r/Fantasy threads (primary) and random-sort r/AskHistorians threads (secondary,
      removed parents dropped): order randomised by the platform, replies collapsed under contest
      mode, so parents are top-level comments; compared with control threads (>= 20 comments) of the
      same subreddit; arrival rank as a placebo covariate
  (d) suggested_sort = new threads in r/philosophy and r/buildapc: newest-first display, position(t)
      = 1 + number of newer top-level siblings so far (time-varying)
HN position is latent (hidden scores drive its sort), so HN gets proxy (b) only.

Model: piecewise-exponential first-reply hazard, M0 = time bins only, M1 = + beta * log(position);
time decay = events-weighted slope of the log baseline hazard on log10(time). Static proxies also get
Weibull and exponential (novelty tau) refits with the covariate. Per-author weighting is primary.
Fitted on the calibration inner fit fold; held-out gain on the inner held-out fold.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model import pem  # noqa: E402
from forum_model.units import build_units, cap_fit, hn_population, load, reddit_population  # noqa: E402

P = load_partitions()
EDGES = H.hazard_edges()                                     # 10 s .. 10 d
EDGES_B = np.r_[0.0, np.logspace(1, np.log10(1800), 13)]     # 10 s .. 30 min
POS_CAP = 500
EXTRA = ["score", "body_status", "time_since_root", "contest_mode", "suggested_sort"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"philosophy": "#2a78d6", "buildapc": "#eb6834", "ask_hn": "#1baf7a", "link": "#eda100",
          "show_hn": "#008300", "Fantasy": "#6250d6", "AskHistorians": "#e34948"}


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def with_thread_attrs(it: pl.DataFrame) -> pl.DataFrame:
    roots = it.filter(pl.col("is_root")).select(
        "thread_id", root_created="created",
        thread_contest=pl.col("contest_mode").fill_null(False) if "contest_mode" in it.columns else pl.lit(False),
        thread_sort=pl.col("suggested_sort") if "suggested_sort" in it.columns else pl.lit(None, pl.String))
    return it.join(roots, on="thread_id", how="left")


def siblings_frame(it: pl.DataFrame) -> pl.DataFrame:
    """All displayed comments (non-root, reaching the root), bots and removed comments included."""
    vis = ~pl.col("excluded").fill_null(False) if "excluded" in it.columns else pl.lit(True)
    return it.filter(~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & vis).select(
        "id", "parent_id", "thread_id", "created", "score", "depth")


def per_author_weights(par: pl.DataFrame) -> np.ndarray:
    na = par.filter(pl.col("author").is_not_null()).group_by("fold", "author").agg(_n=pl.len())
    n = par.join(na, on=["fold", "author"], how="left")["_n"].to_numpy().astype(float)
    return np.where(np.isnan(n), 0.0, 1.0 / np.where(np.isnan(n), 1.0, n))  # unknown authors: weight 0


BOOT = 200  # thread-clustered bootstrap replicates (audit)
BOOT_ROWS: list[dict] = []


def _thread_index(d: pl.DataFrame) -> tuple[np.ndarray, int]:
    _, inv = np.unique(d["thread_id"].to_numpy(), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def fit_proxy(tag: dict, par: pl.DataFrame, edges: np.ndarray, bucket_fn, slope_range, parametric: bool,
              xval=None) -> list[dict]:
    """par: parents with t, d, fold, author, thread_id and a static 'pos' column (unless bucket_fn
    handles time). Estimates on the fit fold; held-out gain on the held-out fold; thread-clustered SE
    for beta and a thread bootstrap (BOOT replicates; fit and held-out folds resampled independently)
    for beta, both decay slopes, the flattening and the held-out gain."""
    rows = []
    fitp = cap_fit(par)
    held = par.filter(pl.col("fold") == "heldout")
    xval = np.log(np.arange(POS_CAP + 1).clip(1)) if xval is None else xval
    prep_f = pem.Prepared(fitp["t"].to_numpy(), fitp["d"].to_numpy(), edges, bucket_fn(fitp), POS_CAP + 1)
    prep_h = pem.Prepared(held["t"].to_numpy(), held["d"].to_numpy(), edges, bucket_fn(held), POS_CAP + 1)
    tf, Gf = _thread_index(fitp)
    th, Gh = _thread_index(held)
    rng = np.random.default_rng(P["seed"])
    for wlab in ["per_author", "raw"]:
        wf = per_author_weights(fitp) if wlab == "per_author" else np.ones(fitp.height)
        wh = per_author_weights(held) if wlab == "per_author" else np.ones(held.height)

        def estimate(mf, mh):
            tab = prep_f.table(wf * mf)
            m0, m1 = pem.fit(tab, None), pem.fit(tab, xval)
            s0 = pem.decay_slope(m0["alpha"], edges, tab["events"], *slope_range)
            s1 = pem.decay_slope(m1["alpha"], edges, tab["events"], *slope_range)
            htab = prep_h.table(wh * mh)
            HW = (wh * mh).sum()
            gain = (pem.heldout_ll(htab, m1, xval) - pem.heldout_ll(htab, m0, None)) / HW if HW > 0 else np.nan
            return tab, m0, m1, s0, s1, gain

        tab, m0, m1, s0, s1, gain = estimate(np.ones(fitp.height), np.ones(held.height))
        se_cl = prep_f.cluster_se_beta(m1, xval, wf, tf)
        reps = []
        for b in range(BOOT):
            mf = rng.multinomial(Gf, np.full(Gf, 1.0 / Gf))[tf].astype(float)
            mh = rng.multinomial(Gh, np.full(Gh, 1.0 / Gh))[th].astype(float) if Gh else np.ones(0)
            _, _, b1, bs0, bs1, bg = estimate(mf, mh)
            reps.append((b1["beta"], bs0, bs1, bs1 - bs0, bg))
            BOOT_ROWS.append({**tag, "weighting": wlab, "rep": b, "beta": b1["beta"], "slope_M0": bs0,
                              "slope_M1": bs1, "flattening": bs1 - bs0, "heldout_gain": bg})
        R = np.array(reps, dtype=float)
        lo, hi = np.nanpercentile(R, 2.5, axis=0), np.nanpercentile(R, 97.5, axis=0)
        row = {**tag, "weighting": wlab, "n_fit": fitp.height, "n_heldout": held.height, "threads_fit": Gf,
               "events_fit": float(tab["events"].sum()), "beta_log_position": m1["beta"],
               "se_beta_model": m1["se_beta"], "se_beta": se_cl, "se_beta_cluster_thread": se_cl,
               "beta_ci_lo": lo[0], "beta_ci_hi": hi[0],
               "hazard_ratio_per_doubling": 2 ** m1["beta"] if m1["beta"] is not None else None,
               "decay_slope_M0": s0, "decay_slope_M0_ci_lo": lo[1], "decay_slope_M0_ci_hi": hi[1],
               "decay_slope_M1": s1, "decay_slope_M1_ci_lo": lo[2], "decay_slope_M1_ci_hi": hi[2],
               "flattening_M1_minus_M0": s1 - s0, "flattening_ci_lo": lo[3], "flattening_ci_hi": hi[3],
               "heldout_ll_gain_per_parent": gain, "heldout_gain_ci_lo": lo[4], "heldout_gain_ci_hi": hi[4],
               "bootstrap_replicates": BOOT, "slope_range_s": f"{slope_range[0]:.0f}-{slope_range[1]:.0f}"}
        for model, m in [("M0", m0), ("M1", m1)]:
            for k, a in enumerate(m["alpha"]):
                rows.append({**tag, "weighting": wlab, "table": "baseline", "model": model, "bin": k,
                             "t_lo": edges[k], "t_hi": edges[k + 1], "log_hazard_per_s": a,
                             "events": float(tab["events"][k].sum())})
        if parametric and wlab == "per_author":
            t, d = fitp["t"].to_numpy(), fitp["d"].to_numpy()
            x = np.log(fitp["pos"].to_numpy().clip(1, POS_CAP).astype(float))
            for fam in ["weibull", "exponential"]:
                f0 = H.fit_first_reply(fam, t, d, wf)
                f1, bb = H.fit_first_reply_cov(fam, t, d, x, wf)
                if fam == "weibull":
                    row.update(weibull_shape_M0=float(np.exp(f0.theta[0])), weibull_shape_M1=float(np.exp(f1.theta[0])),
                               weibull_beta=bb, mu_weibull_M0=f0.mu)
                else:
                    row.update(novelty_tau_h_M0=float(1 / np.exp(f0.theta[0]) / 3600),
                               novelty_tau_h_M1=float(1 / np.exp(f1.theta[0]) / 3600), exponential_beta=bb)
        rows.append({**row, "table": "summary"})
        print(f"  [{tag['proxy']}] {tag['community']} {tag.get('threads', '')} {wlab}: beta={m1['beta']:.3f} "
              f"[{lo[0]:.3f},{hi[0]:.3f}] clusterSE={se_cl:.4f} flattening={s1 - s0:+.3f} [{lo[3]:+.3f},{hi[3]:+.3f}]",
              flush=True)
    return rows


def static_bucket(d: pl.DataFrame):
    return d["pos"].to_numpy().clip(1, POS_CAP).astype(int)


# ------------------------------------------------------------------------------------------
# proxies
# ------------------------------------------------------------------------------------------

def proxy_a(sub: str, ties: str = "oldest") -> list[dict]:
    """ties: 'oldest' (equal final scores ordered oldest first) or 'random' (seeded hash of the comment
    id). Both are assumptions: Reddit's tie-break under 'best' is not observable (audit check 9)."""
    it, _ = reddit_population(sub, EXTRA)
    sib = siblings_frame(it).with_columns(pl.col("score").fill_null(0))
    if ties == "random":
        sib = sib.with_columns(tie=pl.concat_str(pl.lit(f"{P['seed']}:tie:"), pl.col("id")).hash(seed=P["seed"]))
    else:
        sib = sib.with_columns(tie=pl.col("created"))
    sib = sib.sort(["parent_id", "score", "tie"], descending=[False, True, False]).with_columns(
        pos=pl.int_range(1, pl.len() + 1).over("parent_id"), n_sib=pl.len().over("parent_id"))
    U = build_units(it, ~pl.col("is_root"), f"reddit:{sub}")
    par = U["parents"].join(sib.select("id", "pos", "n_sib"), on="id", how="inner").filter(pl.col("n_sib") >= 2)
    del it, sib
    gc.collect()
    return fit_proxy({"proxy": "a_final_score_rank", "dataset": "reddit", "community": sub, "ties": ties}, par,
                     EDGES, static_bucket, (60, 86400), parametric=True)


def proxy_b(ds: str, comm: str, it: pl.DataFrame) -> list[dict]:
    it = with_thread_attrs(it)
    # siblings that arrived in the thread's first 30 minutes (all displayed comments, bots included)
    vis = ~pl.col("excluded").fill_null(False) if "excluded" in it.columns else pl.lit(True)
    early = it.filter(~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & vis
                      & (pl.col("time_since_root") <= 1800)).select("id", "parent_id", "created")
    early = early.sort(["parent_id", "created"]).with_columns(
        pos=pl.int_range(1, pl.len() + 1).over("parent_id"), n_sib=pl.len().over("parent_id"))
    # observation ends at 30 min of thread age
    it = it.with_columns(obs_end=pl.min_horizontal("obs_end", pl.col("root_created") + 1800))
    U = build_units(it, ~pl.col("is_root") & (pl.col("time_since_root") <= 1800), f"{ds}:{comm}")
    par = U["parents"].join(early.select("id", "pos", "n_sib"), on="id", how="inner").filter(pl.col("n_sib") >= 2)
    return fit_proxy({"proxy": "b_early_arrival_rank", "dataset": ds, "community": comm}, par, EDGES_B,
                     static_bucket, (60, 1800), parametric=True)


def thread_titles(sub: str) -> pl.DataFrame:
    import duckdb
    f = str(ROOT / "data" / "raw" / "reddit_submissions" / "data" / f"{sub}-*.parquet")
    return duckdb.sql(f"select id as thread_id, lower(any_value(title)) as title from read_parquet('{f}') group by 1").pl()


MATCHED_SERIES = {"AskHistorians": ["friday free-for-all", "thursday reading", "monday methods"]}


def proxy_c(sub: str, sel: pl.Expr, drop_removed: bool) -> list[dict]:
    it, _ = reddit_population(sub, EXTRA)
    it = with_thread_attrs(it)
    tl = it.filter(~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & (pl.col("depth") == 1)).sort(
        ["thread_id", "created"]).with_columns(pos=pl.int_range(1, pl.len() + 1).over("thread_id"),
                                                n_sib=pl.len().over("thread_id"))
    rows = []
    roots = it.filter(pl.col("is_root"))
    proxy_threads = roots.filter(sel)["thread_id"]
    control_threads = roots.filter(~sel & (pl.col("thread_size") >= 20))["thread_id"]
    sets = [("proxy_threads", proxy_threads), ("control_threads_ge20", control_threads)]
    if sub in MATCHED_SERIES:  # same recurring series under both sort modes
        tt = thread_titles(sub)
        pat = "|".join(MATCHED_SERIES[sub])
        series = roots.join(tt, on="thread_id", how="left").filter(pl.col("title").str.contains(pat))
        sets += [("matched_series_proxy", series.filter(sel)["thread_id"]),
                 ("matched_series_control", series.filter(~sel)["thread_id"])]
    for lab, ids in sets:
        f = (~pl.col("is_root") & (pl.col("depth") == 1) & pl.col("thread_id").is_in(ids.implode()))
        if drop_removed:
            f = f & (pl.col("body_status") != "removed")
        U = build_units(it, f, f"reddit:{sub}")
        par = U["parents"].join(tl.select("id", "pos", "n_sib"), on="id", how="inner").filter(pl.col("n_sib") >= 2)
        rows += fit_proxy({"proxy": "c_randomised_order", "dataset": "reddit", "community": sub, "threads": lab,
                           "removed_parents_dropped": drop_removed}, par, EDGES, static_bucket, (60, 86400),
                          parametric=True)
    return rows


def proxy_d(sub: str) -> list[dict]:
    it, _ = reddit_population(sub, EXTRA)
    it = with_thread_attrs(it)
    new_threads = it.filter(pl.col("is_root") & (pl.col("thread_sort") == "new"))["thread_id"]
    tl = it.filter(~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & (pl.col("depth") == 1)
                   & pl.col("thread_id").is_in(new_threads.implode())).select("id", "thread_id", "created")
    U = build_units(it, ~pl.col("is_root") & (pl.col("depth") == 1) & pl.col("thread_id").is_in(new_threads.implode()),
                    f"reddit:{sub}", keep=["created"])
    nsib = tl.group_by("thread_id").agg(n_sib=pl.len())
    par = U["parents"].join(nsib, on="thread_id", how="inner").filter(pl.col("n_sib") >= 2)
    # global sorted key (thread index, creation time) for counting newer siblings
    tidx = {t: i for i, t in enumerate(tl["thread_id"].unique().sort().to_list())}
    key_all = np.sort(np.array([tidx[t] for t in tl["thread_id"].to_list()], dtype=np.float64) * 1e10
                      + tl["created"].to_numpy())

    def bucket_fn(d: pl.DataFrame):
        base = np.array([tidx[t] for t in d["thread_id"].to_list()], dtype=np.float64) * 1e10 + d["created"].to_numpy()
        lo = np.searchsorted(key_all, base, side="right")

        def at_bin(k: int):
            hi = np.searchsorted(key_all, base + EDGES[k], side="right")
            return (1 + hi - lo).clip(1, POS_CAP).astype(int)
        return at_bin

    par = par.with_columns(pos=pl.lit(1))  # unused for the time-varying fit
    return fit_proxy({"proxy": "d_sort_new_time_varying", "dataset": "reddit", "community": sub}, par, EDGES,
                     bucket_fn, (60, 86400), parametric=False)


# ------------------------------------------------------------------------------------------
# drivers and outputs
# ------------------------------------------------------------------------------------------

def save(name: str, rows: list[dict]):
    o = ROOT / "outputs" / "combined"
    o.mkdir(parents=True, exist_ok=True)
    if BOOT_ROWS:
        pl.DataFrame(BOOT_ROWS, infer_schema_length=None).write_parquet(o / f"step4_{name}_bootstrap.parquet")
        BOOT_ROWS.clear()
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.filter(pl.col("table") == "summary").drop("table").write_csv(o / f"step4_{name}_summary.csv")
    df.filter(pl.col("table") == "baseline").select(
        [c for c in ["proxy", "dataset", "community", "threads", "weighting", "model", "bin", "t_lo", "t_hi",
                     "log_hazard_per_s", "events"] if c in df.columns]).write_csv(o / f"step4_{name}_baseline.csv")


def run_a(ties: str = "oldest"):
    rows = []
    for sub in P["reddit"]["subreddits"]:
        rows += proxy_a(sub, ties)
        gc.collect()
    save("a" if ties == "oldest" else "a_ties_random", rows)


def run_b():
    rows = []
    for sub in P["reddit"]["subreddits"]:
        it, _ = reddit_population(sub, EXTRA)
        rows += proxy_b("reddit", sub, it)
        del it
        gc.collect()
    hn_items = load("hn", None, ["body_status", "score", "time_since_root"])
    for comm in ["ask_hn", "link", "show_hn"]:
        it, _ = hn_population(comm, hn_items=hn_items)
        rows += proxy_b("hn", comm, it)
    save("b", rows)


def run_c():
    rows = proxy_c("Fantasy", pl.col("thread_contest"), drop_removed=False)
    rows += proxy_c("AskHistorians", pl.col("thread_sort") == "random", drop_removed=True)
    save("c", rows)


def run_d():
    rows = []
    for sub in P["reddit"]["subreddits"]:
        rows += proxy_d(sub)
        gc.collect()
    save("d", rows)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8)


def run_figures():
    o = ROOT / "outputs" / "combined"
    figd = ROOT / "figures" / "combined"
    figd.mkdir(parents=True, exist_ok=True)
    panels = [("a", "(a) final-score rank — look-ahead-affected"), ("b", "(b) early thread, arrival rank"),
              ("d", "(d) sort = new, time-varying position")]
    f, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, (p, title) in zip(axes[:3], panels):
        fp = o / f"step4_{p}_baseline.csv"
        if not fp.exists():
            continue
        b = pl.read_csv(fp, infer_schema_length=None).filter(pl.col("weighting") == "per_author")
        for comm in b["community"].unique().sort().to_list():
            for model, ls in [("M0", "-"), ("M1", "--")]:
                x = b.filter((pl.col("community") == comm) & (pl.col("model") == model) & (pl.col("events") >= 5)).sort("bin")
                mid = np.sqrt(np.maximum(x["t_lo"].to_numpy(), 1.0) * x["t_hi"].to_numpy())
                ax.plot(mid, np.exp(x["log_hazard_per_s"].to_numpy()) * 3600, ls, color=COLORS.get(comm, "0.5"),
                        lw=2 if model == "M0" else 1.4, label=f"{comm} {'time only' if model == 'M0' else '+ position'}")
        ax.set_xscale("log"); ax.set_yscale("log")
        style(ax, "time since comment (s, log)", "baseline hazard (per hour, log)", title)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fp = o / "step4_c_baseline.csv"
    if fp.exists():
        ax = axes[3]
        b = pl.read_csv(fp, infer_schema_length=None).filter((pl.col("weighting") == "per_author") & (pl.col("model") == "M0"))
        for comm in ["Fantasy", "AskHistorians"]:
            for th, ls in [("proxy_threads", "-"), ("control_threads_ge20", ":")]:
                x = b.filter((pl.col("community") == comm) & (pl.col("threads") == th) & (pl.col("events") >= 5)).sort("bin")
                mid = np.sqrt(np.maximum(x["t_lo"].to_numpy(), 1.0) * x["t_hi"].to_numpy())
                ax.plot(mid, np.exp(x["log_hazard_per_s"].to_numpy()) * 3600, ls, color=COLORS[comm], lw=2,
                        label=f"r/{comm} {'randomised' if th == 'proxy_threads' else 'control (≥20 comments)'}")
        ax.set_xscale("log"); ax.set_yscale("log")
        style(ax, "time since comment (s, log)", "hazard (per hour, log)", "(c) randomised order vs control")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    f.suptitle("Step 4: reply hazard with and without page position (per-author weighted)", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figd / "step4_position_check.png", dpi=140); plt.close(f)


if __name__ == "__main__":
    which = sys.argv[1:] or ["c", "d", "b", "a", "figures"]
    for w in which:
        {"a": run_a, "a_ties_random": lambda: run_a("random"), "b": run_b, "c": run_c, "d": run_d,
         "figures": run_figures}[w]()
