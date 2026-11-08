"""Step 3 — reply-delay hazard and self-excitation kernel.

Usage: python scripts/step3_kernels.py [reddit] [hn] [moltbook_crawl] [figures]

Per dataset x community x unit (non-root comments; roots; Moltbook also depth-1 and depth>=2):
  - time to first direct reply (non-self, non-bot) or censoring at the observation end;
  - non-parametric (actuarial) hazard in log-spaced bins, 10 s to 10 days, raw and per-author;
  - exponential, Weibull and two-exponential-mixture kernels fitted by ML with censoring on the
    calibration FIT fold, compared on the calibration HELD-OUT fold (validation split untouched);
  - branching ratio (a) = kernel integral mu from the fitted first-reply hazard (the estimate),
    also mu from fitting the same kernel to ALL direct replies, and (b) the depth-tail slope
    exp(d log P(depth>=k)/dk) as a comparison only.
Reddit is processed one subreddit at a time.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, items_paths, load_partitions, proc_dir  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model.split import calibration_fold  # noqa: E402

P = load_partitions()
SEED = P["seed"]
RNG = np.random.default_rng(SEED)  # used only for the conditional-Weibull subsample
FIT_CAP = P["step3"]["fit_cap_parents"]
EDGES = H.hazard_edges()

COLORS = {"philosophy": "#2a78d6", "buildapc": "#eb6834", "ask_hn": "#1baf7a", "link": "#eda100",
          "moltbook": "#e87ba4", "show_hn": "#008300"}
LABELS = {"philosophy": "r/philosophy", "buildapc": "r/buildapc", "ask_hn": "Ask HN", "link": "HN link",
          "show_hn": "Show HN", "moltbook": "Moltbook"}
FAM_STYLE = {"exponential": ":", "weibull": "-", "mixture2": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ep(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def dirs(ds: str):
    o, f = ROOT / "outputs" / ds, ROOT / "figures" / ds
    o.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return o, f


def hash_u(s: str, salt: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{salt}:{s}".encode()).hexdigest()[:13], 16) / 16 ** 13


# ------------------------------------------------------------------------------------------
# assembling parents, first replies and all replies
# ------------------------------------------------------------------------------------------

COLS = ["thread_id", "id", "parent_id", "is_root", "created", "author", "depth", "orphan", "self_reply",
        "bot_listed", "top_active", "split", "thread_size", "community"]


def load(ds: str, communities: list[str] | None, extra: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(items_paths(ds, communities))
    names = lf.collect_schema().names()
    df = lf.select([c for c in COLS + extra if c in names]).collect()
    return df.with_columns(pl.col("id").cast(pl.String), pl.col("parent_id").cast(pl.String))


def build_units(items: pl.DataFrame, obs_end: pl.Expr, parent_filter: pl.Expr, dataset: str) -> dict:
    """items must carry an 'obs_end' column (absolute seconds) for every row.

    Returns per unit: parents (id, thread_id, fold, author, top_active, depth, c, t_first, d,
    n_replies) and replies (parent id, fold, delay, replier top_active)."""
    items = items.with_columns(obs_end=obs_end)
    cal = items.filter(pl.col("split") == "calibration")
    threads = cal.filter(pl.col("is_root"))["thread_id"].unique().to_list()
    fold = pl.DataFrame({"thread_id": threads, "fold": calibration_fold(threads, dataset, SEED)})
    cal = cal.join(fold, on="thread_id", how="inner")
    par = cal.filter(parent_filter & ~pl.col("orphan").fill_null(False) & ~pl.col("bot_listed").fill_null(False))
    par = par.with_columns(c=(pl.col("obs_end") - pl.col("created")).clip(lower_bound=H.TMIN)).filter(
        pl.col("obs_end") > pl.col("created"))
    kids = cal.filter(~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & ~pl.col("bot_listed").fill_null(False)
                      & ~pl.col("self_reply").fill_null(False)).select(
        pid="parent_id", kid_created="created", kid_top="top_active")
    rep = par.select("id", "created", "c", "fold").join(kids, left_on="id", right_on="pid", how="inner").with_columns(
        delay=(pl.col("kid_created") - pl.col("created")).clip(lower_bound=0.0)).filter(pl.col("delay") <= pl.col("c"))
    first = rep.group_by("id").agg(t_first=pl.col("delay").min(), n_replies=pl.len())
    par = par.join(first, on="id", how="left").with_columns(
        d=pl.col("t_first").is_not_null(), n_replies=pl.col("n_replies").fill_null(0),
        t=pl.coalesce("t_first", "c"))
    # per-author weights within each fold (unknown authors count as distinct)
    nau = par.filter(pl.col("author").is_not_null()).group_by("fold", "author").agg(_na=pl.len())
    par = par.join(nau, on=["fold", "author"], how="left").with_columns(w_pa=1.0 / pl.col("_na").fill_null(1)).drop("_na")
    return {"parents": par.select("id", "thread_id", "fold", "author", "top_active", "depth", "c", "t", "d",
                                  "n_replies", "w_pa"),
            "replies": rep.select(pl.col("id").alias("parent"), "fold", "delay", "kid_top")}


def cap_fit(par: pl.DataFrame) -> pl.DataFrame:
    fit = par.filter(pl.col("fold") == "fit")
    if fit.height <= FIT_CAP:
        return fit
    u = np.array([hash_u(i, "cap") for i in fit["id"].to_list()])
    thr = np.partition(u, FIT_CAP - 1)[FIT_CAP - 1]
    return fit.filter(pl.Series(u <= thr))


# ------------------------------------------------------------------------------------------
# one unit
# ------------------------------------------------------------------------------------------

def analyse_unit(ds: str, comm: str, unit: str, U: dict, depth_for_tail: dict[str, np.ndarray] | None):
    par, rep = U["parents"], U["replies"]
    fit = cap_fit(par)
    held = par.filter(pl.col("fold") == "heldout")
    t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
    th, dh = held["t"].to_numpy(), held["d"].to_numpy()
    rows, haz = [], []
    base = {"dataset": ds, "community": comm, "unit": unit, "n_fit": fit.height, "n_heldout": held.height,
            "n_calibration": par.height}
    print(f"[{ds}] {comm} {unit}: fit {fit.height:,} held-out {held.height:,}", flush=True)
    # non-parametric hazard: raw and per-author
    for wlab, w in [("raw", None), ("per_author", fit["w_pa"].to_numpy())]:
        lt = H.life_table(t, d, EDGES, w)
        haz.append(pl.DataFrame(lt).with_columns(dataset=pl.lit(ds), community=pl.lit(comm), unit=pl.lit(unit),
                                                 weighting=pl.lit(wlab)))
    # parametric first-reply fits
    fits = {}
    for fam in H.FAMILIES:
        f = H.fit_first_reply(fam, t, d)
        fits[fam] = f
        rows.append({**base, "fit_type": "first_reply", **f.params(),
                     "heldout_ll_per_comment": H.first_reply_loglik(f, th, dh)})
    best = max(fits, key=lambda k: H.first_reply_loglik(fits[k], th, dh))
    fpa = H.fit_first_reply("weibull", t, d, fit["w_pa"].to_numpy())
    rows.append({**base, "fit_type": "first_reply_per_author", **fpa.params(),
                 "heldout_ll_per_comment": H.first_reply_loglik(fpa, th, dh, held["w_pa"].to_numpy())})
    # all direct replies (Poisson process), same kernels
    rfit = rep.join(fit.select(pl.col("id").alias("parent")), on="parent", how="semi")
    r = rfit["delay"].to_numpy()
    for fam in H.FAMILIES:
        fa = H.fit_all_replies(fam, r, np.ones_like(r), fit["c"].to_numpy(), np.ones(fit.height))
        rows.append({**base, "fit_type": "all_replies", **fa.params(), "heldout_ll_per_comment": None})
    # proper Weibull on delays of comments that got a reply (comparison with published shapes)
    ev = t[d]
    if len(ev) > 50:
        sub = ev if len(ev) <= 200_000 else RNG.choice(ev, 200_000, replace=False)
        k, _, lam = stats.weibull_min.fit(np.maximum(sub, H.TMIN), floc=0)
        rows.append({**base, "fit_type": "conditional_weibull_on_replied", "family": "weibull",
                     "weibull_shape": float(k), "weibull_scale_h": float(lam / 3600), "n_weight": float(len(sub))})
    # descriptors
    tail = {}
    for lab, dep in (depth_for_tail or {}).items():
        tail[lab] = H.depth_tail_slope(dep)
    desc = {**base, "best_family_heldout": best, "mu_kernel_first_reply": fits[best].mu,
            "mu_kernel_weibull": fits["weibull"].mu, "weibull_shape": float(np.exp(fits["weibull"].theta[0])),
            "half_life_h_best": H.half_life(best, fits[best].theta) / 3600,
            "mu_all_replies_weibull": [x for x in rows if x["fit_type"] == "all_replies" and x["family"] == "weibull"][0]["mu"],
            "mu_per_author_weibull": fpa.mu, "weibull_shape_per_author": float(np.exp(fpa.theta[0])),
            "share_with_reply": float(d.mean()), "mu_poisson_from_share": float(-np.log1p(-d.mean())) if d.mean() < 1 else None,
            "mean_replies_observed": float(fit["n_replies"].mean()),
            "var_replies_observed": float(fit["n_replies"].var()),
            "median_first_reply_delay_h": float(np.median(ev) / 3600) if len(ev) else None,
            "top1_share_of_parents": float(fit["top_active"].fill_null(False).mean()),
            "top1_share_of_replies": float(rfit["kid_top"].fill_null(False).mean()) if rfit.height else None,
            "heldout_ll_exponential": H.first_reply_loglik(fits["exponential"], th, dh),
            "heldout_ll_weibull": H.first_reply_loglik(fits["weibull"], th, dh),
            "heldout_ll_mixture2": H.first_reply_loglik(fits["mixture2"], th, dh)}
    for lab, tl in tail.items():
        desc[f"tail_mu_{lab}"] = tl["tail_mu"]
        desc[f"tail_k_range_{lab}"] = tl["k_range"]
        desc[f"tail_r2_{lab}"] = tl["r2"]
    return rows, haz, desc, {"fits": fits, "lt": H.life_table(t, d, EDGES)}


def write(ds: str, rows, haz, descs):
    out, _ = dirs(ds)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out / "step3_fits.csv")
    pl.concat(haz, how="diagonal_relaxed").write_csv(out / "step3_hazard.csv")
    pl.DataFrame(descs, infer_schema_length=None).write_csv(out / "step3_summary.csv")


def comment_depths(items: pl.DataFrame, extra_filter: pl.Expr = pl.lit(True)) -> np.ndarray:
    c = items.filter(~pl.col("is_root") & (pl.col("split") == "calibration") & ~pl.col("orphan").fill_null(False)
                     & ~pl.col("bot_listed").fill_null(False) & extra_filter)
    return c["depth"].drop_nulls().to_numpy()


# ------------------------------------------------------------------------------------------
# dataset drivers
# ------------------------------------------------------------------------------------------

def run_reddit():
    ds = "reddit"
    end = ep(P["reddit"]["observation_end_utc"])
    rows, haz, descs = [], [], []
    for sub in P["reddit"]["subreddits"]:  # one subreddit in memory at a time
        it = load(ds, [sub], ["removed_by_category"])
        lt100 = it.filter(pl.col("is_root") & (pl.col("thread_size") < 100))["thread_id"]
        tails = {"full": comment_depths(it), "lt100": comment_depths(it, pl.col("thread_id").is_in(lt100.implode()))}
        U = build_units(it, pl.lit(end), ~pl.col("is_root"), f"{ds}:{sub}")
        r, h, d, _ = analyse_unit(ds, sub, "nonroot", U, tails)
        rows += r; haz += h; descs.append(d)
        U = build_units(it, pl.lit(end), pl.col("is_root") & pl.col("removed_by_category").is_null(), f"{ds}:{sub}")
        r, h, d, _ = analyse_unit(ds, sub, "root_live", U, None)
        rows += r; haz += h; descs.append(d)
        del it, U
        gc.collect()
    write(ds, rows, haz, descs)


def run_hn():
    ds = "hn"
    end = ep(P["hn"]["observation_end_utc"])
    it = load(ds, None, ["body_status"])
    rows, haz, descs = [], [], []
    for comm in ["ask_hn", "link", "show_hn"]:
        sub = it.filter(pl.col("community") == comm)
        live = sub.filter(pl.col("is_root") & (pl.col("body_status") == "ok"))["thread_id"]
        sub = sub.filter(pl.col("thread_id").is_in(live.implode()))
        lt100 = sub.filter(pl.col("is_root") & (pl.col("thread_size") < 100))["thread_id"]
        tails = {"full": comment_depths(sub), "lt100": comment_depths(sub, pl.col("thread_id").is_in(lt100.implode()))}
        U = build_units(sub, pl.lit(end), ~pl.col("is_root"), f"{ds}:{comm}")
        r, h, d, _ = analyse_unit(ds, comm, "nonroot", U, tails)
        rows += r; haz += h; descs.append(d)
        U = build_units(sub, pl.lit(end), pl.col("is_root"), f"{ds}:{comm}")
        r, h, d, _ = analyse_unit(ds, comm, "root_live", U, None)
        rows += r; haz += h; descs.append(d)
    write(ds, rows, haz, descs)


def run_moltbook():
    ds = "moltbook_crawl"
    it = load(ds, None, ["retrieved", "claimed_comments"])
    spam = pl.read_parquet(proc_dir(ds) / "spam_flags.parquet").filter(pl.col("spam"))["thread_id"]
    it = it.filter(~pl.col("thread_id").is_in(spam.implode()))
    roots = it.filter(pl.col("is_root")).select("thread_id", complete=pl.col("thread_size") >= pl.col("claimed_comments"))
    last_stored = it.filter(~pl.col("is_root")).group_by("thread_id").agg(last_stored=pl.col("created").max())
    crawl_end = it.group_by("thread_id").agg(crawl_end=pl.col("retrieved").max())
    it = it.join(roots, on="thread_id", how="left").join(crawl_end, on="thread_id", how="left").join(
        last_stored, on="thread_id", how="left")
    (s1a, s1b), (s2a, _) = [[ep(x) for x in s] for s in P["moltbook"]["contiguous_comment_segments_utc"]]
    in_seg1 = pl.col("created").is_between(s1a, s1b)
    in_seg2 = pl.col("created") >= s2a
    seg_ok = in_seg1 | in_seg2
    thread_end = pl.when(pl.col("complete")).then(pl.col("crawl_end")).otherwise(
        pl.min_horizontal("crawl_end", "last_stored"))
    obs_end = pl.when(in_seg1).then(pl.min_horizontal(thread_end, pl.lit(s1b))).otherwise(thread_end)
    rows, haz, descs = [], [], []
    for pop, thread_sel in [("complete", pl.col("complete")), ("with_incomplete", pl.lit(True))]:
        sub = it.filter(thread_sel)
        tails = {"complete": comment_depths(it.filter(pl.col("complete")))} if pop == "complete" else None
        units = [("nonroot", ~pl.col("is_root") & seg_ok),
                 ("depth1", ~pl.col("is_root") & seg_ok & (pl.col("depth") == 1)),
                 ("depth2plus", ~pl.col("is_root") & seg_ok & (pl.col("depth") >= 2)),
                 ("root_live", pl.col("is_root") & seg_ok)]
        if pop == "with_incomplete":
            units = units[:1]
        for unit, filt in units:
            U = build_units(sub, obs_end, filt, ds)
            r, h, d, _ = analyse_unit(ds, "moltbook", unit if pop == "complete" else f"{unit}__{pop}", U,
                                      tails if unit == "nonroot" else None)
            rows += r; haz += h; descs.append(d)
    write(ds, rows, haz, descs)


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------

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


def fitted_curve(row: dict, edges: np.ndarray) -> np.ndarray:
    fam = row["family"]
    if fam == "exponential":
        th = np.array([np.log(row["rate_per_h"] / 3600)])
    elif fam == "weibull":
        th = np.array([np.log(row["weibull_shape"]), np.log(row["weibull_scale_h"] * 3600)])
    else:
        wf, mf, ms = row["mix_w_fast"], row["mix_mean_fast_h"] * 3600, row["mix_mean_slow_h"] * 3600
        th = np.array([np.log(wf / (1 - wf)), np.log(1 / mf), np.log(1 / ms)])
    return H.kernel_hazard(H.Fit(fam, th, row["mu"], 0, 0, 0), edges)


def panel(ax, hz: pl.DataFrame, fits: pl.DataFrame, color: str, title: str, show_fits: bool = True):
    e = hz.filter(pl.col("weighting") == "raw").sort("t_lo")
    mid = np.sqrt(np.maximum(e["t_lo"].to_numpy(), 1.0) * e["t_hi"].to_numpy())
    h = e["hazard_per_s"].to_numpy() * 3600
    se = e["se"].to_numpy() * 3600
    ok = e["events"].to_numpy() >= 5
    ax.errorbar(mid[ok], h[ok], yerr=se[ok], fmt="o", ms=4, color=color, ecolor=color, elinewidth=0.8,
                label="empirical (life table)")
    if show_fits:
        edges = np.r_[e["t_lo"].to_numpy(), e["t_hi"].to_numpy()[-1]]
        for row in fits.filter(pl.col("fit_type") == "first_reply").iter_rows(named=True):
            ax.plot(mid, fitted_curve(row, edges) * 3600, FAM_STYLE[row["family"]], color=INK, lw=1.4,
                    label=f"{row['family']} (μ={row['mu']:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    hv = h[ok & (h > 0)]
    if len(hv):  # frame the data; fitted curves that collapse (e.g. exponential) are clipped
        ax.set_ylim(hv.min() / 30, hv.max() * 5)
    for tt in (60, 3600, 86400):
        ax.axvline(tt, color="#bdbcb7", lw=0.6, ls=":")
    style(ax, "time since comment (s, log; dotted: 1 min, 1 h, 1 d)", "reply hazard (per hour, log)", title)


def run_figures():
    data = {}
    for ds in ["reddit", "hn", "moltbook_crawl"]:
        o = ROOT / "outputs" / ds
        if (o / "step3_hazard.csv").exists():
            data[ds] = (pl.read_csv(o / "step3_hazard.csv"), pl.read_csv(o / "step3_fits.csv", infer_schema_length=None))
    # per dataset: non-root (top row) and root (bottom row) hazards with fitted curves
    for ds, (hz, ft) in data.items():
        _, fdir = dirs(ds)
        comms = [c for c in COLORS if c in hz["community"].unique().to_list()]
        f, axes = plt.subplots(2, len(comms), figsize=(5.2 * len(comms), 8), squeeze=False)
        for j, comm in enumerate(comms):
            for i, unit in enumerate(["nonroot", "root_live"]):
                sel = (pl.col("community") == comm) & (pl.col("unit") == unit)
                panel(axes[i][j], hz.filter(sel), ft.filter(sel), COLORS[comm],
                      f"{LABELS[comm]}: {'non-root comments' if unit == 'nonroot' else 'root posts'}")
                axes[i][j].legend(fontsize=7, frameon=False, loc="lower left")
        f.suptitle("First-reply hazard and fitted kernels μ·g(t)\n(calibration fit fold)", fontsize=11, x=0.01,
                   ha="left")
        f.tight_layout(); f.savefig(fdir / "step3_hazard_fits.png", dpi=140); plt.close(f)
    # combined: non-root empirical hazards + best (held-out) fit
    figc = ROOT / "figures" / "combined"
    figc.mkdir(parents=True, exist_ok=True)
    f, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, wlab in zip(axes, ["raw", "per_author"]):
        for ds, comm in [("reddit", "philosophy"), ("reddit", "buildapc"), ("hn", "ask_hn"), ("hn", "link"),
                         ("moltbook_crawl", "moltbook")]:
            if ds not in data:
                continue
            hz = data[ds][0].filter((pl.col("community") == comm) & (pl.col("unit") == "nonroot")
                                    & (pl.col("weighting") == wlab)).sort("t_lo")
            mid = np.sqrt(np.maximum(hz["t_lo"].to_numpy(), 1.0) * hz["t_hi"].to_numpy())
            ok = hz["events"].to_numpy() >= 5
            ax.plot(mid[ok], hz["hazard_per_s"].to_numpy()[ok] * 3600, "o-", ms=3.5, lw=2, color=COLORS[comm],
                    label=LABELS[comm])
        ax.set_xscale("log"); ax.set_yscale("log")
        for tt in (60, 3600, 86400):
            ax.axvline(tt, color="#bdbcb7", lw=0.6, ls=":")
        style(ax, "time since comment (s, log; dotted: 1 min, 1 h, 1 d)", "reply hazard (per hour, log)",
              "Raw" if wlab == "raw" else "Per-author normalised")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    f.suptitle("Non-root comments: hazard of the first (non-self, non-bot) reply", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figc / "step3_nonroot_hazard.png", dpi=150); plt.close(f)
    # Moltbook depth-1 vs depth>=2
    if "moltbook_crawl" in data:
        hz, ft = data["moltbook_crawl"]
        f, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, unit, ttl in zip(axes, ["nonroot", "depth1", "depth2plus"],
                                 ["all non-root (single kernel)", "depth-1 comments", "depth ≥ 2 comments"]):
            sel = (pl.col("unit") == unit)
            panel(ax, hz.filter(sel), ft.filter(sel), COLORS["moltbook"], f"Moltbook: {ttl}")
            ax.legend(fontsize=7, frameon=False, loc="lower left")
        f.tight_layout(); f.savefig(ROOT / "figures" / "moltbook_crawl" / "step3_depth_kernels.png", dpi=140)
        plt.close(f)


def run_keynumbers():
    out = []
    for ds in ["reddit", "hn", "moltbook_crawl"]:
        p = ROOT / "outputs" / ds / "step3_summary.csv"
        if p.exists():
            out.append(pl.read_csv(p, infer_schema_length=None))
    o = ROOT / "outputs" / "combined"
    o.mkdir(parents=True, exist_ok=True)
    pl.concat(out, how="diagonal_relaxed").write_csv(o / "step3_key_numbers.csv")


if __name__ == "__main__":
    which = sys.argv[1:] or ["moltbook_crawl", "reddit", "hn", "figures", "keynumbers"]
    for w in which:
        {"reddit": run_reddit, "hn": run_hn, "moltbook_crawl": run_moltbook, "figures": run_figures,
         "keynumbers": run_keynumbers}[w]()
