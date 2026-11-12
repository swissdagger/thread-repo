"""Step 2 — descriptive structure (no fitting).

Usage:
  python scripts/step2_structure.py census            # sort-mode census over all 50 GECLM subreddits
  python scripts/step2_structure.py [reddit] [hn] [moltbook_crawl] [figures]

Scope (partitions.json -> descriptive_scope): CALIBRATION split only; validation threads untouched.

Per dataset x community x author-filter variant x population, computes
  thread size and max-depth distributions, mean direct replies by depth (root fan-out vs
  non-root), share of comments with >= 1 reply, OP return rate, dyadic reciprocity,
  distinct commenters per thread for threads with 20+ comments (bimodality check),
each raw and per-author-normalised.

Variants:   base (explicit bot list removed; nothing for HN/Moltbook), no_top1 (base + top-1 %
            most active authors removed), namerule (Reddit only; base + bot-name-rule hits removed,
            sensitivity, never the primary filter).
Populations: Reddit/HN: full, lt100 (threads with < 100 stored comments, size-matched).
            Moltbook (crawl, spam posts removed): thread size from the API-reported comment count
            over all posts ("claimed"); shape statistics on completely stored threads ("complete")
            and complete threads with < 100 comments ("complete_lt100").
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, items_paths, load_partitions  # noqa: E402
from forum_model import structure as S  # noqa: E402
from forum_model.spam import moltbook_spam_flags  # noqa: E402

P = load_partitions()
SEED = P["seed"]
np.random.seed(SEED)  # nothing random in step 2; logged for completeness
PROC = ROOT / "data" / "processed"

COLORS = {  # fixed entity -> categorical slot (reference palette, light mode)
    "philosophy": "#2a78d6", "buildapc": "#eb6834", "ask_hn": "#1baf7a", "link": "#eda100",
    "moltbook": "#e87ba4", "show_hn": "#008300", "text_other": "#6250d6", "poll": "#8a8985",
    "deleted_root": "#8a8985",
}
LABELS = {"philosophy": "r/philosophy", "buildapc": "r/buildapc", "ask_hn": "Ask HN", "link": "HN link",
          "moltbook": "Moltbook", "show_hn": "Show HN", "text_other": "HN text (other)", "poll": "HN poll",
          "deleted_root": "HN deleted root"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ep(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def dirs(ds: str):
    o, f = ROOT / "outputs" / ds, ROOT / "figures" / ds
    o.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return o, f


# ------------------------------------------------------------------------------------------
# census
# ------------------------------------------------------------------------------------------

def run_census():
    from forum_model.census import sort_census
    lo, hi = (ep(x) for x in P["reddit"]["thread_window_utc"])
    src = P["sources"]["reddit_submissions"]
    agg = sort_census(src["repo"], src["revision"], lo, hi)
    out, _ = dirs("reddit")
    agg.write_csv(out / "step2_sort_census.csv")
    print(agg.filter(pl.col("period") == "window_2018_2021"))


# ------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------

COLS = ["thread_id", "id", "parent_id", "is_root", "author", "parent_author", "depth", "orphan",
        "thread_size", "split", "bot_listed", "bot_name_rule", "top_active", "community"]


def load(ds: str, extra_cols: list[str], communities: list[str] | None = None) -> pl.DataFrame:
    """Reddit is read one subreddit at a time (pass communities=[sub])."""
    lf = pl.scan_parquet(items_paths(ds, communities))
    names = lf.collect_schema().names()
    cols = [c for c in COLS + extra_cols if c in names]
    df = lf.select(cols).collect()
    return df.with_columns(pl.col("id").cast(pl.String), pl.col("parent_id").cast(pl.String))


def exclusion_sets(com: pl.DataFrame, variants: list[str]) -> dict[str, set[str]]:
    a = com.filter(pl.col("author").is_not_null())
    bots = set(a.filter(pl.col("bot_listed"))["author"].unique().to_list())
    top = set(a.filter(pl.col("top_active"))["author"].unique().to_list())
    rule = set(a.filter(pl.col("bot_name_rule"))["author"].unique().to_list())
    sets = {"base": bots, "no_top1": bots | top, "namerule": bots | rule}
    return {v: sets[v] for v in variants}


def roots_table(items: pl.DataFrame, live_expr: pl.Expr) -> pl.DataFrame:
    com = items.filter(~pl.col("is_root"))
    unknown = com.group_by("thread_id").agg(any_unknown=pl.col("author").is_null().any())
    r = items.filter(pl.col("is_root")).with_columns(live=live_expr).select(
        "thread_id", root_author="author", live="live", size_full="thread_size", split="split",
        claimed=pl.col("claimed_comments") if "claimed_comments" in items.columns else pl.lit(None, pl.Int64))
    return r.join(unknown, on="thread_id", how="left").with_columns(
        complete_authorship=pl.col("root_author").is_not_null() & ~pl.col("any_unknown").fill_null(False)
    ).drop("any_unknown")


# ------------------------------------------------------------------------------------------
# one cell: community x variant x population
# ------------------------------------------------------------------------------------------

def cell(com: pl.DataFrame, roots: pl.DataFrame, X: set[str]) -> tuple[dict, pl.DataFrame, dict]:
    R = S.retained(com, X)
    R, roots = S.with_filtered_children(R, roots)
    t = S.thread_table(R, roots)
    out: dict = {"threads": roots.height}
    out.update(S.comment_level(R))
    out.update(S.root_level(roots))
    out.update(S.op_return(t, X))
    out.update(S.reciprocity(R, roots, X))
    bim, distinct = S.bimodality(t)
    out.update(bim)
    size = t["size"].to_numpy().astype(float)
    w = t["w_author"].to_numpy()
    out.update(S.dist_summary(size, None, "size"))
    out.update(S.dist_summary(size, w, "size_pa"))
    md = t.filter(pl.col("size") > 0)
    out.update(S.dist_summary(md["max_depth"].to_numpy().astype(float), None, "maxdepth"))
    out.update(S.dist_summary(md["max_depth"].to_numpy().astype(float), md["w_author"].to_numpy(), "maxdepth_pa"))
    fan = S.fanout_by_depth(R, roots)
    dists = {"size": size, "maxdepth": md["max_depth"].to_numpy().astype(float), "distinct_ge20": distinct}
    return out, fan, dists


def run_dataset(ds: str, communities: list[str], variants: list[str], items: pl.DataFrame,
                pops: dict[str, pl.Expr], live_expr: pl.Expr, extra: dict | None = None):
    out_dir, _ = dirs(ds)
    rows, fans, ccdfs = [], [], []
    for comm in communities:
        it = items(comm) if callable(items) else items.filter(pl.col("community") == comm)
        Xs = exclusion_sets(it.filter(~pl.col("is_root")), variants)
        roots_all = roots_table(it, live_expr)
        com_all = it.filter(~pl.col("is_root") & (pl.col("split") == "calibration"))
        roots_all = roots_all.filter(pl.col("split") == "calibration")
        for pop, expr in pops.items():
            roots = roots_all.filter(expr)
            com = com_all.join(roots.select("thread_id"), on="thread_id", how="semi")
            for v, X in Xs.items():
                print(f"[{ds}] {comm} {pop} {v}: {roots.height:,} threads, {com.height:,} comments", flush=True)
                stats, fan, dists = cell(com, roots, X)
                stats["excluded_authors"] = len(X)
                for k, val in stats.items():
                    rows.append({"community": comm, "population": pop, "variant": v, "metric": k,
                                 "value": float(val) if val is not None else None})
                fans.append(fan.with_columns(community=pl.lit(comm), population=pl.lit(pop), variant=pl.lit(v)))
                for name, x in dists.items():
                    ccdfs.append(S.ccdf(x).with_columns(community=pl.lit(comm), population=pl.lit(pop),
                                                        variant=pl.lit(v), dist=pl.lit(name)))
    if extra:
        for comm, payload in extra.items():
            for k, val in payload["stats"].items():
                rows.append({"community": comm, "population": "claimed", "variant": "unfiltered", "metric": k,
                             "value": float(val)})
            ccdfs.append(S.ccdf(payload["claimed"]).with_columns(
                community=pl.lit(comm), population=pl.lit("claimed"), variant=pl.lit("unfiltered"),
                dist=pl.lit("size")))
    long = pl.DataFrame(rows)
    long.write_csv(out_dir / "step2_summary_long.csv")
    wide = long.pivot(on="metric", index=["community", "population", "variant"], values="value")
    wide.write_csv(out_dir / "step2_summary_wide.csv")
    pl.concat(fans, how="diagonal_relaxed").write_csv(out_dir / "step2_fanout_by_depth.csv")
    pl.concat(ccdfs, how="diagonal_relaxed").write_csv(out_dir / "step2_ccdf.csv")
    return long


def run_reddit():
    items = lambda comm: load("reddit", ["removed_by_category"], [comm])  # one subreddit in memory at a time
    pops = {"full": pl.lit(True), "lt100": pl.col("size_full") < 100}
    run_dataset("reddit", P["reddit"]["subreddits"], ["base", "no_top1", "namerule"], items, pops,
                pl.col("removed_by_category").is_null())


def run_hn():
    items = load("hn", ["body_status"])
    dead = items.filter(pl.col("is_root") & (pl.col("body_status") == "deleted")).select("thread_id")
    items = items.with_columns(
        community=pl.when(pl.col("thread_id").is_in(dead["thread_id"].implode())).then(pl.lit("deleted_root"))
        .otherwise(pl.col("community")))
    pops = {"full": pl.lit(True), "lt100": pl.col("size_full") < 100}
    run_dataset("hn", ["ask_hn", "link", "show_hn", "text_other", "poll", "deleted_root"], ["base", "no_top1"],
                items, pops, pl.col("body_status") == "ok")


def run_moltbook():
    ds = "moltbook_crawl"
    out_dir, _ = dirs(ds)
    items = load(ds, ["claimed_comments"]).with_columns(community=pl.lit("moltbook"))
    roots = items.filter(pl.col("is_root"))
    sf = P["moltbook"]["spam_filter"]
    spam = moltbook_spam_flags(ROOT / "data" / "raw" / "moltbook_crawl" / "moltbook_upload.db",
                               roots["thread_id"].to_list())
    spam.write_parquet(PROC / ds / "spam_flags.parquet")
    r = roots.join(spam, on="thread_id", how="left").with_columns(pl.col("spam").fill_null(False))
    spam_summary = r.select(
        posts=pl.len(), posts_ge5_stored=(pl.col("stored") >= 5).sum(), spam_posts=pl.col("spam").sum(),
        spam_by_content=pl.col("spam_content").fill_null(False).sum(),
        spam_by_authors=pl.col("spam_authors").fill_null(False).sum(),
        spam_stored_comments=pl.col("stored").filter(pl.col("spam")).sum(),
        spam_claimed_comments=pl.col("claimed_comments").filter(pl.col("spam")).sum(),
        all_stored_comments=pl.col("thread_size").sum(), all_claimed_comments=pl.col("claimed_comments").sum(),
        complete_threads=(pl.col("thread_size") >= pl.col("claimed_comments")).sum(),
        complete_nonspam_threads=((pl.col("thread_size") >= pl.col("claimed_comments")) & ~pl.col("spam")).sum(),
        complete_nonspam_ge100=((pl.col("thread_size") >= pl.col("claimed_comments")) & ~pl.col("spam")
                                & (pl.col("thread_size") >= 100)).sum(),
    )
    spam_summary.write_csv(out_dir / "step2_spam_filter.csv")
    keep = r.filter(~pl.col("spam"))["thread_id"]
    items = items.filter(pl.col("thread_id").is_in(keep.implode()))
    # thread size from the API-reported count, all non-spam calibration posts
    cal = items.filter(pl.col("is_root") & (pl.col("split") == "calibration"))
    claimed = cal["claimed_comments"].to_numpy().astype(float)
    na = cal.filter(pl.col("author").is_not_null()).group_by("author").agg(pl.len().alias("n"))
    w = 1.0 / cal.join(na, on="author", how="left")["n"].fill_null(1).to_numpy()
    st = {"threads": len(claimed)}
    st.update(S.dist_summary(claimed, None, "size"))
    st.update(S.dist_summary(claimed, w, "size_pa"))
    complete = pl.col("size_full") >= pl.col("claimed")
    pops = {"complete": complete, "complete_lt100": complete & (pl.col("size_full") < 100)}
    run_dataset(ds, ["moltbook"], ["base", "no_top1"], items, pops, pl.lit(True),
                extra={"moltbook": {"stats": st, "claimed": claimed}})


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------

def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2,
                va="center")
    ax.plot([x], [y], "o", ms=3.5, color=color)


def load_all():
    data = {}
    for ds in ["reddit", "hn", "moltbook_crawl"]:
        d = ROOT / "outputs" / ds
        if (d / "step2_ccdf.csv").exists():
            data[ds] = {"ccdf": pl.read_csv(d / "step2_ccdf.csv"), "fan": pl.read_csv(d / "step2_fanout_by_depth.csv"),
                        "wide": pl.read_csv(d / "step2_summary_wide.csv")}
    return data


SERIES = [("reddit", "philosophy"), ("reddit", "buildapc"), ("hn", "ask_hn"), ("hn", "link"),
          ("moltbook_crawl", "moltbook")]


def shape_pop(ds: str, matched: bool) -> str:
    if ds == "moltbook_crawl":
        return "complete_lt100" if matched else "complete"
    return "lt100" if matched else "full"


def run_figures():
    data = load_all()
    figc = ROOT / "figures" / "combined"
    figc.mkdir(parents=True, exist_ok=True)

    # 1. thread size CCDF (Moltbook: API-reported count)
    f, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, variant in zip(axes, ["base", "no_top1"]):
        for ds, comm in SERIES:
            c = data[ds]["ccdf"].filter((pl.col("dist") == "size") & (pl.col("community") == comm))
            if ds == "moltbook_crawl":
                c = c.filter(pl.col("population") == "claimed")
                lab = f"{LABELS[comm]} (API count, unfiltered)"
            else:
                c = c.filter((pl.col("population") == "full") & (pl.col("variant") == variant))
                lab = LABELS[comm]
            c = c.filter(pl.col("value") >= 1)
            ax.plot(c["value"], c["ccdf"], color=COLORS[comm], lw=2, label=lab, drawstyle="steps-post")
        ax.set_xscale("log"); ax.set_yscale("log")
        style(ax, "comments in thread, k (log)", "P(size ≥ k) (log)",
              f"Thread size, {'bots removed' if variant == 'base' else 'bots + top-1% authors removed'}")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    f.suptitle("Thread size: complementary CDF (calibration threads)", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figc / "step2_size_ccdf.png", dpi=150); plt.close(f)

    # 2. max depth CCDF, full vs size-matched
    f, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, matched in zip(axes, [False, True]):
        for ds, comm in SERIES:
            c = data[ds]["ccdf"].filter((pl.col("dist") == "maxdepth") & (pl.col("community") == comm)
                                        & (pl.col("population") == shape_pop(ds, matched))
                                        & (pl.col("variant") == "base"))
            ax.plot(c["value"], c["ccdf"], color=COLORS[comm], lw=2, label=LABELS[comm], drawstyle="steps-post")
        ax.set_yscale("log"); ax.set_xscale("log")
        style(ax, "max depth of thread, k (log)", "P(max depth ≥ k) (log)",
              "Size-matched: threads < 100 comments" if matched else
              "Full population (Moltbook: completely stored threads)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    f.suptitle("Maximum thread depth, threads with ≥1 comment (bots removed)", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figc / "step2_maxdepth_ccdf.png", dpi=150); plt.close(f)

    # 3. mean direct replies by depth, raw vs per-author, size-matched dashed
    f, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (col, ttl) in zip(axes, [("mean_replies", "Raw (per comment)"),
                                     ("mean_replies_per_author", "Per-author normalised")]):
        for ds, comm in SERIES:
            for matched, ls in [(False, "-"), (True, "--")]:
                fz = data[ds]["fan"].filter((pl.col("community") == comm) & (pl.col("variant") == "base")
                                            & (pl.col("population") == shape_pop(ds, matched))
                                            & (pl.col("n") >= 200) & (pl.col("depth") <= 15)).sort("depth")
                ax.plot(fz["depth"], fz[col], ls, color=COLORS[comm], lw=2 if not matched else 1.4,
                        marker="o" if not matched else None, ms=4,
                        label=LABELS[comm] if not matched else None)
        ax.set_yscale("log")
        style(ax, "depth of the item (0 = root post)", "mean direct replies (log)", ttl)
    axes[0].plot([], [], "k-", lw=2, label="full population"); axes[0].plot([], [], "k--", lw=1.4, label="size-matched (< 100)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    f.suptitle("Fan-out by depth (bots removed; depths with ≥ 200 items)", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figc / "step2_fanout_by_depth.png", dpi=150); plt.close(f)

    # 4. distinct commenters, threads with 20+ comments
    f, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, matched in zip(axes, [False, True]):
        for ds, comm in SERIES:
            c = data[ds]["ccdf"].filter((pl.col("dist") == "distinct_ge20") & (pl.col("community") == comm)
                                        & (pl.col("population") == shape_pop(ds, matched))
                                        & (pl.col("variant") == "base")).sort("value")
            if c.height < 2:
                continue
            # reconstruct the pmf from the ccdf, then a density on log-spaced bins
            v = c["value"].to_numpy(); cc = c["ccdf"].to_numpy()
            pmf = cc - np.r_[cc[1:], 0]
            # integer-aligned log bins: every bin holds >= 1 integer, so no empty-bin spikes
            edges = np.unique(np.round(np.logspace(0, np.log10(max(v.max(), 10)) + 0.05, 30))) + 0.5
            edges = np.r_[0.5, edges]
            h, _ = np.histogram(v, bins=edges, weights=pmf)
            ax.step(edges[:-1], h / np.diff(np.log10(edges)), where="post", color=COLORS[comm], lw=2,
                    label=LABELS[comm])
        ax.set_xscale("log")
        style(ax, "distinct commenters in thread (log)", "density per log10 unit",
              "Size-matched: 20–99 comments" if matched else "All threads with ≥ 20 comments")
    axes[0].legend(fontsize=7.5, frameon=False)
    f.suptitle("Distinct commenters per thread (bimodality check; bots removed)", fontsize=11, x=0.01, ha="left")
    f.tight_layout(); f.savefig(figc / "step2_distinct_commenters.png", dpi=150); plt.close(f)

    # per-dataset: filter-variant sensitivity of fan-out by depth
    for ds in data:
        _, fdir = dirs(ds)
        comms = data[ds]["fan"]["community"].unique().to_list()
        comms = [c for c in ["philosophy", "buildapc", "ask_hn", "link", "show_hn", "moltbook"] if c in comms]
        f, axes = plt.subplots(1, len(comms), figsize=(4.2 * len(comms), 3.8), sharey=True, squeeze=False)
        for ax, comm in zip(axes[0], comms):
            for v, ls in [("base", "-"), ("no_top1", "--"), ("namerule", ":")]:
                fz = data[ds]["fan"].filter((pl.col("community") == comm) & (pl.col("variant") == v)
                                            & (pl.col("population") == shape_pop(ds, False))
                                            & (pl.col("n") >= 200) & (pl.col("depth") <= 15)).sort("depth")
                if fz.height:
                    ax.plot(fz["depth"], fz["mean_replies"], ls, color=COLORS[comm], lw=2, label=v)
            ax.set_yscale("log")
            style(ax, "depth (0 = root)", "mean direct replies (log)", LABELS[comm])
            ax.legend(fontsize=7, frameon=False)
        f.tight_layout(); f.savefig(fdir / "step2_fanout_variants.png", dpi=150); plt.close(f)


KEY_METRICS = [
    "threads", "comments", "size_mean", "size_q50", "size_q99", "size_share_ge20", "size_pa_mean",
    "maxdepth_q50", "maxdepth_q90", "maxdepth_q99", "maxdepth_max", "maxdepth_pa_mean",
    "root_fanout_all", "root_fanout_live", "root_fanout_all_per_author",
    "mean_replies_nonroot", "mean_replies_nonroot_per_author", "share_with_reply", "share_with_reply_per_author",
    "op_return_complete", "op_return_complete_n", "op_return_all", "op_return_all_n",
    "op_return_complete_per_author", "op_return_all_per_author",
    "reciprocity_thread_complete", "reciprocity_thread_all", "reciprocity_thread_complete_per_author",
    "reciprocity_thread_all_per_author", "reciprocity_global_complete", "reciprocity_global_all",
    "bimod_threads_ge20", "distinct_ge20_q50", "bimod_bc_log_distinct", "bimod_bc_ratio",
    "bimod_dip_log_distinct", "bimod_dip_p", "excluded_authors"]


def run_keynumbers():
    """One row per dataset x community x population x variant with the headline metrics."""
    out = []
    for ds, comm in SERIES + [("hn", "show_hn")]:
        lg = pl.read_csv(ROOT / "outputs" / ds / "step2_summary_long.csv").filter(pl.col("community") == comm)
        fan = pl.read_csv(ROOT / "outputs" / ds / "step2_fanout_by_depth.csv").filter(pl.col("community") == comm)
        for (pop, v), g in lg.group_by(["population", "variant"]):
            d = dict(zip(g["metric"].to_list(), g["value"].to_list()))
            fz = fan.filter((pl.col("population") == pop) & (pl.col("variant") == v))
            for k in (1, 2, 3):
                r = fz.filter(pl.col("depth") == k)
                d[f"fanout_d{k}"] = r["mean_replies"][0] if r.height else None
                d[f"fanout_d{k}_pa"] = r["mean_replies_per_author"][0] if r.height else None
            keys = KEY_METRICS + [f"fanout_d{k}" for k in (1, 2, 3)] + [f"fanout_d{k}_pa" for k in (1, 2, 3)]
            out.append({"dataset": ds, "community": comm, "population": pop, "variant": v,
                        **{m: d.get(m) for m in keys}})
    o = ROOT / "outputs" / "combined"
    o.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).sort("dataset", "community", "population", "variant").write_csv(o / "step2_key_numbers.csv")


# ------------------------------------------------------------------------------------------
# proxy (c) data preparation: S10 bot review and deletion rates (decision after step 2)
# ------------------------------------------------------------------------------------------

PROXY_C_THREADS = {"Fantasy": pl.col("contest_mode").fill_null(False),
                   "AskHistorians": pl.col("suggested_sort") == "random"}


def run_proxy_c_review():
    """Authors commenting in > 1 % of threads (whole subreddit window, and within the proxy-(c)
    threads), for manual bot review; deletion/removal rates overall, by depth, and in proxy threads."""
    out, _ = dirs("reddit")
    rev, dele = [], []
    for sub, sel in PROXY_C_THREADS.items():
        it = load("reddit", ["contest_mode", "suggested_sort", "body_status", "removed_by_category",
                             "time_since_parent"], [sub])
        proxy_ids = it.filter(pl.col("is_root") & sel)["thread_id"]
        com = it.filter(~pl.col("is_root")).with_columns(
            in_proxy=pl.col("thread_id").is_in(proxy_ids.implode()))
        for scope, d in [("subreddit", com), ("proxy_threads", com.filter(pl.col("in_proxy")))]:
            n_threads = d["thread_id"].n_unique()
            a = (d.filter(pl.col("author").is_not_null()).group_by("author").agg(
                comments=pl.len(), threads=pl.col("thread_id").n_unique(),
                share_depth1=(pl.col("depth") == 1).mean(),
                median_delay_s=pl.col("time_since_parent").median(),
                bot_listed=pl.col("bot_listed").first(), bot_name_rule=pl.col("bot_name_rule").first())
                 .with_columns(thread_coverage=pl.col("threads") / n_threads)
                 .filter(pl.col("thread_coverage") > 0.01).sort("thread_coverage", descending=True))
            rev.append(a.with_columns(subreddit=pl.lit(sub), scope=pl.lit(scope), scope_threads=pl.lit(n_threads)))
            for grp, g in [("all", d), ("top_level", d.filter(pl.col("depth") == 1)),
                           ("replies", d.filter(pl.col("depth") >= 2))]:
                dele.append({"subreddit": sub, "scope": scope, "comments_group": grp, "comments": g.height,
                             "author_deleted": float(g["author"].is_null().mean()),
                             "body_deleted": float((g["body_status"] == "deleted").mean()),
                             "body_removed": float((g["body_status"] == "removed").mean()),
                             "bot_listed": float(g["bot_listed"].mean())})
        roots = it.filter(pl.col("is_root"))
        for scope, r in [("subreddit", roots), ("proxy_threads", roots.filter(sel))]:
            dele.append({"subreddit": sub, "scope": scope, "comments_group": "roots", "comments": r.height,
                         "author_deleted": float(r["author"].is_null().mean()),
                         "body_deleted": None,
                         "body_removed": float(r["removed_by_category"].is_not_null().mean()),
                         "bot_listed": None})
    pl.concat(rev, how="diagonal_relaxed").write_csv(out / "step2_proxy_c_author_review.csv")
    pl.DataFrame(dele).write_csv(out / "step2_proxy_c_deletion_rates.csv")


if __name__ == "__main__":
    which = sys.argv[1:] or ["moltbook_crawl", "reddit", "hn", "figures", "keynumbers"]
    for w in which:
        {"census": run_census, "reddit": run_reddit, "hn": run_hn, "moltbook_crawl": run_moltbook,
         "figures": run_figures, "keynumbers": run_keynumbers, "proxy_c_review": run_proxy_c_review}[w]()
