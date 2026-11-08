"""Shared population definitions and parent/reply assembly for timing analyses (steps 3, 3b, 4).

A population is (dataset, community) with an `obs_end` column (absolute seconds) on every item and
a set of named parent filters ("units"). build_units() turns that into
  parents: one row per eligible parent with t (first-reply delay or censoring delay), d (event),
           c (censoring delay), n_replies, fold (inner calibration fold), per-author weight w_pa
  replies: one row per eligible direct reply (delay from parent)
Eligibility follows partitions.json -> step3: calibration split only; orphans and explicit-list
bots excluded as parents and as repliers; self-replies are not events.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import items_paths, load_partitions, proc_dir
from .hazard import TMIN
from .split import calibration_fold

P = load_partitions()
SEED = P["seed"]
COLS = ["thread_id", "id", "parent_id", "is_root", "created", "author", "depth", "orphan", "self_reply",
        "bot_listed", "top_active", "split", "thread_size", "community"]


def ep(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def hash_u(s: str, salt: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{salt}:{s}".encode()).hexdigest()[:13], 16) / 16 ** 13


def load(ds: str, communities: list[str] | None, extra: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(items_paths(ds, communities))
    names = lf.collect_schema().names()
    df = lf.select([c for c in dict.fromkeys(COLS + list(extra)) if c in names]).collect()
    return df.with_columns(pl.col("id").cast(pl.String), pl.col("parent_id").cast(pl.String))


# ------------------------------------------------------------------------------------------
# populations (definitions as in step 3)
# ------------------------------------------------------------------------------------------

def reddit_population(sub: str, extra: list[str] = ()) -> tuple[pl.DataFrame, dict]:
    it = load("reddit", [sub], ["removed_by_category", *extra]).with_columns(
        obs_end=pl.lit(ep(P["reddit"]["observation_end_utc"])))
    return it, {"nonroot": ~pl.col("is_root"),
                "root_live": pl.col("is_root") & pl.col("removed_by_category").is_null()}


def hn_population(comm: str, extra: list[str] = (), hn_items: pl.DataFrame | None = None,
                  exclude_dead: bool = True) -> tuple[pl.DataFrame, dict]:
    """exclude_dead (default since the audit, check 6): HN 'dead' comments (flagged or killed; hidden
    unless a reader turns on showdead) are excluded as parents, as reply events and as siblings."""
    it = hn_items if hn_items is not None else load("hn", None, ["body_status", *extra])
    it = it.filter(pl.col("community") == comm)
    if exclude_dead:
        it = it.with_columns(excluded=~pl.col("is_root") & (pl.col("body_status") == "dead"))
    live = it.filter(pl.col("is_root") & (pl.col("body_status") == "ok"))["thread_id"]
    it = it.filter(pl.col("thread_id").is_in(live.implode())).with_columns(
        obs_end=pl.lit(ep(P["hn"]["observation_end_utc"])))
    return it, {"nonroot": ~pl.col("is_root"), "root_live": pl.col("is_root")}


def moltbook_population(pop: str = "complete", extra: list[str] = ()) -> tuple[pl.DataFrame, dict]:
    ds = "moltbook_crawl"
    it = load(ds, None, ["retrieved", "claimed_comments", *extra])
    spam = pl.read_parquet(proc_dir(ds) / "spam_flags.parquet").filter(pl.col("spam"))["thread_id"]
    it = it.filter(~pl.col("thread_id").is_in(spam.implode()))
    roots = it.filter(pl.col("is_root")).select("thread_id", complete=pl.col("thread_size") >= pl.col("claimed_comments"))
    last_stored = it.filter(~pl.col("is_root")).group_by("thread_id").agg(last_stored=pl.col("created").max())
    crawl_end = it.group_by("thread_id").agg(crawl_end=pl.col("retrieved").max())
    it = it.join(roots, on="thread_id", how="left").join(crawl_end, on="thread_id", how="left").join(
        last_stored, on="thread_id", how="left")
    (s1a, s1b), (s2a, _) = [[ep(x) for x in s] for s in P["moltbook"]["contiguous_comment_segments_utc"]]
    in_seg1 = pl.col("created").is_between(s1a, s1b)
    seg_ok = in_seg1 | (pl.col("created") >= s2a)
    thread_end = pl.when(pl.col("complete")).then(pl.col("crawl_end")).otherwise(
        pl.min_horizontal("crawl_end", "last_stored"))
    it = it.with_columns(obs_end=pl.when(in_seg1).then(pl.min_horizontal(thread_end, pl.lit(s1b))).otherwise(thread_end),
                         seg_ok=seg_ok)
    if pop == "complete":
        it = it.filter(pl.col("complete"))
    nr = ~pl.col("is_root") & pl.col("seg_ok")
    return it, {"nonroot": nr, "depth1": nr & (pl.col("depth") == 1), "depth2plus": nr & (pl.col("depth") >= 2),
                "root_live": pl.col("is_root") & pl.col("seg_ok")}


# ------------------------------------------------------------------------------------------
# parents and replies
# ------------------------------------------------------------------------------------------

def build_units(items: pl.DataFrame, parent_filter: pl.Expr, dataset: str, keep: list[str] = ()) -> dict:
    """items must carry 'obs_end'. keep: extra parent columns to carry through."""
    cal = items.filter(pl.col("split") == "calibration")
    threads = cal.filter(pl.col("is_root"))["thread_id"].unique().to_list()
    fold = pl.DataFrame({"thread_id": threads, "fold": calibration_fold(threads, dataset, SEED)})
    cal = cal.join(fold, on="thread_id", how="inner")
    if "excluded" in cal.columns:  # items hidden from readers (e.g. HN dead): neither parents nor events
        cal_vis = ~pl.col("excluded").fill_null(False)
    else:
        cal_vis = pl.lit(True)
    par = cal.filter(parent_filter & cal_vis & ~pl.col("orphan").fill_null(False) & ~pl.col("bot_listed").fill_null(False))
    par = par.with_columns(c=(pl.col("obs_end") - pl.col("created")).clip(lower_bound=TMIN)).filter(
        pl.col("obs_end") > pl.col("created"))
    kids = cal.filter(~pl.col("is_root") & cal_vis & ~pl.col("orphan").fill_null(False) & ~pl.col("bot_listed").fill_null(False)
                      & ~pl.col("self_reply").fill_null(False)).select(
        pid="parent_id", kid_created="created", kid_top="top_active")
    rep = par.select("id", "created", "c", "fold").join(kids, left_on="id", right_on="pid", how="inner").with_columns(
        delay=(pl.col("kid_created") - pl.col("created")).clip(lower_bound=0.0)).filter(pl.col("delay") <= pl.col("c"))
    first = rep.group_by("id").agg(t_first=pl.col("delay").min(), n_replies=pl.len())
    par = par.join(first, on="id", how="left").with_columns(
        d=pl.col("t_first").is_not_null(), n_replies=pl.col("n_replies").fill_null(0),
        t=pl.coalesce("t_first", "c"))
    nau = par.filter(pl.col("author").is_not_null()).group_by("fold", "author").agg(_na=pl.len())
    # per-author weight 1/(author's parents in the fold); comments by deleted/unknown authors get weight 0
    # (they cannot be attributed to an author; counting each as its own author over-weights them)
    par = par.join(nau, on=["fold", "author"], how="left").with_columns(
        w_pa=pl.when(pl.col("author").is_null()).then(0.0).otherwise(1.0 / pl.col("_na"))).drop("_na")
    cols = ["id", "thread_id", "fold", "author", "top_active", "depth", "c", "t", "d", "n_replies", "w_pa"]
    return {"parents": par.select(list(dict.fromkeys(cols + [k for k in keep if k in par.columns]))),
            "replies": rep.select(pl.col("id").alias("parent"), "fold", "delay", "kid_top")}


def cap_fit(par: pl.DataFrame, cap: int | None = None) -> pl.DataFrame:
    cap = cap or P["step3"]["fit_cap_parents"]
    fit = par.filter(pl.col("fold") == "fit")
    if fit.height <= cap:
        return fit
    u = np.array([hash_u(i, "cap") for i in fit["id"].to_list()])
    thr = np.partition(u, cap - 1)[cap - 1]
    return fit.filter(pl.Series(u <= thr))


def step3_populations():
    """Yield (dataset, community, unit, units_dict) exactly as step 3 defines them."""
    for sub in P["reddit"]["subreddits"]:
        it, filters = reddit_population(sub)
        for unit, f in filters.items():
            yield "reddit", sub, unit, build_units(it, f, f"reddit:{sub}")
        del it
    hn_items = load("hn", None, ["body_status"])
    for comm in ["ask_hn", "link", "show_hn"]:
        it, filters = hn_population(comm, hn_items=hn_items)
        for unit, f in filters.items():
            yield "hn", comm, unit, build_units(it, f, f"hn:{comm}")
    del hn_items
    for pop in ["complete", "with_incomplete"]:
        it, filters = moltbook_population(pop)
        for unit, f in filters.items():
            if pop == "with_incomplete" and unit != "nonroot":
                continue
            yield "moltbook_crawl", "moltbook", unit if pop == "complete" else f"{unit}__{pop}", \
                build_units(it, f, "moltbook_crawl")
