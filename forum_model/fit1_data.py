"""First model fit: populations as normalised items (see forum_model/fit1_streams.py).

Calibration split only; each thread carries its inner fold ('fit' / 'heldout'). Filters follow
docs/definitions.md and docs/fit1_brief.md:
  * explicit-list bots, HN dead items and orphans are dropped as items;
  * Moltbook: crawl, complete non-spam threads, outage handled (obs_end and segment rule as in earlier
    steps), accounts in config/moltbook_scaffolding_accounts.txt dropped as items;
  * HN: Ask / Show / link, post_type carried for the joint fit.
"""
from __future__ import annotations

import hashlib

import polars as pl

from . import ROOT, load_partitions
from .split import calibration_fold
from .units import hn_population, load, moltbook_population, reddit_population

P = load_partitions()
SEED = P["seed"]
POPULATIONS = ["philosophy", "buildapc", "ask_hn", "link", "show_hn", "moltbook"]
DATASET_KEY = {"philosophy": "reddit:philosophy", "buildapc": "reddit:buildapc", "ask_hn": "hn:ask_hn",
               "link": "hn:link", "show_hn": "hn:show_hn", "moltbook": "moltbook_crawl"}


def scaffolding_names() -> tuple[set[str], list[str]]:
    exact, prefix = set(), []
    for line in (ROOT / "config" / "moltbook_scaffolding_accounts.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name = line.split("\t")[0].strip()
        (prefix.append(name[:-1]) if name.endswith("*") else exact.add(name))
    return exact, prefix


def _normalise(it: pl.DataFrame, post_type: str | pl.Expr) -> pl.DataFrame:
    keep = ~pl.col("orphan").fill_null(False) & ~pl.col("bot_listed").fill_null(False)
    if "excluded" in it.columns:
        keep = keep & ~pl.col("excluded").fill_null(False)
    it = it.filter((pl.col("split") == "calibration") & keep)
    pt = pl.lit(post_type) if isinstance(post_type, str) else post_type
    return it.select(pl.col("thread_id").cast(pl.String), pl.col("id").cast(pl.String),
                     pl.col("parent_id").cast(pl.String), "is_root", "created", "author", "depth", "obs_end",
                     post_type=pt)


def load_population(name: str, hn_items: pl.DataFrame | None = None) -> pl.DataFrame:
    if name in ("philosophy", "buildapc"):
        it, _ = reddit_population(name)
        it = it.filter(~(pl.col("is_root") & pl.col("removed_by_category").is_not_null()))  # live roots
        out = _normalise(it, name)
    elif name in ("ask_hn", "link", "show_hn"):
        it, _ = hn_population(name, hn_items=hn_items)
        out = _normalise(it, name)
    elif name == "moltbook":
        it, f = moltbook_population("complete", ["author_name"])
        exact, prefix = scaffolding_names()
        scaff = pl.col("author_name").is_in(list(exact))
        for p in prefix:
            scaff = scaff | pl.col("author_name").str.starts_with(p)
        it = it.filter(~(scaff.fill_null(False) & ~pl.col("is_root")))
        # comments outside the two contiguous segments are not units (seg_ok); they stay as items for structure
        out = _normalise(it, "moltbook")
    else:
        raise ValueError(name)
    threads = out.filter(pl.col("is_root"))["thread_id"].unique().to_list()
    fold = pl.DataFrame({"thread_id": threads, "fold": calibration_fold(threads, DATASET_KEY[name], SEED)})
    return out.join(fold, on="thread_id", how="inner")


def thread_hash(tid: str) -> float:
    return int(hashlib.sha256(f"{SEED}:fit1:{tid}".encode()).hexdigest()[:13], 16) / 16 ** 13


def cap_threads(items: pl.DataFrame, cap_comments: int, H: float = 3 * 86400.0) -> pl.DataFrame:
    """Whole threads in sha256('<seed>:fit1:<thread_id>') order until the number of comments created in the
    first H of their thread would exceed cap_comments."""
    roots = items.filter(pl.col("is_root")).select("thread_id", T0="created")
    n = items.filter(~pl.col("is_root")).join(roots, on="thread_id").filter(
        pl.col("created") - pl.col("T0") < H).group_by("thread_id").len()
    th = roots.join(n, on="thread_id", how="left").with_columns(pl.col("len").fill_null(0)).with_columns(
        u=pl.col("thread_id").map_elements(thread_hash, return_dtype=pl.Float64)).sort("u").with_columns(
        cum=pl.col("len").cum_sum())
    keep = th.filter(pl.col("cum") <= cap_comments)["thread_id"]
    return items.filter(pl.col("thread_id").is_in(keep.implode()))
