"""Descriptive tree structure (step 2). No fitting.

All functions take
  comments: thread_id, id, parent_id, depth, author, parent_author, orphan
  roots:    thread_id, root_author, live, size_full, complete_authorship
and an author exclusion set X (bots, top-1 %, ...). Exclusion semantics: a comment whose author
is in X is removed as a unit (it is neither counted nor counted as a parent); its children stay
at their original depth. Orphans (no path to root) are always removed.

Per-author normalisation ("per_author" weighting): comment-level statistics are first averaged
within author, then across authors (unknown authors excluded); thread-level statistics weight
each thread by 1 / (number of threads by the same root author in the population); threads whose root
author is unknown (deleted) get weight 0 (corrected in step 3b; previously each counted as its own author).
"""
from __future__ import annotations

import numpy as np
import polars as pl

QS = [0.5, 0.9, 0.99]


def retained(comments: pl.DataFrame, X: set[str]) -> pl.DataFrame:
    keep = ~pl.col("orphan") & (pl.col("author").is_null() | ~pl.col("author").is_in(list(X)))
    return comments.filter(keep)


def with_filtered_children(R: pl.DataFrame, roots: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    kids = R.group_by("parent_id").agg(pl.len().cast(pl.Int64).alias("kids"))
    R = R.join(kids.rename({"parent_id": "id"}), on="id", how="left").with_columns(pl.col("kids").fill_null(0))
    roots = roots.join(kids.rename({"parent_id": "thread_id"}), on="thread_id", how="left").with_columns(
        pl.col("kids").fill_null(0))
    return R, roots


def thread_table(R: pl.DataFrame, roots: pl.DataFrame) -> pl.DataFrame:
    """One row per root: size, max depth, distinct commenters, OP / non-OP comment counts."""
    j = R.join(roots.select("thread_id", "root_author"), on="thread_id", how="inner")
    agg = j.group_by("thread_id").agg(
        size=pl.len(),
        max_depth=pl.col("depth").max(),
        distinct_commenters=pl.col("author").drop_nulls().n_unique(),
        op_comments=(pl.col("root_author").is_not_null() & (pl.col("author") == pl.col("root_author"))).sum(),
        nonop_comments=(pl.col("root_author").is_null() | pl.col("author").is_null()
                        | (pl.col("author") != pl.col("root_author"))).sum(),
    )
    t = roots.join(agg, on="thread_id", how="left").with_columns(
        pl.col("size", "op_comments", "nonop_comments", "distinct_commenters").fill_null(0),
        pl.col("max_depth").fill_null(0),
    )
    n_by_author = t.filter(pl.col("root_author").is_not_null()).group_by("root_author").agg(pl.len().alias("_na"))
    return t.join(n_by_author, on="root_author", how="left").with_columns(
        # unknown (deleted) root authors get weight 0 in per-author versions
        w_author=pl.when(pl.col("root_author").is_null()).then(0.0).otherwise(1.0 / pl.col("_na"))).drop("_na")


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if len(x) == 0 or w.sum() <= 0:
        return float("nan")
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q, side="left").clip(max=len(x) - 1)])


def dist_summary(x: np.ndarray, w: np.ndarray | None, prefix: str) -> dict:
    if w is None:
        w = np.ones(len(x))
    if len(x) == 0:
        return {f"{prefix}_n": 0}
    if w.sum() <= 0:  # e.g. per-author weights when every root author is unknown
        return {f"{prefix}_n": int(len(x)), f"{prefix}_mean": float("nan")}
    out = {f"{prefix}_n": int(len(x)), f"{prefix}_mean": float(np.average(x, weights=w))}
    for q in QS:
        out[f"{prefix}_q{int(q * 100)}"] = wquantile(x, w, q)
    out[f"{prefix}_max"] = float(x.max())
    out[f"{prefix}_share_ge20"] = float(np.average(x >= 20, weights=w))
    out[f"{prefix}_share_ge100"] = float(np.average(x >= 100, weights=w))
    return out


def fanout_by_depth(R: pl.DataFrame, roots: pl.DataFrame, max_depth: int = 40) -> pl.DataFrame:
    """Mean filtered direct replies by depth; depth 0 = roots. Raw and per-author."""
    root_rows = roots.select(depth=pl.lit(0, pl.Int32), kids="kids", author="root_author")
    com_rows = R.select(pl.col("depth").clip(upper_bound=max_depth), "kids", "author")
    both = pl.concat([root_rows, com_rows.with_columns(pl.col("depth").cast(pl.Int32))])
    raw = both.group_by("depth").agg(n=pl.len(), mean_replies=pl.col("kids").mean(),
                                     share_ge1=(pl.col("kids") > 0).mean())
    pa = (both.filter(pl.col("author").is_not_null()).group_by("depth", "author")
          .agg(m=pl.col("kids").mean(), s=(pl.col("kids") > 0).mean())
          .group_by("depth").agg(n_authors=pl.len(), mean_replies_per_author=pl.col("m").mean(),
                                 share_ge1_per_author=pl.col("s").mean()))
    return raw.join(pa, on="depth", how="left").sort("depth")


def comment_level(R: pl.DataFrame) -> dict:
    nonroot = R
    pa = (nonroot.filter(pl.col("author").is_not_null()).group_by("author")
          .agg(m=pl.col("kids").mean(), s=(pl.col("kids") > 0).mean()))
    return {
        "comments": nonroot.height,
        "mean_replies_nonroot": float(nonroot["kids"].mean()) if nonroot.height else float("nan"),
        "share_with_reply": float((nonroot["kids"] > 0).mean()) if nonroot.height else float("nan"),
        "mean_replies_nonroot_per_author": float(pa["m"].mean()) if pa.height else float("nan"),
        "share_with_reply_per_author": float(pa["s"].mean()) if pa.height else float("nan"),
        "authors": pa.height,
    }


def root_level(roots: pl.DataFrame) -> dict:
    out = {}
    for lab, r in [("all", roots), ("live", roots.filter(pl.col("live")))]:
        out[f"root_fanout_{lab}"] = float(r["kids"].mean()) if r.height else float("nan")
        out[f"root_fanout_{lab}_n"] = r.height
        pa = r.filter(pl.col("root_author").is_not_null()).group_by("root_author").agg(m=pl.col("kids").mean())
        out[f"root_fanout_{lab}_per_author"] = float(pa["m"].mean()) if pa.height else float("nan")
    return out


def op_return(t: pl.DataFrame, X: set[str]) -> dict:
    """OP return over threads with >= 1 retained non-OP comment and a non-excluded OP.

    complete: threads with complete authorship (root and every comment author known)
    root_known: root author known
    all: every thread; unknown root author counts as 'did not return' (lower bound)
    """
    base = t.filter((pl.col("nonop_comments") > 0)
                    & (pl.col("root_author").is_null() | ~pl.col("root_author").is_in(list(X))))
    base = base.with_columns(returned=pl.col("op_comments") > 0)
    out = {}
    for lab, d in [("complete", base.filter(pl.col("complete_authorship"))),
                   ("root_known", base.filter(pl.col("root_author").is_not_null())),
                   ("all", base)]:
        out[f"op_return_{lab}"] = float(d["returned"].mean()) if d.height else float("nan")
        out[f"op_return_{lab}_n"] = d.height
        pa = d.filter(pl.col("root_author").is_not_null()).group_by("root_author").agg(m=pl.col("returned").mean())
        out[f"op_return_{lab}_per_author"] = float(pa["m"].mean()) if pa.height else float("nan")
        out[f"op_comments_per_thread_{lab}"] = float(d["op_comments"].mean()) if d.height else float("nan")
    return out


def reciprocity(R: pl.DataFrame, roots: pl.DataFrame, X: set[str]) -> dict:
    """Dyadic reciprocity: share of unordered author pairs with replies in both directions.

    Edge a -> b for every retained comment by a whose parent is authored by b (a, b known, a != b,
    b not excluded). Within-thread pairs are (thread, {a, b}); global pairs pool all threads.
    Reported over all threads and over threads with complete authorship.
    """
    Xl = list(X)
    e = R.filter(pl.col("author").is_not_null() & pl.col("parent_author").is_not_null()
                 & (pl.col("author") != pl.col("parent_author")) & ~pl.col("parent_author").is_in(Xl))
    e = e.join(roots.select("thread_id", "complete_authorship"), on="thread_id", how="inner").select(
        "thread_id", "complete_authorship", a="author", b="parent_author")
    e = e.with_columns(lo=pl.min_horizontal("a", "b"), hi=pl.max_horizontal("a", "b"),
                       fwd=pl.col("a") < pl.col("b"))
    out = {}
    for scope in ["all", "complete"]:
        d = e if scope == "all" else e.filter(pl.col("complete_authorship"))
        for level, keys in [("thread", ["thread_id", "lo", "hi"]), ("global", ["lo", "hi"])]:
            pairs = d.group_by(keys).agg(f=pl.col("fwd").any(), r=(~pl.col("fwd")).any()).with_columns(
                recip=pl.col("f") & pl.col("r"))
            key = f"reciprocity_{level}_{scope}"
            out[key] = float(pairs["recip"].mean()) if pairs.height else float("nan")
            out[f"{key}_pairs"] = pairs.height
            # per author: share of the author's pairs that are reciprocated, averaged over authors
            pa = pl.concat([pairs.select(u="lo", recip="recip"), pairs.select(u="hi", recip="recip")]).group_by(
                "u").agg(m=pl.col("recip").mean())
            out[f"{key}_per_author"] = float(pa["m"].mean()) if pa.height else float("nan")
    return out


def bimodality(t: pl.DataFrame, min_size: int = 20) -> tuple[dict, np.ndarray]:
    """Distinct commenters for threads with >= min_size retained comments.

    Sarle's bimodality coefficient (> 5/9 suggests bi/multimodality) on log10(distinct) and on
    distinct/size, plus Hartigan's dip test on log10(distinct)."""
    d = t.filter(pl.col("size") >= min_size)
    x = d["distinct_commenters"].to_numpy().astype(float)
    r = (d["distinct_commenters"] / d["size"]).to_numpy().astype(float)
    out = {"bimod_threads_ge20": int(len(x))}
    for lab, v in [("log_distinct", np.log10(np.maximum(x, 1))), ("ratio", r)]:
        out[f"bimod_bc_{lab}"] = sarle_bc(v)
    try:
        import diptest
        dip, p = diptest.diptest(np.log10(np.maximum(x, 1))) if len(x) > 3 else (np.nan, np.nan)
        out["bimod_dip_log_distinct"], out["bimod_dip_p"] = float(dip), float(p)
    except Exception:  # pragma: no cover
        out["bimod_dip_log_distinct"] = out["bimod_dip_p"] = float("nan")
    w = d["w_author"].to_numpy()
    out.update(dist_summary(x, None, "distinct_ge20"))
    out.update(dist_summary(x, w, "distinct_ge20_pa"))
    return out, x


def sarle_bc(v: np.ndarray) -> float:
    n = len(v)
    if n < 4 or np.std(v) == 0:
        return float("nan")
    m = v - v.mean()
    s2 = (m ** 2).mean()
    g = (m ** 3).mean() / s2 ** 1.5 * np.sqrt(n * (n - 1)) / (n - 2)
    k = (m ** 4).mean() / s2 ** 2 - 3
    k = ((n + 1) * k + 6) * (n - 1) / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def ccdf(x: np.ndarray) -> pl.DataFrame:
    """Empirical P(X >= v) at each distinct v."""
    if len(x) == 0:
        return pl.DataFrame({"value": [], "ccdf": []})
    v, c = np.unique(x, return_counts=True)
    tail = np.cumsum(c[::-1])[::-1] / len(x)
    return pl.DataFrame({"value": v.astype(float), "ccdf": tail})
