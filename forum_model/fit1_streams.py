"""First model fit (docs/fit1_brief.md): normalised items -> the three event streams.

The same code serves real data (Stage 2) and generator output (Stages 1 and 3), so that the estimator
sees both through one definition.

Normalised items: one row per root or comment with columns
  thread_id, id, parent_id, is_root, created (epoch s), author (nullable), depth (root = 0),
  obs_end (epoch s; censoring), post_type.

Stream definitions (docs/fit1_brief.md; docs/definitions.md):
  * observation window per thread: thread age [0, H] (H = 3 days), cut further by obs_end;
  * comment stream of comment c: children of c that are not self-replies (author(c)) and not reply-backs
    (author(parent(c))); unknown authors count as neither;
  * reply-back stream of c: children of c written by author(parent(c)) (for a top-level comment, the OP);
  * root stream: top-level comments not written by the OP;
  * self-replies are never events and never count in "direct replies so far"; they remain units (parents);
  * position rank: arrival rank among all children of the same parent.
"""
from __future__ import annotations

import numpy as np
import polars as pl

H_DEFAULT = 3 * 86400.0


def extract(items: pl.DataFrame, H: float = H_DEFAULT) -> dict:
    roots = items.filter(pl.col("is_root")).select(
        "thread_id", T0=pl.col("created"), op=pl.col("author"), post_type="post_type",
        root_obs_end=pl.col("obs_end"))
    roots = roots.with_columns(U=pl.min_horizontal(pl.col("T0") + H, "root_obs_end") - pl.col("T0")) \
        .filter(pl.col("U") > 0)
    com = items.filter(~pl.col("is_root")).join(roots.select("thread_id", "T0", "op"), on="thread_id", how="inner")
    com = com.with_columns(s=pl.col("created") - pl.col("T0")).filter((pl.col("s") >= 0) & (pl.col("s") < H))
    # parent author (roots and comments) and parent created
    pa = items.select(pl.col("id").alias("parent_id"), parent_author="author", parent_is_root="is_root")
    com = com.join(pa, on="parent_id", how="left")
    com = com.with_columns(
        U=pl.min_horizontal(pl.col("T0") + H, "obs_end") - pl.col("created"),
        op_flag=(pl.col("author") == pl.col("op")).fill_null(False),
        p_is_root=pl.col("parent_is_root").fill_null(False))
    # position: arrival rank among all children of the same parent
    com = com.sort("parent_id", "created", "id").with_columns(pos=pl.int_range(1, pl.len() + 1).over("parent_id"))
    units = com.filter(pl.col("U") > 0)
    # children of comment units
    kids = com.select(pid="parent_id", kid_created="created", kid_author="author")
    ch = units.select("id", "thread_id", "created", "U", "author", "parent_author").join(
        kids, left_on="id", right_on="pid", how="inner").with_columns(
        u=pl.col("kid_created") - pl.col("created")).filter((pl.col("u") >= 0) & (pl.col("u") <= pl.col("U")))
    self_ = (pl.col("kid_author") == pl.col("author")).fill_null(False)
    back = (pl.col("kid_author") == pl.col("parent_author")).fill_null(False)
    ch = ch.filter(~self_).with_columns(stream=pl.when(back).then(pl.lit("reply_back")).otherwise(pl.lit("comment")))
    events_c = ch.select(pl.col("id").alias("unit"), "u", "stream")
    # root stream events: top-level comments, not by the OP, inside the root window
    tl = com.filter(pl.col("p_is_root") & ~pl.col("op_flag")).join(roots.select("thread_id", pl.col("U").alias("UR")),
                                                                    on="thread_id", how="inner")
    events_r = tl.filter(pl.col("s") <= pl.col("UR")).select("thread_id", tau="s")
    units = units.select("id", "thread_id", "created", "T0", "s", "U", "depth", "op_flag", "p_is_root", "pos",
                         "author", "parent_author")
    return {"roots": roots, "units": units, "events_c": events_c, "events_r": events_r, "H": H}


def summary(S: dict) -> dict:
    nr, nu = S["roots"].height, S["units"].height
    ec = S["events_c"]
    return {"threads": nr, "comments": nu,
            "top_level_per_root": S["events_r"].height / max(nr, 1),
            "comment_stream_per_comment": ec.filter(pl.col("stream") == "comment").height / max(nu, 1),
            "reply_backs_per_comment": ec.filter(pl.col("stream") == "reply_back").height / max(nu, 1)}
