"""Reply-tree reconstruction.

Input is a node table with one row per item (root posts and comments). Required columns:

    thread_id  str     thread the platform says the item belongs to (Reddit link_id, Moltbook
                       post_id); all-null for HN, where the thread is inferred from the ancestry
    id         str|int item id, unique
    parent_id  same    id of the parent item, null for roots (prefixes already stripped)
    is_root    bool
    created    f64     seconds since epoch
    author     str     null if deleted / unknown

Any other columns are passed through unchanged.

Conventions
-----------
- depth: root = 0, top-level comment = 1.
- missing parent: a comment whose parent_id does not match any item in the table.
- orphan: a comment whose ancestor chain does not reach a root (the missing-parent comment and
  every descendant of it). Orphans keep a lower-bound depth ("depth_lb") but depth is null.
- cross-thread parent: parent exists but sits in a different thread than the item claims.
- is_root_author: comment author equals root author; null if either author is unknown.
- self_reply: comment author equals parent author (both known).
- n_direct_replies counts every child in the table (including bots / deleted authors);
  filters are applied later at analysis time.

Implementation: ids are mapped to dense u32 indices once, and ancestry is resolved by pointer
jumping (O(log max_depth) joins on integers).
"""
from __future__ import annotations

import polars as pl

MAX_JUMPS = 64
IX = pl.UInt32


def _resolve(parent_ix: pl.DataFrame) -> pl.DataFrame:
    """parent_ix: columns ix, pix (null when parent absent or root). Returns ix, top, dist, cyclic."""
    df = parent_ix.select(
        "ix",
        anc=pl.coalesce("pix", "ix"),
        dist=pl.col("pix").is_not_null().cast(pl.Int32),
        done=pl.col("pix").is_null(),
    )
    for _ in range(MAX_JUMPS):
        if df["done"].all():
            break
        look = df.select(pl.col("ix").alias("anc"), pl.col("anc").alias("a2"),
                         pl.col("dist").alias("d2"), pl.col("done").alias("f2"))
        df = df.join(look, on="anc", how="left").select(
            "ix",
            anc=pl.when(pl.col("done")).then(pl.col("anc")).otherwise(pl.col("a2")),
            dist=pl.when(pl.col("done")).then(pl.col("dist")).otherwise(pl.col("dist") + pl.col("d2")),
            done=pl.col("done") | pl.col("f2"),
        )
    return df.select("ix", top="anc", dist="dist", cyclic=~pl.col("done"))


def build_trees(nodes: pl.DataFrame) -> pl.DataFrame:
    n = nodes.with_row_index("ix").with_columns(pl.col("ix").cast(IX))
    key = n.select(pl.col("id").alias("parent_id"), pl.col("ix").alias("pix"))
    n = n.join(key, on="parent_id", how="left", maintain_order="left")
    n = n.with_columns(missing_parent=~pl.col("is_root") & pl.col("pix").is_null())

    anc = _resolve(n.select("ix", "pix"))
    n = n.join(anc, on="ix", how="left", maintain_order="left")

    # attributes of the top node and of the parent, fetched by integer index
    base = n.select("ix", "is_root", "created", "author", "thread_id", "parent_id")
    top = base.select(pl.col("ix").alias("top"), pl.col("is_root").alias("_top_is_root"),
                      pl.col("created").alias("_top_created"), pl.col("author").alias("_top_author"),
                      pl.col("thread_id").alias("_top_thread"), pl.col("parent_id").alias("top_missing_parent"))
    par = base.select(pl.col("ix").alias("pix"), pl.col("created").alias("parent_created"),
                      pl.col("author").alias("parent_author"), pl.col("thread_id").alias("_par_thread"))
    kids = n.filter(pl.col("pix").is_not_null()).group_by("pix").agg(pl.len().cast(pl.Int32).alias("n_direct_replies"))
    n = (n.join(top, on="top", how="left", maintain_order="left")
          .join(par, on="pix", how="left", maintain_order="left")
          .join(kids.rename({"pix": "ix"}), on="ix", how="left", maintain_order="left"))
    n = n.with_columns(
        pl.col("n_direct_replies").fill_null(0),
        reaches_root=pl.col("_top_is_root") & ~pl.col("cyclic"),
        top_missing_parent=pl.when(pl.col("_top_is_root")).then(None).otherwise(pl.col("top_missing_parent")),
    )

    infer_thread = nodes["thread_id"].null_count() == nodes.height
    if infer_thread:  # HN: thread = id of the root reached
        ids = n.select(pl.col("ix").alias("top"), pl.col("id").alias("_top_id"))
        n = n.join(ids, on="top", how="left", maintain_order="left").with_columns(
            thread_id=pl.when(pl.col("reaches_root")).then(pl.col("_top_id").cast(pl.String))
        ).drop("_top_id")
        rr = pl.col("reaches_root")
        root_created = pl.when(rr).then(pl.col("_top_created"))
        root_author = pl.when(rr).then(pl.col("_top_author"))
    else:  # thread given; root attributes come from the claimed thread root
        roots = n.filter(pl.col("is_root")).select(
            pl.col("id").cast(pl.String).alias("thread_id"),
            pl.col("created").alias("_c_created"), pl.col("author").alias("_c_author"))
        n = n.join(roots, on="thread_id", how="left", maintain_order="left")
        root_created, root_author = pl.col("_c_created"), pl.col("_c_author")
        n = n.with_columns(
            cross_thread_parent=~pl.col("is_root") & pl.col("_par_thread").is_not_null()
            & (pl.col("_par_thread") != pl.col("thread_id")))
    if "cross_thread_parent" not in n.columns:
        n = n.with_columns(cross_thread_parent=pl.lit(False))

    is_c = ~pl.col("is_root")
    n = n.with_columns(
        root_created=root_created,
        root_author=root_author,
    ).with_columns(
        orphan=is_c & ~pl.col("reaches_root"),
        depth=pl.when(pl.col("reaches_root")).then(pl.col("dist")),
        depth_lb=pl.col("dist") + pl.when(pl.col("reaches_root")).then(0).otherwise(1),
        time_since_parent=pl.col("created") - pl.col("parent_created"),
        time_since_root=pl.when(is_c).then(pl.col("created") - pl.col("root_created")),
        is_root_author=pl.when(is_c & pl.col("author").is_not_null() & pl.col("root_author").is_not_null())
        .then(pl.col("author") == pl.col("root_author")),
        self_reply=pl.when(is_c & pl.col("author").is_not_null() & pl.col("parent_author").is_not_null())
        .then(pl.col("author") == pl.col("parent_author")),
    )
    return n.drop("ix", "pix", "top", "dist", "_top_is_root", "_top_created", "_top_author", "_top_thread",
                  "_par_thread", "_c_created", "_c_author", strict=False)
