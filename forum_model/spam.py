"""Moltbook spam filter (De Marzo & Garcia 2026).

For posts with >= min_comments stored comments, a post is spam if the share of unique comment
contents or the share of unique comment authors among its stored comments is below threshold.
Posts with fewer stored comments are never flagged (the rule is undefined for them).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl


def moltbook_spam_flags(db: Path, post_ids: list[str], min_comments: int = 5,
                        min_unique_content: float = 0.5, min_unique_authors: float = 0.2) -> pl.DataFrame:
    sq = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sq.execute("create temp table want(id text primary key)")
    sq.executemany("insert into want values (?)", [(p,) for p in post_ids])
    rows = sq.execute("""
        select c.post_id, count(*), count(distinct c.content), count(distinct c.author_id)
        from comments c join want w on w.id = c.post_id group by c.post_id
    """).fetchall()
    sq.close()
    df = pl.DataFrame(rows, schema=["thread_id", "stored", "unique_content", "unique_authors"], orient="row")
    return df.with_columns(
        share_unique_content=pl.col("unique_content") / pl.col("stored"),
        share_unique_authors=pl.col("unique_authors") / pl.col("stored"),
    ).with_columns(
        spam=(pl.col("stored") >= min_comments)
        & ((pl.col("share_unique_content") < min_unique_content) | (pl.col("share_unique_authors") < min_unique_authors)),
        spam_content=(pl.col("stored") >= min_comments) & (pl.col("share_unique_content") < min_unique_content),
        spam_authors=(pl.col("stored") >= min_comments) & (pl.col("share_unique_authors") < min_unique_authors),
    )
