"""Author filters: bot list, bot-name rule, top-1% most active authors."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import ROOT

BOT_NAME_RULE = r"(?i)(^bot[_-]|[_-]bot$|bot$)"


def load_bot_list(path: str | Path) -> set[str]:
    p = ROOT / path
    return {ln.strip().lower() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}


def flag_authors(comments: pl.DataFrame, bots: set[str] | None, top_frac: float,
                 by: list[str]) -> pl.DataFrame:
    """Add bot_listed, bot_name_rule, top_active columns to a comment table.

    top_active: author is in the top `top_frac` of authors by comment count within `by` groups
    (unknown/deleted authors excluded from the ranking). Ties broken by author name.
    """
    known = pl.col("author").is_not_null()
    if not by:  # rank over the whole dataset
        out = flag_authors(comments.with_columns(_all=pl.lit(1)), bots, top_frac, ["_all"])
        return out.drop("_all")
    df = comments.with_columns(
        bot_listed=known & pl.col("author").str.to_lowercase().is_in(list(bots or [])),
        bot_name_rule=known & pl.col("author").str.contains(BOT_NAME_RULE),
    )
    counts = (
        df.filter(known).group_by([*by, "author"]).agg(pl.len().alias("_n"))
        .sort([*by, "_n", "author"], descending=[False] * len(by) + [True, False])
        .with_columns(
            _r=pl.int_range(1, pl.len() + 1).over(by),
            _na=pl.len().over(by),
        )
        .with_columns(top_active=pl.col("_r") <= (pl.col("_na") * top_frac).ceil())
        .select(*by, "author", "top_active")
    )
    return df.join(counts, on=[*by, "author"], how="left").with_columns(
        pl.col("top_active").fill_null(False)
    )
