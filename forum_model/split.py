"""Deterministic thread-level calibration / validation split, stratified by month."""
from __future__ import annotations

import hashlib

import polars as pl


def assign_split(threads: pl.DataFrame, dataset: str, seed: int, frac: float,
                 strata: list[str]) -> pl.DataFrame:
    """threads: one row per thread with `thread_id`, `root_created` (epoch s) and `strata` cols.

    Within each stratum (month + strata), threads are ranked by sha256(seed:dataset:id) and the
    lowest round(frac * n) go to validation. Returns thread_id, month, split.
    """
    h = [int(hashlib.sha256(f"{seed}:{dataset}:{t}".encode()).hexdigest()[:15], 16)
         for t in threads["thread_id"].to_list()]
    df = threads.with_columns(
        month=pl.from_epoch(pl.col("root_created").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m"),
        _h=pl.Series(h, dtype=pl.Int64),
    )
    keys = ["month", *strata]
    df = df.with_columns(
        _rank=pl.col("_h").rank("ordinal").over(keys),
        _n=pl.len().over(keys),
    ).with_columns(
        split=pl.when(pl.col("_rank") <= (pl.col("_n") * frac).round(0))
        .then(pl.lit("validation")).otherwise(pl.lit("calibration"))
    )
    return df.select("thread_id", "month", *strata, "split")


def calibration_fold(thread_ids: list[str], dataset: str, seed: int, heldout_frac: float = 0.2) -> list[str]:
    """Inner fit / held-out fold WITHIN the calibration split (for model selection by held-out
    likelihood), so the 20 % validation split stays untouched. Deterministic hash."""
    out = []
    for t in thread_ids:
        u = int(hashlib.sha256(f"{seed}:fold:{dataset}:{t}".encode()).hexdigest()[:13], 16) / 16 ** 13
        out.append("heldout" if u < heldout_frac else "fit")
    return out
