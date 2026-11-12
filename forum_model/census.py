"""Sort-mode census over the GECLM Reddit submissions table (download, read columns, delete)."""
from __future__ import annotations

import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import pyarrow.parquet as pq
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

COLS = ["id", "created_utc", "contest_mode", "suggested_sort", "num_comments"]


def _read(fs: HfFileSystem, path: str, repo: str, revision: str, tmp: Path) -> pl.DataFrame:
    """Download one file to a scratch dir, read the needed columns, delete it (disk-bounded)."""
    rel = path.split(f"@{revision}/", 1)[1]
    local = Path(hf_hub_download(repo, rel, repo_type="dataset", revision=revision, local_dir=tmp))
    try:
        t = pq.read_table(local, columns=COLS)
    finally:
        local.unlink(missing_ok=True)
    sub = re.sub(r"-\d{5}-of-\d{5}-[0-9a-f]+\.parquet$", "", path.rsplit("/", 1)[-1])
    return pl.from_arrow(t).with_columns(subreddit=pl.lit(sub))


def _agg(d: pl.DataFrame) -> pl.DataFrame:
    return d.group_by("subreddit", "period").agg(
        threads=pl.len(),
        contest_mode_true=pl.col("contest").sum(),
        contest_mode_comments=pl.col("ncom").filter(pl.col("contest")).sum(),
        contest_threads_ge20_comments=(pl.col("contest") & (pl.col("ncom") >= 20)).sum(),
        sort_new=(pl.col("sort") == "new").sum(),
        sort_new_comments=pl.col("ncom").filter(pl.col("sort") == "new").sum(),
        sort_new_threads_ge20_comments=((pl.col("sort") == "new") & (pl.col("ncom") >= 20)).sum(),
        sort_random=(pl.col("sort") == "random").sum(),
        sort_qa=(pl.col("sort") == "qa").sum(),
        sort_top=(pl.col("sort") == "top").sum(),
        sort_old=(pl.col("sort") == "old").sum(),
        sort_controversial=(pl.col("sort") == "controversial").sum(),
        sort_confidence=(pl.col("sort") == "confidence").sum(),
        sort_live=(pl.col("sort") == "live").sum(),
        sort_null=pl.col("sort").is_null().sum(),
    )


def sort_census(repo: str, revision: str, lo: float, hi: float, workers: int = 3) -> pl.DataFrame:
    """Per subreddit x period (window / all_time): threads, contest_mode, suggested_sort counts.

    Aggregated per file (memory-bounded); ids are unique across files (checked in step 1 for the
    two study subreddits; GECLM splits are one subreddit per file set)."""
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}@{revision}/data/*.parquet"))
    parts = []
    tmp = Path(tempfile.mkdtemp(prefix="census_"))
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_read, fs, p, repo, revision, tmp): p for p in files}
        for i, fu in enumerate(as_completed(futs), 1):
            d = fu.result().with_columns(
                created=pl.col("created_utc").replace("None", None).cast(pl.Float64, strict=False),
                contest=pl.col("contest_mode") == "True",
                sort=pl.col("suggested_sort").replace("None", None),
                ncom=pl.col("num_comments").replace("None", None).cast(pl.Int64, strict=False),
            ).select("subreddit", "id", "created", "contest", "sort", "ncom")
            parts.append(_agg(pl.concat([
                d.with_columns(period=pl.lit("all_time")),
                d.filter(pl.col("created").is_between(lo, hi)).with_columns(period=pl.lit("window_2018_2021")),
            ])))
            del d
            print(f"  census {i}/{len(files)} {futs[fu].rsplit('/', 1)[-1]}", flush=True)
    agg = pl.concat(parts).group_by("subreddit", "period").agg(pl.all().sum())
    return agg.sort("period", "contest_mode_true", descending=[False, True])
