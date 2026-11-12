"""Shared code for the forum-model exploration (tree building, splits, author filters)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_partitions() -> dict:
    return json.loads((ROOT / "partitions.json").read_text())


def proc_dir(ds: str, community: str | None = None) -> Path:
    """data/processed/<ds>/ ; Reddit is stored one subreddit per directory."""
    d = ROOT / "data" / "processed" / ds
    return d / community if (ds == "reddit" and community) else d


def items_paths(ds: str, communities: list[str] | None = None) -> list[Path]:
    """Per-item parquet files for a dataset (Reddit: one file per subreddit)."""
    if ds != "reddit":
        return [proc_dir(ds) / "items.parquet"]
    base = proc_dir("reddit")
    subs = communities or sorted(p.name for p in base.iterdir() if (p / "items.parquet").exists())
    return [base / s / "items.parquet" for s in subs]


def reddit_subreddits(include_proxy_c: bool = False) -> list[str]:
    P = load_partitions()["reddit"]
    subs = list(P["subreddits"])
    if include_proxy_c:
        subs += [k for k, v in P.get("proxy_c_subreddits", {}).items() if isinstance(v, dict)]
    return subs
