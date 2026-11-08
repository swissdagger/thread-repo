"""Download the raw files needed for the exploration, pinned to dataset revisions.

Files go to data/raw/<dataset>/ (gitignored). Revisions are read from partitions.json so
that every later script knows exactly which snapshot it ran on.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "partitions.json").read_text())

PROXY_C = [k for k, v in P["reddit"].get("proxy_c_subreddits", {}).items() if isinstance(v, dict)]
SUBS = P["reddit"]["subreddits"] + PROXY_C

JOBS = {
    "reddit_comments": (P["sources"]["reddit_comments"], [f"data/{s}-*.parquet" for s in SUBS]),
    "reddit_submissions": (P["sources"]["reddit_submissions"], [f"data/{s}-*.parquet" for s in SUBS]),
    "hn": (P["sources"]["hn"], ["part.*.parquet"]),
    "moltbook": (P["sources"]["moltbook"], ["data/comments/*.parquet", "data/posts/*.parquet", "README.md"]),
    "moltbook_crawl": (P["sources"]["moltbook_crawl"], ["moltbook_upload.db", "README.md"]),
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(JOBS)
    for name in which:
        src, patterns = JOBS[name]
        out = ROOT / "data" / "raw" / name
        print(f"[{name}] {src['repo']}@{src['revision'][:10]} -> {out}", flush=True)
        snapshot_download(repo_id=src["repo"], repo_type="dataset", revision=src["revision"],
                          allow_patterns=patterns, local_dir=out, max_workers=4)
        print(f"[{name}] done", flush=True)
