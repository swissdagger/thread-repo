"""Shared code for the forum-model exploration (tree building, splits, author filters)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_partitions() -> dict:
    return json.loads((ROOT / "partitions.json").read_text())
