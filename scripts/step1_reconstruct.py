"""Step 1 — thread reconstruction (no fitting).

Usage: python scripts/step1_reconstruct.py [reddit] [hn] [moltbook] [moltbook_crawl]

For each dataset: select in-window threads (partitions.json), build reply trees, compute per
comment depth / direct replies / time since parent / time since root / is_root_author, flag
authors (bot list, bot-name rule, top 1%), assign the 20% thread-level validation split, and
write integrity counts, contest-mode counts, and score-snapshot staleness.

Per-item tables  -> data/processed/<dataset>/items.parquet   (gitignored, regenerable)
Thread split     -> data/processed/<dataset>/split.parquet   (deterministic from partitions.json)
Summary tables   -> outputs/<dataset>/step1_*.csv
Figures          -> figures/<dataset>/step1_*.png
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model.authors import flag_authors, load_bot_list  # noqa: E402
from forum_model.split import assign_split  # noqa: E402
from forum_model.trees import build_trees  # noqa: E402

P = load_partitions()
SEED = P["seed"]
np.random.seed(SEED)  # nothing in step 1 is random; logged for completeness
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def ep(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def dirs(ds: str):
    o, f, p = ROOT / "outputs" / ds, ROOT / "figures" / ds, PROC / ds
    for d in (o, f, p):
        d.mkdir(parents=True, exist_ok=True)
    return o, f, p


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.sql("SET threads TO 4; SET memory_limit='10GB'; SET preserve_insertion_order=false;")
    return c


# --------------------------------------------------------------------------------------------
# Loaders: each returns (nodes, extra) where nodes has the columns trees.build_trees expects plus
# dataset-specific columns, and extra is a dict of dataset-level diagnostics.
# --------------------------------------------------------------------------------------------

def load_reddit(sub: str) -> tuple[pl.DataFrame, dict]:
    c = con()
    lo, hi = (ep(x) for x in P["reddit"]["thread_window_utc"])
    sfiles = sorted(str(x) for x in (RAW / "reddit_submissions" / "data").glob(f"{sub}-*.parquet"))
    cfiles = sorted(str(x) for x in (RAW / "reddit_comments" / "data").glob(f"{sub}-*.parquet"))
    num = "try_cast(nullif({0}, 'None') as double)"
    c.sql(f"""
        create temp table subs_all as
        select id,
               max({num.format('created_utc')}) as created,
               max({num.format('retrieved_on')}) as retrieved,
               any_value(author) as author_raw,
               bool_or(contest_mode = 'True') as contest_mode,
               any_value(nullif(suggested_sort, 'None')) as suggested_sort,
               max(try_cast(nullif(num_comments,'None') as bigint)) as claimed_comments,
               max(try_cast(nullif(score,'None') as bigint)) as score,
               any_value(nullif(removed_by_category,'None')) as removed_by_category,
               count(*) as n_dup
        from read_parquet({sfiles})
        group by id
    """)
    c.sql(f"create temp table subs as select * from subs_all where created between {lo} and {hi}")
    c.sql(f"""
        create view com_raw as
        select id,
               substr(link_id, 4) as thread_id,
               substr(parent_id, 4) as parent_id,
               left(parent_id, 3) as parent_prefix,
               {num.format('created_utc')} as created,
               {num.format('retrieved_on')} as retrieved,
               author as author_raw,
               case when body = '[deleted]' then 'deleted' when body = '[removed]' then 'removed'
                    else 'ok' end as body_status,
               try_cast(nullif(score,'None') as bigint) as score
        from read_parquet({cfiles})
    """)
    # comments on in-window threads (dedupe: keep the latest retrieval of each id)
    c.sql("""
        create temp table com as
        select * exclude (rn) from (
          select r.*, row_number() over (partition by r.id order by r.retrieved desc nulls last) rn
          from com_raw r semi join subs s on r.thread_id = s.id) where rn = 1
    """)
    extra = {}
    extra["raw_comment_rows_dup_ids"] = c.sql(
        "select count(*) - count(distinct id) from com_raw r semi join subs s on r.thread_id = s.id").fetchall()[0][0]
    # threads whose root submission is absent from the submissions dump but whose earliest
    # comment falls in the window (cannot be placed in the window -> excluded)
    extra["threads_without_root_record"], extra["comments_without_root_record"] = c.sql(f"""
        select count(*), coalesce(sum(n),0) from (
          select thread_id, min(created) first_c, count(*) n from com_raw
          where thread_id not in (select id from subs_all) group by 1)
        where first_c between {lo} and {hi}
    """).fetchall()[0]
    extra["submissions_dup_ids"] = c.sql("select coalesce(sum(n_dup - 1),0) from subs").fetchall()[0][0]
    roots = c.sql("""
        select id as thread_id, id, null::varchar as parent_id, true as is_root, created, retrieved,
               author_raw, 'ok' as body_status, score, contest_mode, suggested_sort,
               claimed_comments, removed_by_category, null::varchar as parent_prefix
        from subs
    """).pl()
    comments = c.sql("""
        select thread_id, id, parent_id, false as is_root, created, retrieved, author_raw,
               body_status, score, null::boolean as contest_mode, null::varchar as suggested_sort,
               null::bigint as claimed_comments, null::varchar as removed_by_category, parent_prefix
        from com
    """).pl()
    nodes = pl.concat([roots, comments]).with_columns(
        author=pl.when(pl.col("author_raw").is_in(["[deleted]", "None"]) | pl.col("author_raw").is_null())
        .then(None).otherwise(pl.col("author_raw")),
        community=pl.lit(sub),
    )
    c.close()
    return nodes, extra


def load_hn() -> tuple[pl.DataFrame, dict]:
    c = con()
    lo, hi = (ep(x) for x in P["hn"]["thread_window_utc"])
    files = sorted(str(x) for x in (RAW / "hn").glob("part.*.parquet"))
    c.sql(f"""
        create view raw as
        select id, parent, time, "by" as author_by, type, coalesce(dead,false) dead, coalesce(deleted,false) deleted,
               title, url, score, descendants
        from read_parquet({files})
        where type in ('story','comment','poll')
    """)
    min_id = c.sql(f"select min(id) from raw where time >= {lo}").fetchall()[0][0]
    # every item from the start of the window on (deleted items can lack a time; keep by id)
    c.sql(f"""
        create temp table items as
        select * exclude (rn) from (
          select *, row_number() over (partition by id order by time nulls last) rn from raw
          where time >= {lo} or (time is null and id >= {min_id})) where rn = 1
    """)
    extra = {"hn_min_window_id": int(min_id)}
    extra["raw_dup_ids"] = c.sql(
        f"select count(*) - count(distinct id) from raw where time >= {lo} or (time is null and id >= {min_id})"
    ).fetchall()[0][0]
    nodes = c.sql(f"""
        select null::varchar as thread_id,
               id,
               case when type = 'comment' then parent end as parent_id,
               type in ('story','poll') as is_root,
               cast(time as double) as created,
               null::double as retrieved,
               author_by as author_raw,
               case when deleted then 'deleted' when dead then 'dead' else 'ok' end as body_status,
               score,
               case when type = 'poll' then 'poll'
                    when regexp_matches(coalesce(title,''), '^(?i)ask hn') then 'ask_hn'
                    when regexp_matches(coalesce(title,''), '^(?i)show hn') then 'show_hn'
                    when coalesce(url,'') <> '' then 'link'
                    else 'text_other' end as post_class,
               descendants as claimed_comments,
               (type in ('story','poll') and time between {lo} and {hi}) as root_in_window
        from items
    """).pl()
    nodes = nodes.with_columns(
        author=pl.col("author_raw"),
        post_class=pl.when(pl.col("is_root")).then(pl.col("post_class")),
    )
    c.close()
    return nodes, extra


def load_moltbook() -> tuple[pl.DataFrame, dict]:
    c = con()
    lo, hi = (ep(x) for x in P["moltbook"]["thread_window_utc"])
    pf = str(RAW / "moltbook" / "data" / "posts" / "*.parquet")
    cf = str(RAW / "moltbook" / "data" / "comments" / "*.parquet")
    # latest snapshot of each row: order by dump_date then fetched_at
    snap = "coalesce(fetched_at::timestamptz, (try_cast(dump_date as date) + interval 1 day)::timestamptz)"
    c.sql(f"""
        create temp table posts_all as
        select * exclude (rn) from (
          select id, agent_id, agent_name, submolt, score, comment_count, is_pinned, created_at,
                 fetched_at, dump_date, {snap} as snap_at, (fetched_at is null) as snap_is_dump_bound,
                 row_number() over (partition by id order by dump_date desc nulls last,
                                    fetched_at desc nulls last) rn,
                 count(*) over (partition by id) as n_snapshots
          from read_parquet('{pf}', union_by_name=true)) where rn = 1
    """)
    c.sql(f"""
        create temp table com_all as
        select * exclude (rn) from (
          select id, post_id, agent_id, agent_name, parent_id, score, created_at, fetched_at, dump_date,
                 {snap} as snap_at, (fetched_at is null) as snap_is_dump_bound,
                 (content is null or content in ('[deleted]','[removed]','')) as content_missing,
                 row_number() over (partition by id order by dump_date desc nulls last,
                                    fetched_at desc nulls last) rn,
                 count(*) over (partition by id) as n_snapshots
          from read_parquet('{cf}', union_by_name=true)) where rn = 1
    """)
    extra = {}
    extra["posts_total_unique"] = c.sql("select count(*) from posts_all").fetchall()[0][0]
    extra["comments_total_unique"] = c.sql("select count(*) from com_all").fetchall()[0][0]
    extra["comments_null_created"] = c.sql("select count(*) from com_all where created_at is null").fetchall()[0][0]
    extra["comments_on_unknown_posts"] = c.sql(
        "select count(*) from com_all where post_id not in (select id from posts_all)").fetchall()[0][0]
    # coverage: gaps in the creation-time stream of all posts+comments around the window
    stream = c.sql(f"""
        select epoch(created_at::timestamptz) t, 'comment' kind from com_all
          where epoch(created_at::timestamptz) between {lo - 86400} and {hi + 14 * 86400}
        union all
        select epoch(created_at::timestamptz), 'post' from posts_all
          where epoch(created_at::timestamptz) between {lo - 86400} and {hi + 14 * 86400}
    """).pl()
    roots = c.sql(f"""
        select id as thread_id, id, null::varchar as parent_id, true as is_root,
               epoch(created_at::timestamptz) as created, epoch(snap_at::timestamptz) as retrieved, snap_is_dump_bound,
               agent_id as author_raw, agent_name, 'ok' as body_status, score,
               comment_count as claimed_comments, submolt, is_pinned::boolean as is_pinned, n_snapshots
        from posts_all where epoch(created_at::timestamptz) between {lo} and {hi}
    """).pl()
    comments = c.sql(f"""
        select post_id as thread_id, id, coalesce(nullif(parent_id,''), post_id) as parent_id, false as is_root,
               epoch(created_at::timestamptz) as created, epoch(snap_at::timestamptz) as retrieved, snap_is_dump_bound,
               agent_id as author_raw, agent_name,
               case when content_missing then 'deleted' else 'ok' end as body_status, score,
               null::bigint as claimed_comments, null::varchar as submolt, null::boolean as is_pinned,
               n_snapshots
        from com_all where post_id in (select id from posts_all where epoch(created_at::timestamptz) between {lo} and {hi})
    """).pl()
    nodes = pl.concat([roots, comments]).with_columns(
        author=pl.col("author_raw"), community=pl.lit("moltbook")
    )
    extra["_stream"] = stream
    c.close()
    return nodes, extra


# --------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------

STALE_Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
STALE_T = {"lt_1min": 60, "lt_1h": 3600, "lt_1d": 86400, "lt_7d": 7 * 86400, "lt_30d": 30 * 86400}


def staleness_table(df: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    d = df.with_columns(lag=pl.col("retrieved") - pl.col("created"),
                        kind=pl.when(pl.col("is_root")).then(pl.lit("root")).otherwise(pl.lit("comment")))
    aggs = [pl.len().alias("n"), pl.col("lag").is_null().sum().alias("n_missing"),
            (pl.col("lag") < 0).sum().alias("n_negative")]
    aggs += [pl.col("lag").quantile(q, "nearest").alias(f"q{int(q * 100):02d}_s") for q in STALE_Q]
    aggs += [(pl.col("lag") < t).mean().alias(f"share_{k}") for k, t in STALE_T.items()]
    return d.group_by([*group, "kind"]).agg(aggs).sort([*group, "kind"])


def integrity(df: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    com = ~pl.col("is_root")
    rt = pl.col("is_root")
    g = df.group_by(group).agg(
        threads=rt.sum(),
        threads_with_comments=(rt & (pl.col("thread_size") > 0)).sum(),
        comments=com.sum(),
        comments_reaching_root=(com & pl.col("reaches_root")).sum(),
        missing_parents=pl.col("missing_parent").sum(),
        orphans=pl.col("orphan").sum(),
        orphans_excl_missing=(pl.col("orphan") & ~pl.col("missing_parent")).sum(),
        cross_thread_parents=pl.col("cross_thread_parent").sum(),
        cyclic=pl.col("cyclic").sum(),
        negative_time_since_parent=(com & (pl.col("time_since_parent") < 0)).sum(),
        zero_time_since_parent=(com & (pl.col("time_since_parent") == 0)).sum(),
        comment_author_unknown=(com & pl.col("author").is_null()).sum(),
        comment_body_deleted=(com & (pl.col("body_status") == "deleted")).sum(),
        comment_body_removed=(com & (pl.col("body_status") == "removed")).sum(),
        comment_dead=(com & (pl.col("body_status") == "dead")).sum(),
        root_author_unknown=(rt & pl.col("author").is_null()).sum(),
        comments_by_root_author=(com & pl.col("is_root_author").fill_null(False)).sum(),
        comments_root_author_undetermined=(com & pl.col("is_root_author").is_null()).sum(),
        self_replies=(com & pl.col("self_reply").fill_null(False)).sum(),
        distinct_comment_authors=pl.col("author").filter(com).n_unique(),
        max_depth=pl.col("depth").max(),
        claimed_comments=pl.col("claimed_comments").filter(rt).sum(),
        comments_bot_listed=(com & pl.col("bot_listed")).sum(),
        comments_bot_name_rule=(com & pl.col("bot_name_rule")).sum(),
        authors_bot_listed=pl.col("author").filter(com & pl.col("bot_listed")).n_unique(),
        authors_bot_name_rule=pl.col("author").filter(com & pl.col("bot_name_rule")).n_unique(),
        authors_top1pct=pl.col("author").filter(com & pl.col("top_active")).n_unique(),
        comments_top1pct=(com & pl.col("top_active")).sum(),
        threads_validation=(rt & (pl.col("split") == "validation")).sum(),
        threads_calibration=(rt & (pl.col("split") == "calibration")).sum(),
    ).with_columns(
        share_comments_top1pct=pl.col("comments_top1pct") / pl.col("comments"),
        comments_after_bot_filter=pl.col("comments") - pl.col("comments_bot_listed"),
        comments_after_bot_and_top1pct=None,
        reconstructed_over_claimed=pl.col("comments") / pl.col("claimed_comments"),
    )
    after = df.filter(~pl.col("is_root") & ~pl.col("bot_listed") & ~pl.col("top_active")).group_by(group).agg(
        pl.len().alias("comments_after_bot_and_top1pct"))
    return g.drop("comments_after_bot_and_top1pct").join(after, on=group, how="left").sort(group)


def to_long(t: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    return t.unpivot(index=group, variable_name="metric").with_columns(pl.col("value").cast(pl.Float64))


def finish(ds: str, items: pl.DataFrame, group: list[str], split_strata: list[str], bots: set | None,
           top_by: list[str]) -> pl.DataFrame:
    """Shared tail: author flags, thread sizes, split, integrity table, write outputs."""
    out, fig, proc = dirs(ds)
    items = flag_authors(items, bots, P["author_filters"]["top_active_fraction"], top_by)
    sizes = items.filter(~pl.col("is_root") & pl.col("thread_id").is_not_null()).group_by("thread_id").agg(
        pl.len().alias("thread_size"))
    items = items.join(sizes, on="thread_id", how="left").with_columns(pl.col("thread_size").fill_null(0))
    threads = items.filter(pl.col("is_root")).select("thread_id", pl.col("created").alias("root_created"),
                                                     *split_strata)
    split = assign_split(threads, ds, SEED, P["holdout"]["validation_fraction"], split_strata)
    split.write_parquet(proc / "split.parquet")
    items = items.join(split.select("thread_id", "split"), on="thread_id", how="left")
    items.write_parquet(proc / "items.parquet")
    integ = integrity(items, group)
    integ.write_csv(out / "step1_integrity.csv")
    to_long(integ, group).write_csv(out / "step1_integrity_long.csv")
    sc = split.group_by(["month", *split_strata, "split"]).len().sort(["month", *split_strata, "split"])
    sc.write_csv(out / "step1_split_counts.csv")
    return items


def plot_staleness(items: pl.DataFrame, ds: str, group: str, title: str, note: str = ""):
    _, fig, _ = dirs(ds)
    d = items.with_columns(lag=pl.col("retrieved") - pl.col("created")).filter(pl.col("lag") > 0)
    groups = sorted(d[group].unique().to_list())
    f, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    bins = np.logspace(0, np.log10(max(d["lag"].max(), 10)) + 0.1, 70)
    for ax, kind in zip(axes, [False, True]):
        for gname in groups:
            x = d.filter((pl.col(group) == gname) & (pl.col("is_root") == kind))["lag"].to_numpy()
            if len(x):
                ax.hist(x, bins=bins, histtype="step", density=True, label=f"{gname} (n={len(x):,})", lw=1.4)
        for t in (60, 3600, 86400, 30 * 86400, 365 * 86400):  # 1 min, 1 h, 1 d, 30 d, 1 y
            ax.axvline(t, color="0.7", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("retrieved − created (s, log; dotted: 1 min, 1 h, 1 d, 30 d, 1 y)"); ax.set_ylabel("density (log)")
        ax.set_title(("roots" if kind else "comments"))
        ax.legend(fontsize=7, frameon=False)
    f.suptitle(title + (f"\n{note}" if note else ""), fontsize=10)
    f.tight_layout()
    f.savefig(fig / "step1_staleness.png", dpi=130)
    plt.close(f)


# --------------------------------------------------------------------------------------------
# Dataset drivers
# --------------------------------------------------------------------------------------------

def run_reddit():
    ds = "reddit"
    out, fig, proc = dirs(ds)
    bots = load_bot_list(P["author_filters"]["bot_list_file"])
    parts, extras = [], {}
    for sub in P["reddit"]["subreddits"]:
        print(f"[reddit] loading r/{sub}", flush=True)
        nodes, extra = load_reddit(sub)
        extras[sub] = extra
        print(f"[reddit] r/{sub}: {nodes.height:,} nodes; building trees", flush=True)
        t = build_trees(nodes)
        del nodes
        # Reddit parent prefix check: t3_ parent must be the thread root
        t = t.with_columns(bad_t3_parent=(pl.col("parent_prefix") == "t3_") & (pl.col("parent_id") != pl.col("thread_id")))
        extras[sub]["t3_parent_not_thread_root"] = int(t["bad_t3_parent"].sum())
        parts.append(t)
    items = pl.concat(parts, how="diagonal_relaxed")
    items = finish(ds, items, ["community"], ["community"], bots, ["community"])

    # contest mode / suggested sort per subreddit
    roots = items.filter(pl.col("is_root"))
    cm = roots.group_by("community").agg(
        threads=pl.len(),
        contest_mode_threads=pl.col("contest_mode").fill_null(False).sum(),
        contest_mode_threads_with_comments=(pl.col("contest_mode").fill_null(False) & (pl.col("thread_size") > 0)).sum(),
        contest_mode_comments=pl.col("thread_size").filter(pl.col("contest_mode").fill_null(False)).sum(),
        contest_mode_null=pl.col("contest_mode").is_null().sum(),
        suggested_sort_new=(pl.col("suggested_sort") == "new").sum(),
        suggested_sort_new_comments=pl.col("thread_size").filter(pl.col("suggested_sort") == "new").sum(),
        suggested_sort_random=(pl.col("suggested_sort") == "random").sum(),
        suggested_sort_qa=(pl.col("suggested_sort") == "qa").sum(),
        suggested_sort_confidence=(pl.col("suggested_sort") == "confidence").sum(),
        suggested_sort_other=(~pl.col("suggested_sort").is_in(["new", "random", "qa", "confidence"])
                              & pl.col("suggested_sort").is_not_null()).sum(),
        suggested_sort_null=pl.col("suggested_sort").is_null().sum(),
    ).sort("community")
    cm.write_csv(out / "step1_contest_mode.csv")
    cmy = roots.with_columns(year=pl.from_epoch(pl.col("created").cast(pl.Int64), "s").dt.year()).group_by(
        "community", "year").agg(threads=pl.len(), contest_mode=pl.col("contest_mode").fill_null(False).sum(),
                                 suggested_new=(pl.col("suggested_sort") == "new").sum()).sort("community", "year")
    cmy.write_csv(out / "step1_contest_mode_by_year.csv")
    roots.filter(pl.col("contest_mode").fill_null(False)).select(
        "community", "thread_id", "created", "thread_size", "claimed_comments", "suggested_sort").write_csv(
        out / "step1_contest_mode_threads.csv")

    # staleness
    st = staleness_table(items, ["community"])
    st.write_csv(out / "step1_staleness_quantiles.csv")
    sty = staleness_table(items.with_columns(
        year=pl.from_epoch(pl.col("created").cast(pl.Int64), "s").dt.year()), ["community", "year"])
    sty.write_csv(out / "step1_staleness_by_year.csv")
    plot_staleness(items, ds, "community", "Reddit: score-snapshot staleness (retrieved_on − created_utc)",
                   "threads with submission created 2018-01-01..2021-12-31")
    plot_lag_by_month(items, ds)
    (out / "step1_extra.json").write_text(json.dumps(extras, indent=2, default=str))
    bot_hits(items, out)
    print(integ_print(out))


def plot_lag_by_month(items: pl.DataFrame, ds: str):
    _, fig, _ = dirs(ds)
    d = items.filter(~pl.col("is_root")).with_columns(
        lag=pl.col("retrieved") - pl.col("created"),
        month=pl.from_epoch(pl.col("created").cast(pl.Int64), "s").dt.truncate("1mo"),
    ).group_by("community", "month").agg(
        q10=pl.col("lag").quantile(0.1), q50=pl.col("lag").median(), q90=pl.col("lag").quantile(0.9),
        missing=pl.col("lag").is_null().mean(),
    ).sort("community", "month")
    d.write_csv(ROOT / "outputs" / ds / "step1_staleness_by_month.csv")
    f, ax = plt.subplots(figsize=(10, 4))
    for i, gname in enumerate(sorted(d["community"].unique())):
        x = d.filter(pl.col("community") == gname)
        ax.plot(x["month"], x["q50"], label=f"r/{gname} median", color=f"C{i}")
        ax.fill_between(x["month"], x["q10"], x["q90"], alpha=0.15, color=f"C{i}", label=f"r/{gname} 10–90%")
    for t, lab in [(60, "1 min"), (3600, "1 h"), (86400, "1 d"), (30 * 86400, "30 d"), (365 * 86400, "1 y")]:
        ax.axhline(t, color="0.7", lw=0.8, ls=":"); ax.text(ax.get_xlim()[0], t, " " + lab, fontsize=7, color="0.4", va="bottom")
    ax.set_yscale("log"); ax.set_ylabel("retrieved_on − created_utc (s, log)")
    ax.set_xlabel("comment month"); ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Reddit comments: how stale is the score snapshot, by month of comment")
    f.tight_layout(); f.savefig(ROOT / "figures" / ds / "step1_staleness_by_month.png", dpi=130); plt.close(f)


def bot_hits(items: pl.DataFrame, out: Path):
    h = items.filter(~pl.col("is_root") & (pl.col("bot_listed") | pl.col("bot_name_rule"))).group_by(
        "community", "author", "bot_listed", "bot_name_rule").len().sort("len", descending=True)
    h.write_csv(out / "step1_bot_author_hits.csv")


def integ_print(out: Path) -> str:
    return (out / "step1_integrity.csv").read_text()


def run_hn():
    ds = "hn"
    out, fig, proc = dirs(ds)
    print("[hn] loading", flush=True)
    nodes, extra = load_hn()
    print(f"[hn] {nodes.height:,} items since window start; building trees", flush=True)
    t = build_trees(nodes)
    del nodes
    # root class / window attaches through the inferred thread
    rootinfo = t.filter(pl.col("is_root")).select(
        pl.col("id").cast(pl.String).alias("thread_id"), pl.col("post_class").alias("_pc"), pl.col("root_in_window").alias("_rw"))
    t = t.join(rootinfo, on="thread_id", how="left").with_columns(
        community=pl.coalesce("post_class", "_pc"), in_window=pl.coalesce("_rw", pl.lit(False))).drop("_pc", "_rw")
    min_id = extra["hn_min_window_id"]
    # comments not reaching a root: either their chain leads to a pre-window thread (top's parent id
    # is below the first window id: out of scope) or to an item genuinely absent from the dump.
    unres = t.filter(~pl.col("is_root") & ~pl.col("reaches_root"))
    unres = unres.with_columns(pre_window=pl.col("top_missing_parent").cast(pl.Int64, strict=False) < min_id)
    extra["comments_on_pre_window_threads"] = int(unres["pre_window"].sum())
    extra["unattributable_orphans"] = int((~unres["pre_window"]).sum())
    extra["unattributable_missing_parents"] = int(unres.filter(~pl.col("pre_window") & pl.col("missing_parent")).height)
    extra["comments_on_post_window_threads"] = int(t.filter(~pl.col("is_root") & pl.col("reaches_root") & ~pl.col("in_window")).height)
    items = t.filter(pl.col("in_window")).drop("in_window")
    items = finish(ds, items, ["community"], ["community"], None, [])
    # HN has no per-item retrieval time: staleness of the score snapshot cannot be computed.
    (out / "step1_extra.json").write_text(json.dumps(extra, indent=2, default=str))
    print(integ_print(out))


def run_moltbook():
    ds = "moltbook"
    out, fig, proc = dirs(ds)
    print("[moltbook] loading", flush=True)
    nodes, extra = load_moltbook()
    stream = extra.pop("_stream")
    print(f"[moltbook] {nodes.height:,} nodes; building trees", flush=True)
    t = build_trees(nodes)
    del nodes
    items = finish(ds, t, ["community"], [], None, [])
    st = staleness_table(items.with_columns(bound=pl.col("snap_is_dump_bound").fill_null(True)),
                         ["community", "bound"])
    st.write_csv(out / "step1_staleness_quantiles.csv")
    plot_staleness(items, ds, "community", "Moltbook: score-snapshot staleness (latest snapshot − created_at)",
                   "snapshot time = fetched_at when present, else end of dump_date (upper bound)")
    gaps = coverage(stream, ds)
    extra["largest_gaps"] = gaps.group_by("stream", maintain_order=True).head(3).to_dicts()
    (out / "step1_extra.json").write_text(json.dumps(extra, indent=2, default=str))
    print(integ_print(out))


def load_moltbook_crawl() -> tuple[pl.DataFrame, dict]:
    """giordano-dm/moltbook-crawl (SQLite). Score = upvotes - downvotes at crawled_at."""
    import sqlite3
    lo, hi = (ep(x) for x in P["moltbook"]["thread_window_utc"])
    db = RAW / "moltbook_crawl" / "moltbook_upload.db"
    sq = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    def ts(col):  # ISO strings with or without offset; crawled_at is naive UTC
        return (pl.col(col).str.replace(r"\+00:00$", "").str.replace("T", " ")
                .str.to_datetime(time_zone="UTC", strict=False, time_unit="us").dt.epoch("us") / 1e6)

    posts = pl.read_database(
        "select id, author_id, author_name, submolt, upvotes, downvotes, comment_count, created_at, crawled_at from posts",
        sq).with_columns(created=ts("created_at"), retrieved=ts("crawled_at"))
    com = pl.read_database(
        "select id, post_id, author_id, author_name, parent_id, upvotes, downvotes, created_at, crawled_at, depth, "
        "(content is null or content in ('', '[deleted]', '[removed]')) as content_missing from comments",
        sq).with_columns(created=ts("created_at"), retrieved=ts("crawled_at"))
    sq.close()
    extra = {"posts_total": posts.height, "comments_total": com.height,
             "posts_unparsed_created": int(posts["created"].null_count()),
             "comments_unparsed_created": int(com["created"].null_count()),
             "comments_on_unknown_posts": int((~com["post_id"].is_in(posts["id"].implode())).sum())}
    stream = pl.concat([com.select(t="created", kind=pl.lit("comment")),
                        posts.select(t="created", kind=pl.lit("post"))]).drop_nulls()
    win = posts.filter(pl.col("created").is_between(lo, hi))
    roots = win.select(
        thread_id="id", id="id", parent_id=pl.lit(None, pl.String), is_root=pl.lit(True), created="created",
        retrieved="retrieved", author=pl.col("author_id"), author_name="author_name",
        body_status=pl.lit("ok"), score=(pl.col("upvotes") - pl.col("downvotes")),
        claimed_comments="comment_count", submolt="submolt", stored_depth=pl.lit(None, pl.Int64))
    comments = com.filter(pl.col("post_id").is_in(win["id"].implode())).select(
        thread_id="post_id", id="id",
        parent_id=pl.when(pl.col("parent_id").is_null() | (pl.col("parent_id") == "")).then(pl.col("post_id"))
        .otherwise(pl.col("parent_id")),
        is_root=pl.lit(False), created="created", retrieved="retrieved", author=pl.col("author_id"),
        author_name="author_name",
        body_status=pl.when(pl.col("content_missing").cast(pl.Boolean)).then(pl.lit("deleted")).otherwise(pl.lit("ok")),
        score=(pl.col("upvotes") - pl.col("downvotes")), claimed_comments=pl.lit(None, pl.Int64),
        submolt=pl.lit(None, pl.String), stored_depth=pl.col("depth").cast(pl.Int64))
    nodes = pl.concat([roots, comments], how="vertical_relaxed").with_columns(community=pl.lit("moltbook_crawl"))
    extra["_stream"] = stream
    return nodes, extra


def run_moltbook_crawl():
    ds = "moltbook_crawl"
    out, fig, proc = dirs(ds)
    print("[moltbook_crawl] loading", flush=True)
    nodes, extra = load_moltbook_crawl()
    stream = extra.pop("_stream")
    print(f"[moltbook_crawl] {nodes.height:,} nodes; building trees", flush=True)
    t = build_trees(nodes)
    del nodes
    items = finish(ds, t, ["community"], [], None, [])
    com = items.filter(~pl.col("is_root"))
    # cross-check reconstructed depth against the crawler's stored depth (0 = top-level)
    chk = com.filter(pl.col("depth").is_not_null() & pl.col("stored_depth").is_not_null())
    extra["depth_check_n"] = chk.height
    extra["depth_check_mismatch"] = int((chk["depth"] != chk["stored_depth"] + 1).sum())
    # 100-stored-comments cap: threads whose stored comments fall short of the API-reported count
    roots = items.filter(pl.col("is_root"))
    cap = roots.select(
        threads_at_cap=((pl.col("thread_size") >= 100) & (pl.col("claimed_comments") > pl.col("thread_size"))).sum(),
        threads_any_shortfall=(pl.col("claimed_comments") > pl.col("thread_size")).sum(),
        comments_in_capped_threads=pl.col("thread_size").filter(
            (pl.col("thread_size") >= 100) & (pl.col("claimed_comments") > pl.col("thread_size"))).sum(),
        claimed_in_capped_threads=pl.col("claimed_comments").filter(
            (pl.col("thread_size") >= 100) & (pl.col("claimed_comments") > pl.col("thread_size"))).sum(),
        max_stored_size=pl.col("thread_size").max(),
    )
    cap.write_csv(out / "step1_comment_cap.csv")
    extra["comment_cap"] = cap.to_dicts()[0]
    st = staleness_table(items, ["community"])
    st.write_csv(out / "step1_staleness_quantiles.csv")
    plot_staleness(items, ds, "community", "Moltbook crawl: score-snapshot staleness (crawled_at − created_at)",
                   "threads with post created 2026-01-28..2026-02-04")
    gaps = coverage(stream, ds)
    extra["largest_gaps"] = gaps.group_by("stream", maintain_order=True).head(3).to_dicts()
    (out / "step1_extra.json").write_text(json.dumps(extra, indent=2, default=str))
    print(integ_print(out))



def coverage(stream: pl.DataFrame, ds: str) -> pl.DataFrame:
    out, fig, _ = dirs(ds)
    rows = []
    for kind in ["comment", "post", "any"]:
        s = stream if kind == "any" else stream.filter(pl.col("kind") == kind)
        t = np.sort(s["t"].to_numpy())
        if len(t) < 2:
            continue
        dt = np.diff(t)
        for i in np.argsort(dt)[::-1][:10]:
            rows.append({"stream": kind, "gap_hours": dt[i] / 3600,
                         "gap_start_utc": datetime.fromtimestamp(t[i], timezone.utc).isoformat(),
                         "gap_end_utc": datetime.fromtimestamp(t[i + 1], timezone.utc).isoformat()})
    gaps = pl.DataFrame(rows).sort(["stream", "gap_hours"], descending=[False, True])
    gaps.write_csv(out / "step1_coverage_gaps.csv")
    hourly = stream.with_columns(hour=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.truncate("1h")).group_by(
        "hour", "kind").len()
    grid = pl.DataFrame({"hour": pl.datetime_range(hourly["hour"].min(), hourly["hour"].max(), "1h", eager=True)})
    hourly = (grid.join(pl.DataFrame({"kind": ["comment", "post"]}), how="cross")
              .join(hourly, on=["hour", "kind"], how="left").with_columns(pl.col("len").fill_null(0)).sort("hour"))
    hourly.write_csv(out / "step1_hourly_counts.csv")
    # contiguous comment segments inside the thread window (split at comment-stream gaps > 6 h)
    lo, hi = (ep(x) for x in P["moltbook"]["thread_window_utc"])
    tc = np.sort(stream.filter(pl.col("kind") == "comment")["t"].to_numpy())
    tc = tc[(tc >= lo) & (tc <= hi)]
    segs = []
    if len(tc):
        cuts = np.where(np.diff(tc) > 6 * 3600)[0]
        starts = np.r_[0, cuts + 1]; ends = np.r_[cuts, len(tc) - 1]
        for a, b in zip(starts, ends):
            segs.append({"segment_start_utc": datetime.fromtimestamp(tc[a], timezone.utc).isoformat(),
                         "segment_end_utc": datetime.fromtimestamp(tc[b], timezone.utc).isoformat(),
                         "hours": (tc[b] - tc[a]) / 3600, "comments": int(b - a + 1)})
    pl.DataFrame(segs).write_csv(out / "step1_contiguous_comment_segments.csv")
    f, ax = plt.subplots(figsize=(11, 3.8))
    for i, kind in enumerate(["post", "comment"]):
        h = hourly.filter(pl.col("kind") == kind)
        ax.step(h["hour"], h["len"], where="post", label=f"{kind}s per hour", color=f"C{i}", lw=1)
    lo, hi = (datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in P["moltbook"]["thread_window_utc"])
    ax.axvspan(lo, hi, color="0.9", zorder=0, label="thread window")
    ax.set_yscale("symlog", linthresh=10); ax.set_ylabel("items created per hour")
    ax.legend(fontsize=8, frameon=False); ax.set_title(f"{ds}: creation-time coverage (all posts and comments in the source)")
    f.tight_layout(); f.savefig(fig / "step1_coverage.png", dpi=130); plt.close(f)
    return gaps


def refresh_reddit_bot_flags():
    """Recompute bot_listed from the current bot list without rebuilding trees (streamed), then
    regenerate the integrity and bot-hit tables. Used when the explicit list changes."""
    ds = "reddit"
    out, _, proc = dirs(ds)
    bots = sorted(load_bot_list(P["author_filters"]["bot_list_file"]))
    src, tmp = proc / "items.parquet", proc / "items.refresh.parquet"
    (pl.scan_parquet(src)
     .with_columns(bot_listed=pl.col("author").is_not_null() & pl.col("author").str.to_lowercase().is_in(bots))
     .sink_parquet(tmp))
    tmp.replace(src)
    cols = ["community", "is_root", "thread_size", "reaches_root", "missing_parent", "orphan", "cross_thread_parent",
            "cyclic", "time_since_parent", "author", "body_status", "is_root_author", "self_reply", "depth",
            "claimed_comments", "bot_listed", "bot_name_rule", "top_active", "split"]
    items = pl.read_parquet(src, columns=cols)
    integ = integrity(items, ["community"])
    integ.write_csv(out / "step1_integrity.csv")
    to_long(integ, ["community"]).write_csv(out / "step1_integrity_long.csv")
    bot_hits(items, out)
    print(integ_print(out))


if __name__ == "__main__":
    which = sys.argv[1:] or ["moltbook", "moltbook_crawl", "reddit", "hn"]
    for w in which:
        {"reddit": run_reddit, "hn": run_hn, "moltbook": run_moltbook, "moltbook_crawl": run_moltbook_crawl,
         "refresh_reddit_bot_flags": refresh_reddit_bot_flags}[w]()
