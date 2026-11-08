"""Audit data checks (run before step 4b).

Usage: python scripts/checks.py [1 2 3 4 5 6 7 8 9 10]   (default: all)

Each check writes a record {id, name, threshold, result, status, notes} to
outputs/audit/checks/<id>.json and its tables to outputs/audit/check<id>_*.csv. reports/audit.md
is written from these records. Status values: PASS, FAIL (fixed: see notes), FLAG (needs a
decision or cannot be verified), INFO (documentation item).
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, items_paths, load_partitions, proc_dir  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model.units import (build_units, cap_fit, ep, hn_population, load, moltbook_population,  # noqa: E402
                               reddit_population)

P = load_partitions()
OUT = ROOT / "outputs" / "audit"
(OUT / "checks").mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(P["seed"])
STUDY_SUBS = P["reddit"]["subreddits"]
ALL_SUBS = STUDY_SUBS + [k for k, v in P["reddit"]["proxy_c_subreddits"].items() if isinstance(v, dict)]


def record(cid: int, name: str, threshold: str, result: dict, status: str, notes: str = ""):
    rec = {"id": cid, "name": name, "threshold": threshold, "result": result, "status": status, "notes": notes}
    (OUT / "checks" / f"{cid:02d}.json").write_text(json.dumps(rec, indent=2, default=float))
    print(f"[check {cid}] {status}: {name}", flush=True)
    return rec


def mcf_mu(U: dict, H_s: float, weight: str = "raw") -> float:
    par, rep = U["parents"], U["replies"]
    w = par["w_pa"].to_numpy() if weight == "per_author" else np.ones(par.height)
    wmap = dict(zip(par["id"].to_list(), w)) if weight == "per_author" else None
    r = rep["delay"].to_numpy()
    wr = np.array([wmap[x] for x in rep["parent"].to_list()]) if wmap else np.ones(len(r))
    return H.mean_cumulative(H.SortedCensor(par["c"].to_numpy()), w, r, wr, H_s)


# ------------------------------------------------------------------------------------------
def check1():
    """GECLM completeness: stored comments per thread vs submission num_comments."""
    rows = []
    for sub in ALL_SUBS:
        r = pl.scan_parquet(items_paths("reddit", [sub])).filter(pl.col("is_root")).select(
            "thread_size", "claimed_comments", "created").collect()
        r = r.with_columns(year=pl.from_epoch(pl.col("created").cast(pl.Int64), "s").dt.year(),
                           ratio=pl.col("thread_size") / pl.col("claimed_comments"))
        g = r.filter(pl.col("claimed_comments") >= 10).group_by("year").agg(
            threads=pl.len(), median_ratio=pl.col("ratio").median(), p10=pl.col("ratio").quantile(0.1),
            p90=pl.col("ratio").quantile(0.9), share_lt_0_5=(pl.col("ratio") < 0.5).mean(),
            share_gt_1_5=(pl.col("ratio") > 1.5).mean()).with_columns(subreddit=pl.lit(sub)).sort("year")
        rows.append(g)
    t = pl.concat(rows)
    t.write_csv(OUT / "check01_completeness.csv")
    big = t.filter(pl.col("threads") >= 100)
    bad = big.filter(~pl.col("median_ratio").is_between(0.9, 1.15) | (pl.col("share_lt_0_5") > 0.05))
    return record(1, "GECLM completeness: stored comments vs submission num_comments",
                  "every subreddit-year with >= 100 threads (num_comments >= 10): median stored/claimed in [0.90, 1.15] "
                  "and <= 5% of threads with ratio < 0.5",
                  {"subreddit_years": big.height, "failing": bad.to_dicts(),
                   "median_ratio_range": [float(big["median_ratio"].min()), float(big["median_ratio"].max())],
                   "max_share_lt_0_5": float(big["share_lt_0_5"].max())},
                  "PASS" if bad.height == 0 else "FLAG",
                  "num_comments is the submission snapshot at its retrieval time; stored comments include deleted/removed "
                  "ones and any posted after that snapshot, so ratios slightly above 1 are expected.")


# ------------------------------------------------------------------------------------------
def check2():
    """Negative or zero parent-to-child delays; timestamp units and time zones."""
    rows = []
    sources = [("reddit", s) for s in ALL_SUBS] + [("hn", None), ("moltbook_crawl", None)]
    for ds, sub in sources:
        lf = pl.scan_parquet(items_paths(ds, [sub] if sub else None)).filter(
            ~pl.col("is_root") & pl.col("time_since_parent").is_not_null())
        a = lf.select(n=pl.len(), neg=(pl.col("time_since_parent") < 0).sum(),
                      zero=(pl.col("time_since_parent") == 0).sum(),
                      min_neg=pl.col("time_since_parent").min(),
                      created_min=pl.col("created").min(), created_max=pl.col("created").max(),
                      frac_nonint=((pl.col("created") - pl.col("created").floor()) > 0).mean()).collect().row(0, named=True)
        rows.append({"dataset": ds, "community": sub or ds, **a, "neg_share": a["neg"] / a["n"], "zero_share": a["zero"] / a["n"]})
    t = pl.DataFrame(rows).with_columns(
        created_min_utc=pl.from_epoch(pl.col("created_min").cast(pl.Int64), "s").cast(pl.String),
        created_max_utc=pl.from_epoch(pl.col("created_max").cast(pl.Int64), "s").cast(pl.String))
    t.write_csv(OUT / "check02_delays.csv")
    bad = t.filter(pl.col("neg_share") >= 1e-4)
    return record(2, "Negative or zero parent-to-child delays; timestamp units and zones",
                  "negative delays < 0.01% of comments in every dataset; all timestamps epoch seconds (UTC) inside the "
                  "expected observation range",
                  {"per_dataset": t.select("community", "n", "neg", "neg_share", "zero", "zero_share", "min_neg",
                                           "created_min_utc", "created_max_utc", "frac_nonint").to_dicts()},
                  "PASS" if bad.height == 0 else "FLAG",
                  "Units: Reddit created_utc and HN time are integer epoch seconds (UTC by definition); Moltbook "
                  "created_at is ISO 8601 with +00:00, parsed to UTC epoch seconds with microsecond fractions. Negative "
                  "delays are clipped to 0 and zero delays placed at 0.5 s in the timing fits. Zero delays are mostly "
                  "same-second bot replies (Reddit AutoModerator, step 1) and HN same-second replies.")


# ------------------------------------------------------------------------------------------
def _cap_compare(U, salt_rng):
    par = U["parents"]
    fit_all = par.filter(pl.col("fold") == "fit")
    capped = cap_fit(par)
    idx = salt_rng.choice(fit_all.height, size=min(capped.height, fit_all.height), replace=False)
    rnd = fit_all[np.sort(idx)]

    def desc(d):
        return {"n": d.height, "share_replied": float(d["d"].mean()), "mean_log_c": float(np.log(d["c"]).mean()),
                "share_depth1": float((d["depth"] == 1).mean()), "top1_share": float(d["top_active"].fill_null(False).mean()),
                "mean_replies": float(d["n_replies"].mean())}
    return fit_all, capped, rnd, {"full_fit_fold": desc(fit_all), "hash_cap": desc(capped), "numpy_random": desc(rnd)}


def check3():
    """How the 2M-parent cap was sampled; reconfirm step 3/4 numbers on an independent random sample."""
    import importlib.util
    res, rows = {}, []
    # step 3: r/buildapc and HN link non-root (the two capped units)
    for ds, comm in [("reddit", "buildapc"), ("hn", "link")]:
        if ds == "reddit":
            it, f = reddit_population(comm)
            U = build_units(it, f["nonroot"], f"reddit:{comm}")
        else:
            it, f = hn_population(comm, exclude_dead=False)  # as in step 3
            U = build_units(it, f["nonroot"], f"hn:{comm}")
        fit_all, capped, rnd, desc = _cap_compare(U, RNG)
        fh = H.fit_first_reply("weibull", capped["t"].to_numpy(), capped["d"].to_numpy())
        fr = H.fit_first_reply("weibull", rnd["t"].to_numpy(), rnd["d"].to_numpy())
        res[f"{ds}:{comm}"] = {"describe": desc, "weibull_hash_cap": {"mu": fh.mu, "shape": float(np.exp(fh.theta[0]))},
                               "weibull_numpy_random": {"mu": fr.mu, "shape": float(np.exp(fr.theta[0]))}}
        rows.append({"unit": f"step3 {comm} nonroot", "mu_hash": fh.mu, "mu_random": fr.mu,
                     "rel_diff_mu": fr.mu / fh.mu - 1, "shape_hash": float(np.exp(fh.theta[0])),
                     "shape_random": float(np.exp(fr.theta[0])),
                     "abs_diff_shape": abs(float(np.exp(fr.theta[0]) - np.exp(fh.theta[0])))})
        del it, U
        gc.collect()
    # step 4 (a) r/buildapc: refit with a numpy-random cap instead of the hash cap
    spec = importlib.util.spec_from_file_location("s4", ROOT / "scripts" / "step4_position.py")
    s4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s4)
    import forum_model.units as units_mod
    orig = s4.cap_fit

    def random_cap(par, cap=None):
        cap = cap or P["step3"]["fit_cap_parents"]
        fit = par.filter(pl.col("fold") == "fit")
        if fit.height <= cap:
            return fit
        idx = np.sort(np.random.default_rng(P["seed"] + 1).choice(fit.height, cap, replace=False))
        return fit[idx]
    s4.cap_fit = random_cap
    rows_a = [r for r in s4.proxy_a("buildapc") if r.get("table") == "summary" and r["weighting"] == "per_author"][0]
    s4.cap_fit = orig
    ref = pl.read_csv(ROOT / "outputs" / "combined" / "step4_a_summary.csv").filter(
        (pl.col("community") == "buildapc") & (pl.col("weighting") == "per_author")).row(0, named=True)
    rows.append({"unit": "step4 (a) buildapc per-author", "beta_hash": ref["beta_log_position"],
                 "beta_random": rows_a["beta_log_position"], "se_beta": ref["se_beta"],
                 "flattening_hash": ref["flattening_M1_minus_M0"], "flattening_random": rows_a["flattening_M1_minus_M0"],
                 "weibull_shape_M0_hash": ref["weibull_shape_M0"], "weibull_shape_M0_random": rows_a["weibull_shape_M0"]})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "check03_cap.csv")
    ok = all(abs(r.get("rel_diff_mu", 0)) < 0.01 and r.get("abs_diff_shape", 0) < 0.01 for r in rows[:2])
    b = rows[2]
    ok_b = abs(b["beta_random"] - b["beta_hash"]) < max(0.01, 4 * b["se_beta"]) and \
        abs(b["flattening_random"] - b["flattening_hash"]) < 0.01
    return record(3, "2M-parent cap: sampling method and reconfirmation",
                  "independent numpy-random sample of the same size reproduces step-3 mu within 1% and Weibull shape "
                  "within 0.01, and step-4 (a) beta within max(0.01, 4 SE) and flattening within 0.01",
                  {"units": rows, "describe": res}, "PASS" if (ok and ok_b) else "FLAG",
                  "The cap keeps the 2,000,000 fit-fold parents with the lowest sha256('<seed>:cap:<comment id>'): a "
                  "deterministic simple random sample of comments, independent of every covariate (not stratified by "
                  "thread). Thread clustering is handled in the bootstrap, not the cap.")


# ------------------------------------------------------------------------------------------
def check4():
    """Per-author weight computation across inner folds."""
    rows = []
    for ds, comm, U in _nonroot_units():
        par = U["parents"]
        known = par.filter(pl.col("author").is_not_null()).group_by("fold", "author").agg(s=pl.col("w_pa").sum())
        rows.append({"dataset": ds, "community": comm, "max_abs_sum_minus_1": float((known["s"] - 1).abs().max()),
                     "unknown_weight_max": float(par.filter(pl.col("author").is_null())["w_pa"].max() or 0.0),
                     "authors_in_both_folds": int(known.group_by("author").len().filter(pl.col("len") == 2).height),
                     "authors": int(known["author"].n_unique())})
    t = pl.DataFrame(rows)
    t.write_csv(OUT / "check04_weights.csv")
    ok = (t["max_abs_sum_minus_1"].max() < 1e-9) and (t["unknown_weight_max"].max() == 0)
    return record(4, "Per-author weights across inner folds",
                  "within each inner fold every known author's weights sum to 1 (|sum - 1| < 1e-9); unknown authors weight 0",
                  {"per_population": t.to_dicts()}, "PASS" if ok else "FAIL",
                  "Weights are 1/(the author's parents in that fold, in that population), so fit-fold and held-out-fold "
                  "weights never mix; an author present in both folds is weighted separately in each. After the 2M cap "
                  "weights are not renormalised: the cap is a simple random sample, so each author keeps the same "
                  "expected share. Step 4 recomputes weights within each proxy's parent set and fold; the early/late "
                  "split pools the calibration split (descriptive) and recomputes there.")


def _nonroot_units():
    for sub in STUDY_SUBS:
        it, f = reddit_population(sub, ["body_status"])
        yield "reddit", sub, build_units(it, f["nonroot"], f"reddit:{sub}", keep=["body_status"])
        del it
        gc.collect()
    hn = load("hn", None, ["body_status"])
    for comm in ["ask_hn", "link", "show_hn"]:
        it, f = hn_population(comm, hn_items=hn)
        yield "hn", comm, build_units(it, f["nonroot"], f"hn:{comm}", keep=["body_status"])
    del hn
    it, f = moltbook_population("complete", ["body_status"])
    yield "moltbook_crawl", "moltbook", build_units(it, f["nonroot"], "moltbook_crawl", keep=["body_status"])


# ------------------------------------------------------------------------------------------
def check5():
    """Removed-comment handling."""
    rows = []
    for ds, comm, U in _nonroot_units():
        par = U["parents"]
        removed = (par["body_status"].is_in(["removed", "deleted", "dead"])) | par["author"].is_null()
        rep = U["replies"]
        rem_ids = par.filter(removed)["id"]
        n_rep_to_removed = rep.filter(pl.col("parent").is_in(rem_ids.implode())).height
        mu_all = mcf_mu(U, 3 * 86400)
        keep = {"parents": par.filter(~removed), "replies": rep.filter(~pl.col("parent").is_in(rem_ids.implode()))}
        mu_kept = mcf_mu(keep, 3 * 86400)
        rows.append({"dataset": ds, "community": comm, "parents": par.height,
                     "share_removed_or_deleted_or_unknown": float(removed.mean()),
                     "share_body_removed": float((par["body_status"] == "removed").mean()),
                     "share_body_deleted": float((par["body_status"] == "deleted").mean()),
                     "share_author_unknown": float(par["author"].is_null().mean()),
                     "replies_to_these_parents": n_rep_to_removed, "share_of_all_replies": n_rep_to_removed / max(rep.height, 1),
                     "mu_mcf3d_all_parents": mu_all, "mu_mcf3d_excluding": mu_kept, "rel_change": mu_kept / mu_all - 1})
    t = pl.DataFrame(rows)
    t.write_csv(OUT / "check05_removed.csv")
    worst = float(t["rel_change"].abs().max())
    return record(5, "Removed-comment handling",
                  "removed/deleted parents are handled identically to others (kept as parents; their replies are events); "
                  "excluding them changes mu (MCF, 3 days) by < 5% in every population",
                  {"per_population": t.to_dicts(), "max_abs_rel_change": worst},
                  "PASS" if worst < 0.05 else "FLAG",
                  "Removed (by moderators) and deleted (by authors) comments stay in the trees, as parents and as reply "
                  "events, because they were visible before removal and their replies are real. Exceptions: proxy (c) "
                  "r/AskHistorians drops removed parents (decision after step 3); HN 'dead' items are excluded since the "
                  "audit (check 6).")


# ------------------------------------------------------------------------------------------
def check6():
    """HN dead/flagged items in parents, events and sibling positions."""
    hn = load("hn", None, ["body_status", "time_since_root"])
    rows = []
    for comm in ["ask_hn", "link", "show_hn"]:
        it_in, f = hn_population(comm, hn_items=hn, exclude_dead=False)
        U_in = build_units(it_in, f["nonroot"], f"hn:{comm}", keep=["body_status"])
        it_ex, f2 = hn_population(comm, hn_items=hn, exclude_dead=True)
        U_ex = build_units(it_ex, f2["nonroot"], f"hn:{comm}")
        par = U_in["parents"]
        dead_par = par.filter(pl.col("body_status") == "dead")
        com = it_in.filter(~pl.col("is_root") & (pl.col("split") == "calibration"))
        rows.append({"community": comm,
                     "dead_share_of_comments": float((com["body_status"] == "dead").mean()),
                     "dead_parents_before_fix": dead_par.height, "dead_parent_share_before": dead_par.height / par.height,
                     "mu_mcf3d_dead_included": mcf_mu(U_in, 3 * 86400),
                     "mu_mcf3d_dead_excluded": mcf_mu(U_ex, 3 * 86400),
                     "share_replied_included": float(par["d"].mean()),
                     "share_replied_excluded": float(U_ex["parents"]["d"].mean())})
        rows[-1]["mu_rel_change"] = rows[-1]["mu_mcf3d_dead_excluded"] / rows[-1]["mu_mcf3d_dead_included"] - 1
        del it_in, it_ex, U_in, U_ex
        gc.collect()
    t = pl.DataFrame(rows)
    t.write_csv(OUT / "check06_hn_dead.csv")
    return record(6, "HN dead/flagged items excluded from parents, events and sibling positions",
                  "dead items appear in none of parents, reply events, or sibling positions",
                  {"per_class": t.to_dicts()}, "FAIL",
                  "Before the audit dead comments were kept (only bots and orphans were excluded): they were parents, "
                  "reply events and siblings in proxy (b). FIXED: forum_model.units.hn_population now excludes them by "
                  "default (column 'excluded'), and step 4 sibling positions skip them. Step 3, 3b and 4 HN outputs are "
                  "recomputed with the fix (see reports/audit.md). 'Flagged' items are the dead ones in this dump; "
                  "'deleted' items (removed by their author after being visible) are kept.")


# ------------------------------------------------------------------------------------------
def check7():
    """Moltbook created_at semantics and the 30-second root reply median."""
    import sqlite3
    db = ROOT / "data" / "raw" / "moltbook_crawl" / "moltbook_upload.db"
    sq = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sample = sq.execute("select created_at, crawled_at from comments limit 5").fetchall()
    fmt = sq.execute("""select sum(created_at like '%+00:00'), sum(created_at like '%Z'), count(*),
                               sum(length(created_at)) / count(*) from comments""").fetchone()
    sq.close()
    it = pl.read_parquet(proc_dir("moltbook_crawl") / "items.parquet",
                         columns=["thread_id", "id", "is_root", "created", "retrieved", "author_name", "author",
                                  "time_since_parent", "depth", "parent_id"])
    com = it.filter(~pl.col("is_root"))
    after_crawl = float((com["created"] > com["retrieved"]).mean())
    dup_ts = com.group_by("created").len().filter(pl.col("len") > 1)["len"].sum()
    it_c, f = moltbook_population("complete", ["author_name", "time_since_parent"])
    U = build_units(it_c, f["root_live"], "moltbook_crawl")
    par = U["parents"]
    first = par.filter(pl.col("d"))
    q = {f"q{int(p * 100)}_s": float(first["t"].quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    shares = {k: float((first["t"] < v).mean()) for k, v in [("lt_5s", 5), ("lt_30s", 30), ("lt_60s", 60), ("lt_10min", 600)]}
    # who writes the first reply on posts
    kids = it_c.filter(~pl.col("is_root") & (pl.col("depth") == 1)).sort("created").group_by("thread_id").agg(
        first_author=pl.col("author_name").first(), first_delay=pl.col("time_since_parent").first())
    top = kids.group_by("first_author").agg(n=pl.len(), med_delay_s=pl.col("first_delay").median()).sort("n", descending=True)
    top10_share = float(top.head(10)["n"].sum() / kids.height)
    top.head(20).write_csv(OUT / "check07_first_repliers.csv")
    # same check excluding the top-10 first repliers
    rest = kids.filter(~pl.col("first_author").is_in(top.head(10)["first_author"].implode()))
    res = {"timestamp_format": {"sample": sample, "share_with_offset": fmt[0] / fmt[2], "mean_length": fmt[3]},
           "share_comments_created_after_crawled": after_crawl, "duplicate_timestamp_comments": int(dup_ts or 0),
           "root_first_reply_quantiles": q, "root_first_reply_shares": shares,
           "top10_first_repliers_share": top10_share, "top10_first_repliers": top.head(10).to_dicts(),
           "median_first_reply_s_excluding_top10": float(rest["first_delay"].median())}
    status = "PASS" if after_crawl < 1e-4 else "FLAG"
    return record(7, "Moltbook created_at semantics; 30-second root reply median",
                  "created_at is a UTC server timestamp: no comment created after it was crawled (< 0.01%), parent-to-child "
                  "delays non-negative (check 2); the fast root replies are reported with who writes them",
                  res, status,
                  "created_at strings carry microseconds and an explicit +00:00 offset (server-assigned UTC). The root "
                  "first-reply speed is interpreted in reports/audit.md together with the share written by the most "
                  "frequent first repliers.")


# ------------------------------------------------------------------------------------------
def check8():
    """Stability of contest_mode / suggested_sort for proxy (c) and (d) threads."""
    out = {}
    # (c) r/Fantasy contest mode: API obscures comment scores to 1 for non-moderators while contest mode is on
    for sub, sel in [("Fantasy", pl.col("contest_mode").fill_null(False)), ("AskHistorians", pl.col("suggested_sort") == "random")]:
        it = load("reddit", [sub], ["contest_mode", "suggested_sort", "score", "retrieved"])
        roots = it.filter(pl.col("is_root"))
        proxy_ids = roots.filter(sel)["thread_id"]
        ctrl_ids = roots.filter(~sel & (pl.col("thread_size") >= 20))["thread_id"]
        com = it.filter(~pl.col("is_root"))
        per_thread = com.filter(pl.col("thread_id").is_in(proxy_ids.implode())).group_by("thread_id").agg(
            n=pl.len(), share_score_1=(pl.col("score") == 1).mean(),
            median_retrieval_lag_d=((pl.col("retrieved") - pl.col("created")) / 86400).median())
        ctrl = com.filter(pl.col("thread_id").is_in(ctrl_ids.implode()))
        out[sub] = {"proxy_threads": proxy_ids.len(), "threads_with_comments": per_thread.height,
                    "median_share_score_1_proxy": float(per_thread["share_score_1"].median()),
                    "share_threads_score1_ge_0_9": float((per_thread["share_score_1"] >= 0.9).mean()),
                    "share_score_1_control_comments": float((ctrl["score"] == 1).mean()),
                    "median_retrieval_lag_days": float(per_thread["median_retrieval_lag_d"].median())}
        per_thread.write_csv(OUT / f"check08_{sub}_per_thread.csv")
    # recurring-series consistency (title patterns) for (c) and (d)
    series = {"Fantasy": ["self-promotion", "general discussion", "friday social"],
              "AskHistorians": ["sunday digest", "friday free-for-all", "thursday reading", "monday methods", "tuesday trivia"],
              "buildapc": ["simple questions", "build ready", "discussion thread", "megathread"],
              "philosophy": ["open discussion thread", "weekly discussion", "ama"]}
    lo, hi = (ep(x) for x in P["reddit"]["thread_window_utc"])
    srows = []
    for sub, pats in series.items():
        f = str(ROOT / "data" / "raw" / "reddit_submissions" / "data" / f"{sub}-*.parquet")
        for pat in pats:
            q = duckdb.sql(f"""select strftime(to_timestamp(try_cast(created_utc as double))::timestamp, '%Y') yr,
                    count(*) n, sum((contest_mode='True')::int) contest, sum((suggested_sort='random')::int) random_,
                    sum((suggested_sort='new')::int) new_
                    from read_parquet('{f}') where try_cast(created_utc as double) between {lo} and {hi}
                    and lower(title) like '%{pat}%' group by 1 order by 1""").pl()
            for r in q.iter_rows(named=True):
                srows.append({"subreddit": sub, "series": pat, **r})
    st = pl.DataFrame(srows)
    st.write_csv(OUT / "check08_series.csv")
    fan = out["Fantasy"]
    status = "PASS" if fan["share_threads_score1_ge_0_9"] >= 0.9 else "FLAG"
    return record(8, "Stability of contest_mode / suggested_sort (proxy c and d threads)",
                  "r/Fantasy contest threads: >= 90% of threads show score-obscuring (>= 90% of comments with score 1) "
                  "at comment retrieval; recurring series consistent in mode; flags otherwise",
                  {"score_obscuring": out, "series_by_year": st.to_dicts()}, status,
                  "GECLM holds one snapshot per submission and per comment, so a toggle cannot be observed directly. "
                  "Contest mode leaves a trace (API scores obscured to 1 for non-moderators while it is on); random and "
                  "new sorts leave none, so (d) and r/AskHistorians (c) rest on the snapshot value and series consistency.")


# ------------------------------------------------------------------------------------------
def check9():
    """Sort assumptions for proxies (a) and (d)."""
    it = load("reddit", ["buildapc"], ["suggested_sort", "score"])
    roots = it.filter(pl.col("is_root"))
    sort_counts = roots.group_by("suggested_sort").len().sort("len", descending=True).to_dicts()
    com = it.filter(~pl.col("is_root"))
    sib = com.group_by("parent_id").agg(n=pl.len(), n_score_tied_top=(pl.col("score") == pl.col("score").max()).sum())
    tie_share = float((sib.filter(pl.col("n") >= 2)["n_score_tied_top"] >= 2).mean())
    sc1 = float((com["score"] == 1).mean())
    res = {"buildapc_suggested_sort_counts": sort_counts, "share_sibling_groups_with_tied_top_final_score": tie_share,
           "share_comments_final_score_1": sc1,
           "assumptions": {
               "a": "Display order approximated by rank of FINAL net score among siblings (Reddit 'best' = Wilson lower bound "
                    "on up/down votes; only net score is available). Ties broken by creation time, oldest first. Stickied "
                    "and moderator-distinguished comments (pinned on top) cannot be identified (no 'stickied' field in "
                    "GECLM comments) and are ranked by score like others. Collapsed/'load more' children are not modelled.",
               "b": "Early-thread display order = arrival order (oldest first) among siblings: under 'best' new comments "
                    "all start at score 1 (one self-vote), so their confidence is equal and Reddit's tie-break order is "
                    "assumed chronological. HN early order assumed to follow arrival the same way.",
               "d": "suggested_sort = 'new' shows top-level comments newest first; position(t) = 1 + newer top-level "
                    "siblings so far. Assumes readers use the suggested sort (a reader can override it), stickied comments "
                    "are not identified, and the sort applies from thread creation (snapshot value; see check 8)."}}
    return record(9, "Sort assumptions for proxies (a) and (d)", "documented; unverifiable parts flagged",
                  res, "FLAG",
                  "The assumptions cannot be verified with GECLM: no per-comment vote split, no stickied flag, no reader-side "
                  "sort. Final-score ties among siblings are common (see result), so (a)'s tie-break matters.")


# ------------------------------------------------------------------------------------------
def check10():
    """HN snapshot completeness against id ranges; title-prefix classification."""
    files = sorted(str(x) for x in (ROOT / "data" / "raw" / "hn").glob("part.*.parquet"))
    lo, hi = (ep(x) for x in P["hn"]["thread_window_utc"])
    # ids are allocated sequentially to every item type; comments are never re-dated (unlike second-chance
    # stories), so the window's id range is taken from comments and coverage = distinct ids / id span,
    # per 1M-id block and overall
    a, b = duckdb.sql(f"""select min(id), max(id) from read_parquet({files})
                          where type = 'comment' and time between {lo} and {hi}""").fetchall()[0]
    cov = duckdb.sql(f"""
        select (id // 1000000) as block, min(id) a, max(id) b, count(distinct id) n
        from read_parquet({files}) where id between {a} and {b} group by 1 order by 1""").pl().with_columns(
        span=pl.min_horizontal(pl.lit(b), (pl.col("block") + 1) * 1000000 - 1)
        - pl.max_horizontal(pl.lit(a), pl.col("block") * 1000000) + 1).with_columns(coverage=pl.col("n") / pl.col("span"))
    cov.write_csv(OUT / "check10_id_coverage.csv")
    titles = duckdb.sql(f"""
        select
          sum(regexp_matches(lower(title), '^ask hn'))::bigint ask_prefix,
          sum(regexp_matches(lower(title), '^(ask hn|ask yc)'))::bigint ask_or_yc,
          sum(regexp_matches(lower(title), '.+ask hn') and not regexp_matches(lower(title), '^ask hn'))::bigint ask_not_prefix,
          sum(regexp_matches(lower(title), '^ask hn') and coalesce(url,'') <> '')::bigint ask_with_url,
          sum(regexp_matches(lower(title), '^show hn'))::bigint show_prefix,
          sum(regexp_matches(lower(title), '^show hn') and coalesce(url,'') = '')::bigint show_without_url,
          sum(regexp_matches(lower(title), '^(tell|launch) hn'))::bigint tell_launch,
          sum(regexp_matches(lower(title), '^\\s+(ask|show) hn'))::bigint leading_space,
          count(*) stories
        from read_parquet({files}) where type = 'story' and time between {lo} and {hi} and coalesce(deleted, false) = false
    """).pl().row(0, named=True)
    min_cov = float(cov["coverage"].min())
    overall_cov = float(cov["n"].sum() / cov["span"].sum())
    ambiguous = (titles["ask_not_prefix"] + titles["leading_space"] + titles["ask_or_yc"] - titles["ask_prefix"]) / titles["stories"]
    status = "PASS" if (min_cov >= 0.99 and ambiguous < 0.01) else "FLAG"
    return record(10, "HN snapshot completeness (id ranges); title-prefix classification",
                  "every 1M-id block in the window's comment id range: distinct ids / id span >= 0.99; ambiguous "
                  "Ask/Show titles < 1% of stories",
                  {"id_range": [int(a), int(b)], "min_block_coverage": min_cov, "overall_coverage": overall_cov,
                   "titles": titles, "ambiguous_share": ambiguous}, status,
                  "HN ids are allocated sequentially to every item type, so distinct ids over the span measure what the "
                  "dump is missing. Ask HN = title starting 'Ask HN' (case-insensitive); 'Ask YC' and titles with 'Ask HN' "
                  "later in the title stay in link/text_other; Ask HN posts with a url stay Ask HN.")


CHECKS = {i: f for i, f in enumerate([check1, check2, check3, check4, check5, check6, check7, check8, check9, check10], 1)}

if __name__ == "__main__":
    which = [int(x) for x in sys.argv[1:]] or list(CHECKS)
    for i in which:
        CHECKS[i]()
        gc.collect()
