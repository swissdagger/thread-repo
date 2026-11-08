"""Step 3b — additions requested after step 3 (no change to step-3 outputs).

Usage: python scripts/step3b_extensions.py [reddit] [hn] [moltbook_crawl] [combine]

For every step-3 population and unit (same parents, events, censoring and inner folds):
  1. Kernel competitors with an onset: log-normal and gamma (shape > 1), fitted to first replies
     on the inner fit fold; held-out log-likelihood; change in mu vs the Weibull reference.
  2. K4 check (non-root units): does author activity predict replies WITHIN a thread? Poisson with
     thread fixed effects (conditional likelihood = conditional logit on parent choice) vs a pooled
     Poisson. Outcome: direct non-self, non-bot replies observed before censoring. Offset: log G(c)
     from the step-3 Weibull kernel. Covariates: log(1 + author's comments in OTHER calibration
     threads), depth dummies, log(1 + minutes from thread start to the comment), OP's own comment.
  3. Early/late split: share of first replies within 10 min, 10-60 min, 1 h to horizon (1 day
     primary, 3 days sensitivity), per comment and among replied comments; raw and per-author.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model.fe_poisson import fit_fe_poisson, fit_pooled_poisson  # noqa: E402
from forum_model.units import (build_units, cap_fit, hn_population, load, moltbook_population,  # noqa: E402
                               reddit_population)

P = load_partitions()
NEW_FAMILIES = ("lognormal", "gamma")
EXTRA = ["time_since_root", "is_root_author"]


def weibull_ref(ds: str, comm: str, unit: str) -> dict:
    f = pl.read_csv(ROOT / "outputs" / ds / "step3_fits.csv", infer_schema_length=None)
    r = f.filter((pl.col("community") == comm) & (pl.col("unit") == unit) & (pl.col("fit_type") == "first_reply")
                 & (pl.col("family") == "weibull")).row(0, named=True)
    s = pl.read_csv(ROOT / "outputs" / ds / "step3_summary.csv", infer_schema_length=None)
    srow = s.filter((pl.col("community") == comm) & (pl.col("unit") == unit)).row(0, named=True)
    return {"mu": r["mu"], "shape": r["weibull_shape"], "scale_s": r["weibull_scale_h"] * 3600,
            "heldout_ll": srow["heldout_ll_weibull"], "best_step3": srow["best_family_heldout"],
            "heldout_ll_best_step3": srow[f"heldout_ll_{srow['best_family_heldout']}"]}


# ------------------------------------------------------------------------------------------
# 1. kernel competitors
# ------------------------------------------------------------------------------------------

def competitors(ds, comm, unit, U, ref) -> list[dict]:
    par = U["parents"]
    fit, held = cap_fit(par), par.filter(pl.col("fold") == "heldout")
    t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
    th, dh = held["t"].to_numpy(), held["d"].to_numpy()
    rows = []
    for fam in NEW_FAMILIES:
        f = H.fit_first_reply(fam, t, d)
        ll = H.first_reply_loglik(f, th, dh)
        rows.append({"dataset": ds, "community": comm, "unit": unit, **f.params(),
                     "heldout_ll_per_comment": ll, "heldout_ll_minus_weibull": ll - ref["heldout_ll"],
                     "heldout_ll_minus_step3_best": ll - ref["heldout_ll_best_step3"],
                     "step3_best": ref["best_step3"], "mu_weibull_ref": ref["mu"],
                     "mu_change_vs_weibull": f.mu - ref["mu"], "mu_rel_change_vs_weibull": f.mu / ref["mu"] - 1})
        print(f"  {fam}: mu={f.mu:.4f} (weibull {ref['mu']:.4f}) dLL vs weibull={ll - ref['heldout_ll']:+.4f}", flush=True)
    return rows


# ------------------------------------------------------------------------------------------
# 2. K4: author activity within threads
# ------------------------------------------------------------------------------------------

def author_activity(it: pl.DataFrame) -> pl.DataFrame:
    """Comments by each author in OTHER calibration threads (bots and orphans excluded)."""
    c = it.filter(~pl.col("is_root") & (pl.col("split") == "calibration") & pl.col("author").is_not_null()
                  & ~pl.col("bot_listed").fill_null(False) & ~pl.col("orphan").fill_null(False))
    tot = c.group_by("author").agg(_tot=pl.len())
    per = c.group_by("thread_id", "author").agg(_in=pl.len())
    return per.join(tot, on="author").select("thread_id", "author", activity_other=pl.col("_tot") - pl.col("_in"))


def k4(ds, comm, unit, U, act, ref) -> list[dict]:
    par = cap_fit(U["parents"]).filter(pl.col("author").is_not_null())
    par = par.join(act, on=["thread_id", "author"], how="left").with_columns(pl.col("activity_other").fill_null(0))
    k, lam = ref["shape"], ref["scale_s"]
    c = par["c"].to_numpy()
    G = -np.expm1(-np.power(c / lam, k))
    off = np.log(np.maximum(G, 1e-300))
    depth = par["depth"].to_numpy()
    names = ["log1p_author_activity_other", "depth_2", "depth_3", "depth_4plus", "log1p_minutes_after_thread_start",
             "op_comment"]
    X = np.column_stack([np.log1p(par["activity_other"].to_numpy()),
                         depth == 2, depth == 3, depth >= 4,
                         np.log1p(np.maximum(par["time_since_root"].to_numpy(), 0) / 60),
                         par["is_root_author"].fill_null(False).to_numpy()]).astype(float)
    # depth baseline = lowest depth present (avoids a full dummy set collinear with the thread effects)
    dmin = min(int(depth.min()), 4)
    base_col = {1: None, 2: 1, 3: 2, 4: 3}[dmin]
    keep = X.std(axis=0) > 0
    if base_col is not None:
        keep[base_col] = False
    X, names = X[:, keep], [n for n, k_ in zip(names, keep) if k_]
    _, g = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
    y = par["n_replies"].to_numpy().astype(float)
    rows = []
    for model, res in [("thread_fixed_effects", fit_fe_poisson(g, y, X, off)),
                       ("pooled", fit_pooled_poisson(g, y, X, off))]:
        for j, n in enumerate(names):
            rows.append({"dataset": ds, "community": comm, "unit": unit, "model": model, "term": n,
                         "coef": float(res["coef"][j]), "se_cluster_thread": float(res["se"][j]),
                         "rate_ratio_per_10x_activity": float(10 ** res["coef"][j]) if n.startswith("log1p_author") else None,
                         "n_parents": res["n_parents"], "n_threads": res.get("n_threads"), "n_events": res["n_events"]})
    a = [r for r in rows if r["term"] == names[0]]
    print(f"  K4: activity coef FE={a[0]['coef']:.3f} (se {a[0]['se_cluster_thread']:.3f}) pooled={a[1]['coef']:.3f}",
          flush=True)
    return rows


# ------------------------------------------------------------------------------------------
# 3. early / late split
# ------------------------------------------------------------------------------------------

WINDOWS = [("lt_10min", 0, 600), ("10_60min", 600, 3600)]


def early_late(ds, comm, unit, U) -> list[dict]:
    par = U["parents"]  # whole calibration split (both inner folds): descriptive
    rows = []
    for horizon_lab, hz in [("1d", 86400.0), ("3d", 3 * 86400.0)]:
        e = par.filter(pl.col("c") >= hz)
        if e.height == 0:
            continue
        na = e.filter(pl.col("author").is_not_null()).group_by("author").agg(_n=pl.len())
        e = e.join(na, on="author", how="left").with_columns(
            w=pl.when(pl.col("author").is_null()).then(0.0).otherwise(1.0 / pl.col("_n")))  # unknown authors: 0
        t, d = e["t"].to_numpy(), e["d"].to_numpy() & (e["t"].to_numpy() < hz)
        for wlab, w in [("raw", np.ones(e.height)), ("per_author", e["w"].to_numpy())]:
            W = w.sum()
            rep = (w * d).sum()
            row = {"dataset": ds, "community": comm, "unit": unit, "horizon": horizon_lab, "weighting": wlab,
                   "eligible_comments": e.height, "share_replied_by_horizon": rep / W}
            for lab, a, b in WINDOWS + [(f"1h_{horizon_lab}", 3600, hz)]:
                inw = d & (t >= a) & (t < b)
                row[f"per_comment_{lab}"] = (w * inw).sum() / W
                row[f"among_replied_{lab}"] = (w * inw).sum() / rep if rep > 0 else None
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------
# drivers
# ------------------------------------------------------------------------------------------

ONLY: set[str] | None = None  # e.g. {"early_late"} to recompute just that part


def process(ds, comm, unit_name, it, f, name, act, out):
    print(f"[{ds}] {comm} {unit_name}", flush=True)
    U = build_units(it, f, name, keep=EXTRA)
    ref = weibull_ref(ds, comm, unit_name)
    if ONLY is None or "competitors" in ONLY:
        out["competitors"] += competitors(ds, comm, unit_name, U, ref)
    if ONLY is None or "early_late" in ONLY:
        out["early_late"] += early_late(ds, comm, unit_name, U)
    if (ONLY is None or "k4" in ONLY) and not unit_name.startswith("root") and act is not None:
        out["k4"] += k4(ds, comm, unit_name, U, act, ref)
    del U
    gc.collect()


def write(ds, out):
    o = ROOT / "outputs" / ds
    for k, v in out.items():
        if v:
            pl.DataFrame(v, infer_schema_length=None).write_csv(o / f"step3b_{k}.csv")


def new_out():
    return {"competitors": [], "early_late": [], "k4": []}


def run_reddit():
    out = new_out()
    for sub in P["reddit"]["subreddits"]:
        it, filters = reddit_population(sub, EXTRA)
        act = author_activity(it)
        for unit, f in filters.items():
            process("reddit", sub, unit, it, f, f"reddit:{sub}", act, out)
        del it, act
        gc.collect()
    write("reddit", out)


def run_hn():
    out = new_out()
    hn_items = load("hn", None, ["body_status", *EXTRA])
    for comm in ["ask_hn", "link", "show_hn"]:
        it, filters = hn_population(comm, hn_items=hn_items)
        act = author_activity(it)
        for unit, f in filters.items():
            process("hn", comm, unit, it, f, f"hn:{comm}", act, out)
    write("hn", out)


def run_moltbook():
    out = new_out()
    for pop in ["complete", "with_incomplete"]:
        it, filters = moltbook_population(pop, EXTRA)
        act = author_activity(it)
        for unit, f in filters.items():
            if pop == "with_incomplete" and unit != "nonroot":
                continue
            process("moltbook_crawl", "moltbook", unit if pop == "complete" else f"{unit}__{pop}", it, f,
                    "moltbook_crawl", act, out)
    write("moltbook_crawl", out)


def run_combine():
    o = ROOT / "outputs" / "combined"
    for k in ["competitors", "early_late", "k4"]:
        parts = [pl.read_csv(ROOT / "outputs" / ds / f"step3b_{k}.csv", infer_schema_length=None)
                 for ds in ["reddit", "hn", "moltbook_crawl"] if (ROOT / "outputs" / ds / f"step3b_{k}.csv").exists()]
        if parts:
            pl.concat(parts, how="diagonal_relaxed").write_csv(o / f"step3b_{k}.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("--only="):
        ONLY = set(args[0].split("=", 1)[1].split(","))
        args = args[1:]
    which = args or ["moltbook_crawl", "reddit", "hn", "combine"]
    for w in which:
        {"reddit": run_reddit, "hn": run_hn, "moltbook_crawl": run_moltbook, "combine": run_combine}[w]()
