"""Step 4c — calibration split only.

Usage: python scripts/step4c_extensions.py [reply_back] [showhn_op] [fantasy]

  reply_back  4c.1: 4b.1 thread x time-bin FE age model with and without reply-backs (replies written by
              the author of the comment's parent), per population; Reddit and HN reported separately.
  showhn_op   4c.2: Show HN, 4b.2 proxy (b) position model with and without the OP's replies as events.
  fantasy     4c.3: 4b.1 thread x time-bin FE age model on r/Fantasy contest threads vs matched
              non-contest threads (>= 20 comments), top-level parents.

Intervals: 95% thread-clustered bootstrap (multinomial thread counts, B = 200, seed from partitions.json
+ offsets logged below); thread-clustered sandwich SEs alongside. Weightings: per-author (primary; weighted
Poisson score equations, i.e. y -> w*y and offset -> offset + log w, w = 1/(author's parents in the fold),
unknown authors dropped) and raw.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import pem  # noqa: E402
from forum_model.age_fe import cells  # noqa: E402
from forum_model.fe_poisson import FEPoissonBoot  # noqa: E402
from forum_model.units import build_units, cap_fit, load, reddit_population, refirst  # noqa: E402
import step4b_extensions as S4b  # noqa: E402

P = load_partitions()
SEED = P["seed"]
OUT = ROOT / "outputs" / "combined"
B = 200
BINS, HMAX, CAP = S4b.BINS, S4b.HMAX, S4b.CAP
HUMAN = S4b.HUMAN
SEED_OFFSETS = {"reply_back": 41, "showhn_op": 42, "fantasy": 43}


def pct(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def cap_threads(par: pl.DataFrame) -> pl.DataFrame:
    th = par.group_by("thread_id").len().with_columns(
        u=pl.col("thread_id").map_elements(S4b.thread_hash, return_dtype=pl.Float64)).sort("u").with_columns(
        cum=pl.col("len").cum_sum())
    keep_t = th.filter(pl.col("cum") <= CAP)["thread_id"] if th["len"].sum() > CAP else th["thread_id"]
    return par.filter(pl.col("thread_id").is_in(keep_t.implode()))


def weighted_design(D: dict, yname: str, w_row: np.ndarray | None):
    """Per-author version: y -> w*y, offset -> offset + log w, rows with w = 0 dropped."""
    y, off, X, g, thr = D["y"][yname], D["off"], D["X"], D["g"], D["thr"]
    if w_row is None:
        return g, y, X, off, thr
    k = w_row > 0
    return g[k], y[k] * w_row[k], X[k], off[k] + np.log(w_row[k]), thr[k]


def row_weights(D: dict, par: pl.DataFrame) -> np.ndarray:
    """w_pa of the parent behind each (comment, bin) row, rebuilt from the cell construction order."""
    s = np.maximum(par["time_since_root"].to_numpy(), 0.0)
    e = np.minimum(s + par["c"].to_numpy(), HMAX)
    w = par["w_pa"].to_numpy()
    out = []
    for b in range(len(BINS) - 1):
        ov = np.minimum(e, BINS[b + 1]) - np.maximum(s, BINS[b])
        out.append(w[np.nonzero(ov > 0)[0]])
    return np.concatenate(out)


def fit_boot(D: dict, ynames: list[str], w_row, rng) -> dict:
    """Point fit + sandwich + bootstrap for each y in ynames, sharing replicate weights."""
    T = D["n_threads"]
    preps = {}
    for yn in ynames:
        g, y, X, off, thr = weighted_design(D, yn, w_row)
        preps[yn] = FEPoissonBoot(g, y, X, off, thr)
    point = {yn: preps[yn].fit(se=True) for yn in ynames}
    reps = {yn: [] for yn in ynames}
    for _ in range(B):
        wc = rng.multinomial(T, np.full(T, 1.0 / T)).astype(float)
        for yn in ynames:
            reps[yn].append(preps[yn].fit(wc, b0=point[yn][0]))
    return {yn: {"coef": point[yn][0], "se": point[yn][1], "reps": np.array(reps[yn]),
                 "rows": preps[yn].n_rows} for yn in ynames}


# ------------------------------------------------------------------------------------------
# 4c.1 reply-back
# ------------------------------------------------------------------------------------------

def run_reply_back():
    rng = np.random.default_rng(SEED + SEED_OFFSETS["reply_back"])
    rows, diffs = [], []
    extra = ["time_since_root", "is_root_author", "parent_author"]
    hn = load("hn", None, ["body_status", *extra])
    for ds, comm in HUMAN:
        it, f = S4b.population(ds, comm, extra, hn_items=hn if ds == "hn" else None)
        U = build_units(it, f["nonroot"], f"{ds}:{comm}", keep=extra)
        del it
        gc.collect()
        par = cap_threads(U["parents"])
        rep = U["replies"].join(par.select(pl.col("id").alias("parent"), "parent_author"), on="parent", how="inner")
        is_rb = (pl.col("kid_author") == pl.col("parent_author")).fill_null(False)
        share_rb = float(rep.select(is_rb.mean()).item())
        D = cells(par, {"all": rep, "no_reply_back": rep.filter(~is_rb)}, BINS, HMAX)
        for wlab, w_row in [("per_author", row_weights(D, par)), ("raw", None)]:
            R = fit_boot(D, ["all", "no_reply_back"], w_row, rng)
            for yn, r in R.items():
                for j, nm in enumerate(D["names"]):
                    lo, hi = pct(r["reps"][:, j])
                    rows.append({"platform": "Reddit" if ds == "reddit" else "HN", "community": comm,
                                 "weighting": wlab, "replies": yn, "term": nm, "coef": float(r["coef"][j]),
                                 "boot_ci_lo": lo, "boot_ci_hi": hi, "se_cluster_thread": float(r["se"][j]),
                                 "rows": r["rows"], "parents": D["parents"], "threads": D["n_threads"],
                                 "events": float(D["y"][yn].sum()), "share_replies_reply_back": share_rb,
                                 "B": B})
            j = D["names"].index("log1p_age_minutes")
            dr = R["no_reply_back"]["reps"][:, j] - R["all"]["reps"][:, j]
            d0 = R["no_reply_back"]["coef"][j] - R["all"]["coef"][j]
            lo, hi = pct(dr)
            diffs.append({"platform": "Reddit" if ds == "reddit" else "HN", "community": comm, "weighting": wlab,
                          "age_all": float(R["all"]["coef"][j]), "age_no_reply_back": float(R["no_reply_back"]["coef"][j]),
                          "delta": float(d0), "delta_ci_lo": lo, "delta_ci_hi": hi,
                          "share_replies_reply_back": share_rb, "B": B})
            print(f"  [4c.1] {comm} {wlab}: age all {R['all']['coef'][j]:+.3f}, no reply-back "
                  f"{R['no_reply_back']['coef'][j]:+.3f}, delta {d0:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"(reply-back share {share_rb:.3f})", flush=True)
            pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4c_reply_back_terms.csv")
            pl.DataFrame(diffs).write_csv(OUT / "step4c_reply_back.csv")
        del U, par, rep, D
        gc.collect()


# ------------------------------------------------------------------------------------------
# 4c.2 Show HN without OP replies
# ------------------------------------------------------------------------------------------

def proxy_b_units(it: pl.DataFrame, name: str):
    roots = it.filter(pl.col("is_root")).select("thread_id", root_created="created")
    it = it.join(roots, on="thread_id", how="left").with_columns(
        obs_end=pl.min_horizontal("obs_end", pl.col("root_created") + 1800))
    ar = S4b.arrival_rank(it, early_only=True)
    f = ~pl.col("is_root") & (pl.col("time_since_root") <= 1800)
    U = build_units(it, f, name, keep=["time_since_root", "root_author"])
    par = U["parents"].join(ar.select("id", "pos", "n_sib"), on="id", how="inner").filter(pl.col("n_sib") >= 2)
    rep = U["replies"].join(par.select(pl.col("id").alias("parent"), "root_author"), on="parent", how="inner")
    return par, rep


def pem_design(fit: pl.DataFrame):
    t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
    prep = pem.Prepared(t, d, S4b.EDGES_B, np.zeros(fit.height, dtype=int), 1)
    depth = fit["depth"].to_numpy()
    xpos = np.log(np.clip(fit["pos"].to_numpy(), 1, S4b.POS_CAP))
    late = np.log1p(np.maximum(fit["time_since_root"].to_numpy(), 0) / 60)
    return prep, depth, xpos, late, d


def run_showhn_op():
    rng = np.random.default_rng(SEED + SEED_OFFSETS["showhn_op"])
    it, _ = S4b.population("hn", "show_hn", ["time_since_root", "root_author"])
    par, rep = proxy_b_units(it, "hn:show_hn")
    is_op = (pl.col("kid_author") == pl.col("root_author")).fill_null(False)
    par_ex = refirst(par, rep.filter(~is_op))
    fit_all = cap_fit(par)
    fit_ex = fit_all.select("id").join(par_ex, on="id", how="left", maintain_order="left")  # same rows/order
    assert fit_ex["id"].equals(fit_all["id"])
    first_by_op = rep.sort("delay").group_by("parent").first().select(pl.col("kid_author") == pl.col("root_author")).to_series()
    share_first_op = float(first_by_op.fill_null(False).mean())
    _, tix = np.unique(fit_all["thread_id"].to_numpy(), return_inverse=True)
    T = int(tix.max()) + 1
    designs = {}
    for lab, fdf in [("all replies", fit_all), ("OP replies excluded", fit_ex)]:
        prep, depth, xpos, late, d = pem_design(fdf)
        dcols, dnames = [], []
        for dv, nm in [(2, "depth_2"), (3, "depth_3")]:
            if dv > int(depth.min()) and (d & (depth == dv)).sum() >= 10:
                dcols.append((depth == dv).astype(float))
                dnames.append(nm)
        designs[lab] = (prep, {"position_only": (xpos[:, None], ["log_position"]),
                               "position_plus_controls": (np.column_stack([xpos, late, *dcols]),
                                                          ["log_position", "log1p_minutes_since_thread_start", *dnames])},
                        int(d.sum()))
    rows, diffs = [], []
    for wlab in ["per_author", "raw"]:
        w0 = S4b.per_author_weights(fit_all) if wlab == "per_author" else np.ones(fit_all.height)
        reps_w = [rng.multinomial(T, np.full(T, 1.0 / T))[tix] for _ in range(B)]
        for model in ["position_only", "position_plus_controls"]:
            res = {}
            for lab, (prep, models, nev) in designs.items():
                X, names = models[model]
                pt = pem.fit_multi(prep, X, w0, tix)
                bs = np.array([pem.fit_multi(prep, X, w0 * rw, tix)["beta"] for rw in reps_w])
                res[lab] = (pt, bs, names)
                for j, nm in enumerate(names):
                    lo, hi = pct(bs[:, j])
                    rows.append({"community": "show_hn", "weighting": wlab, "model": model, "events": lab, "term": nm,
                                 "coef": float(pt["beta"][j]), "boot_ci_lo": lo, "boot_ci_hi": hi,
                                 "se_cluster_thread": float(pt["se"][j]), "parents": fit_all.height, "threads": T,
                                 "first_reply_events": nev, "share_first_replies_by_op": share_first_op, "B": B})
            a, e = res["all replies"], res["OP replies excluded"]
            dr = a[1][:, 0] - e[1][:, 0]
            lo, hi = pct(dr)
            diffs.append({"weighting": wlab, "model": model, "beta_all": float(a[0]["beta"][0]),
                          "beta_op_excluded": float(e[0]["beta"][0]), "diff_all_minus_excluded": float(a[0]["beta"][0] - e[0]["beta"][0]),
                          "diff_ci_lo": lo, "diff_ci_hi": hi, "share_first_replies_by_op": share_first_op, "B": B})
            print(f"  [4c.2] show_hn {wlab} {model}: beta all {a[0]['beta'][0]:+.3f}, OP excluded {e[0]['beta'][0]:+.3f} "
                  f"{pct(e[1][:, 0])}, diff [{lo:+.3f},{hi:+.3f}]", flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4c_showhn_op_terms.csv")
    pl.DataFrame(diffs).write_csv(OUT / "step4c_showhn_op.csv")


# ------------------------------------------------------------------------------------------
# 4c.3 r/Fantasy contest vs matched non-contest
# ------------------------------------------------------------------------------------------

def run_fantasy():
    rng = np.random.default_rng(SEED + SEED_OFFSETS["fantasy"])
    extra = ["time_since_root", "is_root_author", "contest_mode", "thread_size"]
    it, _ = reddit_population("Fantasy", extra)
    roots = it.filter(pl.col("is_root"))
    sets = {"contest": roots.filter(pl.col("contest_mode").fill_null(False))["thread_id"],
            "non-contest, 20+ comments": roots.filter(~pl.col("contest_mode").fill_null(False)
                                                      & (pl.col("thread_size") >= 20))["thread_id"]}
    rows, store = [], {}
    for lab, ids in sets.items():
        f = ~pl.col("is_root") & (pl.col("depth") == 1) & pl.col("thread_id").is_in(ids.implode())
        U = build_units(it, f, "reddit:Fantasy", keep=["time_since_root", "is_root_author"])
        par = cap_threads(U["parents"])
        rep = U["replies"].join(par.select(pl.col("id").alias("parent")), on="parent", how="semi")
        D = cells(par, {"all": rep}, BINS, HMAX)
        for wlab, w_row in [("per_author", row_weights(D, par)), ("raw", None)]:
            r = fit_boot(D, ["all"], w_row, rng)["all"]
            j = D["names"].index("log1p_age_minutes")
            store[(lab, wlab)] = (float(r["coef"][j]), r["reps"][:, j])
            for jj, nm in enumerate(D["names"]):
                lo, hi = pct(r["reps"][:, jj])
                rows.append({"community": "Fantasy", "threads_set": lab, "weighting": wlab, "term": nm,
                             "coef": float(r["coef"][jj]), "boot_ci_lo": lo, "boot_ci_hi": hi,
                             "se_cluster_thread": float(r["se"][jj]), "rows": r["rows"], "parents": D["parents"],
                             "threads": D["n_threads"], "events": float(D["y"]["all"].sum()), "B": B})
            print(f"  [4c.3] Fantasy {lab} {wlab}: age {r['coef'][j]:+.3f} {pct(r['reps'][:, j])}", flush=True)
    for wlab in ["per_author", "raw"]:
        c0, cr = store[("contest", wlab)]
        k0, kr = store[("non-contest, 20+ comments", wlab)]
        lo, hi = pct(cr - kr)
        rows.append({"community": "Fantasy", "threads_set": "contest minus non-contest", "weighting": wlab,
                     "term": "log1p_age_minutes", "coef": c0 - k0, "boot_ci_lo": lo, "boot_ci_hi": hi, "B": B})
        print(f"  [4c.3] contest - non-contest {wlab}: {c0 - k0:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4c_fantasy.csv")


if __name__ == "__main__":
    print(f"seed {SEED}, offsets {SEED_OFFSETS}, B={B}", flush=True)
    which = sys.argv[1:] or ["showhn_op", "fantasy", "reply_back"]
    for w in which:
        {"reply_back": run_reply_back, "showhn_op": run_showhn_op, "fantasy": run_fantasy}[w]()
        gc.collect()
