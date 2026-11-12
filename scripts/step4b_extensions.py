"""Step 4b — calibration split only (decisions after the audit).

Usage: python scripts/step4b_extensions.py [age_fe] [position_b] [roots_early_late] [speed] [ties]

  age_fe            4b.1: replies per (comment, thread-age bin) ~ log comment age + depth + OP comment
                    (+ display position where known) with no FE, thread FE, and thread x time-bin FE;
                    Poisson, offset log exposure; thread-clustered CIs. Five human populations and
                    r/Fantasy contest threads (top-level comments).
  position_b        4b.2: proxy (b) first-reply PEM (first 30 minutes), position exponent with and
                    without controls log1p(minutes since thread start) and depth; per-author weighted;
                    thread-clustered CIs. Placebo: r/Fantasy contest threads (top-level), and r/Fantasy
                    control threads.
  roots_early_late  4b.3: early/late split for root posts (1-day horizon), raw and per-author, bootstrap.
  speed             4b.4: speed reconciliation (roots vs comments; self-replies, bots, spam, incomplete
                    threads, outage segments; archive vs crawl; exponential / Weibull half-life,
                    log-normal median, KM median delay; naive parent-to-child medians).
  ties              proxy (a) tie-break sensitivity (oldest-first vs random), materiality rule from
                    partitions.json.
"""
from __future__ import annotations

import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import hazard as H  # noqa: E402
from forum_model import pem  # noqa: E402
from forum_model.fe_poisson import fit_fe_poisson, fit_pooled_poisson  # noqa: E402
from forum_model.units import (build_units, cap_fit, hn_population, load, moltbook_archive_population,  # noqa: E402
                               moltbook_population, reddit_population)

P = load_partitions()
SEED = P["seed"]
OUT = ROOT / "outputs" / "combined"
OUT.mkdir(parents=True, exist_ok=True)
DAY = 86400.0
BINS = np.array(P["step4b"]["age_fe"]["thread_age_bins_s"], dtype=float)
HMAX = float(P["step4b"]["age_fe"]["horizon_s"])
CAP = int(P["step4b"]["age_fe"]["cap_parents"])
EDGES_B = np.r_[0.0, np.logspace(1, np.log10(1800), 13)]
POS_CAP = 500
Z = 1.959964
HUMAN = [("reddit", "philosophy"), ("reddit", "buildapc"), ("hn", "ask_hn"), ("hn", "link"), ("hn", "show_hn")]


def thread_hash(t: str) -> float:
    return int(hashlib.sha256(f"{SEED}:4b:{t}".encode()).hexdigest()[:13], 16) / 16 ** 13


def population(ds: str, comm: str, extra: list[str], hn_items=None):
    if ds == "reddit":
        return reddit_population(comm, extra)
    return hn_population(comm, extra, hn_items=hn_items)


def arrival_rank(it: pl.DataFrame, early_only: bool = False) -> pl.DataFrame:
    vis = ~pl.col("excluded").fill_null(False) if "excluded" in it.columns else pl.lit(True)
    f = ~pl.col("is_root") & ~pl.col("orphan").fill_null(False) & vis
    if early_only:
        f = f & (pl.col("time_since_root") <= 1800)
    return it.filter(f).select("id", "parent_id", "created").sort(["parent_id", "created"]).with_columns(
        pos=pl.int_range(1, pl.len() + 1).over("parent_id"), n_sib=pl.len().over("parent_id"))


def ci(coef, se):
    return coef - Z * se, coef + Z * se


# ------------------------------------------------------------------------------------------
# 4b.1
# ------------------------------------------------------------------------------------------

def age_fe_one(tag: dict, it: pl.DataFrame, parent_filter: pl.Expr, name: str, with_pos: bool) -> list[dict]:
    U = build_units(it, parent_filter, name, keep=["time_since_root", "is_root_author"])
    par, rep = U["parents"], U["replies"]
    # cap: whole threads, ordered by hash
    th = par.group_by("thread_id").len().with_columns(
        u=pl.col("thread_id").map_elements(thread_hash, return_dtype=pl.Float64)).sort("u").with_columns(
        cum=pl.col("len").cum_sum())
    keep_t = th.filter(pl.col("cum") <= CAP)["thread_id"] if th["len"].sum() > CAP else th["thread_id"]
    par = par.filter(pl.col("thread_id").is_in(keep_t.implode()))
    if with_pos:
        ar = arrival_rank(it)
        par = par.join(ar.select("id", "pos"), on="id", how="left").with_columns(pl.col("pos").fill_null(1))
    rep = rep.join(par.select(pl.col("id").alias("parent")), on="parent", how="semi")
    s = np.maximum(par["time_since_root"].to_numpy(), 0.0)
    e = np.minimum(s + par["c"].to_numpy(), HMAX)
    pid = {k: i for i, k in enumerate(par["id"].to_list())}
    rp = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
    rt = s[rp] + rep["delay"].to_numpy()
    ok = rt < e[rp]
    nb = len(BINS) - 1
    rb = np.clip(np.searchsorted(BINS, rt[ok], side="right") - 1, 0, nb - 1)
    counts = np.bincount(rp[ok] * nb + rb, minlength=len(s) * nb)
    _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
    depth = par["depth"].to_numpy()
    op = par["is_root_author"].fill_null(False).to_numpy().astype(float)
    pos = np.log(np.clip(par["pos"].to_numpy(), 1, POS_CAP)) if with_pos else None
    rows = {"y": [], "off": [], "age": [], "bin": [], "i": []}
    for b in range(nb):
        lo, hi = BINS[b], BINS[b + 1]
        ov = np.minimum(e, hi) - np.maximum(s, lo)
        m = ov > 0
        idx = np.nonzero(m)[0]
        mid = (np.maximum(s[idx], lo) + np.minimum(e[idx], hi)) / 2 - s[idx]
        rows["y"].append(counts[idx * nb + b])
        rows["off"].append(np.log(ov[idx]))
        rows["age"].append(np.log1p(mid / 60))
        rows["bin"].append(np.full(len(idx), b))
        rows["i"].append(idx)
    y = np.concatenate(rows["y"]).astype(float)
    off = np.concatenate(rows["off"])
    age = np.concatenate(rows["age"])
    bn = np.concatenate(rows["bin"])
    ii = np.concatenate(rows["i"])
    thr = tix[ii]
    dmin = int(depth.min())
    cols, names = [age], ["log1p_age_minutes"]
    for dv, nm in [(2, "depth_2"), (3, "depth_3")]:
        if dv > dmin:
            cols.append((depth[ii] == dv).astype(float))
            names.append(nm)
    if dmin < 4 and (depth >= 4).any():
        cols.append((depth[ii] >= 4).astype(float))
        names.append("depth_4plus")
    if op.std() > 0:
        cols.append(op[ii])
        names.append("op_comment")
    base = np.column_stack(cols)
    bin_d = np.column_stack([(bn == b).astype(float) for b in range(1, nb)])
    out = []
    specs = [("no_fixed_effects", "pooled", np.column_stack([base, bin_d]), None),
             ("thread_fe", "fe", np.column_stack([base, bin_d]), thr),
             ("thread_x_timebin_fe", "fe", base, thr * nb + bn)]
    if with_pos:
        early = (BINS[bn + 1] <= 1800).astype(float)
        pos_early = pos[ii] * early
        specs.append(("thread_x_timebin_fe_with_early_position", "fe", np.column_stack([base, pos_early]), thr * nb + bn))
    for label, kind, X, g in specs:
        keepc = X.std(axis=0) > 0
        Xk = X[:, keepc]
        res = fit_pooled_poisson(thr, y, Xk, off) if kind == "pooled" else fit_fe_poisson(g, y, Xk, off, cluster=thr)
        coef_names = [n for n, k in zip(_names_for(label, names, nb), keepc) if k]
        if res.get("dropped"):
            print(f"  [4b.1] {tag['community']} {label}: absorbed/collinear, dropped "
                  f"{[coef_names[j] for j in res['dropped']]}", flush=True)
        for j, n in enumerate(coef_names):
            if n.startswith("bin_"):
                continue
            lo, hi = ci(res["coef"][j], res["se"][j])
            out.append({**tag, "model": label, "term": n, "coef": float(res["coef"][j]),
                        "se_cluster_thread": float(res["se"][j]), "ci_lo": float(lo), "ci_hi": float(hi),
                        "rows": int(len(y)), "parents": par.height, "threads": int(tix.max() + 1),
                        "events": float(y.sum())})
        a = [r for r in out if r["model"] == label and r["term"] == "log1p_age_minutes"][0]
        print(f"  [4b.1] {tag['community']} {label}: age {a['coef']:+.3f} [{a['ci_lo']:+.3f},{a['ci_hi']:+.3f}]", flush=True)
    return out


def _names_for(label: str, names: list[str], nb: int) -> list[str]:
    if label in ("no_fixed_effects", "thread_fe"):
        return names + [f"bin_{b}" for b in range(1, nb)]
    if "position" in label:
        return names + ["log_position_x_first_30min"]
    return list(names)


def run_age_fe():
    rows = []
    hn = load("hn", None, ["body_status", "time_since_root", "is_root_author"])
    for ds, comm in HUMAN:
        it, f = population(ds, comm, ["time_since_root", "is_root_author"], hn_items=hn if ds == "hn" else None)
        rows += age_fe_one({"dataset": ds, "community": comm, "parents_definition": "all non-root comments"},
                           it, f["nonroot"], f"{ds}:{comm}", with_pos=True)
        del it
        gc.collect()
    del hn
    it, _ = reddit_population("Fantasy", ["time_since_root", "is_root_author", "contest_mode"])
    contest = it.filter(pl.col("is_root") & pl.col("contest_mode").fill_null(False))["thread_id"]
    rows += age_fe_one({"dataset": "reddit", "community": "Fantasy_contest",
                        "parents_definition": "top-level comments in contest-mode threads (order randomised)"},
                       it, ~pl.col("is_root") & (pl.col("depth") == 1) & pl.col("thread_id").is_in(contest.implode()),
                       "reddit:Fantasy", with_pos=False)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4b_age_fe.csv")


# ------------------------------------------------------------------------------------------
# 4b.2
# ------------------------------------------------------------------------------------------

def per_author_weights(par: pl.DataFrame) -> np.ndarray:
    na = par.filter(pl.col("author").is_not_null()).group_by("fold", "author").agg(_n=pl.len())
    n = par.join(na, on=["fold", "author"], how="left")["_n"].to_numpy().astype(float)
    return np.where(np.isnan(n), 0.0, 1.0 / np.where(np.isnan(n), 1.0, n))


def position_fit(tag: dict, par: pl.DataFrame, edges: np.ndarray) -> list[dict]:
    fit = cap_fit(par)
    t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
    prep = pem.Prepared(t, d, edges, np.zeros(fit.height, dtype=int), 1)
    _, tix = np.unique(fit["thread_id"].to_numpy(), return_inverse=True)
    depth = fit["depth"].to_numpy()
    xpos = np.log(np.clip(fit["pos"].to_numpy(), 1, POS_CAP))
    late = np.log1p(np.maximum(fit["time_since_root"].to_numpy(), 0) / 60)
    dcols, dnames = [], []
    dmin = int(depth.min())
    for dv, nm in [(2, "depth_2"), (3, "depth_3")]:
        if dv > dmin and (d & (depth == dv)).sum() >= 10:  # dummies need events (no separation)
            dcols.append((depth == dv).astype(float))
            dnames.append(nm)
    if dmin < 4 and (d & (depth >= 4)).sum() >= 10:
        dcols.append((depth >= 4).astype(float))
        dnames.append("depth_4plus")
    out = []
    for wlab in ["per_author", "raw"]:
        w = per_author_weights(fit) if wlab == "per_author" else np.ones(fit.height)
        for model, X, names in [("position_only", xpos[:, None], ["log_position"]),
                                ("position_plus_controls", np.column_stack([xpos, late, *dcols]),
                                 ["log_position", "log1p_minutes_since_thread_start", *dnames])]:
            res = pem.fit_multi(prep, X, w, tix)
            for j, n in enumerate(names):
                lo, hi = ci(res["beta"][j], res["se"][j])
                out.append({**tag, "weighting": wlab, "model": model, "term": n, "coef": float(res["beta"][j]),
                            "se_cluster_thread": float(res["se"][j]), "ci_lo": float(lo), "ci_hi": float(hi),
                            "hazard_ratio_per_doubling": float(2 ** res["beta"][j]) if n == "log_position" else None,
                            "parents": fit.height, "threads": res["threads"], "events_weighted": res["events"]})
            b = [r for r in out if r["weighting"] == wlab and r["model"] == model and r["term"] == "log_position"][0]
            print(f"  [4b.2] {tag['community']} {tag['thread_set']} {tag['parents_definition']} {wlab} {model}: "
                  f"pos {b['coef']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}]", flush=True)
    return out


def proxy_b_parents(it: pl.DataFrame, name: str, parent_extra: pl.Expr = pl.lit(True), top_level: bool = False):
    roots = it.filter(pl.col("is_root")).select("thread_id", root_created="created")
    it = it.join(roots, on="thread_id", how="left").with_columns(
        obs_end=pl.min_horizontal("obs_end", pl.col("root_created") + 1800))
    ar = arrival_rank(it, early_only=True)
    f = ~pl.col("is_root") & (pl.col("time_since_root") <= 1800) & parent_extra
    if top_level:
        f = f & (pl.col("depth") == 1)
    U = build_units(it, f, name, keep=["time_since_root"])
    return U["parents"].join(ar.select("id", "pos", "n_sib"), on="id", how="inner").filter(pl.col("n_sib") >= 2)


def run_position_b():
    rows = []
    hn = load("hn", None, ["body_status", "time_since_root"])
    for ds, comm in HUMAN:
        it, _ = population(ds, comm, ["time_since_root"], hn_items=hn if ds == "hn" else None)
        par = proxy_b_parents(it, f"{ds}:{comm}")
        rows += position_fit({"dataset": ds, "community": comm, "thread_set": "all", "window": "first 30 min",
                              "parents_definition": "all non-root comments"}, par, EDGES_B)
        par_tl = proxy_b_parents(it, f"{ds}:{comm}", top_level=True)
        rows += position_fit({"dataset": ds, "community": comm, "thread_set": "all", "window": "first 30 min",
                              "parents_definition": "top-level comments"}, par_tl, EDGES_B)
        del it
        gc.collect()
    del hn
    it, _ = reddit_population("Fantasy", ["time_since_root", "contest_mode"])
    roots = it.filter(pl.col("is_root"))
    contest = roots.filter(pl.col("contest_mode").fill_null(False))["thread_id"]
    control = roots.filter(~pl.col("contest_mode").fill_null(False) & (pl.col("thread_size") >= 20))["thread_id"]
    for lab, ids in [("contest (placebo)", contest), ("control, 20+ comments", control)]:
        par = proxy_b_parents(it, "reddit:Fantasy", pl.col("thread_id").is_in(ids.implode()), top_level=True)
        rows += position_fit({"dataset": "reddit", "community": "Fantasy", "thread_set": lab, "window": "first 30 min",
                              "parents_definition": "top-level comments"}, par, EDGES_B)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4b_position_b.csv")


# ------------------------------------------------------------------------------------------
# 4b.3
# ------------------------------------------------------------------------------------------

def run_roots_early_late(B: int = 200):
    rng = np.random.default_rng(SEED + 19)
    rows, reps_all = [], {}
    hn = load("hn", None, ["body_status"])
    pops = [(ds, c, lambda ds=ds, c=c: population(ds, c, [], hn_items=hn if ds == "hn" else None)) for ds, c in HUMAN]
    pops.append(("moltbook_crawl", "moltbook", lambda: moltbook_population("complete")))
    for ds, comm, loader in pops:
        it, f = loader()
        U = build_units(it, f["root_live"], ds if ds == "moltbook_crawl" else f"{ds}:{comm}")
        par = U["parents"].filter(pl.col("c") >= DAY)
        na = par.filter(pl.col("author").is_not_null()).group_by("author").agg(_n=pl.len())
        par = par.join(na, on="author", how="left").with_columns(
            wpa=pl.when(pl.col("author").is_null()).then(0.0).otherwise(1.0 / pl.col("_n")))
        _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
        G = int(tix.max()) + 1
        t, d = par["t"].to_numpy(), par["d"].to_numpy() & (par["t"].to_numpy() < DAY)
        win = [d & (t < 600), d & (t >= 600) & (t < 3600), d & (t >= 3600) & (t < DAY)]
        for wlab, w0 in [("raw", np.ones(par.height)), ("per_author", par["wpa"].to_numpy())]:
            def stat(w):
                W = w.sum()
                return [float((w * x).sum() / W) for x in win]
            est = stat(w0)
            R = np.array([stat(w0 * rng.multinomial(G, np.full(G, 1.0 / G))[tix]) for _ in range(B)])
            reps_all[(comm, wlab)] = (est, R)
            for j, name in enumerate(["lt_10min", "10_60min", "1h_1d"]):
                lo, hi = np.percentile(R[:, j], [2.5, 97.5])
                rows.append({"community": comm, "unit": "root_live", "weighting": wlab, "window": name,
                             "share_of_roots": est[j], "ci_lo": float(lo), "ci_hi": float(hi), "eligible_roots": par.height})
        print(f"  [4b.3] {comm} roots done", flush=True)
        del it
        gc.collect()
    for (comm, wlab), (est, R) in reps_all.items():
        if comm == "moltbook":
            continue
        em, M = reps_all[("moltbook", wlab)]
        for j, name in enumerate(["lt_10min", "10_60min", "1h_1d"]):
            lo, hi = np.percentile(R[:, j] / M[:, j], [2.5, 97.5])
            rows.append({"community": comm, "unit": "root_live", "weighting": wlab, "window": f"ratio_vs_moltbook_{name}",
                         "share_of_roots": est[j] / em[j], "ci_lo": float(lo), "ci_hi": float(hi)})
    pl.DataFrame(rows).write_csv(OUT / "step4b_roots_early_late.csv")


# ------------------------------------------------------------------------------------------
# 4b.4
# ------------------------------------------------------------------------------------------

def speed_metrics(U: dict) -> dict:
    par = U["parents"]
    fit = cap_fit(par)
    t, d = fit["t"].to_numpy(), fit["d"].to_numpy()
    out = {"parents_calibration": par.height, "parents_fit": fit.height}
    for fam in ["exponential", "weibull", "lognormal"]:
        f = H.fit_first_reply(fam, t, d)
        out[f"{fam}_mu"] = f.mu
        out[f"{fam}_half_life_min"] = H.half_life(fam, f.theta) / 60
        if fam == "weibull":
            out["weibull_shape"] = float(np.exp(f.theta[0]))
    km = H.FirstReplyKM(par["t"].to_numpy(), par["d"].to_numpy(), 3 * DAY)
    p, med = km.p_reply_and_median(np.ones(par.height))
    out["p_reply_3d"] = p
    out["km_median_delay_min"] = med / 60
    obs = par.filter(pl.col("d"))["t"]
    out["naive_median_first_reply_min"] = float(obs.median() / 60) if obs.len() else None
    rep = U["replies"]["delay"]
    out["naive_median_all_replies_min"] = float(rep.median() / 60) if rep.len() else None
    return out


def run_speed():
    rows = []

    def add(dataset, comm, unit, variant, U):
        m = speed_metrics(U)
        rows.append({"dataset": dataset, "community": comm, "unit": unit, "variant": variant, **m})
        print(f"  [4b.4] {comm} {unit} {variant}: KM median {m['km_median_delay_min']:.2f} min, "
              f"Weibull hl {m['weibull_half_life_min']:.2f}, lognormal median {m['lognormal_half_life_min']:.2f}", flush=True)
        pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step4b_speed.csv")

    # Moltbook crawl variants
    variants = [("default", dict(), dict()),
                ("+ self-replies", dict(), dict(include_self=True)),
                ("+ spam posts", dict(include_spam=True), dict()),
                ("+ incomplete threads", dict(pop="with_incomplete"), dict()),
                ("no outage segments", dict(segments=False), dict()),
                ("all of the above", dict(include_spam=True, pop="with_incomplete", segments=False), dict(include_self=True))]
    for vname, popkw, ukw in variants:
        it, f = moltbook_population(**{"pop": "complete", **popkw})
        for unit in ["nonroot", "root_live"]:
            add("moltbook_crawl", "moltbook", unit, vname, build_units(it, f[unit], "moltbook_crawl", **ukw))
        del it
        gc.collect()
    # archive cross-check
    a = P["moltbook"]["archive_crosscheck"]
    it, f = moltbook_archive_population(*a["posts_created_utc"], a["observation_end_utc"])
    for unit in ["nonroot", "root_live"]:
        add("moltbook_archive", "moltbook_archive", unit, "archive default", build_units(it, f[unit], "moltbook_archive"))
        add("moltbook_archive", "moltbook_archive", unit, "archive + self-replies",
            build_units(it, f[unit], "moltbook_archive", include_self=True))
    del it
    gc.collect()
    # humans
    hn = load("hn", None, ["body_status"])
    for ds, comm in HUMAN:
        it, f = population(ds, comm, [], hn_items=hn if ds == "hn" else None)
        name = f"{ds}:{comm}"
        for unit in ["nonroot", "root_live"]:
            add(ds, comm, unit, "default", build_units(it, f[unit], name))
            add(ds, comm, unit, "+ self-replies", build_units(it, f[unit], name, include_self=True))
            if ds == "reddit":
                add(ds, comm, unit, "+ bots", build_units(it, f[unit], name, include_bots=True))
        del it
        gc.collect()
    del hn
    # naive pooled parent-to-child delays (all stored comments of the calibration split, no filters)
    naive = []
    sources = [("moltbook_crawl (window, all posts incl. spam)", "moltbook_crawl", None),
               ("reddit philosophy", "reddit", ["philosophy"]), ("reddit buildapc", "reddit", ["buildapc"]),
               ("hn (all classes)", "hn", None)]
    for lab, ds, comms in sources:
        it = load(ds, comms, ["time_since_parent"])
        c = it.filter(~pl.col("is_root") & (pl.col("split") == "calibration") & pl.col("time_since_parent").is_not_null())
        for scope, ff in [("all comments", pl.lit(True)), ("replies to posts (depth 1)", pl.col("depth") == 1),
                          ("replies to comments (depth >= 2)", pl.col("depth") >= 2)]:
            x = c.filter(ff)["time_since_parent"]
            naive.append({"source": lab, "scope": scope, "comments": x.len(),
                          "median_parent_to_child_min": float(x.median() / 60),
                          "share_within_1min": float((x < 60).mean())})
        del it
        gc.collect()
    pl.DataFrame(naive).write_csv(OUT / "step4b_speed_naive.csv")


# ------------------------------------------------------------------------------------------
# proxy (a) tie-break sensitivity
# ------------------------------------------------------------------------------------------

def run_ties():
    a = pl.read_csv(OUT / "step4_a_summary.csv", infer_schema_length=None)
    r = pl.read_csv(OUT / "step4_a_ties_random_summary.csv", infer_schema_length=None)
    k = ["community", "weighting"]
    cols = ["beta_log_position", "se_beta_cluster_thread", "beta_ci_lo", "beta_ci_hi", "flattening_M1_minus_M0",
            "flattening_ci_lo", "flattening_ci_hi", "decay_slope_M0", "decay_slope_M1"]
    j = a.select(k + cols).join(r.select(k + cols), on=k, suffix="_random")
    j = j.with_columns(
        d_beta=(pl.col("beta_log_position_random") - pl.col("beta_log_position")).abs(),
        d_flat=(pl.col("flattening_M1_minus_M0_random") - pl.col("flattening_M1_minus_M0")).abs(),
        tol_beta=pl.max_horizontal(pl.lit(0.05), 2 * pl.col("se_beta_cluster_thread")),
        sign_flip=((pl.col("flattening_ci_lo") > 0) & (pl.col("flattening_ci_hi_random") < 0))
        | ((pl.col("flattening_ci_hi") < 0) & (pl.col("flattening_ci_lo_random") > 0)),
    ).with_columns(material=(pl.col("d_beta") > pl.col("tol_beta")) | (pl.col("d_flat") > 0.01) | pl.col("sign_flip"))
    j.write_csv(OUT / "step4b_proxy_a_ties.csv")
    print(j.select(k + ["beta_log_position", "beta_log_position_random", "flattening_M1_minus_M0",
                        "flattening_M1_minus_M0_random", "material"]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["ties", "roots_early_late", "position_b", "age_fe", "speed"]
    for w in which:
        {"age_fe": run_age_fe, "position_b": run_position_b, "roots_early_late": run_roots_early_late,
         "speed": run_speed, "ties": run_ties}[w]()
        gc.collect()
