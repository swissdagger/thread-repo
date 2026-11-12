"""First model fit, Stage 1: generator and estimator recovery (synthetic only). docs/fit1_brief.md.

Usage: python scripts/fit1_stage1.py roots | calibrate | recover [pop ...] | hn_joint | bootcheck | summary | smoke <pop>

  roots      real root arrival times and post types of each population's capped inner fit fold
             (partitions.json -> fit1.cap_comments), saved to outputs/fit1/stage1_roots_<pop>.parquet
  calibrate  set alpha_c, alpha_R and rho so the generator matches each population's real volumes
             (top-level comments per root, comment-stream replies and reply-backs per comment); the true
             parameter set is written to outputs/fit1/stage1_truth.json (committed before recovery runs)
  recover    20 replicate datasets per population: simulate, fit the full model and B1, B = 100 one-step
             thread bootstrap; outputs/fit1/stage1_recovery_<pop>.parquet
  hn_joint   joint HN root-stream fit (gamma shared, kappa on Ask HN) on the same replicate datasets
  bootcheck  one-step vs full-refit thread bootstrap (Stage A) on one Ask HN replicate
  summary    coverage table, pass rule, figures
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import fit1_estimator as EST  # noqa: E402
from forum_model.fit1_data import POPULATIONS, cap_threads, load_population  # noqa: E402
from forum_model.fit1_generator import Generator, clock_integral, lognormal_coefs  # noqa: E402
from forum_model.fit1_streams import extract, summary  # noqa: E402

P = load_partitions()
F1 = P["fit1"]
S1 = F1["stage1"]
SEED = P["seed"]
OUT = ROOT / "outputs" / "fit1"
FIG = ROOT / "figures" / "fit1"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
HN = ["ask_hn", "link", "show_hn"]


def hour_profile(amp: float, peak: float) -> np.ndarray:
    h = np.arange(24)
    v = amp * np.cos(2 * np.pi * (h - peak) / 24.0)
    return v - v[0]


def gen_params(pop: str, alphas: dict) -> dict:
    sp = S1["truth_spec"][pop]
    hn = S1["truth_spec"]["hn_shared"]
    gamma = hn["gamma"] if pop in HN else sp["gamma"]
    kappa = hn["kappa"] if pop == "ask_hn" else 0.0
    hour = hour_profile(*(hn["hour_amp_peak"] if pop in HN else sp["hour_amp_peak"]))
    return {"alpha_c": alphas["alpha_c"], "alpha_R": alphas["alpha_R"], "r": sp["r"], "b": sp["b"],
            "k": lognormal_coefs(*sp["k_median_sigma"]), "a": lognormal_coefs(*sp["a_median_sigma"]),
            "aR": lognormal_coefs(*sp["aR_median_sigma"]), "hour": hour,
            "beta": dict(zip(["dep2", "dep3p", "op", "logn", "pos"], sp["beta"])),
            "gamma": gamma, "kappa": kappa, "rho": alphas["rho"], "back_median_min": sp["back_median_sigma"][0],
            "back_sigma": sp["back_median_sigma"][1], "beta_back": tuple(sp["beta_back"]), "pi_prev": sp["pi_prev"]}


def truth_vector(pop: str, gp: dict) -> dict:
    """True values in the estimator's parameterisation."""
    t = {"k1_log_u": gp["k"][0], "k2_log_u_sq": gp["k"][1], "a1_log_tau": gp["a"][0], "a2_log_tau_sq": gp["a"][1],
         "dep2": gp["beta"]["dep2"], "dep3p": gp["beta"]["dep3p"], "op": gp["beta"]["op"], "logn": gp["beta"]["logn"],
         "pos_early": gp["beta"]["pos"], "aR1": gp["aR"][0], "aR2": gp["aR"][1], "gamma": gp["gamma"] - gp["kappa"],
         "alpha_c": gp["alpha_c"], "alpha_R": gp["alpha_R"], "log_r": np.log(gp["r"]), "b": gp["b"],
         "log_rho": np.log(gp["rho"]), "back_m": np.log(gp["back_median_min"]), "log_back_sigma": np.log(gp["back_sigma"]),
         "beta_back_root": gp["beta_back"][0], "beta_back_dep3p": gp["beta_back"][1]}
    for h in range(1, 24):
        t[f"hour_{h}"] = gp["hour"][h]
    return {k: float(v) for k, v in t.items()}


def roots_of(pop: str) -> pl.DataFrame:
    return pl.read_parquet(OUT / f"stage1_roots_{pop}.parquet")


# ------------------------------------------------------------------------------------------
def run_roots():
    from forum_model.units import load
    hn = load("hn", None, ["body_status"])
    rows = []
    for pop in POPULATIONS:
        it = load_population(pop, hn_items=hn if pop in HN else None)
        capd = cap_threads(it.filter(pl.col("fold") == "fit"), int(F1["cap_comments"]))
        S = extract(capd)
        sm = summary(S)
        r = capd.filter(pl.col("is_root")).select("thread_id", T0="created", post_type="post_type") \
            .with_columns(question=pl.col("post_type") == "ask_hn")
        r.write_parquet(OUT / f"stage1_roots_{pop}.parquet")
        rows.append({"population": pop, **sm})
        print(pop, sm, flush=True)
    pl.DataFrame(rows).write_csv(OUT / "stage1_real_targets.csv")


def run_calibrate():
    targets = {r["population"]: r for r in pl.read_csv(OUT / "stage1_real_targets.csv").to_dicts()}
    rng = np.random.default_rng(SEED + int(S1["seed_offset"]))
    truth = {}
    for pop in POPULATIONS:
        sp = S1["truth_spec"][pop]
        tg = targets[pop]
        al = {"alpha_R": float(np.log(tg["top_level_per_root"] / clock_integral(*lognormal_coefs(*sp["aR_median_sigma"])))),
              "alpha_c": float(np.log(tg["comment_stream_per_comment"] / two_clock_integral(sp))),
              "rho": float(tg["reply_backs_per_comment"])}
        roots = roots_of(pop)
        for it in range(int(S1["calibration_iterations"])):
            gp = gen_params(pop, al)
            try:
                items, diag = Generator(gp, max_total=1_500_000).simulate(roots, rng)
            except RuntimeError as e:
                print(f"  [calibrate] {pop} it {it}: {e}; lowering alpha_c by 1", flush=True)
                al["alpha_c"] -= 1.0
                continue
            sm = summary(extract(items))
            al["alpha_R"] += float(np.log(tg["top_level_per_root"] / max(sm["top_level_per_root"], 1e-9)))
            al["alpha_c"] += float(np.log(tg["comment_stream_per_comment"] / max(sm["comment_stream_per_comment"], 1e-9)))
            al["rho"] *= float(tg["reply_backs_per_comment"] / max(sm["reply_backs_per_comment"], 1e-9))
            print(f"  [calibrate] {pop} it {it}: sim {sm['top_level_per_root']:.3f}/{sm['comment_stream_per_comment']:.3f}/"
                  f"{sm['reply_backs_per_comment']:.3f} target {tg['top_level_per_root']:.3f}/"
                  f"{tg['comment_stream_per_comment']:.3f}/{tg['reply_backs_per_comment']:.3f}; max thread {diag['max_thread_size']}",
                  flush=True)
        gp = gen_params(pop, al)
        truth[pop] = {"alphas": al, "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gp.items()},
                      "truth_vector": truth_vector(pop, gp)}
    (OUT / "stage1_truth.json").write_text(json.dumps(truth, indent=1))


def two_clock_integral(sp) -> float:
    """Expected comment-stream integral of exp(k(u) + a(s + u)) over u in [0, 3 d - s], for a comment created at
    the root clock's median age s (starting value for alpha_c only; calibration iterates from it)."""
    k = lognormal_coefs(*sp["k_median_sigma"])
    a = lognormal_coefs(*sp["a_median_sigma"])
    s0 = sp["aR_median_sigma"][0] * 60.0
    u = np.logspace(-1, np.log10(3 * 86400 - s0), 4000)
    du = np.diff(np.r_[0.0, u])
    xu, xt = np.log(u / 60.0), np.log((s0 + u) / 60.0)
    return float(np.sum(np.exp(k[0] * xu + k[1] * xu ** 2 + a[0] * xt + a[1] * xt ** 2) * du))


def load_truth():
    return json.loads((OUT / "stage1_truth.json").read_text())


def dataset(pop: str, rep: int):
    T = load_truth()[pop]
    gp = dict(T["params"])
    gp["hour"] = np.array(gp["hour"])
    gp["k"], gp["a"], gp["aR"] = tuple(gp["k"]), tuple(gp["a"]), tuple(gp["aR"])
    # pre-registered: a dataset in which any thread reaches the generator's per-thread guard (or the global
    # guard) is redrawn with the next attempt seed; redraws are counted and reported
    for attempt in range(50):
        rng = np.random.default_rng([SEED, int(S1["dataset_seed_offset"]), POPULATIONS.index(pop), rep, attempt])
        try:
            items, diag = Generator(gp).simulate(roots_of(pop), rng)
        except RuntimeError:
            continue
        if diag["capped_events"] == 0 and diag["max_thread_size"] < 20000:
            diag["redraws"] = attempt
            return items, diag, T["truth_vector"], rng
    raise RuntimeError(f"{pop} rep {rep}: no non-exploding dataset in 50 attempts")


def rows_from_fit(pop, rep, R, truth, diag, secs):
    """One row per (arm, parameter). Arms: RE (random effects), FE (the brief's estimator), B1; the reply-back
    stream is estimated once and scored under both RE and FE."""
    rows = []
    blocks = []
    if "RE" in R:
        blocks.append(("RE", "joint", R["RE"]["names"], R["RE"]["coef"], R["RE"]["reps"], R["RE"]["se_sandwich"]))
        blocks.append(("RE", "reply_back", R["reply_back"]["names"], R["reply_back"]["coef"], R["reply_back"]["reps"], None))
    if "FE" in R:
        blocks.append(("FE", "stageA", R["FE"]["names"], R["FE"]["coef"], R["FE"]["reps"], R["FE"]["se_sandwich"]))
        blocks.append(("FE", "fitness", R["FE"]["fitness"]["names"], R["FE"]["fitness"]["coef"], R["FE"]["fitness"]["reps"], None))
        blocks.append(("FE", "reply_back", R["reply_back"]["names"], R["reply_back"]["coef"], R["reply_back"]["reps"], None))
    if "B1" in R:
        blocks.append(("B1", "stageA", R["B1"]["names"], R["B1"]["coef"], R["B1"]["reps"], R["B1"]["se_sandwich"]))
    for model, stage, names, coef, reps, se in blocks:
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
            tv = truth.get(nm)
            rows.append({"population": pop, "rep": rep, "model": model, "stage": stage, "param": nm,
                         "truth": tv, "estimate": float(coef[j]), "ci_lo": float(lo), "ci_hi": float(hi),
                         "boot_sd": float(np.nanstd(reps[:, j])),
                         "se_sandwich": float(se[j]) if se is not None else None,
                         "covered": (bool(lo <= tv <= hi) if tv is not None else None),
                         "max_thread_size": diag["max_thread_size"], "redraws": diag["redraws"],
                         "re_iterations": R.get("RE", {}).get("n_iter"), "seconds": secs})
    return rows


def run_recover(pops):
    n_rep, B = int(S1["replicates"]), int(S1["B"])
    n_fe = int(S1["fe_arm_replicates"])
    for pop in pops:
        rows, curves, done = [], [], set()
        fr, fc = OUT / f"stage1_recovery_{pop}.parquet", OUT / f"stage1_curves_{pop}.parquet"
        if fr.exists() and fc.exists():  # resume (deterministic seeds: identical to an uninterrupted run)
            rows = pl.read_parquet(fr).to_dicts()
            cprev = pl.read_parquet(fc).to_dicts()
            done = {r["rep"] for r in rows} & {c["rep"] for c in cprev}
            rows = [r for r in rows if r["rep"] in done]
            curves = [[c for c in cprev if c["rep"] in done]]
            print(f"  [recover] {pop}: resuming, replicates already done {sorted(done)}", flush=True)
        for rep in range(n_rep):
            if rep in done:
                continue
            t0 = time.time()
            items, diag, truth, rng = dataset(pop, rep)
            S = extract(items)
            R = EST.fit_all(S, B, rng, arms=("RE", "FE") if rep < n_fe else ("RE",), with_b1=True)
            rows += rows_from_fit(pop, rep, R, truth, diag, time.time() - t0)
            curves.append(clock_curves(R, truth, pop, rep, S))
            cov = np.mean([r["covered"] for r in rows if r["rep"] == rep and r["model"] == "RE" and r["covered"] is not None])
            print(f"  [recover] {pop} rep {rep}: {S['units'].height} comments, RE coverage {cov:.2f}, "
                  f"iters {R['RE']['n_iter']}, redraws {diag['redraws']}, {time.time() - t0:.0f}s", flush=True)
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(OUT / f"stage1_recovery_{pop}.parquet")
            pl.DataFrame([c for cc in curves for c in cc]).write_parquet(OUT / f"stage1_curves_{pop}.parquet")


AGES = np.logspace(np.log10(60), np.log10(3 * 86400), 25)


def clock_curves(R, truth, pop, rep, S):
    """log-shape curves of the two clocks, relative to their value at 1 h, with pointwise bootstrap bands."""
    out = []
    names = R["RE"]["names"]
    reps = R["RE"]["reps"]
    coef = R["RE"]["coef"]
    ev = S["events_c"].filter(pl.col("stream") == "comment").join(
        S["units"].select(pl.col("id").alias("unit"), "s"), on="unit")
    u_ev = ev["u"].to_numpy()
    tau_ev = (ev["u"] + ev["s"]).to_numpy()
    for clock, (i1, i2), evs in [("comment_clock_k", ("k1_log_u", "k2_log_u_sq"), u_ev),
                                 ("thread_clock_a", ("a1_log_tau", "a2_log_tau_sq"), tau_ev)]:
        j1, j2 = names.index(i1), names.index(i2)
        x = np.log(AGES / 60.0)
        x0 = np.log(60.0)  # reference: 1 hour
        f = lambda c1, c2: c1 * (x - x0) + c2 * (x * x - x0 * x0)
        true = f(truth[i1], truth[i2])
        est = f(coef[j1], coef[j2])
        band = np.array([f(r[j1], r[j2]) for r in reps])
        lo, hi = np.nanpercentile(band, [2.5, 97.5], axis=0)
        p5, p95 = (np.percentile(evs, [5, 95]) if len(evs) else (np.nan, np.nan))
        med = np.median(evs) if len(evs) else np.nan
        for k, a in enumerate(AGES):
            out.append({"population": pop, "rep": rep, "clock": clock, "age_s": a, "true": true[k], "estimate": est[k],
                        "lo": lo[k], "hi": hi[k], "in_event_range": bool(p5 <= a <= p95), "event_age_p5": p5,
                        "event_age_p95": p95, "event_age_median": med})
    return out


def run_smoke(pop):
    t0 = time.time()
    items, diag, truth, rng = dataset(pop, 0)
    print("simulated", items.height, diag["max_thread_size"], f"{time.time() - t0:.0f}s", flush=True)
    S = extract(items)
    print(summary(S), flush=True)
    t1 = time.time()
    R = EST.fit_all(S, 20, rng)
    print("fit", f"{time.time() - t1:.0f}s", R["design"], flush=True)
    rows = rows_from_fit(pop, 0, R, truth, diag, 0)
    df = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("model") == "full")
    pl.Config.set_tbl_rows(80)
    print(df.select("param", pl.col("truth").round(3), pl.col("estimate").round(3), pl.col("ci_lo").round(3),
                    pl.col("ci_hi").round(3), pl.col("boot_sd").round(3), pl.col("se_sandwich").round(3), "covered"))


# ------------------------------------------------------------------------------------------
def run_hn_joint():
    """Joint HN root stream on the same replicate datasets (regenerated deterministically), random-effects
    likelihood: root clock and intercept per post type, gamma shared, kappa on Ask HN, hours shared, fitness
    (r, b) shared."""
    n_rep, B = int(S1["replicates"]), int(S1["B"])
    truth = load_truth()
    tv = {"gamma": S1["truth_spec"]["hn_shared"]["gamma"], "minus_kappa": -S1["truth_spec"]["hn_shared"]["kappa"]}
    for pop in HN:
        tv[f"aR1_{pop}"] = truth[pop]["truth_vector"]["aR1"]
        tv[f"aR2_{pop}"] = truth[pop]["truth_vector"]["aR2"]
        tv[f"alpha_R_{pop}"] = truth[pop]["truth_vector"]["alpha_R"]
    tv["log_r"] = truth["ask_hn"]["truth_vector"]["log_r"]
    tv["b"] = truth["ask_hn"]["truth_vector"]["b"]
    for h in range(1, 24):
        tv[f"hour_{h}"] = truth["ask_hn"]["truth_vector"][f"hour_{h}"]
    rows, done = [], set()
    fj = OUT / "stage1_recovery_hn_joint.parquet"
    if fj.exists():  # resume (deterministic seeds)
        rows = pl.read_parquet(fj).to_dicts()
        done = {r["rep"] for r in rows}
        print(f"  [hn_joint] resuming, replicates already done {sorted(done)}", flush=True)
    for rep in range(n_rep):
        if rep in done:
            continue
        t0 = time.time()
        parts = []
        for pop in HN:
            items, diag, _, _ = dataset(pop, rep)
            parts.append(items.filter(pl.col("is_root") | (pl.col("depth") == 1)).with_columns(
                id=pl.when(pl.col("is_root")).then(pl.col("id")).otherwise(pl.lit(pop + ":") + pl.col("id"))))
        S = extract(pl.concat(parts))
        D = EST.root_only_design(S, HN)
        M = EST.REModel(D)
        st = np.zeros(M.p + 2)
        st[M.p], st[M.p + 1] = np.log(5.0), 1.0
        th = M.fit(st, iters=60)
        I = M.info(th)
        sc = M.scores(th)
        rng = np.random.default_rng([SEED, int(S1["dataset_seed_offset"]), 99, rep])
        W = rng.multinomial(D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)
        reps = th[None, :] + ((W - 1.0) @ sc) @ np.linalg.inv(I).T
        names = M.names + ["log_r", "b"]
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
            t = tv.get(nm)
            rows.append({"population": "hn_joint", "rep": rep, "model": "RE", "stage": "joint_root", "param": nm,
                         "truth": t, "estimate": float(th[j]), "ci_lo": float(lo), "ci_hi": float(hi),
                         "boot_sd": float(np.nanstd(reps[:, j])), "covered": bool(lo <= t <= hi) if t is not None else None,
                         "re_iterations": M.n_iter, "seconds": time.time() - t0})
        k = names.index("minus_kappa")
        print(f"  [hn_joint] rep {rep}: kappa {-th[k]:.4f} (truth {-tv['minus_kappa']}), gamma {th[names.index('gamma')]:.4f}, "
              f"iters {M.n_iter}, {time.time() - t0:.0f}s", flush=True)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(OUT / "stage1_recovery_hn_joint.parquet")


def run_bootcheck():
    """Pre-registered check: one-step vs full-refit thread bootstrap for the RE arm, Ask HN rep 0, B = 100."""
    B = int(S1["B"])
    items, diag, truth, rng = dataset("ask_hn", 0)
    S = extract(items)
    D = EST.build(S)
    M = EST.REModel(D)
    st = np.zeros(M.p + 2)
    st[M.p - 2] = np.log(max(D["yc"].sum(), 1) / np.exp(D["oc"][D["yc"] == 0]).sum())
    st[M.p - 1] = np.log(max(D["yr"].sum(), 1) / np.exp(D["or"][D["yr"] == 0]).sum())
    st[M.p], st[M.p + 1] = np.log(5.0), 1.0
    th = M.fit(st, iters=60)
    W = np.random.default_rng([SEED, int(S1["dataset_seed_offset"]), 98]).multinomial(
        D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)
    one = th[None, :] + ((W - 1.0) @ M.scores(th)) @ np.linalg.inv(M.info(th)).T
    full = np.full_like(one, np.nan)
    t0 = time.time()
    for b in range(B):
        full[b] = M.fit_weighted(th, W[b])
        if b % 10 == 0:
            print(f"  [bootcheck] {b}/{B} {time.time() - t0:.0f}s", flush=True)
    names = M.names + ["log_r", "b"]
    rows = []
    for j, nm in enumerate(names):
        lo1, hi1 = np.nanpercentile(one[:, j], [2.5, 97.5])
        lo2, hi2 = np.nanpercentile(full[:, j], [2.5, 97.5])
        rows.append({"param": nm, "estimate": float(th[j]), "onestep_lo": lo1, "onestep_hi": hi1,
                     "refit_lo": lo2, "refit_hi": hi2, "width_ratio": (hi1 - lo1) / (hi2 - lo2),
                     "centre_shift_in_refit_widths": ((lo1 + hi1) / 2 - (lo2 + hi2) / 2) / (hi2 - lo2)})
    df = pl.DataFrame(rows)
    df.write_csv(OUT / "stage1_bootcheck.csv")
    ok = float(((df["width_ratio"] - 1).abs() <= 0.25).mean())
    print(df, f"share within +-25%: {ok:.2f}", flush=True)


# ------------------------------------------------------------------------------------------
def run_summary():
    frames = [pl.read_parquet(f) for f in sorted(OUT.glob("stage1_recovery_*.parquet"))]
    d = pl.concat(frames, how="diagonal_relaxed")
    scored = d.filter(pl.col("model").is_in(["RE", "FE"]) & pl.col("truth").is_not_null())
    cov = scored.group_by("population", "model", "stage", "param").agg(
        n_rep=pl.len(), covered=pl.col("covered").sum(), truth=pl.col("truth").first(),
        mean_est=pl.col("estimate").mean(), mean_boot_sd=pl.col("boot_sd").mean(),
        sd_est=pl.col("estimate").std()).with_columns(
        bias=pl.col("mean_est") - pl.col("truth"),
        bias_in_sd=(pl.col("mean_est") - pl.col("truth")) / pl.col("mean_boot_sd"),
        sd_ratio=pl.col("mean_boot_sd") / pl.col("sd_est"),
        pass_literal=pl.col("covered") >= (0.9 * pl.col("n_rep")).ceil(),
    ).with_columns(
        genuine_fail=(pl.col("covered") <= (0.75 * pl.col("n_rep")).floor()) | (pl.col("bias_in_sd").abs() > 0.5)
    ).sort("population", "model", "stage", "param")
    cov.write_csv(OUT / "stage1_coverage.csv")
    b1 = d.filter(pl.col("model") == "B1").join(
        d.filter(pl.col("model") == "RE").select("population", "rep", "param", pl.col("truth").alias("t_true")),
        on=["population", "rep", "param"], how="left").group_by("population", "param").agg(
        truth=pl.col("t_true").first(), mean_est=pl.col("estimate").mean(), mean_boot_sd=pl.col("boot_sd").mean(),
        covered=(pl.col("ci_lo") <= pl.col("t_true")).and_(pl.col("ci_hi") >= pl.col("t_true")).sum(), n=pl.len())
    b1.write_csv(OUT / "stage1_B1_kernel.csv")
    curv = []
    for f in sorted(OUT.glob("stage1_curves_*.parquet")):
        c = pl.read_parquet(f)
        g = c.filter(pl.col("in_event_range")).group_by("population", "clock", "rep").agg(
            frac_in_band=((pl.col("true") >= pl.col("lo")) & (pl.col("true") <= pl.col("hi"))).mean())
        mid = c.with_columns(dist=(pl.col("age_s").log() - pl.col("event_age_median").log()).abs()).sort("dist") \
            .group_by("population", "clock", "rep").agg(halfwidth_at_median=((pl.col("hi") - pl.col("lo")) / 2).first())
        curv.append(g.join(mid, on=["population", "clock", "rep"]))
    cv = pl.concat(curv)
    cs = cv.group_by("population", "clock").agg(
        reps_band_ok=(pl.col("frac_in_band") >= 0.9).sum(), reps_informative=(pl.col("halfwidth_at_median") < 0.5).sum(),
        median_halfwidth=pl.col("halfwidth_at_median").median(), n=pl.len()).sort("population", "clock")
    cs.write_csv(OUT / "stage1_clock_shapes.csv")
    meta = d.group_by("population").agg(redraws=pl.col("redraws").max(), max_thread=pl.col("max_thread_size").max(),
                                         mean_seconds=pl.col("seconds").mean(), re_iter_max=pl.col("re_iterations").max())
    meta.write_csv(OUT / "stage1_run_meta.csv")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_width_chars(220)
    print(cov.filter(~pl.col("pass_literal") | pl.col("genuine_fail")).select(
        "population", "model", "param", pl.col("truth").round(3), pl.col("mean_est").round(3), "covered", "n_rep",
        pl.col("bias_in_sd").round(2), pl.col("sd_ratio").round(2), "genuine_fail"))
    print(cov.group_by("population", "model").agg(params=pl.len(), literal_pass=pl.col("pass_literal").sum(),
                                                  genuine_fail=pl.col("genuine_fail").sum()).sort("population", "model"))
    print(cs)
    print(b1.sort("population", "param"))
    print(meta)
    figures(cov.filter(pl.col("model") == "RE"))


def figures(cov: pl.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    # 1. coverage per parameter (RE arm), hours pooled into one row (min and mean)
    pops = POPULATIONS + ["hn_joint"]
    re = cov
    params = [p_ for p_ in dict.fromkeys(re.filter(~pl.col("param").str.starts_with("hour"))["param"].to_list())]
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(params) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, pop in enumerate(pops):
        x = re.filter(pl.col("population") == pop)
        for j, prm in enumerate(params + ["hours (min of 23)"]):
            if prm.startswith("hours"):
                h = x.filter(pl.col("param").str.starts_with("hour"))
                v = h["covered"].min() if h.height else None
                n = h["n_rep"].max() if h.height else None
            else:
                r = x.filter(pl.col("param") == prm)
                v = r["covered"][0] if r.height else None
                n = r["n_rep"][0] if r.height else None
            if v is None:
                continue
            col = "#1baf7a" if v >= 0.9 * n else ("#eda100" if v > 0.75 * n else "#e34948")
            ax.scatter(i, j, s=60, color=col, zorder=3)
            ax.text(i + 0.18, j, f"{v}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(params) + 1))
    ax.set_yticklabels(params + ["hours (min of 23)"], fontsize=8, color=INK2)
    ax.set_xticks(range(len(pops)))
    ax.set_xticklabels(pops, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(pops) - 0.2)
    ax.set_title("Stage 1, RE arm: replicates (of 20) whose 95% interval covers the truth", color=INK, fontsize=10, loc="left")
    ax.text(0.0, -0.06, "green >= 18 (pass)   amber 16-17   red <= 15 (genuine failure, F1-S3)", transform=ax.transAxes,
            fontsize=8, color=INK2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_coverage.png", facecolor=SURF)
    # 2. Moltbook clock shapes: truth vs estimates (all replicates)
    c = pl.read_parquet(OUT / "stage1_curves_moltbook.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, (clock, ttl) in zip(axes, [("comment_clock_k", "comment clock k(u)"), ("thread_clock_a", "thread clock a(tau)")]):
        ax.set_facecolor(SURF)
        x = c.filter(pl.col("clock") == clock)
        for rep in x["rep"].unique().to_list():
            r = x.filter(pl.col("rep") == rep).sort("age_s")
            ax.plot(r["age_s"].to_numpy() / 60, r["estimate"].to_numpy(), color="#2a78d6", lw=0.8, alpha=0.35)
        r0 = x.filter(pl.col("rep") == x["rep"].min()).sort("age_s")
        ax.plot(r0["age_s"].to_numpy() / 60, r0["true"].to_numpy(), color=INK, lw=2, label="truth")
        ax.plot([], [], color="#2a78d6", lw=1, label="estimate, each replicate")
        p5, p95 = x["event_age_p5"].median(), x["event_age_p95"].median()
        ax.axvspan(p5 / 60, p95 / 60, color=GRID, alpha=0.5, lw=0)
        ax.set_xscale("log")
        ax.set_title(f"Moltbook (burst-heavy): {ttl}, log shape vs 1 h", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("age (minutes); shaded = 5th-95th percentile of event ages", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    axes[0].set_ylabel("log rate relative to age 1 h", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_moltbook_clocks.png", facecolor=SURF)


# ------------------------------------------------------------------------------------------
# Stage 1 rerun (after review): estimator v2 (Rev-1, 5, 7; Rev-8 tested on Moltbook), outputs stage1b_*
# ------------------------------------------------------------------------------------------
from forum_model import fit1_estimator2 as EST2  # noqa: E402


def rows_from_fit2(pop, rep, R, truth, diag, secs, variant):
    rows = []
    conv = R["RE"]
    blocks = [("RE", "joint", R["RE"]["names"], R["RE"]["coef"], R["RE"]["reps"], R["RE"]["se_sandwich"]),
              ("RE", "reply_back", R["reply_back"]["names"], R["reply_back"]["coef"], R["reply_back"]["reps"], None),
              ("B1", "stageA", R["B1"]["names"], R["B1"]["coef"], R["B1"]["reps"], R["B1"]["se_sandwich"])]
    for model, stage, names, coef, reps, se in blocks:
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
            tv = truth.get(nm)
            rows.append({"population": pop, "variant": variant, "rep": rep, "model": model, "stage": stage, "param": nm,
                         "truth": tv, "estimate": float(coef[j]), "ci_lo": float(lo), "ci_hi": float(hi),
                         "boot_sd": float(np.nanstd(reps[:, j])), "se_sandwich": float(se[j]) if se is not None else None,
                         "covered": (bool(lo <= tv <= hi) if tv is not None else None),
                         "max_thread_size": diag["max_thread_size"], "redraws": diag["redraws"],
                         "re_iterations": conv["n_iter"], "re_converged": conv["converged"], "start_b": conv["start_b"],
                         "seconds": secs})
    return rows


def clock_curves2(R, truth, pop, rep, S, variant):
    """Separate clocks (as in Stage 1) and the combined clock k(u) + a(s_bar + u) (Rev-6), relative to u = 1 h,
    with pointwise bootstrap bands; s_bar = median creation thread age of the dataset's comments."""
    names, reps, coef = R["RE"]["names"], R["RE"]["reps"], R["RE"]["coef"]
    ev = S["events_c"].filter(pl.col("stream") == "comment").join(S["units"].select(pl.col("id").alias("unit"), "s"), on="unit")
    u_ev = ev["u"].to_numpy()
    tau_ev = (ev["u"] + ev["s"]).to_numpy()
    s_bar = float(np.median(S["units"]["s"].to_numpy()))
    x0 = np.log(60.0)
    xs = np.log(AGES / 60.0)
    xt = np.log((s_bar + AGES) / 60.0)
    xt0 = np.log((s_bar + 3600.0) / 60.0)
    j = {k: names.index(k) for k in ["k1_log_u", "k2_log_u_sq", "a1_log_tau", "a2_log_tau_sq"]}

    def kf(c):
        return c[j["k1_log_u"]] * (xs - x0) + c[j["k2_log_u_sq"]] * (xs ** 2 - x0 ** 2)

    def af(c, x=xs, xr=x0):
        return c[j["a1_log_tau"]] * (x - xr) + c[j["a2_log_tau_sq"]] * (x ** 2 - xr ** 2)

    def comb(c):
        return kf(c) + af(c, xt, xt0)
    tv = np.array([truth.get(n, np.nan) for n in names])
    out = []
    for clock, f, evs in [("comment_clock_k", kf, u_ev), ("thread_clock_a", af, tau_ev), ("combined_k_plus_a", comb, u_ev)]:
        band = np.array([f(r) for r in reps])
        lo, hi = np.nanpercentile(band, [2.5, 97.5], axis=0)
        true, est = f(tv), f(coef)
        p5, p95 = (np.percentile(evs, [5, 95]) if len(evs) else (np.nan, np.nan))
        med = np.median(evs) if len(evs) else np.nan
        for k, a in enumerate(AGES):
            out.append({"population": pop, "variant": variant, "rep": rep, "clock": clock, "age_s": a, "true": true[k],
                        "estimate": est[k], "lo": lo[k], "hi": hi[k], "in_event_range": bool(p5 <= a <= p95),
                        "event_age_p5": p5, "event_age_p95": p95, "event_age_median": med, "s_bar": s_bar})
    return out


def run_recover2(pop: str, variant: str = "standard"):
    n_rep, B = int(S1["replicates"]), int(S1["B"])
    rev8 = variant == "rev8"
    fr, fc = OUT / f"stage1b_recovery_{pop}_{variant}.parquet", OUT / f"stage1b_curves_{pop}_{variant}.parquet"
    rows, curves, done = [], [], set()
    if fr.exists() and fc.exists():
        rows = pl.read_parquet(fr).to_dicts()
        cprev = pl.read_parquet(fc).to_dicts()
        done = {r["rep"] for r in rows} & {c["rep"] for c in cprev}
        rows = [r for r in rows if r["rep"] in done]
        curves = [c for c in cprev if c["rep"] in done]
        print(f"  [recover2] {pop} {variant}: resuming, done {sorted(done)}", flush=True)
    for rep in range(n_rep):
        if rep in done:
            continue
        t0 = time.time()
        items, diag, truth, rng = dataset(pop, rep)
        S = extract(items)
        R = EST2.fit_all(S, B, rng, rev8=rev8)
        rows += rows_from_fit2(pop, rep, R, truth, diag, time.time() - t0, variant)
        curves += clock_curves2(R, truth, pop, rep, S, variant)
        cov = np.mean([r["covered"] for r in rows if r["rep"] == rep and r["model"] == "RE" and r["covered"] is not None])
        print(f"  [recover2] {pop} {variant} rep {rep}: RE coverage {cov:.2f}, iters {R['RE']['n_iter']}, "
              f"converged {R['RE']['converged']}, start b {R['RE']['start_b']}, {time.time() - t0:.0f}s", flush=True)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(fr)
        pl.DataFrame(curves).write_parquet(fc)


def run_hn_joint2():
    n_rep, B = int(S1["replicates"]), int(S1["B"])
    truth = load_truth()
    tv = {"gamma": S1["truth_spec"]["hn_shared"]["gamma"], "minus_kappa": -S1["truth_spec"]["hn_shared"]["kappa"]}
    for pop in HN:
        for nm in ["aR1", "aR2", "alpha_R"]:
            tv[f"{nm}_{pop}"] = truth[pop]["truth_vector"][nm]
    tv["log_r"], tv["b"] = truth["ask_hn"]["truth_vector"]["log_r"], truth["ask_hn"]["truth_vector"]["b"]
    for h in range(1, 24):
        tv[f"hour_{h}"] = truth["ask_hn"]["truth_vector"][f"hour_{h}"]
    fj = OUT / "stage1b_recovery_hn_joint.parquet"
    rows = pl.read_parquet(fj).to_dicts() if fj.exists() else []
    done = {r["rep"] for r in rows}
    for rep in range(n_rep):
        if rep in done:
            continue
        t0 = time.time()
        parts = []
        for pop in HN:
            items, diag, _, _ = dataset(pop, rep)
            parts.append(items.filter(pl.col("is_root") | (pl.col("depth") == 1)).with_columns(
                id=pl.when(pl.col("is_root")).then(pl.col("id")).otherwise(pl.lit(pop + ":") + pl.col("id"))))
        S = extract(pl.concat(parts))
        D = EST2.build(S, post_types=HN, comment_stream=False)
        M = EST2.REModel2(D)
        th, conv = M.fit_multistart()
        I, sc = M.info(th), M.scores(th)
        rng = np.random.default_rng([SEED, int(S1["dataset_seed_offset"]), 99, rep])
        W = rng.multinomial(D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)
        reps = th[None, :] + ((W - 1.0) @ sc) @ np.linalg.inv(I).T
        for j, nm in enumerate(M.names):
            lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
            t = tv.get(nm)
            rows.append({"population": "hn_joint", "variant": "standard", "rep": rep, "model": "RE", "stage": "joint_root",
                         "param": nm, "truth": t, "estimate": float(th[j]), "ci_lo": float(lo), "ci_hi": float(hi),
                         "boot_sd": float(np.nanstd(reps[:, j])), "covered": bool(lo <= t <= hi) if t is not None else None,
                         "re_iterations": conv["iterations"], "re_converged": conv["converged"], "start_b": conv["start_b"],
                         "seconds": time.time() - t0})
        k = M.names.index("minus_kappa")
        print(f"  [hn_joint2] rep {rep}: kappa {-th[k]:.4f}, gamma {th[M.names.index('gamma')]:.4f}, "
              f"iters {conv['iterations']}, converged {conv['converged']}, {time.time() - t0:.0f}s", flush=True)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(fj)


def run_bootcheck2():
    B = int(S1["B"])
    items, diag, truth, rng = dataset("ask_hn", 0)
    S = extract(items)
    D = EST2.build(S)
    M = EST2.REModel2(D)
    th, conv = M.fit_multistart()
    W = np.random.default_rng([SEED, int(S1["dataset_seed_offset"]), 98]).multinomial(
        D["nT"], np.full(D["nT"], 1.0 / D["nT"]), size=B).astype(float)
    one = th[None, :] + ((W - 1.0) @ M.scores(th)) @ np.linalg.inv(M.info(th)).T
    full = np.full_like(one, np.nan)
    t0 = time.time()
    for b in range(B):
        full[b] = M.fit_weighted(th, W[b])
        if b % 10 == 0:
            print(f"  [bootcheck2] {b}/{B} {time.time() - t0:.0f}s", flush=True)
    rows = []
    for j, nm in enumerate(M.names):
        lo1, hi1 = np.nanpercentile(one[:, j], [2.5, 97.5])
        lo2, hi2 = np.nanpercentile(full[:, j], [2.5, 97.5])
        rows.append({"param": nm, "estimate": float(th[j]), "onestep_lo": lo1, "onestep_hi": hi1, "refit_lo": lo2,
                     "refit_hi": hi2, "width_ratio": (hi1 - lo1) / (hi2 - lo2),
                     "centre_shift_in_refit_widths": ((lo1 + hi1) / 2 - (lo2 + hi2) / 2) / (hi2 - lo2)})
    df = pl.DataFrame(rows)
    df.write_csv(OUT / "stage1b_bootcheck.csv")
    print(df, f"share within +-25%: {float(((df['width_ratio'] - 1).abs() <= 0.25).mean()):.2f}", flush=True)


ROOT_STREAM = ["aR1", "aR2", "gamma", "alpha_R", "b", "log_r"]


def _coverage(d: pl.DataFrame) -> pl.DataFrame:
    return d.filter((pl.col("model") == "RE") & pl.col("truth").is_not_null()).group_by(
        "population", "variant", "stage", "param").agg(
        n_rep=pl.len(), covered=pl.col("covered").sum(), truth=pl.col("truth").first(),
        mean_est=pl.col("estimate").mean(), mean_boot_sd=pl.col("boot_sd").mean(), sd_est=pl.col("estimate").std()
    ).with_columns(bias_in_sd=(pl.col("mean_est") - pl.col("truth")) / pl.col("mean_boot_sd"),
                   sd_ratio=pl.col("mean_boot_sd") / pl.col("sd_est"),
                   pass_literal=pl.col("covered") >= (0.9 * pl.col("n_rep")).ceil()).with_columns(
        genuine_fail=(pl.col("covered") <= (0.75 * pl.col("n_rep")).floor()) | (pl.col("bias_in_sd").abs() > 0.5))


def run_summary2():
    d = pl.concat([pl.read_parquet(f) for f in sorted(OUT.glob("stage1b_recovery_*.parquet"))], how="diagonal_relaxed")
    cov = _coverage(d).sort("population", "variant", "stage", "param")
    # Rev-8 decision (pre-stated rule)
    m = cov.filter(pl.col("population") == "moltbook")
    dec = {}
    for v in ["standard", "rev8"]:
        x = m.filter(pl.col("variant") == v)
        dec[v] = {"root_genuine": int(x.filter(pl.col("param").is_in(ROOT_STREAM))["genuine_fail"].sum()),
                  "total_genuine": int(x["genuine_fail"].sum())}
    keep = dec["rev8"]["root_genuine"] < dec["standard"]["root_genuine"] and dec["rev8"]["total_genuine"] <= dec["standard"]["total_genuine"]
    kept = "rev8" if keep else "standard"
    (OUT / "stage1b_rev8_decision.json").write_text(json.dumps({"counts": dec, "keep_rev8": keep, "moltbook_variant": kept}, indent=1))
    cov = cov.with_columns(primary=(pl.col("population") != "moltbook") | (pl.col("variant") == kept))
    cov.write_csv(OUT / "stage1b_coverage.csv")
    # clock shapes incl. combined clock (F1-S2b)
    c = pl.concat([pl.read_parquet(f) for f in sorted(OUT.glob("stage1b_curves_*.parquet"))], how="diagonal_relaxed")
    g = c.filter(pl.col("in_event_range")).group_by("population", "variant", "clock", "rep").agg(
        frac_in_band=((pl.col("true") >= pl.col("lo")) & (pl.col("true") <= pl.col("hi"))).mean())
    cs = g.group_by("population", "variant", "clock").agg(reps_band_ok=(pl.col("frac_in_band") >= 0.9).sum(), n=pl.len()) \
        .sort("population", "variant", "clock")
    cs.write_csv(OUT / "stage1b_clock_shapes.csv")
    b1 = d.filter(pl.col("model") == "B1").join(
        d.filter(pl.col("model") == "RE").select("population", "variant", "rep", "param", pl.col("truth").alias("t_true")),
        on=["population", "variant", "rep", "param"], how="left").group_by("population", "variant", "param").agg(
        truth=pl.col("t_true").first(), mean_est=pl.col("estimate").mean(),
        covered=((pl.col("ci_lo") <= pl.col("t_true")) & (pl.col("ci_hi") >= pl.col("t_true"))).sum(), n=pl.len())
    b1.write_csv(OUT / "stage1b_B1_kernel.csv")
    meta = d.filter(pl.col("model") == "RE").group_by("population", "variant", "rep").agg(
        pl.col("re_iterations").first(), pl.col("re_converged").first(), pl.col("start_b").first(),
        pl.col("seconds").first()).group_by("population", "variant").agg(
        fits=pl.len(), converged=pl.col("re_converged").sum(), max_iter=pl.col("re_iterations").max(),
        start_b=pl.col("start_b").unique().sort().cast(pl.String).str.join(","), mean_seconds=pl.col("seconds").mean()) \
        .sort("population", "variant")
    meta.write_csv(OUT / "stage1b_run_meta.csv")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_width_chars(220)
    print(cov.group_by("population", "variant").agg(params=pl.len(), literal_pass=pl.col("pass_literal").sum(),
                                                    genuine_fail=pl.col("genuine_fail").sum()).sort("population", "variant"))
    print(cov.filter(pl.col("genuine_fail")).select("population", "variant", "param", pl.col("truth").round(3),
          pl.col("mean_est").round(3), "covered", pl.col("bias_in_sd").round(2), pl.col("sd_ratio").round(2)))
    print(dec, "keep rev8:", keep)
    print(cs)
    print(b1.sort("population", "param"))
    print(meta)
    figures2(cov.filter(pl.col("primary")), c.filter((pl.col("population") == "moltbook") & (pl.col("variant") == kept)))


def figures2(cov: pl.DataFrame, cm: pl.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    pops = POPULATIONS + ["hn_joint"]
    params = list(dict.fromkeys(cov.filter(~pl.col("param").str.starts_with("hour"))["param"].to_list()))
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(params) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, pop in enumerate(pops):
        x = cov.filter(pl.col("population") == pop)
        for j, prm in enumerate(params + ["hours (min of 23)"]):
            if prm.startswith("hours"):
                h = x.filter(pl.col("param").str.starts_with("hour"))
                v, n = (h["covered"].min(), h["n_rep"].max()) if h.height else (None, None)
            else:
                r = x.filter(pl.col("param") == prm)
                v, n = (r["covered"][0], r["n_rep"][0]) if r.height else (None, None)
            if v is None:
                continue
            col = "#1baf7a" if v >= 0.9 * n else ("#eda100" if v > 0.75 * n else "#e34948")
            ax.scatter(i, j, s=60, color=col, zorder=3)
            ax.text(i + 0.18, j, f"{v}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(params) + 1))
    ax.set_yticklabels(params + ["hours (min of 23)"], fontsize=8, color=INK2)
    ax.set_xticks(range(len(pops)))
    ax.set_xticklabels(pops, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(pops) - 0.2)
    ax.set_title("Stage 1 rerun: replicates (of 20) whose 95% interval covers the truth", color=INK, fontsize=10, loc="left")
    ax.text(0.0, -0.06, "green >= 18 (pass)   amber 16-17   red <= 15", transform=ax.transAxes, fontsize=8, color=INK2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "stage1b_coverage.png", facecolor=SURF)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x = cm.filter(pl.col("clock") == "combined_k_plus_a")
    for rep in x["rep"].unique().to_list():
        r = x.filter(pl.col("rep") == rep).sort("age_s")
        a, t = r["age_s"].to_numpy() / 60, r["true"].to_numpy()
        ax.fill_between(a, r["lo"].to_numpy() - t, r["hi"].to_numpy() - t, color="#2a78d6", alpha=0.04, lw=0)
        ax.plot(a, r["estimate"].to_numpy() - t, color="#2a78d6", lw=0.8, alpha=0.5)
    ax.axhline(0, color=INK, lw=1.5, label="truth")
    ax.plot([], [], color="#2a78d6", lw=1, label="estimate − truth, each replicate (band shaded)")
    ax.axvspan(x["event_age_p5"].median() / 60, x["event_age_p95"].median() / 60, color=GRID, alpha=0.5, lw=0)
    ax.set_xscale("log")
    ax.set_title("Moltbook: combined clock k(u) + a(s̄ + u), error in log shape vs 1 h", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("comment age (minutes); grey = 5th-95th percentile of event ages", color=INK2, fontsize=9)
    ax.set_ylabel("estimate − truth (log rate)", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "stage1b_moltbook_combined_clock.png", facecolor=SURF)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "roots":
        run_roots()
    elif cmd == "calibrate":
        run_calibrate()
    elif cmd == "recover":
        run_recover(sys.argv[2:] or POPULATIONS)
    elif cmd == "smoke":
        run_smoke(sys.argv[2])
    elif cmd == "hn_joint":
        run_hn_joint()
    elif cmd == "bootcheck":
        run_bootcheck()
    elif cmd == "summary":
        run_summary()
    elif cmd == "recover2":
        run_recover2(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "standard")
    elif cmd == "hn_joint2":
        run_hn_joint2()
    elif cmd == "bootcheck2":
        run_bootcheck2()
    elif cmd == "summary2":
        run_summary2()
