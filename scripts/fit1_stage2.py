"""First model fit, Stage 2: fit and compare (docs/fit1_brief.md; pre-stated tests F2-* in docs/results_register.md).

Usage:
  python3 scripts/fit1_stage2.py run [pop ...]   # per population (human populations by default); resumable
  python3 scripts/fit1_stage2.py joint            # joint HN root-stream fit (kappa)
  python3 scripts/fit1_stage2.py summary          # pass rule, tables, figures
  python3 scripts/fit1_stage2.py smoke            # code check on synthetic data only (no real data read)

Data: calibration split only. Fit on the capped inner fit fold, compare on the capped inner held-out fold
(cap and seeds in partitions.json -> fit1). The validation split is not read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model import fit1_estimator2 as EST2  # noqa: E402
from forum_model import fit1_stage2 as F2  # noqa: E402
from forum_model.fit1_data import POPULATIONS, cap_threads, load_population  # noqa: E402
from forum_model.fit1_streams import extract  # noqa: E402

P = load_partitions()
F1 = P["fit1"]
S2 = F1["stage2"]
SEED = P["seed"]
OUT = ROOT / "outputs" / "fit1"
FIG = ROOT / "figures" / "fit1"
HUMANS = ["philosophy", "buildapc", "ask_hn", "link", "show_hn"]
HN = ["ask_hn", "link", "show_hn"]
B_FIT, B_HO, N_SIM = int(S2["B_fit"]), int(S2["B_heldout"]), int(S2["sim_replicates"])
_HN_ITEMS = None


def hn_items():
    global _HN_ITEMS
    if _HN_ITEMS is None:
        from forum_model.units import load
        _HN_ITEMS = load("hn", None, ["body_status"])
    return _HN_ITEMS


def folds(pop: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    it = load_population(pop, hn_items=hn_items() if pop in HN else None)
    cap = int(F1["cap_comments"])
    return cap_threads(it.filter(pl.col("fold") == "fit"), cap), cap_threads(it.filter(pl.col("fold") == "heldout"), cap)


def param_rows(pop, model, fit):
    rows = []
    for j, nm in enumerate(fit["names"]):
        lo, hi = F2.interval(fit["reps"][:, j])
        rows.append({"population": pop, "model": model, "param": nm, "estimate": float(fit["coef"][j]), "ci_lo": lo,
                     "ci_hi": hi, "boot_sd": float(np.nanstd(fit["reps"][:, j])), "n_iter": fit.get("n_iter"),
                     "converged": fit.get("converged"), "start_b": fit.get("start_b")})
    if model == "full":  # daily hour profile (hours centred on their 24-hour mean)
        hk = [fit["names"].index(f"hour_{h}") for h in range(1, 24)]
        Hb = np.c_[np.zeros(len(fit["reps"])), fit["reps"][:, hk]]
        Hb -= Hb.mean(axis=1, keepdims=True)
        h0 = np.r_[0.0, fit["coef"][hk]]
        h0 -= h0.mean()
        for h in range(24):
            lo, hi = F2.interval(Hb[:, h])
            rows.append({"population": pop, "model": model, "param": f"hourc_{h}", "estimate": float(h0[h]),
                         "ci_lo": lo, "ci_hi": hi, "boot_sd": float(np.std(Hb[:, h])), "n_iter": None,
                         "converged": None, "start_b": None})
    return rows


def run_pop(pop: str, S_fit: dict, S_ho: dict, items_ho: pl.DataFrame, tag: str = "stage2", B_fit=B_FIT, B_ho=B_HO,
            n_sim=N_SIM):
    t0 = time.time()
    pidx = POPULATIONS.index(pop) if pop in POPULATIONS else 99
    log = lambda m: print(f"  [{tag}] {pop}: {m} ({time.time() - t0:.0f}s)", flush=True)
    W = F2.boot_weights(S_fit["roots"].height, B_fit, [SEED, int(S2["seed_offset"]), pidx, 0])
    fits = {}
    for m in ("full", "B1", "B3"):
        fits[m] = F2.fit_re(S_fit, m, W)
        log(f"{m} fitted: iters {fits[m]['n_iter']}, converged {fits[m]['converged']}, start b {fits[m]['start_b']}")
    rb = F2.fit_reply_back(S_fit, W)
    b2 = F2.fit_b2(S_fit)
    log(f"reply-back and B2 fitted (B2 iters {b2['n_iter']})")
    rows = []
    for m in ("full", "B1", "B3"):
        rows += param_rows(pop, m, fits[m])
    rows += param_rows(pop, "reply_back", rb)
    rows += [{"population": pop, "model": "B2", "param": n, "estimate": float(c), "ci_lo": None, "ci_hi": None,
              "boot_sd": None, "n_iter": b2["n_iter"], "converged": None, "start_b": None}
             for n, c in zip(b2["names"], b2["coef"])]
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(OUT / f"{tag}_params_{pop}.parquet")
    # held-out timing
    tm = F2.heldout_timing(S_ho, fits, rb)
    log("held-out timing done")
    pc = F2.parent_choice(S_ho, fits["full"], rb, b2)
    log("held-out parent choice done")
    # simulations of the held-out threads (real root times and post types; fitness drawn fresh)
    roots = S_ho["roots"].select("thread_id", "T0", "post_type").with_columns(question=pl.col("post_type") == "ask_hn")
    obs = F2.thread_stats(items_ho, roots, observed=True)
    sims, sdiag = {}, {}
    for k, m in enumerate(("full", "B1", "B3")):
        sims[m], sdiag[m] = F2.simulate_stats(F2.gen_params(fits[m], rb, m), roots, n_sim,
                                              [SEED, int(S2["seed_offset"]), pidx, 1 + k])
        log(f"{m} simulated: {sdiag[m]}")
    # thread bootstrap over held-out threads
    nT = S_ho["roots"].height
    Wh = F2.boot_weights(nT, B_ho, [SEED, int(S2["seed_offset"]), pidx, 9])
    ones = np.ones(nT)
    res = {"population": pop, "threads_fit": S_fit["roots"].height, "threads_heldout": nT,
           "events_heldout": int(tm["n_events"].sum()), "sim_diag": sdiag,
           "fits": {m: {k: fits[m][k] for k in ("n_iter", "converged", "start_b", "loglik")} for m in fits},
           "b2": {"coef": b2["coef"].tolist(), "n_iter": b2["n_iter"]},
           "timing_B3_check_max_abs_diff": tm["B3_check_max_abs_diff"]}

    def stat(f):
        pt = f(ones[None, :])[0]
        bs = f(Wh)
        lo, hi = F2.interval(bs)
        return {"estimate": float(pt), "ci_lo": lo, "ci_hi": hi}
    n_ev = tm["n_events"]
    for m in ("B1", "B3"):
        d = tm["full"] - tm[m]
        res[f"timing_full_minus_{m}"] = stat(lambda w, d=d: (w @ d) / (w @ n_ev))
    for m in ("full", "B1", "B3"):
        res[f"timing_ll_per_event_{m}"] = stat(lambda w, m=m: (w @ tm[m]) / (w @ n_ev))
    pcd = pc.sort("ti")
    ti = pcd["ti"].to_numpy()
    full_pc = np.zeros(nT); b2_pc = np.zeros(nT); ne = np.zeros(nT)
    t3f = np.zeros(nT); t3b = np.zeros(nT); n4 = np.zeros(nT); t3f4 = np.zeros(nT); t3b4 = np.zeros(nT)
    for arr, col in ((full_pc, "ll_full"), (b2_pc, "ll_b2"), (ne, "events"), (t3f, "top3_full"), (t3b, "top3_b2"),
                     (n4, "events_4plus"), (t3f4, "top3_full_4plus"), (t3b4, "top3_b2_4plus")):
        arr[ti] = pcd[col].to_numpy()
    res["parent_full_minus_B2"] = stat(lambda w: (w @ (full_pc - b2_pc)) / (w @ ne))
    res["parent_ll_per_reply_full"] = stat(lambda w: (w @ full_pc) / (w @ ne))
    res["parent_ll_per_reply_B2"] = stat(lambda w: (w @ b2_pc) / (w @ ne))
    res["top3_full"] = stat(lambda w: (w @ t3f) / (w @ ne))
    res["top3_B2"] = stat(lambda w: (w @ t3b) / (w @ ne))
    res["top3_full_minus_B2"] = stat(lambda w: (w @ (t3f - t3b)) / (w @ ne))
    res["top3_full_4plus"] = stat(lambda w: (w @ t3f4) / (w @ n4))
    res["top3_B2_4plus"] = stat(lambda w: (w @ t3b4) / (w @ n4))
    res["share_replies_4plus_candidates"] = stat(lambda w: (w @ n4) / (w @ ne))
    # depth tail P(depth >= 10) and mu, mu0
    res["depth10_observed"] = stat(lambda w: (w @ obs["n_deep"]) / (w @ obs["n_com"]))
    for m in ("full", "B1", "B3"):
        nd = sum(s["n_deep"] for s in sims[m]) if sims[m] else np.zeros(nT)
        nc = sum(s["n_com"] for s in sims[m]) if sims[m] else np.ones(nT)
        res[f"depth10_{m}"] = stat(lambda w, nd=nd, nc=nc: (w @ nd) / (w @ nc))
        res[f"depth10_error_{m}"] = stat(lambda w, nd=nd, nc=nc: (w @ nd) / (w @ nc) - (w @ obs["n_deep"]) / (w @ obs["n_com"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            res[f"depth10_logratio_{m}"] = stat(lambda w, nd=nd, nc=nc: np.log(((w @ nd) / (w @ nc)) / ((w @ obs["n_deep"]) / (w @ obs["n_com"]))))
    for q in ("mu", "mu0"):
        mc = lambda W_, st: np.array([F2.mcf(st[q], w) for w in W_])
        res[f"{q}_observed_heldout"] = stat(lambda w: mc(w, obs))
        for m in ("full", "B1", "B3"):
            if sims[m]:
                res[f"{q}_{m}"] = stat(lambda w, m=m: np.mean([mc(w, s) for s in sims[m]], axis=0))
    (OUT / f"{tag}_result_{pop}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    pl.DataFrame({"ti": np.arange(nT), "n_events": n_ev, **{f"ll_{m}": tm[m] for m in ("full", "B1", "B3")},
                  "pc_full": full_pc, "pc_b2": b2_pc}).write_parquet(OUT / f"{tag}_heldout_threads_{pop}.parquet")
    log("done")
    return res


def cmd_run(pops):
    for pop in pops:
        if (OUT / f"stage2_result_{pop}.json").exists():
            print(f"  [stage2] {pop}: done, skipped", flush=True)
            continue
        fit, ho = folds(pop)
        S_fit, S_ho = extract(fit), extract(ho)
        del fit
        print(f"  [stage2] {pop}: fit threads {S_fit['roots'].height}, held-out threads {S_ho['roots'].height}", flush=True)
        run_pop(pop, S_fit, S_ho, ho)


def cmd_joint():
    f = OUT / "stage2_params_hn_joint.parquet"
    if f.exists():
        print("  [stage2] joint: done, skipped", flush=True)
        return
    t0 = time.time()
    parts = []
    for pop in HN:
        fit, _ = folds(pop)
        parts.append(fit.filter(pl.col("is_root") | (pl.col("depth") == 1)).with_columns(
            id=pl.when(pl.col("is_root")).then(pl.col("id")).otherwise(pl.lit(pop + ":") + pl.col("id")),
            parent_id=pl.when(pl.col("is_root") | (pl.col("depth") == 1)).then(pl.col("parent_id")).otherwise(pl.lit(pop + ":") + pl.col("parent_id"))))
    S = extract(pl.concat(parts, how="diagonal_relaxed"))
    D = EST2.build(S, post_types=HN, comment_stream=False)
    M = EST2.REModel2(D)
    th, conv = M.fit_multistart()
    W = F2.boot_weights(D["nT"], B_FIT, [SEED, int(S2["seed_offset"]), 98, 0])
    reps = th[None, :] + ((W - 1.0) @ M.scores(th)) @ np.linalg.inv(M.info(th)).T
    fit = {"names": M.names, "coef": th, "reps": reps, "n_iter": conv["iterations"], "converged": conv["converged"],
           "start_b": conv["start_b"]}
    pl.DataFrame(param_rows("hn_joint", "full", fit), infer_schema_length=None).write_parquet(f)
    k = M.names.index("minus_kappa")
    print(f"  [stage2] joint: threads {D['nT']}, kappa {-th[k]:.4f}, iters {conv['iterations']}, converged "
          f"{conv['converged']}, {time.time() - t0:.0f}s", flush=True)


def stage1_rev10_flags() -> set[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location("s1", ROOT / "scripts" / "fit1_stage1.py")
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    f = s1.score_rev10(s1._load("stage1b").filter(pl.col("variant") == "standard"))
    return {(r["population"], r["param"]) for r in f.filter(pl.col("genuine_fail")).iter_rows(named=True)}


def cmd_summary():
    res = {p: json.loads((OUT / f"stage2_result_{p}.json").read_text()) for p in HUMANS if (OUT / f"stage2_result_{p}.json").exists()}
    rows, passes = [], {}
    for p, r in res.items():
        ok = {}
        for key in ("timing_full_minus_B1", "timing_full_minus_B3", "parent_full_minus_B2"):
            ok[key] = r[key]["ci_lo"] > 0
            rows.append({"population": p, "comparison": key, **r[key], "beats_with_interval_excluding_0": ok[key]})
        passes[p] = all(ok.values())
    comp = pl.DataFrame(rows)
    comp.write_csv(OUT / "stage2_comparisons.csv")
    n_pass = sum(passes.values())
    verdict = {"per_population": passes, "populations_passing": n_pass, "populations_scored": len(passes),
               "F2_S1_passes": n_pass >= 4}
    (OUT / "stage2_verdict.json").write_text(json.dumps(verdict, indent=1))
    metrics = []
    for p, r in res.items():
        for key, v in r.items():
            if isinstance(v, dict) and "estimate" in v:
                metrics.append({"population": p, "quantity": key, **v})
    pl.DataFrame(metrics).write_csv(OUT / "stage2_metrics.csv")
    flags = stage1_rev10_flags()
    pars = pl.concat([pl.read_parquet(f) for f in sorted(OUT.glob("stage2_params_*.parquet"))], how="diagonal_relaxed")
    pars = pars.with_columns(stage1_rev10_flag=pl.struct("population", "param").map_elements(
        lambda s: (s["population"], s["param"]) in flags, return_dtype=pl.Boolean))
    pars.write_csv(OUT / "stage2_params.csv")
    audit = pl.read_csv(ROOT / "outputs" / "audit" / "mu_bootstrap.csv").filter(pl.col("unit").is_in(["nonroot", "root_live"]))
    audit.write_csv(OUT / "stage2_audit_mu_reference.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(220)
    print(comp)
    print(verdict)
    figures(pars, res)


def figures(pars: pl.DataFrame, res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a5cd6"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, pop in enumerate(HUMANS):
        x = pars.filter((pl.col("population") == pop) & (pl.col("model") == "full") & pl.col("param").str.starts_with("hourc_"))
        if not x.height:
            continue
        h = x["param"].str.replace("hourc_", "").cast(pl.Int64).to_numpy()
        o = np.argsort(h)
        ax.fill_between(h[o], x["ci_lo"].to_numpy()[o], x["ci_hi"].to_numpy()[o], color=COL[i], alpha=0.15, lw=0)
        ax.plot(h[o], x["estimate"].to_numpy()[o], color=COL[i], lw=1.6, label=pop)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("hour of day (UTC)", color=INK2, fontsize=9)
    ax.set_ylabel("log rate relative to the daily mean", color=INK2, fontsize=9)
    ax.set_title("Stage 2: daily activity profile c(h), full model, 95% bootstrap band", color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=5)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_hour_profiles.png", facecolor=SURF)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, key, lab in zip(axs, ("timing_full_minus_B1", "timing_full_minus_B3", "parent_full_minus_B2"),
                            ("timing: full − B1 (per event)", "timing: full − B3 (per event)", "parent choice: full − B2 (per reply)")):
        ax.set_facecolor(SURF)
        for i, pop in enumerate(HUMANS):
            if pop not in res:
                continue
            v = res[pop][key]
            ax.errorbar(v["estimate"], i, xerr=[[v["estimate"] - v["ci_lo"]], [v["ci_hi"] - v["estimate"]]], fmt="o",
                        color=COL[i], ms=5, capsize=2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(HUMANS)))
        ax.set_yticklabels(HUMANS if ax is axs[0] else [], fontsize=8, color=INK2)
        ax.invert_yaxis()
        ax.set_title(lab, color=INK, fontsize=9, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axs[0].set_xlabel("log-likelihood difference, 95% thread-bootstrap interval", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_comparisons.png", facecolor=SURF)


def cmd_smoke():
    """Synthetic-only code check: Ask HN truth, development seed, a subset of roots, split by thread hash."""
    spec = importlib.util.spec_from_file_location("s1", ROOT / "scripts" / "fit1_stage1.py")
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    from forum_model.fit1_generator import Generator
    pop = "ask_hn"
    T = s1.load_truth()[pop]
    gp = dict(T["params"])
    gp["hour"] = np.array(gp["hour"])
    gp["k"], gp["a"], gp["aR"] = tuple(gp["k"]), tuple(gp["a"]), tuple(gp["aR"])
    roots = s1.roots_of(pop).head(3000)
    items, _ = Generator(gp).simulate(roots, np.random.default_rng([SEED, 9999, 7, 0]))
    h = items["thread_id"].hash(SEED) % 5
    it_fit, it_ho = items.filter(h != 0), items.filter(h == 0)
    S_fit, S_ho = extract(it_fit), extract(it_ho)
    print("smoke truth (selected):", {k: round(v, 3) for k, v in T["truth_vector"].items() if not k.startswith("hour")})
    r = run_pop("smoke_" + pop, S_fit, S_ho, it_ho, tag="stage2smoke", B_fit=20, B_ho=100, n_sim=2)
    print(json.dumps({k: v for k, v in r.items() if isinstance(v, dict) and "estimate" in v}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2:] or HUMANS)
    elif cmd == "joint":
        cmd_joint()
    elif cmd == "summary":
        cmd_summary()
    elif cmd == "smoke":
        cmd_smoke()
