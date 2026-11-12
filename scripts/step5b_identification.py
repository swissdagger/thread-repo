"""Step 5b — saturation identification (calibration split only). Pre-registered in docs/results_register.md
(R23-R25) and partitions.json -> step5b, committed before this script was run.

Usage: python scripts/step5b_identification.py [dose] [show] [placebo]

  dose     5b.1: Step 5 model with steps at the 1st, 2nd and 3rd top-level comments (Ask HN, HN link, Show HN).
  show     5b.2: Step 5 model unchanged on Show HN; contrasts Ask - Show and Show - link.
  placebo  5b.3: synthetic threads with random exposure spikes, no saturation, no popularity feedback, matched to
           Ask HN and HN link volumes; Step 5 estimator unchanged.
Outputs: outputs/hn/step5b_dose.csv, step5b_show.csv, step5b_placebo.csv, step5b_placebo_summary.csv
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model.fe_poisson import FEPoissonBoot  # noqa: E402
from forum_model.units import build_units, hn_population, load  # noqa: E402
import step4b_extensions as S4b  # noqa: E402
import step5_saturation as S5  # noqa: E402

P = load_partitions()
Q = P["step5b"]
SEED = P["seed"]
B = int(Q["B"])
H, BINS, CAP = S5.H, S5.BINS, S5.CAP
NB = len(BINS) - 1
OUT = ROOT / "outputs" / "hn"
pct = S5.pct


def roots(comm: str, hn: pl.DataFrame, cap: bool = True):
    it, f = hn_population(comm, hn_items=hn)
    U = build_units(it, f["root_live"], f"hn:{comm}")
    del it
    par = U["parents"]
    n_all = par.height
    if cap and par.height > CAP:
        par = par.with_columns(u=pl.col("thread_id").map_elements(S4b.thread_hash, return_dtype=pl.Float64)) \
            .sort("u").head(CAP).drop("u")
    rep = U["replies"].join(par.select(pl.col("id").alias("parent")), on="parent", how="semi")
    return par, rep, n_all


def cells_k(par: pl.DataFrame, rep: pl.DataFrame, K: int = 3) -> dict:
    """States s = number of earlier top-level comments (0..K, K = 'K or more'). State s ends with comment s+1
    (its event). Boundaries tau_j = max(d_j, tau_{j-1} + 1 s), tau_0 = 0; the last state runs to cH."""
    n = par.height
    cH = np.minimum(par["c"].to_numpy(), H)
    pid = {k: i for i, k in enumerate(par["id"].to_list())}
    rp = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
    dl = rep["delay"].to_numpy()
    o = np.lexsort((dl, rp))
    rp, dl = rp[o], dl[o]
    keep = dl <= cH[rp]
    rp, dl = rp[keep], dl[keep]
    start = np.r_[0, np.nonzero(rp[1:] != rp[:-1])[0] + 1] if len(rp) else np.array([], dtype=int)
    rank = np.arange(len(rp)) - np.repeat(start, np.diff(np.r_[start, len(rp)])) if len(rp) else np.array([], dtype=int)
    tau = np.full((n, K + 1), np.inf)  # tau[:, j] = end of state j (time of comment j+1), j < K
    tau_prev = np.zeros(n)
    for j in range(K):
        m = rank == j
        tj = np.full(n, np.inf)
        tj[rp[m]] = dl[m]
        tj = np.where(np.isfinite(tj), np.maximum(tj, tau_prev + 1.0), np.inf)
        tj = np.where(tj <= cH, tj, np.inf)
        tau[:, j] = tj
        tau_prev = np.where(np.isfinite(tj), tj, tau_prev)
    rows = {"i": [], "bin": [], "state": [], "y": [], "ex": []}
    lo_s = np.zeros(n)
    for s in range(K + 1):
        hi_s = np.minimum(tau[:, s], cH) if s < K else cH
        alive = lo_s < hi_s
        if s < K:
            ev_t = tau[:, s]
            has_ev = np.isfinite(ev_t)
            ev_bin = np.clip(np.searchsorted(BINS, np.where(has_ev, ev_t, 0), side="left") - 1, 0, NB - 1)
        else:
            m = rank >= K
            dd = np.maximum(dl[m], lo_s[rp[m]] + 0.5)
            later = np.bincount(rp[m] * NB + np.clip(np.searchsorted(BINS, dd, side="left") - 1, 0, NB - 1),
                                minlength=n * NB).reshape(n, NB)
        for b in range(NB):
            ex = np.clip(np.minimum(hi_s, BINS[b + 1]) - np.maximum(lo_s, BINS[b]), 0, None) * alive
            y = ((has_ev & (ev_bin == b)).astype(float) if s < K else later[:, b].astype(float))
            idx = np.nonzero(ex > 0)[0]
            for k, v in [("i", idx), ("bin", np.full(len(idx), b)), ("state", np.full(len(idx), s)),
                         ("y", y[idx]), ("ex", ex[idx])]:
                rows[k].append(v)
        lo_s = np.where(np.isfinite(tau[:, s]), tau[:, s], cH) if s < K else lo_s
    R = {k: np.concatenate(v) for k, v in rows.items()}
    return R


def fit_states(R: dict, w0: np.ndarray, K: int, rng, n_threads: int):
    X = np.column_stack([(R["state"] >= j).astype(float) for j in range(1, K + 1)]
                        + [(R["bin"] == b).astype(float) for b in range(1, NB)])
    prep = FEPoissonBoot(R["i"], R["y"], X, np.log(R["ex"]), R["i"])
    coef, se = prep.fit(w0, se=True)
    reps = np.array([prep.fit(w0 * rng.multinomial(n_threads, np.full(n_threads, 1.0 / n_threads)), b0=coef)[:K]
                     for _ in range(B)])
    return coef[:K], se[:K], reps


# ------------------------------------------------------------------------------------------
# 5b.1 dose-response and 5b.2 Show HN control
# ------------------------------------------------------------------------------------------

def run_dose_and_show():
    rng = np.random.default_rng(SEED + int(Q["seed_offset"]))
    hn = load("hn", None, ["body_status"])
    dose, show, store = [], [], {}
    for comm in ["ask_hn", "link", "show_hn"]:
        par, rep, n_all = roots(comm, hn)
        Rk = cells_k(par, rep, K=3)
        R1 = S5.cells(par, rep)
        R1 = {"i": R1["i"], "bin": R1["bin"], "state": R1["after"], "y": R1["y"], "ex": R1["ex"]}
        for wlab, w0 in [("per_author", par["w_pa"].to_numpy()), ("raw", np.ones(par.height))]:
            c3, s3, r3 = fit_states(Rk, w0, 3, rng, par.height)
            store[("dose", comm, wlab)] = (c3, r3)
            for j in range(3):
                lo, hi = pct(r3[:, j])
                dose.append({"community": comm, "weighting": wlab, "step": f"comment_{j + 1}", "beta": float(c3[j]),
                             "hazard_ratio": float(np.exp(c3[j])), "boot_ci_lo": lo, "boot_ci_hi": hi,
                             "se_cluster_thread": float(s3[j]), "roots_used": par.height, "roots_population": n_all,
                             "events": float(Rk["y"].sum()), "B": B})
                print(f"  [5b.1] {comm} {wlab} step {j + 1}: {c3[j]:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
            c1, s1, r1 = fit_states(R1, w0, 1, rng, par.height)
            store[("one", comm, wlab)] = (c1, r1)
            lo, hi = pct(r1[:, 0])
            show.append({"community": comm, "weighting": wlab, "beta_after": float(c1[0]), "hazard_ratio": float(np.exp(c1[0])),
                         "boot_ci_lo": lo, "boot_ci_hi": hi, "se_cluster_thread": float(s1[0]),
                         "roots_used": par.height, "roots_population": n_all, "B": B})
            print(f"  [5b.2] {comm} {wlab} after: {c1[0]:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
        del par, rep, Rk, R1
        gc.collect()
    for wlab in ["per_author", "raw"]:
        for a, b in [("ask_hn", "link"), ("ask_hn", "show_hn"), ("show_hn", "link")]:
            ca, ra = store[("dose", a, wlab)]
            cb, rb = store[("dose", b, wlab)]
            for j in range(3):
                lo, hi = pct(ra[:, j] - rb[:, j])
                dose.append({"community": f"{a} minus {b}", "weighting": wlab, "step": f"comment_{j + 1}",
                             "beta": float(ca[j] - cb[j]), "boot_ci_lo": lo, "boot_ci_hi": hi, "B": B})
            # is the step-1 contrast more negative than steps 2 and 3?
            for j in (1, 2):
                d0 = (ca[0] - cb[0]) - (ca[j] - cb[j])
                lo, hi = pct((ra[:, 0] - rb[:, 0]) - (ra[:, j] - rb[:, j]))
                dose.append({"community": f"{a} minus {b}", "weighting": wlab,
                             "step": f"contrast step1 minus step{j + 1}", "beta": float(d0),
                             "boot_ci_lo": lo, "boot_ci_hi": hi, "B": B})
            c1a, r1a = store[("one", a, wlab)]
            c1b, r1b = store[("one", b, wlab)]
            lo, hi = pct(r1a[:, 0] - r1b[:, 0])
            show.append({"community": f"{a} minus {b}", "weighting": wlab, "beta_after": float(c1a[0] - c1b[0]),
                         "hazard_ratio": float(np.exp(c1a[0] - c1b[0])), "boot_ci_lo": lo, "boot_ci_hi": hi, "B": B})
            print(f"  [5b] {a}-{b} {wlab}: after {c1a[0] - c1b[0]:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    pl.DataFrame(dose, infer_schema_length=None).write_csv(OUT / "step5b_dose.csv")
    pl.DataFrame(show, infer_schema_length=None).write_csv(OUT / "step5b_show.csv")


# ------------------------------------------------------------------------------------------
# 5b.3 synthetic placebo
# ------------------------------------------------------------------------------------------

PL = Q["placebo"]
MED, SIG = 7200.0, 1.5
TARGET = {"ask_hn": (6.424, 0.736), "link": (1.755, 0.236)}


def G(t):
    return norm.cdf((np.log(np.maximum(t, 1e-9)) - np.log(MED)) / SIG)


def Ginv(u):
    return np.exp(np.log(MED) + SIG * norm.ppf(u))


def draw_spikes(n, pi, rng):
    has = rng.random(n) < pi
    T = np.exp(rng.uniform(np.log(300.0), np.log(43200.0), n))
    return has, T


def lam_total(has, T, M):
    D = float(PL["spike"]["duration_D_s"])
    extra = (M - 1) * (G(np.minimum(T + D, H)) - G(np.minimum(T, H))) * has
    return G(H) + extra, extra


def calibrate(pi, M, target, rng):
    has, T = draw_spikes(200_000, pi, rng)
    lam, _ = lam_total(has, T, M)
    mean, share = target
    A = mean / lam.mean()

    def f(logk):
        k = np.exp(logk)
        return 1 - np.mean((1 + A * lam / k) ** (-k)) - share
    logk = brentq(f, np.log(1e-4), np.log(1e4))
    return A, float(np.exp(logk))


def simulate(n, A, k, pi, M, rng):
    D = float(PL["spike"]["duration_D_s"])
    has, T = draw_spikes(n, pi, rng)
    lam, extra = lam_total(has, T, M)
    a = rng.gamma(k, 1.0 / k, n)
    cnt = rng.poisson(A * a * lam)
    tid = np.repeat(np.arange(n), cnt)
    from_spike = rng.random(len(tid)) < (extra / lam)[tid]
    u = rng.random(len(tid))
    lo = np.where(from_spike, G(np.minimum(T, H))[tid], 0.0)
    hi = np.where(from_spike, G(np.minimum(T + D, H))[tid], G(H))
    t = Ginv(lo + u * (hi - lo))
    ids = np.array([f"s{i}" for i in range(n)])
    first = np.full(n, np.inf)
    np.minimum.at(first, tid, t)
    par = pl.DataFrame({"id": ids, "c": np.full(n, H), "t": np.where(np.isfinite(first), first, H),
                        "d": np.isfinite(first)})
    rep = pl.DataFrame({"parent": ids[tid], "delay": t})
    return par, rep


def run_placebo():
    rng = np.random.default_rng(SEED + int(Q["seed_offset"]) + 1)
    rows = []
    n_sim = int(PL["simulations_per_setting"])
    for pi in PL["spike"]["probability_pi"]:
        for M in PL["spike"]["multiplier_M"]:
            cal = {p: calibrate(pi, M, TARGET[p], rng) for p in TARGET}
            for s in range(n_sim):
                betas = {}
                for p, n in PL["threads"].items():
                    A, k = cal[p]
                    par, rep = simulate(int(n), A, k, pi, M, rng)
                    R = S5.cells(par, rep)
                    X = np.column_stack([R["after"].astype(float)] + [(R["bin"] == b).astype(float) for b in range(1, NB)])
                    prep = FEPoissonBoot(R["i"], R["y"], X, np.log(R["ex"]), R["i"])
                    betas[p] = float(prep.fit()[0])
                    rows.append({"pi": pi, "M": M, "sim": s, "population": p, "A": A, "k": k, "beta_after": betas[p],
                                 "mean_comments": float(rep.height / par.height), "share_with_comment": float(par["d"].mean())})
                rows.append({"pi": pi, "M": M, "sim": s, "population": "ask_hn minus link",
                             "beta_after": betas["ask_hn"] - betas["link"]})
            x = pl.DataFrame(rows, infer_schema_length=None).filter((pl.col("pi") == pi) & (pl.col("M") == M))
            c = x.filter(pl.col("population") == "ask_hn minus link")["beta_after"].to_numpy()
            print(f"  [5b.3] pi={pi} M={M}: contrast mean {c.mean():+.3f} range [{np.percentile(c, 2.5):+.3f},"
                  f"{np.percentile(c, 97.5):+.3f}] (A,k ask {cal['ask_hn']}, link {cal['link']})", flush=True)
            pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "step5b_placebo.csv")
    d = pl.DataFrame(rows, infer_schema_length=None)
    summ = d.group_by("pi", "M", "population").agg(
        mean=pl.col("beta_after").mean(), p2_5=pl.col("beta_after").quantile(0.025, "linear"),
        p97_5=pl.col("beta_after").quantile(0.975, "linear"), mean_comments=pl.col("mean_comments").mean(),
        share_with_comment=pl.col("share_with_comment").mean(), n_sim=pl.len()).sort("pi", "M", "population")
    summ.write_csv(OUT / "step5b_placebo_summary.csv")
    print(summ)


if __name__ == "__main__":
    print(f"seed {SEED} + {Q['seed_offset']} (+1 placebo), B={B}", flush=True)
    which = sys.argv[1:] or ["placebo", "dose"]
    for w in which:
        {"dose": run_dose_and_show, "show": run_dose_and_show, "placebo": run_placebo}[w]()
