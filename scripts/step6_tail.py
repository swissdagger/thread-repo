"""Step 6 — External-exposure tail (calibration split only).

All-reply intensity (and first-reply hazard) per parent out to 30 days, for comments and roots, on
Reddit (r/philosophy, r/buildapc) and Moltbook (crawl to its observable horizon; archive to 30 days).
Definitions: docs/definitions.md -> Late tail. Validation test pre-stated in docs/results_register.md (R4).

Usage: python scripts/step6_tail.py [compute] [figure]
Outputs: outputs/combined/step6_tail_bins.csv, step6_tail_summary.csv, figures/combined/step6_tail.png
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forum_model import ROOT, load_partitions  # noqa: E402
from forum_model.units import build_units, moltbook_archive_population, moltbook_population, reddit_population  # noqa: E402

P = load_partitions()
SEED = P["seed"]
SEED_OFFSET = 61
B = 200
DAY = 86400.0
H = float(P["step6"]["horizon_s"])
EDGES = np.r_[0.0, np.logspace(1, np.log10(H), 40)]
K = len(EDGES) - 1
MID = np.sqrt(np.maximum(EDGES[:-1], 5.0) * EDGES[1:])
TAIL = (MID >= P["step6"]["tail_fit_range_s"][0]) & (MID <= P["step6"]["tail_fit_range_s"][1])
FITR = EDGES[1:] <= P["step6"]["kernel_fit_range_s"][1]
MIN_AT_RISK = int(P["step6"]["min_parents_at_risk_per_bin"])
OUT = ROOT / "outputs" / "combined"
FIG = ROOT / "figures" / "combined"
FIG.mkdir(parents=True, exist_ok=True)


def pct(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def thread_tables(U: dict):
    """Per-thread x bin arrays: all-reply events, exposure; first-reply events, exposure; for raw and
    per-author weights. Returns dict of (T, K) float32 arrays + parents at risk per bin."""
    par, rep = U["parents"], U["replies"]
    cH = np.minimum(par["c"].to_numpy(), H)
    t1 = np.minimum(par["t"].to_numpy(), cH)
    d1 = par["d"].to_numpy() & (par["t"].to_numpy() <= cH)
    _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
    T = int(tix.max()) + 1
    wpa = par["w_pa"].to_numpy()
    pid = {k: i for i, k in enumerate(par["id"].to_list())}
    rp = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
    dl = rep["delay"].to_numpy()
    ok = dl <= cH[rp]
    rp, dl = rp[ok], dl[ok]
    rb = np.clip(np.searchsorted(EDGES, dl, side="right") - 1, 0, K - 1)
    fb = np.clip(np.searchsorted(EDGES, t1, side="right") - 1, 0, K - 1)
    out = {f"{k}_{w}": np.zeros((T, K), dtype=np.float32) for k in ["ev", "ex", "ev1", "ex1"] for w in ["raw", "pa"]}
    at_risk = np.zeros(K, dtype=np.int64)
    for k in range(K):
        lo, hi = EDGES[k], EDGES[k + 1]
        ex = np.clip(cH - lo, 0, hi - lo)
        ex1 = np.clip(t1 - lo, 0, hi - lo)
        at_risk[k] = int((ex > 0).sum())
        m = rb == k
        e1 = (d1 & (fb == k)).astype(float)
        for w, wv in [("raw", None), ("pa", wpa)]:
            ww = np.ones(len(cH)) if wv is None else wv
            out[f"ex_{w}"][:, k] = np.bincount(tix, weights=ww * ex, minlength=T)
            out[f"ex1_{w}"][:, k] = np.bincount(tix, weights=ww * ex1, minlength=T)
            out[f"ev1_{w}"][:, k] = np.bincount(tix, weights=ww * e1, minlength=T)
            out[f"ev_{w}"][:, k] = np.bincount(tix[rp[m]], weights=ww[rp[m]], minlength=T)
    return out, at_risk, T, par.height


def tail_slope(ev, ex, sel):
    """Poisson ML of log rate = a + b log t on the selected bins."""
    x = np.log(MID[sel])
    e, E = ev[sel], ex[sel]
    if (E > 0).sum() < 3 or e.sum() <= 0:
        return np.nan
    x0 = x - x.mean()
    a, b = np.log(e.sum() / E.sum()), 0.0
    for _ in range(100):
        mu = E * np.exp(a + b * x0)
        g = np.array([(e - mu).sum(), ((e - mu) * x0).sum()])
        Hm = np.array([[mu.sum(), (mu * x0).sum()], [(mu * x0).sum(), (mu * x0 ** 2).sum()]])
        try:
            st = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            return np.nan
        a, b = a + st[0], b + st[1]
        if np.abs(st).max() < 1e-10:
            break
    return float(b)


def ln_bin_mass(th):
    """Average log-normal density over each bin (per second), parameters (log mu, m, log s)."""
    m, s = th[1], np.exp(th[2])
    lo = np.log(np.maximum(EDGES[:-1], 1e-3))
    hi = np.log(EDGES[1:])
    Gm = norm.cdf((hi - m) / s) - norm.cdf((lo - m) / s)
    return np.exp(th[0]) * Gm / (EDGES[1:] - EDGES[:-1])


def fit_lognormal(ev, ex, th0=None):
    sel = FITR & (ex > 0)

    def nll(th):
        lam = ln_bin_mass(th)[sel] * ex[sel]
        return float((lam - ev[sel] * np.log(np.maximum(lam, 1e-300))).sum())
    if th0 is None:
        th0 = np.array([np.log(max(ev[sel].sum() / max(ex[sel].sum(), 1e-9) * DAY, 1e-6)), np.log(1800.0), np.log(2.0)])
    r = minimize(nll, th0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    return r.x


def summarise(ev, ex, ev1, ex1, th0=None):
    sel = TAIL & (ex > 0)
    b = tail_slope(ev, ex, sel)
    b1 = tail_slope(ev1, ex1, TAIL & (ex1 > 0))
    th = fit_lognormal(ev, ex, th0)
    pred = (ln_bin_mass(th) * ex)[sel].sum()
    exc = float(ev[sel].sum() / pred) if pred > 0 else np.nan
    share_tail = float(ev[sel].sum() / ev[ex > 0].sum()) if ev.sum() > 0 else np.nan
    return {"slope": b, "slope_first_reply": b1, "excess": exc, "share_replies_3_30d": share_tail,
            "ln_mu": float(np.exp(th[0])), "ln_median_min": float(np.exp(th[1]) / 60), "ln_sigma": float(np.exp(th[2]))}, th


def pops():
    for sub in ["philosophy", "buildapc"]:
        def ld(sub=sub):
            return reddit_population(sub)
        yield "reddit", sub, ld, f"reddit:{sub}"
    yield "moltbook_crawl", "moltbook_crawl", lambda: moltbook_population("complete"), "moltbook_crawl"
    a = P["moltbook"]["archive_crosscheck"]
    yield "moltbook_archive", "moltbook_archive", lambda: moltbook_archive_population(
        *a["posts_created_utc"], a["observation_end_utc"]), "moltbook_archive"


def compute():
    rng = np.random.default_rng(SEED + SEED_OFFSET)
    bins_rows, summ = [], []
    reps_store = {}
    for ds, comm, loader, name in pops():
        it, f = loader()
        for unit in ["nonroot", "root_live"]:
            U = build_units(it, f[unit], name)
            tabs, at_risk, T, npar = thread_tables(U)
            del U
            gc.collect()
            for w in ["pa", "raw"]:
                ev_t, ex_t, ev1_t, ex1_t = tabs[f"ev_{w}"], tabs[f"ex_{w}"], tabs[f"ev1_{w}"], tabs[f"ex1_{w}"]
                ev, ex, ev1, ex1 = (a.sum(0).astype(float) for a in (ev_t, ex_t, ev1_t, ex1_t))
                ok = at_risk >= MIN_AT_RISK
                ev, ex, ev1, ex1 = ev * ok, ex * ok, ev1 * ok, ex1 * ok
                s0, th0 = summarise(ev, ex, ev1, ex1)
                R = {"lam": [], "lam1": [], "slope": [], "slope_first_reply": [], "excess": []}
                for _ in range(B):
                    wv = rng.multinomial(T, np.full(T, 1.0 / T)).astype(np.float32)
                    evr, exr = (ev_t.T @ wv).astype(float) * ok, (ex_t.T @ wv).astype(float) * ok
                    ev1r, ex1r = (ev1_t.T @ wv).astype(float) * ok, (ex1_t.T @ wv).astype(float) * ok
                    sr, _ = summarise(evr, exr, ev1r, ex1r, th0)
                    for k in ["slope", "slope_first_reply", "excess"]:
                        R[k].append(sr[k])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        R["lam"].append(evr / exr)
                        R["lam1"].append(ev1r / ex1r)
                reps_store[(comm, unit, w)] = np.array(R["slope"])
                lam_lo, lam_hi = np.nanpercentile(np.array(R["lam"]), [2.5, 97.5], axis=0)
                l1_lo, l1_hi = np.nanpercentile(np.array(R["lam1"]), [2.5, 97.5], axis=0)
                pred = ln_bin_mass(th0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    for k in range(K):
                        if not ok[k] or ex[k] <= 0:
                            continue
                        bins_rows.append({"dataset": ds, "community": comm, "unit": unit, "weighting": w,
                                          "bin_lo_s": EDGES[k], "bin_hi_s": EDGES[k + 1], "mid_s": MID[k],
                                          "parents_at_risk": int(at_risk[k]),
                                          "intensity_per_day": ev[k] / ex[k] * DAY, "ci_lo": lam_lo[k] * DAY, "ci_hi": lam_hi[k] * DAY,
                                          "first_reply_hazard_per_day": ev1[k] / ex1[k] * DAY if ex1[k] > 0 else None,
                                          "fr_ci_lo": l1_lo[k] * DAY, "fr_ci_hi": l1_hi[k] * DAY,
                                          "lognormal_fit_per_day": pred[k] * DAY, "events": ev[k]})
                horizon = float(EDGES[1:][ok].max() / DAY) if ok.any() else 0.0
                row = {"dataset": ds, "community": comm, "unit": unit, "weighting": w, "parents": npar, "threads": T,
                       "observable_horizon_days": horizon, **s0, "B": B}
                for k in ["slope", "slope_first_reply", "excess"]:
                    row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = pct(R[k])
                summ.append(row)
                print(f"  [6] {comm} {unit} {w}: horizon {horizon:.1f} d, slope {s0['slope']:+.2f} "
                      f"[{row['slope_ci_lo']:+.2f},{row['slope_ci_hi']:+.2f}], excess {s0['excess']:.2f} "
                      f"[{row['excess_ci_lo']:.2f},{row['excess_ci_hi']:.2f}], share 3-30d {s0['share_replies_3_30d']:.4f}",
                      flush=True)
            del tabs
            gc.collect()
            pl.DataFrame(bins_rows, infer_schema_length=None).write_csv(OUT / "step6_tail_bins.csv")
            pl.DataFrame(summ, infer_schema_length=None).write_csv(OUT / "step6_tail_summary.csv")
        del it
        gc.collect()
    # pre-stated contrast: Moltbook archive slope minus each Reddit slope (independent samples)
    diffs = []
    for unit in ["nonroot", "root_live"]:
        for w in ["pa", "raw"]:
            m = reps_store.get(("moltbook_archive", unit, w))
            for sub in ["philosophy", "buildapc"]:
                r = reps_store.get((sub, unit, w))
                if m is None or r is None:
                    continue
                m0 = [x for x in summ if x["community"] == "moltbook_archive" and x["unit"] == unit and x["weighting"] == w][0]["slope"]
                r0 = [x for x in summ if x["community"] == sub and x["unit"] == unit and x["weighting"] == w][0]["slope"]
                lo, hi = pct(m - r)
                diffs.append({"unit": unit, "weighting": w, "contrast": f"moltbook_archive - {sub}",
                              "slope_diff": m0 - r0, "ci_lo": lo, "ci_hi": hi, "B": B})
    pl.DataFrame(diffs).write_csv(OUT / "step6_tail_contrasts.csv")
    print(pl.DataFrame(diffs))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    C = {"philosophy": "#2a78d6", "buildapc": "#eb6834", "moltbook_crawl": "#1baf7a", "moltbook_archive": "#6250d6"}
    LAB = {"philosophy": "r/philosophy", "buildapc": "r/buildapc", "moltbook_crawl": "Moltbook crawl",
           "moltbook_archive": "Moltbook archive"}
    t = pl.read_csv(OUT / "step6_tail_bins.csv").filter(pl.col("weighting") == "pa")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, unit, ttl in [(axes[0], "nonroot", "Comments"), (axes[1], "root_live", "Roots (posts)")]:
        ax.set_facecolor(SURF)
        for comm in C:
            x = t.filter((pl.col("community") == comm) & (pl.col("unit") == unit) & (pl.col("intensity_per_day") > 0))
            if x.height == 0:
                continue
            mid = x["mid_s"].to_numpy() / DAY
            ax.fill_between(mid, x["ci_lo"].to_numpy(), x["ci_hi"].to_numpy(), color=C[comm], alpha=0.15, lw=0)
            ax.plot(mid, x["intensity_per_day"].to_numpy(), "-", color=C[comm], lw=2, label=LAB[comm])
            ax.plot(mid, x["lognormal_fit_per_day"].to_numpy(), ":", color=C[comm], lw=1.2)
        ax.axvspan(3, 30, color=GRID, alpha=0.5, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 / DAY, 30)
        ymin = t.filter((pl.col("unit") == unit) & (pl.col("intensity_per_day") > 0))["ci_lo"].min()
        ax.set_ylim(bottom=max(ymin / 3, 1e-7))
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
        ax.set_title(f"{ttl}: replies per parent per day, by parent age", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("parent age (days); shaded = 3–30 d tail window", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("all-reply intensity (per-author, 95% thread bootstrap)", color=INK2, fontsize=9)
    axes[1].text(0.99, 0.98, "dotted: log-normal fitted on 0–3 d, extrapolated", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "step6_tail.png", facecolor=SURF)


if __name__ == "__main__":
    print(f"seed {SEED} + {SEED_OFFSET}, B={B}", flush=True)
    which = sys.argv[1:] or ["compute", "figure"]
    for w in which:
        {"compute": compute, "figure": figure}[w]()
