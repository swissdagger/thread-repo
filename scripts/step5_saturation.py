"""Step 5 — Saturation without scores (calibration split only).

In Ask HN threads the first top-level comment is the candidate answer. The reply hazard to the root
(top-level comments) before vs after its arrival is compared with HN link posts as the control.
Definitions: docs/definitions.md -> Saturation. Validation test pre-stated in docs/results_register.md (R3).

Usage: python scripts/step5_saturation.py
Outputs: outputs/hn/step5_saturation.csv, outputs/hn/step5_saturation_bins.csv,
         figures/hn/step5_saturation.png
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
from forum_model.fe_poisson import FEPoissonBoot  # noqa: E402
from forum_model.units import build_units, hn_population, load  # noqa: E402
import step4b_extensions as S4b  # noqa: E402

P = load_partitions()
SEED = P["seed"]
SEED_OFFSET = 51
B = 200
H = float(P["step5"]["horizon_s"])
BINS = np.array(P["step5"]["root_age_bins_s"], dtype=float)
CAP = int(P["step5"]["cap_roots"])
OUT = ROOT / "outputs" / "hn"
FIG = ROOT / "figures" / "hn"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def pct(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def cells(par: pl.DataFrame, rep: pl.DataFrame) -> dict:
    """Rows (root, age bin, state). before = [0, tb], tb = max(t_a, 1 s), ends with the candidate answer;
    after = (tb, cH]. Roots with no top-level comment by cH contribute before-exposure only."""
    n = par.height
    cH = np.minimum(par["c"].to_numpy(), H)
    d = par["d"].to_numpy() & (par["t"].to_numpy() <= cH)
    tb = np.where(d, np.maximum(par["t"].to_numpy(), 1.0), cH)
    pid = {k: i for i, k in enumerate(par["id"].to_list())}
    rp = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
    dl = rep["delay"].to_numpy()
    # drop each root's first reply (it is the before-event) and replies beyond the horizon
    order = np.lexsort((dl, rp))
    rp, dl = rp[order], dl[order]
    first = np.r_[True, rp[1:] != rp[:-1]]
    keep = ~first & (dl <= cH[rp])
    rp_a, dl_a = rp[keep], np.maximum(dl[keep], tb[rp[keep]])
    nb = len(BINS) - 1
    after_ev = np.bincount(rp_a * nb + np.clip(np.searchsorted(BINS, dl_a, side="right") - 1, 0, nb - 1),
                           minlength=n * nb).reshape(n, nb)
    before_bin = np.clip(np.searchsorted(BINS, tb, side="left") - 1, 0, nb - 1)
    rows = {"i": [], "bin": [], "after": [], "y": [], "ex": []}
    for b in range(nb):
        lo, hi = BINS[b], BINS[b + 1]
        exb = np.clip(np.minimum(tb, hi) - lo, 0, None)
        exa = np.clip(np.minimum(cH, hi) - np.maximum(tb, lo), 0, None) * d
        for st, ex, y in [(0, exb, (d & (before_bin == b)).astype(float)), (1, exa, after_ev[:, b].astype(float))]:
            m = ex > 0
            idx = np.nonzero(m)[0]
            rows["i"].append(idx)
            rows["bin"].append(np.full(len(idx), b))
            rows["after"].append(np.full(len(idx), st))
            rows["y"].append(y[idx])
            rows["ex"].append(ex[idx])
    R = {k: np.concatenate(v) for k, v in rows.items()}
    return R


def run():
    rng = np.random.default_rng(SEED + SEED_OFFSET)
    hn = load("hn", None, ["body_status"])
    results, bins_out, store = [], [], {}
    for comm in ["ask_hn", "link"]:
        it, f = hn_population(comm, hn_items=hn)
        U = build_units(it, f["root_live"], f"hn:{comm}")
        del it
        gc.collect()
        par = U["parents"]
        n_all = par.height
        if par.height > CAP:
            par = par.with_columns(u=pl.col("thread_id").map_elements(S4b.thread_hash, return_dtype=pl.Float64)) \
                .sort("u").head(CAP).drop("u")
        rep = U["replies"].join(par.select(pl.col("id").alias("parent")), on="parent", how="semi")
        R = cells(par, rep)
        nb = len(BINS) - 1
        g = R["i"]  # thread (= root) fixed effects
        X = np.column_stack([R["after"].astype(float)] + [(R["bin"] == b).astype(float) for b in range(1, nb)])
        prep = FEPoissonBoot(g, R["y"], X, np.log(R["ex"]), g)
        cH = np.minimum(par["c"].to_numpy(), H)
        d = par["d"].to_numpy() & (par["t"].to_numpy() <= cH)
        desc = {"roots_population": n_all, "roots_used": par.height, "share_with_top_level_comment_3d": float(d.mean()),
                "median_first_top_level_min": float(np.median(par["t"].to_numpy()[d]) / 60),
                "top_level_comments_3d": float(R["y"].sum())}
        for wlab, w0 in [("per_author", par["w_pa"].to_numpy()), ("raw", np.ones(par.height))]:
            coef, se = prep.fit(w0, se=True)
            reps = np.array([prep.fit(w0 * rng.multinomial(par.height, np.full(par.height, 1.0 / par.height)),
                                      b0=coef)[0] for _ in range(B)])
            lo, hi = pct(reps)
            store[(comm, wlab)] = (coef[0], reps)
            results.append({"community": comm, "weighting": wlab, "beta_after": float(coef[0]),
                            "hazard_ratio_after": float(np.exp(coef[0])), "boot_ci_lo": lo, "boot_ci_hi": hi,
                            "se_cluster_thread": float(se[0]), "B": B, **desc})
            print(f"  [5] {comm} {wlab}: beta_after {coef[0]:+.3f} [{lo:+.3f},{hi:+.3f}] (HR {np.exp(coef[0]):.2f})", flush=True)
            # descriptive hazard by root-age bin and state (weighted events / exposure), bootstrap CIs
            w_rows = w0[R["i"]]
            for st in (0, 1):
                for b in range(nb):
                    m = (R["after"] == st) & (R["bin"] == b)
                    ev, ex = (w_rows[m] * R["y"][m]).sum(), (w_rows[m] * R["ex"][m]).sum()
                    bins_out.append({"community": comm, "weighting": wlab, "state": "after" if st else "before",
                                     "bin_lo_s": BINS[b], "bin_hi_s": BINS[b + 1], "events": float(ev),
                                     "exposure_s": float(ex), "hazard_per_hour": float(ev / ex * 3600) if ex > 0 else None,
                                     "roots_at_risk": int(m.sum())})
        del U, par, rep, R, prep
        gc.collect()
    for wlab in ["per_author", "raw"]:
        a0, ar = store[("ask_hn", wlab)]
        l0, lr = store[("link", wlab)]
        lo, hi = pct(ar - lr)
        results.append({"community": "ask_hn minus link (saturation contrast)", "weighting": wlab,
                        "beta_after": float(a0 - l0), "hazard_ratio_after": float(np.exp(a0 - l0)),
                        "boot_ci_lo": lo, "boot_ci_hi": hi, "B": B})
        print(f"  [5] ask_hn - link {wlab}: {a0 - l0:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    pl.DataFrame(results, infer_schema_length=None).write_csv(OUT / "step5_saturation.csv")
    pl.DataFrame(bins_out).write_csv(OUT / "step5_saturation_bins.csv")
    figure()


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    C = {"ask_hn": "#1baf7a", "link": "#eda100"}
    LAB = {"ask_hn": "Ask HN", "link": "HN link"}
    t = pl.read_csv(OUT / "step5_saturation_bins.csv").filter((pl.col("weighting") == "per_author")
                                                              & (pl.col("roots_at_risk") >= 50))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, comm in zip(axes, ["ask_hn", "link"]):
        ax.set_facecolor("#fcfcfb")
        for st, ls in [("before", "--"), ("after", "-")]:
            x = t.filter((pl.col("community") == comm) & (pl.col("state") == st) & pl.col("hazard_per_hour").is_not_null())
            mid = np.sqrt(np.maximum(x["bin_lo_s"].to_numpy(), 60) * x["bin_hi_s"].to_numpy()) / 3600
            ax.plot(mid, x["hazard_per_hour"].to_numpy(), ls, marker="o", ms=4, lw=2, color=C[comm],
                    label=f"{st} the first top-level comment")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{LAB[comm]}: top-level comment rate on the root", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("root age (hours)", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    axes[0].set_ylabel("top-level comments per root per hour (per-author)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "step5_saturation.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    print(f"seed {SEED} + {SEED_OFFSET}, B={B}, horizon {H}s, cap {CAP} roots", flush=True)
    run()
