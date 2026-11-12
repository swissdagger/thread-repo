"""Step 6b — Moltbook late-reply checks (calibration split only). Pre-registered in docs/results_register.md
(R26, R27) and partitions.json -> step6b, committed before this script was run.

Usage: python scripts/step6b_agents_tail.py [sweepers] [archive_authors]

  sweepers         6b.1: crawl root all-reply intensity with and without replies by Stromfee and FinallyOffline
                   (audit check 7); bump ratio R at 1-2 days, thread bootstrap.
  archive_authors  6b.2: who writes the archive's late (> 3 d) replies; author concentration vs early replies;
                   authors in spike bins.
Outputs: outputs/moltbook_crawl/step6b_sweepers*.csv, figures/moltbook_crawl/step6b_sweepers.png,
         outputs/moltbook_archive/step6b_late_authors*.csv
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
from forum_model.units import build_units, moltbook_archive_population, moltbook_population, refirst  # noqa: E402
import step6_tail as S6  # noqa: E402

P = load_partitions()
Q = P["step6b"]
SEED = P["seed"]
B = 200
DAY = S6.DAY
EDGES, MID, K = S6.EDGES, S6.MID, S6.K
SWEEPERS = Q["sweepers"]["author_names"]
OUT_C = ROOT / "outputs" / "moltbook_crawl"
OUT_A = ROOT / "outputs" / "moltbook_archive"
FIG_C = ROOT / "figures" / "moltbook_crawl"
for d in (OUT_C, OUT_A, FIG_C):
    d.mkdir(parents=True, exist_ok=True)
pct = S6.pct
BUMP = (MID >= 1 * DAY) & (MID <= 2 * DAY)
NEIGH = ((MID >= 0.25 * DAY) & (MID <= 0.9 * DAY)) | ((MID >= 2.2 * DAY) & (MID <= 6 * DAY))


def loglin(ev, ex, sel):
    """Poisson ML of log rate = a + b log t on selected bins; returns (a, b) on centred log t, and the centre."""
    x = np.log(MID[sel])
    xc = x.mean()
    x0 = x - xc
    e, E = ev[sel], ex[sel]
    a, b = np.log(max(e.sum(), 1e-9) / E.sum()), 0.0
    for _ in range(100):
        mu = E * np.exp(a + b * x0)
        g = np.array([(e - mu).sum(), ((e - mu) * x0).sum()])
        Hm = np.array([[mu.sum(), (mu * x0).sum()], [(mu * x0).sum(), (mu * x0 ** 2).sum()]])
        st = np.linalg.solve(Hm, g)
        a, b = a + st[0], b + st[1]
        if np.abs(st).max() < 1e-10:
            break
    return a, b, xc


def bump_ratio(ev, ex, ok):
    sel_f, sel_b = NEIGH & ok & (ex > 0), BUMP & ok & (ex > 0)
    if sel_f.sum() < 3 or sel_b.sum() < 1:
        return np.nan
    a, b, xc = loglin(ev, ex, sel_f)
    exp_b = (ex[sel_b] * np.exp(a + b * (np.log(MID[sel_b]) - xc))).sum()
    return float(ev[sel_b].sum() / exp_b)


# ------------------------------------------------------------------------------------------
# 6b.1
# ------------------------------------------------------------------------------------------

def run_sweepers():
    rng = np.random.default_rng(SEED + int(Q["seed_offset"]))
    it, f = moltbook_population("complete", ["author_name"])
    names = it.select("author", "author_name").unique("author")
    U = build_units(it, f["root_live"], "moltbook_crawl")
    del it
    gc.collect()
    rep = U["replies"].join(names.rename({"author": "kid_author", "author_name": "kid_name"}), on="kid_author", how="left")
    is_sw = pl.col("kid_name").is_in(SWEEPERS).fill_null(False)
    share_sw = float(rep.select(is_sw.mean()).item())
    in_bump = rep.filter((pl.col("delay") >= EDGES[np.nonzero(BUMP)[0][0]]) & (pl.col("delay") < EDGES[np.nonzero(BUMP)[0][-1] + 1]))
    share_sw_bump = float(in_bump.select(is_sw.mean()).item()) if in_bump.height else np.nan
    rep_ex = rep.filter(~is_sw)
    versions = {"all replies": U, "sweepers excluded": {"parents": refirst(U["parents"], rep_ex), "replies": rep_ex}}
    tabs = {}
    for v, UU in versions.items():
        t, at_risk, T, npar = S6.thread_tables(UU)
        tabs[v] = (t, at_risk)
    ok = tabs["all replies"][1] >= S6.MIN_AT_RISK
    rows, bins_rows = [], []
    for w in ["pa", "raw"]:
        point = {v: bump_ratio(tabs[v][0][f"ev_{w}"].sum(0).astype(float), tabs[v][0][f"ex_{w}"].sum(0).astype(float), ok)
                 for v in versions}
        reps = {v: [] for v in versions}
        for _ in range(B):
            wv = rng.multinomial(T, np.full(T, 1.0 / T)).astype(np.float32)
            for v in versions:
                ev = (tabs[v][0][f"ev_{w}"].T @ wv).astype(float)
                ex = (tabs[v][0][f"ex_{w}"].T @ wv).astype(float)
                reps[v].append(bump_ratio(ev, ex, ok))
        for v in versions:
            lo, hi = pct(reps[v])
            rows.append({"weighting": w, "replies": v, "bump_ratio": point[v], "ci_lo": lo, "ci_hi": hi,
                         "share_root_replies_by_sweepers": share_sw, "share_bump_window_replies_by_sweepers": share_sw_bump,
                         "roots": npar, "threads": T, "B": B})
            print(f"  [6b.1] {w} {v}: R = {point[v]:.2f} [{lo:.2f},{hi:.2f}]", flush=True)
            ev = tabs[v][0][f"ev_{w}"].sum(0).astype(float)
            ex = tabs[v][0][f"ex_{w}"].sum(0).astype(float)
            for k in range(K):
                if ok[k] and ex[k] > 0:
                    bins_rows.append({"weighting": w, "replies": v, "mid_s": MID[k], "intensity_per_day": ev[k] / ex[k] * DAY,
                                      "events": ev[k]})
        d = np.array(reps["all replies"]) - np.array(reps["sweepers excluded"])
        lo, hi = pct(d)
        rows.append({"weighting": w, "replies": "difference (all - excluded)",
                     "bump_ratio": point["all replies"] - point["sweepers excluded"], "ci_lo": lo, "ci_hi": hi, "B": B})
    print(f"  [6b.1] sweepers write {share_sw:.3%} of root replies, {share_sw_bump:.3%} in the 1-2 d window", flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT_C / "step6b_sweepers.csv")
    pl.DataFrame(bins_rows).write_csv(OUT_C / "step6b_sweepers_bins.csv")
    figure_sweepers()


def figure_sweepers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    t = pl.read_csv(OUT_C / "step6b_sweepers_bins.csv").filter(pl.col("weighting") == "pa")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for v, col, ls in [("all replies", "#1baf7a", "-"), ("sweepers excluded", "#6250d6", "--")]:
        x = t.filter((pl.col("replies") == v) & (pl.col("intensity_per_day") > 0))
        ax.plot(x["mid_s"].to_numpy() / DAY, x["intensity_per_day"].to_numpy(), ls, color=col, lw=2, label=v)
    ax.axvspan(1, 2, color=GRID, alpha=0.6, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Moltbook crawl roots: replies per post per day (per-author)", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("post age (days); shaded = 1–2 d bump window", color=INK2, fontsize=9)
    ax.set_ylabel("all-reply intensity", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, title="Stromfee and FinallyOffline", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_C / "step6b_sweepers.png", facecolor=SURF)


# ------------------------------------------------------------------------------------------
# 6b.2
# ------------------------------------------------------------------------------------------

def concentration(counts: np.ndarray) -> dict:
    tot = counts.sum()
    if tot <= 0:
        return {"replies": 0.0, "distinct_authors": 0, "top1": np.nan, "top5": np.nan, "top10": np.nan, "hhi": np.nan}
    s = np.sort(counts)[::-1] / tot
    return {"replies": float(tot), "distinct_authors": int((counts > 0).sum()), "top1": float(s[:1].sum()),
            "top5": float(s[:5].sum()), "top10": float(s[:10].sum()), "hhi": float((s ** 2).sum())}


def run_archive_authors():
    import duckdb
    rng = np.random.default_rng(SEED + int(Q["seed_offset"]) + 1)
    a = P["moltbook"]["archive_crosscheck"]
    it, f = moltbook_archive_population(*a["posts_created_utc"], a["observation_end_utc"])
    cf = str(ROOT / "data" / "raw" / "moltbook" / "data" / "comments" / "*.parquet")
    c = duckdb.connect()
    names = c.sql(f"select agent_id author, any_value(agent_name) name from read_parquet('{cf}', union_by_name=true) "
                  f"group by agent_id").pl()
    c.close()
    rows, top_rows, spike_rows = [], [], []
    for unit in ["root_live", "nonroot"]:
        U = build_units(it, f[unit], "moltbook_archive")
        par = U["parents"].select("id", "thread_id")
        rep = U["replies"].join(par, left_on="parent", right_on="id", how="left")
        _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
        tmap = dict(zip(par["thread_id"].to_list(), tix.tolist()))
        T = int(tix.max()) + 1
        rt = np.array([tmap[x] for x in rep["thread_id"].to_list()], dtype=np.int64)
        auth = rep["kid_author"].fill_null("<unknown>").to_numpy()
        uniq, ai = np.unique(auth, return_inverse=True)
        dl = rep["delay"].to_numpy()
        windows = {"late (3-30 d)": (dl > 3 * DAY) & (dl <= 30 * DAY), "early (<= 3 d)": dl <= 3 * DAY}
        for wname, m in windows.items():
            cnt = np.bincount(ai[m], minlength=len(uniq)).astype(float)
            pt = concentration(cnt)
            R = {k: [] for k in ["top1", "top5", "top10", "hhi"]}
            for _ in range(B):
                wv = rng.multinomial(T, np.full(T, 1.0 / T)).astype(float)
                cr = np.bincount(ai[m], weights=wv[rt[m]], minlength=len(uniq))
                cc = concentration(cr)
                for k in R:
                    R[k].append(cc[k])
            row = {"unit": unit, "window": wname, **pt, "B": B}
            for k in R:
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = pct(R[k])
            rows.append(row)
            print(f"  [6b.2] {unit} {wname}: {pt['replies']:.0f} replies, {pt['distinct_authors']} authors, "
                  f"top1 {pt['top1']:.3f}, top5 {pt['top5']:.3f} [{row['top5_ci_lo']:.3f},{row['top5_ci_hi']:.3f}], "
                  f"hhi {pt['hhi']:.3f}", flush=True)
        # top-10 late authors, with their early-reply counts
        late, early = windows["late (3-30 d)"], windows["early (<= 3 d)"]
        cl = np.bincount(ai[late], minlength=len(uniq))
        ce = np.bincount(ai[early], minlength=len(uniq))
        th_late = np.array([len(np.unique(rt[late & (ai == j)])) for j in np.argsort(cl)[::-1][:10]])
        for r_, j in enumerate(np.argsort(cl)[::-1][:10]):
            top_rows.append({"unit": unit, "rank": r_ + 1, "author_id": uniq[j], "late_replies": int(cl[j]),
                             "late_share": float(cl[j] / max(cl.sum(), 1)), "threads_with_late_replies": int(th_late[r_]),
                             "early_replies": int(ce[j]), "late_fraction_of_own_replies": float(cl[j] / max(cl[j] + ce[j], 1))})
        # spike bins: tail bins whose events exceed 2x the fitted 3-30 d power law
        tabs, at_risk, _, _ = S6.thread_tables(U)
        ev, ex = tabs["ev_raw"].sum(0).astype(float), tabs["ex_raw"].sum(0).astype(float)
        sel = S6.TAIL & (ex > 0)
        a0, b0, xc = loglin(ev, ex, sel)
        fit = ex * np.exp(a0 + b0 * (np.log(MID) - xc))
        for k in np.nonzero(sel & (ev > 2 * fit))[0]:
            m = (dl >= EDGES[k]) & (dl < EDGES[k + 1])
            ck = np.bincount(ai[m], minlength=len(uniq))
            order = np.argsort(ck)[::-1][:5]
            spike_rows.append({"unit": unit, "bin_lo_days": EDGES[k] / DAY, "bin_hi_days": EDGES[k + 1] / DAY,
                               "events": float(ev[k]), "fitted": float(fit[k]), "distinct_authors": int((ck > 0).sum()),
                               "threads": int(len(np.unique(rt[m]))),
                               **{f"author{i + 1}": uniq[j] for i, j in enumerate(order) if ck[j] > 0},
                               **{f"author{i + 1}_count": int(ck[j]) for i, j in enumerate(order) if ck[j] > 0}})
        del U, rep, tabs
        gc.collect()
    top = pl.DataFrame(top_rows).join(names.rename({"author": "author_id"}), on="author_id", how="left")
    top.write_csv(OUT_A / "step6b_late_authors_top10.csv")
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT_A / "step6b_late_authors.csv")
    sp = pl.DataFrame(spike_rows, infer_schema_length=None)
    for i in range(1, 6):
        col = f"author{i}"
        if col in sp.columns:
            sp = sp.join(names.rename({"author": col, "name": f"{col}_name"}), on=col, how="left")
    sp.write_csv(OUT_A / "step6b_spike_bins.csv")
    print(top)
    print(sp)


if __name__ == "__main__":
    print(f"seed {SEED} + {Q['seed_offset']}, B={B}", flush=True)
    which = sys.argv[1:] or ["sweepers", "archive_authors"]
    for w in which:
        {"sweepers": run_sweepers, "archive_authors": run_archive_authors}[w]()
