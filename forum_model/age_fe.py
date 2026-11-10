"""(comment, thread-age bin) cell design for the Step 4b/4c age models.

Each parent comment is followed over thread-age bins up to a horizon; one row per (comment, bin) with
y = replies in the bin, offset = log exposure in the bin, covariates log1p(comment age in minutes) at the
cell midpoint, depth dummies (baseline: lowest depth present), OP comment, and optionally log arrival rank
active in the first-30-minute bins. Same construction as scripts/step4b_extensions.py::age_fe_one.
"""
from __future__ import annotations

import numpy as np
import polars as pl

POS_CAP = 500


def cells(par: pl.DataFrame, rep_sets: dict[str, pl.DataFrame], bins: np.ndarray, hmax: float,
          with_pos: bool = False) -> dict:
    s = np.maximum(par["time_since_root"].to_numpy(), 0.0)
    e = np.minimum(s + par["c"].to_numpy(), hmax)
    nb = len(bins) - 1
    pid = {k: i for i, k in enumerate(par["id"].to_list())}
    counts = {}
    for name, rep in rep_sets.items():
        rp = np.array([pid[x] for x in rep["parent"].to_list()], dtype=np.int64)
        rt = s[rp] + rep["delay"].to_numpy()
        ok = rt < e[rp]
        rb = np.clip(np.searchsorted(bins, rt[ok], side="right") - 1, 0, nb - 1)
        counts[name] = np.bincount(rp[ok] * nb + rb, minlength=len(s) * nb)
    _, tix = np.unique(par["thread_id"].to_numpy(), return_inverse=True)
    depth = par["depth"].to_numpy()
    op = par["is_root_author"].fill_null(False).to_numpy().astype(float)
    ys = {k: [] for k in counts}
    off, age, bn, ii = [], [], [], []
    for b in range(nb):
        lo, hi = bins[b], bins[b + 1]
        ov = np.minimum(e, hi) - np.maximum(s, lo)
        idx = np.nonzero(ov > 0)[0]
        mid = (np.maximum(s[idx], lo) + np.minimum(e[idx], hi)) / 2 - s[idx]
        for k in counts:
            ys[k].append(counts[k][idx * nb + b])
        off.append(np.log(ov[idx]))
        age.append(np.log1p(mid / 60))
        bn.append(np.full(len(idx), b))
        ii.append(idx)
    off, age, bn, ii = map(np.concatenate, (off, age, bn, ii))
    ys = {k: np.concatenate(v).astype(float) for k, v in ys.items()}
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
    if with_pos:
        pos = np.log(np.clip(par["pos"].to_numpy(), 1, POS_CAP))
        early = (bins[bn + 1] <= 1800).astype(float)
        cols.append(pos[ii] * early)
        names.append("log_position_x_first_30min")
    return {"y": ys, "off": off, "X": np.column_stack(cols), "names": names, "thr": tix[ii],
            "g": tix[ii] * nb + bn, "n_threads": int(tix.max()) + 1, "parents": par.height}
