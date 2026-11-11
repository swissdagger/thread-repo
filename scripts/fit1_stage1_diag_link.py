"""Stage 1 diagnostic (after the pre-registered runs, not a pre-registered test): why the RE arm fails on
HN link. On replicate 0: (1) marginal log-likelihood at the truth vs at the reported estimate (optimiser
failure if the truth is higher); (2) a refit started at the truth; (3) the profile over b with the other
parameters re-maximised at b in {0.5, 1, 2, 3.5, 5}."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit1_stage1 as F  # noqa: E402
from forum_model import fit1_estimator as EST  # noqa: E402
from forum_model.fit1_streams import extract  # noqa: E402

items, diag, truth, rng = F.dataset("link", 0)
S = extract(items)
D = EST.build(S)
M = EST.REModel(D)
names = M.names + ["log_r", "b"]
th_true = np.array([truth[n] for n in names])
rec = pl.read_parquet(F.OUT / "stage1_recovery_link.parquet").filter((pl.col("model") == "RE") & (pl.col("rep") == 0)
                                                                     & (pl.col("stage") == "joint"))
est = dict(zip(rec["param"].to_list(), rec["estimate"].to_list()))
th_est = np.array([est[n] for n in names])
rows = [{"point": "truth", "loglik": M.loglik(th_true)}, {"point": "reported estimate", "loglik": M.loglik(th_est)}]
th_t = M.fit(th_true, iters=100, warm=False)
rows.append({"point": "refit from truth", "loglik": M.loglik(th_t), "iters": M.n_iter,
             **{f"est_{k}": float(th_t[names.index(k)]) for k in ["b", "log_r", "gamma", "alpha_R", "aR1"]}})
print(pl.DataFrame(rows), flush=True)
prof = []
for bval in [0.5, 1.0, 2.0, 3.5, 5.0]:
    st = th_t.copy()
    st[M.p + 1] = bval
    # re-maximise everything except b (b held by restoring it after each Newton step)
    th = st.copy()
    for _ in range(40):
        g, Hm = M.grad_hess(th)
        k = M.p + 1
        idx = [i for i in range(len(th)) if i != k]
        step = np.zeros(len(th))
        step[idx] = -np.linalg.solve(Hm[np.ix_(idx, idx)], g[idx])
        step /= max(1.0, np.abs(step).max() / 0.5)
        lam, ll0 = 1.0, M.loglik(th)
        while lam > 1e-4 and M.loglik(th + lam * step) < ll0:
            lam /= 2
        th = th + lam * step
        if np.abs(lam * step).max() < 1e-7:
            break
    prof.append({"b": bval, "profile_loglik": M.loglik(th), "gamma": float(th[names.index("gamma")]),
                 "log_r": float(th[names.index("log_r")])})
    print(prof[-1], flush=True)
out = pl.concat([pl.DataFrame(rows), pl.DataFrame(prof)], how="diagonal_relaxed")
out.write_csv(F.OUT / "stage1_diag_link.csv")
