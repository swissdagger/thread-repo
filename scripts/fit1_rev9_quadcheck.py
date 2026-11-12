"""fit1 Rev-9 development check (truth only, no recovery data). Rev-9 development (truth only): estimator exposure mass vs the generator's, with hour splits (10 phases).
Rules: midpoint (current); GL-n in log age on pieces wider than delta in log age, midpoint on narrower ones;
pieces starting at 0: composite GL-n in log age over [1e-3 s, b] in pieces of log width <= W0."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user")
from forum_model.fit1_estimator2 import BASE_EDGES, REV8_EDGES
from forum_model.fit1_generator import LOG_GRID
T = json.load(open("/home/user/outputs/fit1/stage1_truth.json"))
def lx(t): return np.log(np.maximum(t, 1e-3) / 60.0)
def gm(a, b): return np.where(a > 0, np.sqrt(np.maximum(a, 1e-9) * b), b / 2)

def pieces(edges, phase, H=3 * 86400.0):
    hb = np.arange(phase, H, 3600.0)
    g = np.unique(np.r_[edges[edges <= H], hb])
    return g[:-1], g[1:]

def gen_mass(f, phase, H=3 * 86400.0):
    hb = np.arange(phase, H, 3600.0)
    g = np.unique(np.r_[LOG_GRID[LOG_GRID <= H], hb, H])
    return g, f(gm(g[:-1], g[1:])) * np.diff(g)

def gl(f, a, b, n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    la, lb = np.log(a), np.log(b)
    x = (lb - la)[:, None] / 2 * xg[None, :] + ((la + lb) / 2)[:, None]
    return np.sum((lb - la)[:, None] / 2 * wg[None, :] * np.exp(x) * f(np.exp(x)), axis=1)

def est_mass(f, a, b, rule, n, delta, W0):
    m = f(gm(a, b)) * (b - a)
    if rule == "mid":
        return m
    out = m.copy()
    lw = np.where(a > 0, np.log(b / np.maximum(a, 1e-12)), np.inf)
    w = (lw > delta) & (a > 0)
    out[w] = gl(f, a[w], b[w], n)
    for i in np.nonzero(a == 0)[0]:
        la, lb = np.log(1e-3), np.log(b[i]); k = int(np.ceil((lb - la) / W0)); e = np.exp(np.linspace(la, lb, k + 1))
        out[i] = gl(f, e[:-1], e[1:], n).sum() + f(np.array([5e-4]))[0] * 1e-3
    return out

def score(f, edges, rule, n=2, delta=0.05, W0=2.0):
    errs, tot = [], []
    for phase in np.linspace(0, 3600, 10, endpoint=False) + 1.0:
        a, b = pieces(edges, phase)
        g, gmass = gen_mass(f, phase)
        cg = np.r_[0, np.cumsum(gmass)]
        ref = np.interp(b, g, cg) - np.interp(a, g, cg)   # generator mass per estimator piece (piecewise-linear cum)
        e = est_mass(f, a, b, rule, n, delta, W0)
        # compare on estimator cells (merge pieces of the same base cell)
        cell = np.searchsorted(edges, a, side="right") - 1
        R = np.bincount(cell, weights=ref); Em = np.bincount(cell, weights=e)
        ok = R > 0
        lr = np.log(Em[ok] / R[ok]); wt = R[ok] / R[ok].sum()
        errs.append(np.sum(wt * np.abs(lr))); tot.append(np.max(np.abs(lr) * (wt > 1e-3)))
    return np.mean(errs), np.mean(tot)

for pop, edges in [("moltbook", REV8_EDGES), ("buildapc", BASE_EDGES)]:
    p = T[pop]["params"]
    fs = {"root": lambda u: np.exp(p["aR"][0] * lx(u) + p["aR"][1] * lx(u) ** 2)}
    for s in [60.0, 3600.0]:
        fs[f"comment s={s:.0f}"] = (lambda u, s=s: np.exp(p["k"][0] * lx(u) + p["k"][1] * lx(u) ** 2 + p["a"][0] * lx(s + u) + p["a"][1] * lx(s + u) ** 2))
    for nm, f in fs.items():
        line = [f"{pop:9s} {nm:15s} mid {score(f, edges, 'mid')[0]:.5f}"]
        for n, d in [(2, 0.05), (2, 0.1), (3, 0.05), (3, 0.1)]:
            line.append(f"GL{n} d={d}: {score(f, edges, 'gl', n, d)[0]:.5f}")
        print("  ".join(line))
