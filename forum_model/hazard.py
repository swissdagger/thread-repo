"""Reply-delay hazards and self-excitation kernels (step 3).

Model: replies to a comment arrive as an inhomogeneous Poisson process with intensity mu * g(t),
where g is a normalised kernel (integrates to 1) and t is time since the comment. Then

  - time to FIRST reply has hazard h(t) = mu * g(t) and (defective) survival exp(-mu * G(t)),
    so a comment never gets a reply with probability exp(-mu);
  - ALL direct replies have log-likelihood sum_r log(mu g(t_r)) - mu * sum_i G(c_i).

mu is the kernel integral = expected direct replies per comment = the branching ratio. In both
likelihoods mu has a closed-form profile, so only the kernel shape parameters are optimised.

Kernels (all integrate to 1): exponential, Weibull, two-exponential mixture.
Times are in seconds; censoring times c are observation ends measured from the comment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

TMIN = 0.5  # s; same-second replies are placed at 0.5 s


# ------------------------------------------------------------------------------------------
# kernel families
# ------------------------------------------------------------------------------------------

def _exp_parts(th, t):
    lb = th[0]
    b = np.exp(lb)
    return lb - b * t, -np.expm1(-b * t)


def _weib_parts(th, t):
    lk, ll = th
    k, lam = np.exp(lk), np.exp(ll)
    lt = np.log(t)
    z = np.exp(k * (lt - ll))
    return lk - ll + (k - 1) * (lt - ll) - z, -np.expm1(-z)


def _mix_parts(th, t):
    a, lb1, lb2 = th
    w = special.expit(a)
    b1, b2 = np.exp(lb1), np.exp(lb2)
    lg = np.logaddexp(np.log(w) + lb1 - b1 * t, np.log1p(-w) + lb2 - b2 * t)
    G = w * -np.expm1(-b1 * t) + (1 - w) * -np.expm1(-b2 * t)
    return lg, G


def _lognorm_parts(th, t):
    m, ls = th
    sd = np.exp(ls)
    lt = np.log(t)
    z = (lt - m) / sd
    return -lt - ls - 0.5 * np.log(2 * np.pi) - 0.5 * z * z, special.ndtr(z)


def _gamma_parts(th, t):
    """Gamma kernel with shape k = 1 + exp(a) > 1 (density rises from 0, peaks at (k-1)*scale)."""
    a, lsc = th
    k = 1.0 + np.exp(a)
    sc = np.exp(lsc)
    return (k - 1) * np.log(t) - t / sc - special.gammaln(k) - k * lsc, special.gammainc(k, t / sc)


FAMILIES = {"exponential": _exp_parts, "weibull": _weib_parts, "mixture2": _mix_parts,
            "lognormal": _lognorm_parts, "gamma": _gamma_parts}
STEP3_FAMILIES = ("exponential", "weibull", "mixture2")


def starts(family: str, t_event: np.ndarray) -> list[np.ndarray]:
    m = float(np.median(t_event)) if len(t_event) else 3600.0
    m = max(m, 1.0)
    if family == "exponential":
        return [np.array([np.log(np.log(2) / m)])]
    if family == "weibull":
        return [np.array([np.log(k), np.log(m)]) for k in (0.3, 0.7, 1.0)]
    if family == "lognormal":
        return [np.array([np.log(m), np.log(sd)]) for sd in (1.0, 2.0, 3.0)]
    if family == "gamma":
        return [np.array([np.log(k - 1), np.log(m / k)]) for k in (1.2, 2.0, 4.0)]
    return [np.array([a, np.log(f1 / m), np.log(f2 / m)])
            for a in (-1.0, 0.0, 1.0) for f1, f2 in ((10.0, 0.1), (3.0, 0.03))]


def describe(family: str, th: np.ndarray) -> dict:
    if family == "exponential":
        return {"rate_per_h": float(np.exp(th[0]) * 3600), "mean_delay_h": float(1 / np.exp(th[0]) / 3600)}
    if family == "weibull":
        return {"weibull_shape": float(np.exp(th[0])), "weibull_scale_h": float(np.exp(th[1]) / 3600)}
    if family == "lognormal":
        sd = float(np.exp(th[1]))
        return {"lognormal_median_h": float(np.exp(th[0]) / 3600), "lognormal_sigma": sd,
                "kernel_mode_h": float(np.exp(th[0] - sd * sd) / 3600)}
    if family == "gamma":
        k, sc = 1.0 + float(np.exp(th[0])), float(np.exp(th[1]))
        return {"gamma_shape": k, "gamma_scale_h": sc / 3600, "kernel_mode_h": (k - 1) * sc / 3600}
    w = float(special.expit(th[0]))
    b = np.exp(th[1:])
    fast, slow = (0, 1) if b[0] >= b[1] else (1, 0)
    wf = w if fast == 0 else 1 - w
    return {"mix_w_fast": wf, "mix_mean_fast_h": float(1 / b[fast] / 3600),
            "mix_mean_slow_h": float(1 / b[slow] / 3600)}


# ------------------------------------------------------------------------------------------
# fitting
# ------------------------------------------------------------------------------------------

@dataclass
class Fit:
    family: str
    theta: np.ndarray
    mu: float
    loglik: float
    n: float
    events: float

    def params(self) -> dict:
        d = {"family": self.family, "mu": self.mu, "loglik": self.loglik, "n_weight": self.n,
             "events_weight": self.events, "half_life_h": half_life(self.family, self.theta) / 3600}
        d.update(describe(self.family, self.theta))
        return d


def _first_reply_nll(th, parts, t, d, w):
    lg, G = parts(th, t)
    Wd = (w * d).sum()
    SG = (w * G).sum()
    if SG <= 0 or not np.isfinite(SG):
        return np.inf
    mu = Wd / SG
    ll = (w * d * lg).sum() + Wd * np.log(mu) - Wd
    return -ll if np.isfinite(ll) else np.inf


def fit_first_reply(family: str, t: np.ndarray, d: np.ndarray, w: np.ndarray | None = None) -> Fit:
    """Defective-survival MLE on time to first reply (t: event or censoring time; d: event flag)."""
    t = np.maximum(t, TMIN)
    w = np.ones_like(t) if w is None else w
    parts = FAMILIES[family]
    best = None
    for x0 in starts(family, t[d]):
        r = optimize.minimize(_first_reply_nll, x0, args=(parts, t, d, w), method="Nelder-Mead",
                              options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    lg, G = parts(best.x, t)
    mu = (w * d).sum() / (w * G).sum()
    return Fit(family, best.x, float(mu), float(-best.fun), float(w.sum()), float((w * d).sum()))


def first_reply_loglik(fit: Fit, t: np.ndarray, d: np.ndarray, w: np.ndarray | None = None) -> float:
    """Per-comment (weighted mean) log-likelihood of held-out data under a fitted model."""
    t = np.maximum(t, TMIN)
    w = np.ones_like(t) if w is None else w
    lg, G = FAMILIES[fit.family](fit.theta, t)
    ll = (w * d * (np.log(fit.mu) + lg)).sum() - fit.mu * (w * G).sum()
    return float(ll / w.sum())


def _all_replies_nll(th, parts, r, wr, c, wc):
    lgr, _ = parts(th, r)
    _, Gc = parts(th, c)
    R = wr.sum()
    SG = (wc * Gc).sum()
    if SG <= 0 or not np.isfinite(SG):
        return np.inf
    mu = R / SG
    ll = (wr * lgr).sum() + R * np.log(mu) - R
    return -ll if np.isfinite(ll) else np.inf


def fit_all_replies(family: str, r: np.ndarray, wr: np.ndarray, c: np.ndarray, wc: np.ndarray) -> Fit:
    """Poisson-process MLE on every direct reply delay r (each <= its parent's censoring time),
    with parent censoring times c."""
    r, c = np.maximum(r, TMIN), np.maximum(c, TMIN)
    parts = FAMILIES[family]
    best = None
    for x0 in starts(family, r):
        res = optimize.minimize(_all_replies_nll, x0, args=(parts, r, wr, c, wc), method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    _, Gc = parts(best.x, c)
    mu = wr.sum() / (wc * Gc).sum()
    return Fit(family, best.x, float(mu), float(-best.fun), float(wc.sum()), float(wr.sum()))


def fit_first_reply_cov(family: str, t: np.ndarray, d: np.ndarray, x: np.ndarray,
                        w: np.ndarray | None = None) -> tuple[Fit, float]:
    """h_i(t) = mu * exp(beta * x_i) * g(t) with a static covariate x; mu profiled out."""
    t = np.maximum(t, TMIN)
    w = np.ones_like(t) if w is None else w
    parts = FAMILIES[family]
    Wd = (w * d).sum()
    wdx = (w * d * x).sum()

    def nll(z):
        beta, th = z[0], z[1:]
        lg, G = parts(th, t)
        SG = (w * np.exp(beta * x) * G).sum()
        if SG <= 0 or not np.isfinite(SG):
            return np.inf
        mu = Wd / SG
        ll = (w * d * lg).sum() + beta * wdx + Wd * np.log(mu) - Wd
        return -ll if np.isfinite(ll) else np.inf

    best = None
    for x0 in starts(family, t[d]):
        r = optimize.minimize(nll, np.r_[0.0, x0], method="Nelder-Mead",
                              options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    beta, th = best.x[0], best.x[1:]
    _, G = parts(th, t)
    mu = Wd / (w * np.exp(beta * x) * G).sum()
    return Fit(family, th, float(mu), float(-best.fun), float(w.sum()), float(Wd)), float(beta)


def half_life(family: str, th: np.ndarray) -> float:
    """Time by which half the kernel mass has arrived: G(t) = 0.5 (seconds)."""
    f = lambda lt: FAMILIES[family](th, np.array([np.exp(lt)]))[1][0] - 0.5  # noqa: E731
    try:
        return float(np.exp(optimize.brentq(f, np.log(1e-3), np.log(1e10))))
    except ValueError:
        return float("nan")


def kernel_hazard(fit: Fit, edges: np.ndarray) -> np.ndarray:
    """Bin-averaged fitted hazard mu * (G(b) - G(a)) / (b - a), per second."""
    a, b = edges[:-1], edges[1:]
    _, Ga = FAMILIES[fit.family](fit.theta, np.maximum(a, TMIN))
    _, Gb = FAMILIES[fit.family](fit.theta, b)
    return fit.mu * (Gb - Ga) / (b - a)


# ------------------------------------------------------------------------------------------
# non-parametric hazard
# ------------------------------------------------------------------------------------------

def hazard_edges(t_min: float = 10.0, t_max: float = 10 * 86400.0, n: int = 25) -> np.ndarray:
    """[0, 10 s) then n log-spaced bins from 10 s to 10 days."""
    return np.r_[0.0, np.logspace(np.log10(t_min), np.log10(t_max), n + 1)]


def life_table(t: np.ndarray, d: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Actuarial hazard: weighted events in bin / weighted time at risk in bin (per second)."""
    w = np.ones_like(t) if w is None else w
    ev, ex, ev2 = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        exp_ = np.clip(t - a, 0.0, b - a)
        inb = d & (t >= a) & (t < b)
        ex.append(float((w * exp_).sum()))
        ev.append(float((w * inb).sum()))
        ev2.append(float((w * w * inb).sum()))
    ev, ex, ev2 = map(np.asarray, (ev, ex, ev2))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = ev / ex
        se = np.sqrt(ev2) / ex
    return {"t_lo": edges[:-1], "t_hi": edges[1:], "events": ev, "exposure_s": ex, "hazard_per_s": h, "se": se}


# ------------------------------------------------------------------------------------------
# depth-tail slope (comparison only)
# ------------------------------------------------------------------------------------------

def depth_tail_slope(depth: np.ndarray, kmin: int = 2, min_count: int = 100) -> dict:
    """OLS of log P(depth >= k) on k for k in [kmin, K], K = last k with >= min_count comments.
    exp(slope) is the depth-tail 'effective branching' (comparison only, never the estimate)."""
    depth = depth[depth >= 1]
    if len(depth) == 0:
        return {"tail_mu": float("nan"), "k_range": "", "r2": float("nan"), "n_points": 0}
    kmax_obs = int(depth.max())
    ks = np.arange(1, kmax_obs + 1)
    cnt = np.array([(depth >= k).sum() for k in ks])
    keep = (ks >= kmin) & (cnt >= min_count)
    if keep.sum() < 2:
        return {"tail_mu": float("nan"), "k_range": "", "r2": float("nan"), "n_points": int(keep.sum())}
    x, y = ks[keep], np.log(cnt[keep] / len(depth))
    slope, icpt = np.polyfit(x, y, 1)
    yhat = icpt + slope * x
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(x) > 2 else float("nan")
    return {"tail_mu": float(np.exp(slope)), "k_range": f"{x.min()}-{x.max()}", "r2": float(r2),
            "n_points": int(len(x))}


# ------------------------------------------------------------------------------------------
# censoring-adjusted non-parametric estimators (audit; see docs/definitions.md)
# ------------------------------------------------------------------------------------------

class SortedCensor:
    """Pre-sorts censoring times so risk sets Y(t) = sum_i w_i [c_i >= t] are cheap to recompute
    under new weights (bootstrap)."""

    def __init__(self, c: np.ndarray):
        self.order = np.argsort(c, kind="stable")
        self.cs = c[self.order]

    def risk(self, w: np.ndarray, t: np.ndarray) -> np.ndarray:
        cw = np.r_[0.0, np.cumsum(w[self.order])]
        k = np.searchsorted(self.cs, t, side="left")   # number of c_i < t
        return cw[-1] - cw[k]


def mean_cumulative(sc: SortedCensor, w: np.ndarray, r: np.ndarray, wr: np.ndarray, H: float) -> float:
    """Nelson-Aalen mean cumulative function for recurrent events (replies) at horizon H:
    MCF(H) = sum_{replies r <= H} wr / Y(r). With no censoring this is the plain mean count."""
    m = r <= H
    Y = sc.risk(w, r[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(Y > 0, wr[m] / Y, 0.0)))


class FirstReplyKM:
    """Kaplan-Meier for time to first reply; precomputes the event-time grid once."""

    def __init__(self, t: np.ndarray, d: np.ndarray, H: float):
        self.sc = SortedCensor(t)          # at-risk: observed time t_i >= u
        ev = d & (t <= H)
        self.ev_idx = np.nonzero(ev)[0]
        self.u, self.inv = np.unique(t[ev], return_inverse=True)

    def curve(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dj = np.bincount(self.inv, weights=w[self.ev_idx], minlength=len(self.u))
        Y = self.sc.risk(w, self.u)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(Y > 0, 1 - dj / Y, 1.0)
        return self.u, np.cumprod(np.clip(q, 0, 1))

    def p_reply_and_median(self, w: np.ndarray) -> tuple[float, float]:
        u, S = self.curve(w)
        if len(u) == 0:
            return 0.0, float("nan")
        p = 1 - S[-1]
        F = 1 - S
        k = np.searchsorted(F, p / 2, side="left")
        return float(p), float(u[min(k, len(u) - 1)])


def thread_multipliers(n_threads: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B x n_threads multinomial bootstrap counts (resampling threads with replacement)."""
    return rng.multinomial(n_threads, np.full(n_threads, 1.0 / n_threads), size=B).astype(np.float64)


def fit_first_reply_from(family: str, t: np.ndarray, d: np.ndarray, w: np.ndarray, theta0: np.ndarray) -> Fit:
    """Single warm-started fit (bootstrap replicates)."""
    t = np.maximum(t, TMIN)
    parts = FAMILIES[family]
    r = optimize.minimize(_first_reply_nll, theta0, args=(parts, t, d, w), method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-4})
    lg, G = parts(r.x, t)
    mu = (w * d).sum() / (w * G).sum()
    return Fit(family, r.x, float(mu), float(-r.fun), float(w.sum()), float((w * d).sum()))
