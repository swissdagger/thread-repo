# Step 4: The position check (Reddit and HN)

> **Corrections (audit, `reports/audit.md` §3):**
> (1) The β SEs below are model-based and understate uncertainty by up to 95× where few large threads
> dominate. Thread-clustered SEs and 95% bootstrap intervals are in the audit and in
> `outputs/combined/step4_*_summary.csv`.
> (2) **"About half of the arrival-rank effect is display position" is not established:** the r/Fantasy
> contest-vs-control difference in β has a 95% CI of [−0.04, +0.38].
> (3) The r/AskHistorians "matched series" control is withdrawn: the series changed sort mode over time
> (check 8).
> (4) HN (b) numbers were recomputed without dead items: Ask HN β −0.273, HN link +0.056, Show HN +0.707.
> The flattening conclusions are unchanged.


Status: done. Waiting for confirmation before Step 5.

Script: `scripts/step4_position.py`. Shared code: `forum_model/pem.py` (piecewise-exponential hazard with
a position covariate) and `forum_model/hazard.py` (`fit_first_reply_cov`).

- Tables: `outputs/combined/step4_{a,b,c,d}_summary.csv`, plus `step4_*_baseline.csv` with the time-bin
  hazards.
- Figure: `figures/combined/step4_position_check.png`.

## The question

When page position among siblings enters the reply hazard, does the time decay flatten? If so, position
was absorbing what looked like novelty decay. The brief asks for this under each proxy, especially (b) and
(c), not just (a).

## What was computed

**Model.** A piecewise-exponential first-reply hazard:

    log h_i(t) = α_k + β · log(position_i(t)),  for t in time bin k

- **M0** has the time bins α_k only: the non-parametric time decay.
- **M1** adds β · log(position).
- **Time decay** is the events-weighted slope of the log baseline hazard against log₁₀(time). The fit range
  is 1 min to 1 day, or 1 to 30 min for proxy (b).
- **Flattening** = slope(M1) − slope(M0). Positive means the decay flattened once position was included.

For the static proxies I also refit the Weibull kernel (shape) and the exponential novelty kernel (τ,
`model_reference.md`'s F term) with and without the covariate.

**Data and estimation.**

- Parents, first-reply events and censoring follow Step 3: non-self, non-bot replies; bots and orphans
  excluded.
- Position counts every displayed sibling: bots and removed comments occupy slots.
- Only sibling groups with 2+ comments are used. Position is capped at 500.
- Estimates are on the calibration inner fit fold. The held-out gain is M1 − M0 log-likelihood per parent
  on the inner held-out fold.
- **Per-author weighting is primary** (unknown authors weight 0; see Step 3b). Raw versions are in the
  tables.
- Standard errors for β are model-based and understate the uncertainty, because they don't cluster by
  thread.

**Proxies, each reported separately.**

| proxy | data | position | note |
|---|---|---|---|
| (a) | r/philosophy, r/buildapc, all non-root comments | rank among siblings by **final** score | **Look-ahead-affected**: final scores accumulate for months (Step 1) and include votes caused by the replies being predicted |
| (b) | r/philosophy, r/buildapc, Ask HN, HN link, Show HN; comments made in the thread's **first 30 minutes** | arrival rank among siblings | Censored at 30 minutes of thread age. **Main identification.** HN gets only (b): its sort uses hidden scores, so HN position is latent |
| (c) | **r/Fantasy contest-mode threads** (primary), **r/AskHistorians random-sort threads with removed parents dropped** (secondary) | randomized by the platform | Contest mode collapses replies, so parents are **top-level comments**. Compared with control threads (20+ comments) of the same subreddit. Arrival rank is used as a placebo covariate: it should matter less where display is random |
| (d) | `suggested_sort = new` threads in r/philosophy and r/buildapc, top-level comments | **time-varying**: 1 + number of newer top-level siblings so far (newest first) | Evaluated at each time bin |

**r/AskHistorians matched control.** For r/AskHistorians I added a control from the **same recurring
series** run under both sort modes: Friday Free-for-All, Thursday Reading, Monday Methods.

**r/Fantasy has no such control.** Its recurring threads (self-promotion, general discussion, Friday
social) are almost always in contest mode: 118 contest vs 1 non-contest for "general discussion".

**Both proxy (c) sources are social, promotional or digest threads, not ordinary Q&A or link threads.**
Their decay and placebo results describe those thread types.

## Key numbers (per-author weighted)

### (a) Final-score rank. Look-ahead-affected.

| | parents (fit fold) | β per log position (SE) | hazard ratio per doubling | decay slope M0 → M1 | flattening | held-out gain | Weibull shape M0 → M1 | novelty τ M0 → M1 |
|---|---:|---|---:|---|---:|---:|---|---|
| r/philosophy | 251,222 | −0.362 (0.004) | 0.78 | −0.851 → −0.831 | **+0.021** | +0.046 | 0.523 → 0.531 | 10.9 h → 11.2 h |
| r/buildapc | 2,000,000 | −0.268 (0.001) | 0.83 | −1.053 → −1.038 | **+0.016** | +0.036 | 0.378 → 0.383 | 46.9 h → 47.2 h |

### (b) Early thread, arrival rank. Main identification.

| | parents (fit fold) | β (SE) | hazard ratio per doubling | decay slope, 1–30 min, M0 → M1 | flattening | held-out gain | Weibull shape, 30-min window, M0 → M1 |
|---|---:|---|---:|---|---:|---:|---|
| r/philosophy | 7,237 | +0.037 (0.051) | 1.03 | −0.580 → −0.576 | +0.004 | 0.000 | 1.31 → 1.31 |
| r/buildapc | 1,145,824 | **−0.320** (0.009) | 0.80 | −0.836 → −0.848 | −0.012 | +0.005 | 1.16 → 1.17 |
| Ask HN | 13,917 | **−0.260** (0.035) | 0.84 | −0.410 → −0.450 | −0.039 | +0.002 | 1.45 → 1.44 |
| HN link | 83,695 | +0.065 (0.014) | 1.05 | −0.485 → −0.475 | +0.010 | 0.000 | 1.18 → 1.19 |
| Show HN | 2,782 | +0.663 (0.068) | 1.58 | −0.605 → −0.499 | +0.106 | 0.000 (raw +0.004) | 1.44 → 1.46 |

The Weibull shapes above 1 reflect the onset within the first 30 minutes (Step 3, K2).

### (c) Randomized order vs control

Top-level comments as parents; replies to top-level comments as events.

| | threads | parents (fit fold) | decay slope M0 | Weibull shape | novelty τ | placebo: arrival-rank β (SE) |
|---|---|---:|---:|---:|---:|---|
| **r/Fantasy** | contest mode | 12,564 | **−0.758** | 0.518 | 17.2 h | **−0.219** (0.016) |
| r/Fantasy | control, 20+ comments | 440,183 | −0.793 | 0.523 | 12.4 h | −0.459 (0.004) |
| r/AskHistorians | random sort, removed parents dropped | 3,604 | −0.783 | 0.521 | 12.5 h | −0.141 (0.023) |
| r/AskHistorians | control, 20+ comments | 7,900 | −0.558 | 0.592 | 37.6 h | −0.233 (0.017) |
| r/AskHistorians | **same series, random sort** | 738 | −0.735 | 0.556 | 17.4 h | −0.167 (0.095) |
| r/AskHistorians | **same series, not random** | 591 | −0.827 | 0.608 | 16.9 h | −0.274 (0.101) |

Raw (unweighted) slopes for the same series: −0.727 (random) and −0.706 (not random).

### (d) Sort = new, time-varying position

| | parents (fit fold) | β (SE) | hazard ratio per doubling | decay slope M0 → M1 | flattening | held-out gain |
|---|---:|---|---:|---|---:|---:|
| r/philosophy | 2,614 | −0.096 (0.058) | 0.94 | −0.410 → −0.392 | +0.018 | −0.001 |
| r/buildapc | 123,026 | **−0.262** (0.006) | 0.83 | −0.763 → **−0.612** | **+0.151** | **−0.005** |

## Does time decay flatten?

**Under (a), barely.** Even with the look-ahead-affected final-score rank, which predicts strongly (0.78–0.83×
per doubling, positive held-out gain), the decay slope moves only +0.02. The Weibull shape moves by
0.005–0.008 and τ by about 2%. The circularity the reference feared, position absorbing time decay
mechanically, is small here. That's because final-score rank is **static**: it can absorb decay only by
separating fast-reply comments from slow ones. It can't follow a comment sliding down the page.

**Under (b), no.** The flattening ranges from −0.04 to +0.01, with one exception.

- Show HN flattens by +0.11, but on 2,782 parents and with no held-out gain under per-author weighting.
- Arrival rank lowers the reply hazard on r/buildapc (0.80× per doubling) and Ask HN (0.84×). It has no
  effect on r/philosophy or HN link.
- Including it leaves the 1–30 min decay unchanged, or slightly steeper.

**Under (c), no clear flattening.**

- r/Fantasy contest threads decay slightly flatter than its controls (−0.758 vs −0.793), with the same
  Weibull shape (0.518 vs 0.523).
- In the r/AskHistorians matched series, randomized threads are flatter per-author (−0.735 vs −0.827) but
  slightly steeper raw (−0.727 vs −0.706), on fewer than 1,000 parents each: inconclusive.
- The unmatched r/AskHistorians control is a different kind of thread, so its large difference (−0.783 vs
  −0.558) is not interpretable.

**Under (c), the placebo works as expected.** Where display order is randomized, the arrival-rank
association roughly halves:

- r/Fantasy: −0.22 vs −0.46;
- r/AskHistorians: −0.14 vs −0.23, and in the matched series −0.17 vs −0.27.

So **about half of the arrival-rank effect in ordinary threads is display position, and about half is
something intrinsic to arriving later**, such as declining thread activity (Step 3b: lateness effect within
threads).

**Under (d), flattening in the fit that does not survive held-out data.** On r/buildapc, time-varying
position flattens the decay from −0.76 to −0.61. But adding position makes the *held-out* likelihood worse.
In `new`-sorted megathreads, the count of newer siblings mainly tracks how busy the thread is, which
differs between threads; the effect doesn't transfer across them. r/philosophy shows no effect.

**Overall.** Position matters for *which* comment gets replies: about 0.73–0.86× per doubling of sibling
rank where display order is known, and about half of that survives randomization. But it **does not
absorb the time decay** under the main identification (b) or the randomized check (c). The reference's
expected finding "time decay flattens under (b) and (c)" is **not supported**. The one time-varying proxy
(d) suggests position can mimic decay inside busy newest-first threads, but that doesn't generalize.

## Fitted-on / validated-on

| quantity | fitted on | compared or validated on |
|---|---|---|
| α_k, β (PEM); Weibull shape, novelty τ, β (parametric) | Calibration **inner fit fold** for each population and proxy (capped at 2M parents for (a) r/buildapc) | Held-out gain on the calibration **inner held-out fold** |
| Decay slopes, flattening, placebo contrasts | Derived from the fit-fold estimates | – |

The 20% validation split was not used. Reply timing and position are calibration quantities; no
validation-list quantity was computed.

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`.

- **P1. Keep a novelty kernel in F; position does not replace it.** The reference predicts that τ shrinks
  once position enters exposure, because position was absorbing the decay. I found no such change: τ and
  the Weibull shape move by at most about 2% under every static proxy, and the time-varying proxy doesn't
  hold up out of sample. Novelty decay and position should be modelled as separate factors (F's
  exp(−age/τ) and E's e_thread(pos)).
- **P2. The position-bias curve, over rank among siblings, is a power law of about pos^(−0.2…−0.3).**
  Proxies (a), (b) and (d) give 0.73–0.86× per doubling. The randomized-order placebo shows that only
  about half of the arrival-rank association is display position, so **e_thread should be estimated with
  arrival-time (lateness) controls, or from randomized threads.** Otherwise it takes up the intrinsic
  lateness effect.
- **P3. Proxy (c) only covers social and digest threads.** Both sources are recurring social, promotional
  or digest threads, and contest mode also collapses replies (parents = top-level comments only). A
  position-bias curve estimated there should be checked against (b) before being applied to ordinary
  threads.
- **P4. r/AskHistorians is a weak secondary.** 44% of top-level parents in its random threads were removed
  (dropped here), which leaves 3,604 parents and only 738 in the matched series.
- **P5. The HN position curve stays latent beyond the first 30 minutes.** Within that window arrival rank
  matters on Ask HN (0.84× per doubling) but not on HN link threads. HN's sort applies hidden scores, so
  later position must be inferred jointly with attractiveness, as the reference already plans.
