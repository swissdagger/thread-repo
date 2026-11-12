# First model fit, Stage 1 rerun: revised estimator

**Synthetic data only**, as in Stage 1. The real data entered only as root arrival times, post types and the
volumes the truth was calibrated to. The validation split was not read.

**What changed since Stage 1** (decisions after the Stage 1 review, committed in `6c075a4` before any rerun
replicate ran):
- Rev-1: fitness integrated out is the only estimator.
- Rev-5: multi-start over b, with convergence reported for every fit.
- Rev-7: exposure cells split at UTC hour boundaries.
- Rev-8: bins below 10 s, tested on Moltbook under a pre-stated keep rule.
- Rev-6 and F1-S2b: Moltbook reports and is scored on the combined clock k(u) + a(s̄ + u).

The pass rules are otherwise unchanged. The datasets, truth and seeds are identical to Stage 1, so every
difference from Stage 1 comes from the estimator.

Code: `forum_model/fit1_estimator2.py`, `scripts/fit1_stage1.py` (`recover2`, `hn_joint2`, `bootcheck2`,
`summary2`). Outputs: `outputs/fit1/stage1b_*`. Figures: `figures/fit1/stage1b_*`.

## Verdict

| rule | outcome |
|---|---|
| Rev-8 keep rule | **Kept for Moltbook.** Root-stream genuine failures fall from 6 to 3, and total genuine failures from 10 to 3. |
| F1-S1 (literal) | **Fails.** As in Stage 1, it is unreachable for a correct estimator: about 50 parameters × 7 fits at p = 0.925 each. |
| F1-S3 (genuine failures) | At chance level in Ask HN, Show HN, r/philosophy, HN link and the joint HN fit. **Not** at chance level in Moltbook (root clock and γ) or r/buildapc (clocks, root intercept, a common hour shift). |
| F1-S2b (Moltbook combined clock) | **Passes, at the threshold:** 18/20. |
| F1-S4 (one-step bootstrap) | **Passes.** Widths within 5% for all 39 parameters. |
| F1-S5 (B1) | Unchanged. The single-clock comment kernel is wrong in every human population. |
| Convergence | **160/160 fits converged**, in ≤ 14 iterations. Stage 1 had 24 non-converged fits. |

Stage 1 failed because of the optimiser in HN link and the joint fit. That is fixed. What remains is small
and concentrated in the clock terms, and it looks like a cell-integration error (Rev-9 proposed). Details
follow.

## F1-S1 and F1-S3, by population

Out of 44 parameters per population (36 for the joint HN fit). Moltbook figures are for the kept Rev-8 variant.

| population | ≥ 18/20 covered (Stage 1) | genuine failures (Stage 1) | what fails now |
|---|---|---|---|
| HN link | 41 (15) | **1** (28) | b: 19/20 covered, bias +0.52 SD |
| Ask HN | 40 (41) | **1** (0) | hour 9: 20/20 covered, bias +0.53 SD |
| Show HN | 38 (40) | **1** (1) | hour 14: 18/20 covered, bias +0.57 SD |
| r/philosophy | 36 (36) | **1** (1) | hour 4: 15/20 covered, no bias, interval too narrow (SD ratio 0.69) |
| joint HN (κ) | 31 of 36 (20) | **2** (11) | α_R (link) −0.59 SD, b +0.53 SD. Both covered ≥ 18/20. |
| Moltbook (Rev-8) | 38 (34) | **3** (10) | aR1 14/20 (+1.5 SD), aR2 11/20 (−1.8 SD), γ 17/20 (−1.2 SD) |
| r/buildapc | 41 (41) | **10** (11) | 6 hours (+0.5 to +0.6 SD), k1 15/20 (+0.85 SD), k2 −0.74 SD, aR1 +0.6, aR2 −0.6, α_R −0.73 |

Figure: `figures/fit1/stage1b_coverage.png` (coverage only; bias flags are listed above).

**What chance level means here.** Under exact 95% coverage and no bias, F1-S3 flags a parameter with
probability ≈ 0.04:
- coverage ≤ 15/20 has p ≈ 0.016;
- a mean bias above 0.5 bootstrap SD over 20 replicates is a 2.24-standard-error event, p ≈ 0.025.

That is about 1.8 flags per 44-parameter fit. The first five rows above are at or below that level. Their
flags are isolated hour effects, b, and one intercept, all covered in ≥ 18/20 except r/philosophy hour 4.

**HN link and the joint fit are fixed by Rev-5.** Every fit converges, from b = 3.5 (all HN link fits) or
from b = 3.5 or 5 (joint fit). In Stage 1, γ was inflated to 0.21 on link and to 0.024 on the joint fit
(truth 0.010). The joint fit now recovers:

| parameter | truth | mean estimate | covered |
|---|---|---|---|
| κ | 0.030 | 0.029 | 19/20 |
| γ | 0.010 | 0.009 | 19/20 |
| b | 3.5 | 3.76 | 19/20 |
| log r | 2.48 | 2.61 | 19/20 |

The per-type root clocks are covered 18–20/20, and the per-type intercepts 17–19/20.

**Moltbook: Rev-8 helped but did not finish the job.**

| | standard bins | Rev-8 bins |
|---|---|---|
| aR1 | 15/20, −1.25 SD | 14/20, +1.51 SD |
| aR2 | 2/20, +3.54 SD | 11/20, −1.78 SD |
| γ | 7/20, +2.19 SD | 17/20, −1.16 SD |
| b | 12/20, −1.36 SD | 18/20, +0.41 SD |
| log r | 16/20, −1.05 SD | 17/20, +0.17 SD |
| α_R | 19/20, −0.95 SD | 20/20, +0.02 SD |
| a2 | 15/20, −1.26 SD | 18/20, +0.13 SD |
| k2 | 12/20, +1.44 SD | 19/20, −0.20 SD |

The finer early bins fix fitness (b, log r), the comment-clock quadratics and most of γ. The root-clock
coefficients aR1 and aR2 still miss. Their errors have opposite signs, so they partly cancel in the clock
shape, and both signs flipped when the bins changed. A shape that depends on the binning points to how the
exposure integral is computed in a cell where the root rate changes fastest (the first minutes). It does not
point to identification. Rev-9 proposes quadrature within cells.

**r/buildapc: small biases, the tightest intervals.** r/buildapc is the largest sample (15,183 threads). Its
flags are small in absolute terms:
- k1 −0.242 vs −0.251;
- aR1 −0.163 vs −0.167;
- α_R −6.51 vs −6.49.

The same cell-integration error would show here first.

**Correction to the Stage 1 report on hours.** Stage 1 described the r/buildapc hour effects as "pulled toward
0 by about 0.02", and Rev-7 was proposed on that basis. That description was wrong:
- In both runs, all 23 hours sit about +0.011 above truth on average.
- The deviation hardly depends on the true value (correlation 0.2). Attenuation would give a strong negative
  correlation.

So this is a common shift relative to the hour-0 baseline. A midpoint rule cannot produce that pattern, so
Rev-7 did not change it.

| population | mean common hour shift |
|---|---|
| r/buildapc | +0.011 (SE 0.005, t = 2.3) |
| Ask HN | +0.005 (t = 1.3) |
| Show HN | +0.005 (t = 0.9) |
| r/philosophy | −0.001 (t = −0.2) |
| HN link | −0.006 (t = −1.5) |

Across five populations this is marginal. Stage 1 and the rerun share the same datasets, and therefore the same
realised shift. Rev-7 stays: it is exact by construction and costs nothing in recovery. F1-S3, however, counts
this one baseline deviation as up to 23 separate failures (Rev-10 proposed).

**Reply-back stream:** recovered everywhere, 16–20/20, no bias flags. This is the same as Stage 1.

## F1-S2b: combined clock in the burst setting (Moltbook)

| | combined clock k(u) + a(s̄ + u) in band | comment clock k | thread clock a |
|---|---|---|---|
| Moltbook, Rev-8 (scored) | **18/20** | 17/20 | 16/20 |
| Moltbook, standard bins | 17/20 | 13/20 | 13/20 |
| human populations, for comparison | 17–18/20 | 18–19/20 | 17–20/20 |

"In band" means the truth lies inside the pointwise 95% band at ≥ 90% of event-range ages.

Figure: `figures/fit1/stage1b_moltbook_combined_clock.png`, estimate minus truth per replicate.
- Inside the event range (1–45 min), the errors stay within about ±0.15 in log rate.
- Beyond it, they fan out to ±1 by 3 days, as the bands do. No events constrain that region.

The rule passes at its threshold, so this is a pass without margin.

## F1-S4 and F1-S5

- **F1-S4** (`outputs/fit1/stage1b_bootcheck.csv`): one-step and full-refit 95% interval widths agree within
  5% for all 39 parameters (ratios 0.95–1.03). Centres differ by ≤ 0.04 refit widths. The one-step bootstrap
  stays the Stage 2 default.
- **F1-S5** (`outputs/fit1/stage1b_B1_kernel.csv`): with the thread clock omitted, the comment kernel is
  covered in 0/20 replicates for both terms in every human population. Example, Ask HN: k1 0.68 vs a true 0.15.
  The thread clock's decay moves into the comment kernel, as in Stage 1.

## Run notes

- **Out-of-memory kills.** Three parallel fits exceeded the container's 15 GB memory limit. Five jobs were
  killed after 1–2 replicates each, and the rerun continued with two workers.
- **Container restarts.** The container was reclaimed twice while the session was idle. Each time, the jobs
  resumed from saved replicates, so no replicate was lost or run twice.

Neither affects the results.

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| True parameters (intercepts and ρ calibrated) | Volumes of the calibration inner fit fold (capped); unchanged from Stage 1 | – |
| All rerun estimates | Synthetic datasets only (the Stage 1 datasets) | Recovery against known truth. No real-data validation. |

## Implications for the model (and for Stage 2)

`docs/model_reference.md` is unchanged.

1. **The revised estimator recovers the model in four of the six populations and in the joint HN fit**, to
   within chance-level flags. This includes κ, γ, b and r in the joint fit, which Stage 1 could not recover.
2. **Moltbook's root clock (aR1, aR2) should not be read coefficient by coefficient.** Its fitness, γ, comment
   clock and combined clock recover with Rev-8 bins. Agent-side root-stream timing conclusions should wait for
   Rev-9 or be stated as shapes over the event range.
3. **r/buildapc's clock coefficients carry small biases** (≤ 0.01 in absolute terms) that matter only at its
   precision. They are flagged, not fatal. Stage 2 comparisons of clock coefficients across populations should
   use shapes over the event range, not raw coefficients.
4. **The hour baseline has a small shared shift in some datasets.** Hour effects should be reported as a
   profile, not as single hours against hour 0.
5. **Under F1-S3 as written, Stage 2 proceeds only for parameters without genuine failures**, or after a logged
   revision. The parameters held back are Moltbook aR1, aR2 and γ, and the r/buildapc parameters listed above.

## Proposed before Stage 2 (needs confirmation)

1. **Rev-9:** exposure integral by 3-point Gauss–Legendre quadrature in log age within each cell. Test it on
   Moltbook and r/buildapc under a pre-stated rule like Rev-8's, then rerun those two populations.
2. **Rev-10:** for future stages, the F1-S3 bias criterion would also require more than 3 standard errors of
   the replicate mean, and would treat the 23 hours as one block. The rerun was scored under the rule as
   stated.
3. **Alternatively, proceed to Stage 2 now.** Moltbook aR1, aR2 and γ, and the flagged r/buildapc
   coefficients, would carry a caveat until Rev-9 is tested.

Stopping here as instructed. The rerun results, report and register outcomes are not committed.
