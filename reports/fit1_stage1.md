# First model fit, Stage 1: generator and estimator recovery

**Synthetic data only.** No real reply was used for estimation in this stage. The real data entered only as
inputs:
- each population's root arrival times and post types (the capped inner fit fold of the calibration split);
- its volumes, which the true parameters were calibrated to match.

The validation split was not read.

The pass rules (F1-S1 to S5) and the true parameters were committed **before** the recovery runs, in
`6d550b4`. Development findings that led to estimator revisions before the runs are logged as Rev-1 to Rev-4
in `docs/results_register.md`. They were found on seeds disjoint from the recovery datasets.

**Verdict: Stage 1 fails its pre-stated rule.** The details are below, as the brief asks: which parameters
fail and why. Four revisions are proposed (Rev-5 to Rev-8). None is applied; Stage 1 should be rerun once
they are confirmed.

## What was built

| piece | file | what it does |
|---|---|---|
| Stream definitions | `forum_model/fit1_streams.py` | One function turns real or simulated items into the three streams: comment stream (no self-replies, no reply-backs), reply-back stream, and root stream (no OP top-level comments). |
| Generator | `forum_model/fit1_generator.py` | Draws fitness z ~ Gamma(r, 1/r) per thread and simulates the three streams to thread age 3 d, generation by generation. Events come from exact time rescaling on a 316-point grid per unit, with UTC-hour and 30-minute boundaries. Authors follow the brief's Stage 3 model. Reused in Stage 3 and by the simulator. |
| Generator check | (development) | Mean counts match the analytic integrals: top-level comments 3.024 vs 3.000, comment-stream replies 0.397 vs 0.400, reply-backs 0.299 vs 0.300. Reply-back median 29.3 vs 30 min. |
| Estimator | `forum_model/fit1_estimator.py` | Exposure cells (40 bins, 10 s to 3 d, split at each unit's own events and at thread age 30 min) plus one row per event at its exact time (Rev-2). **RE arm:** fitness integrated out (marginal likelihood, Newton with the analytic Hessian). **FE arm:** the brief's estimator (Rev-1). Reply-back stream by direct maximum likelihood. One-step thread bootstrap, B = 100. |
| Driver | `scripts/fit1_stage1.py` | roots, calibrate, recover, hn_joint, bootcheck, summary. |
| Post-hoc diagnostic | `scripts/fit1_stage1_diag_link.py` | HN link only. Not a pre-registered test. |

**Settings** (`partitions.json` → `fit1`):
- **Sizes:** each population's capped inner fit fold, ≤ 100,000 comments in whole threads. That is 6,058
  threads (r/philosophy), 15,183 (r/buildapc), 7,594 (Ask HN), 9,958 (HN link), 16,555 (Show HN) and
  21,142 (Moltbook).
- **True parameters** (`outputs/fit1/stage1_truth.json`): set near Step 3/4b/4c estimates. The intercepts
  were calibrated so the simulated volumes match the real ones within 2% (top-level comments per root,
  comment-stream replies and reply-backs per comment).
- **Burst-heavy setting:** Moltbook. Its top-level comments arrive within minutes, so thread age and comment
  age are nearly collinear.
- **Replicates:** 20 per population for the RE arm, the first 5 for the FE arm. The joint HN root-stream fit
  for κ uses the same 20 replicates.
- **Redraws:** 2 of 120 datasets were redrawn because a thread hit the 20,000-comment guard (Rev-3): one in
  HN link, one in r/philosophy.

## Results

### F1-S1 Parameter recovery: fails (literal rule), both arms

Out of 44 parameters per population (36 for the joint HN fit):

| population | RE: ≥ 18/20 covered | RE: genuine failures (F1-S3) | FE: covered 5/5 | FE: genuine failures |
|---|---|---|---|---|
| Ask HN | 41 | **0** | 24 | 26 |
| Show HN | 40 | **1** | 12 | 30 |
| r/philosophy | 36 | **1** | 24 | 27 |
| r/buildapc | 41 | 11 | 13 | 33 |
| Moltbook | 34 | 10 | 9 | 34 |
| HN link | 15 | 28 | 17 | 29 |
| joint HN (κ) | 20 of 36 | 11 | – | – |

Figure: `figures/fit1/stage1_coverage.png`.

**The brief's estimator (FE arm) fails broadly in every population**, as the development work predicted.
History-dependent covariates under thread fixed effects carry a dynamic-panel bias, which spills into
depth, OP and fitness. Examples (mean estimate vs truth, Show HN): γ −0.25 vs +0.01; log(1 + replies)
−0.28 vs +0.05; depth 2 −0.92 vs −0.58.

**RE arm, by population:**
- **Ask HN, Show HN, r/philosophy recover.** There are 0–1 genuine failures, each an hour effect, which
  F1-S3 expects by chance. Coverage of the remaining parameters is 16–20/20.
- **r/buildapc recovers to within small biases.** All 11 flags come from the bias rule (|bias| > 0.5 SD),
  not from coverage. Its intervals are the tightest (most threads). The biases are:
  - hour effects pulled toward 0 by about 0.02, out of ±0.3;
  - root clock aR1 +0.005;
  - comment-clock slope k1 +0.009 (coverage 15/20).

  The hour attenuation is what the brief's hour-at-cell-midpoint rule predicts for long late cells. Rev-7
  proposes a fix.
- **HN link fails: the optimiser never converges.** All 20 fits hit the 60-iteration cap. b is pinned at
  its 0.05 floor and γ inflates to 0.21 (truth 0.01), with knock-on errors in log r, α_R, depth and OP.
  - The post-hoc diagnostic (`outputs/fit1/stage1_diag_link.csv`, replicate 0) shows this is the optimiser,
    not identification. The log-likelihood is −833,805 at the truth against −835,839 at the reported
    estimate. A refit started at the truth converges in 4 iterations to b 3.34, log r 2.39, γ 0.016. The
    profile over b peaks at 3.5, the true value.
  - The default start (b = 1, r = 5) leads into a ridge on which feedback (γ) stands in for fitness. Most
    link threads get 0–1 top-level comments, so the ridge is long. Rev-5 proposes a multi-start.
- **Moltbook (the burst setting) shows genuine biases.** These are in the root stream (aR2 2/20, γ 7/20,
  b 12/20, aR1 15/20) and the clock quadratics (a2 15/20, k2 12/20), as covered in F1-S2 below. The fits
  converged (≤ 11 iterations), so this is not an optimiser failure. One possible cause, untested: the root
  stream peaks within the first minute, and the [0, 10 s] exposure cell is coarse there (Rev-8).
- **Joint HN fit for κ is not clean.** κ is 0.033 against a true 0.030, covered 16/20. But 4 of 20 fits did
  not converge (the same ridge as HN link), γ is biased upward (0.024 vs 0.010), and the post-type
  intercepts miss in 14–15/20.

The reply-back stream recovers everywhere: all five parameters, all populations, ≥ 16/20, no bias flags.
It has no fitness and no history dependence.

### F1-S2 Clock shapes in the burst-heavy setting: fails

| | k1 | k2 | a1 | a2 | shapes inside the band | band informative (half-width < 0.5) |
|---|---|---|---|---|---|---|
| Moltbook | 18/20 | 12/20 | 17/20 | 15/20 | 13/20 (each clock) | 20/20 (half-width 0.26–0.28) |
| human populations, for comparison | – | – | – | – | 15–20/20 | 20/20 (half-width 0.003–0.03) |

Figure: `figures/fit1/stage1_moltbook_clocks.png`. Inside the event range, both estimated shapes follow the
truth. Beyond it they diverge in mirror image: the comment clock decays too slowly and the thread clock too
fast. Where τ ≈ u, the data identify the two clocks' sum (k(u) + a(s + u)), not the split between them.
Rev-6 proposes reporting the sum for burst populations.

### F1-S4 One-step vs full-refit bootstrap: passes

On Ask HN replicate 0 (`outputs/fit1/stage1_bootcheck.csv`), all 39 parameters have one-step and full-refit
95% interval widths within 5% of each other (ratios 0.95–1.03). Centres differ by at most 0.04 refit widths.
The one-step bootstrap can be used in Stage 2.

### F1-S5 B1 (single clock) on the same data

With the thread clock omitted, the comment kernel is wrong in every human population (0/20 coverage for
both terms). Example, Ask HN: k1 0.68 vs a true 0.15, k2 −0.23 vs −0.125. The thread clock's decay is
pushed into the comment kernel, so B1's "comment kernel" mixes the two clocks. On Moltbook, k1 is covered
12/20. This is the brief's expectation for B1, and the Stage 2 B1 comparison should be read with it in mind.

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| True parameters (intercepts and ρ calibrated) | Volumes of the calibration inner fit fold (capped) | – |
| All Stage 1 estimates | Synthetic datasets only | Recovery against known truth (this stage). No real-data validation. |

## Implications for the model (and for Stage 2)

`docs/model_reference.md` is unchanged.

1. **Estimate fitness by integrating it out, not with thread fixed effects** (Rev-1). The brief's two-step
   fixed-effects estimator is inconsistent for the model's own history terms: γ·n_top and log(1 + replies
   so far). The model is unchanged; only the estimator differs.
2. **Where threads are sparse, popularity feedback (γ) and fitness (b, r) are separable but hard to reach
   numerically.** Stage 2 needs the multi-start (Rev-5), and every fit should report convergence. This
   matters for κ, which rides on γ in the joint HN fit.
3. **In burst populations (Moltbook), the comment clock and the thread clock cannot be told apart** beyond
   the event range. Agent-side conclusions should use the identified sum (Rev-6), not the separate clocks.
4. **Hour-of-day at the cell midpoint attenuates hour effects** in large samples (Rev-7). The bias is small
   but detectable.
5. **The reply-back stream (ρ, g_back) is the most robust part of the fit.**

## Proposed before Stage 2 (needs confirmation)

1. Apply Rev-5 (multi-start) and Rev-7 (hour-exact cells). Test Rev-8 (sub-10 s bins) on Moltbook.
2. Rerun Stage 1 for the populations those revisions touch: HN link, the joint HN fit and Moltbook. A full
   rerun is cleaner but costs about 6 hours. Pass rules unchanged.
3. Decide Rev-6: report the identified clock sum for Moltbook.

Stopping here as instructed. The Stage 1 outputs, report and register outcomes are not committed yet.
