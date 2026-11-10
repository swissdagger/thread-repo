# Step 6: External-exposure tail

**Calibration split only. The outcome is exploratory** and is recorded as R4 in
`docs/results_register.md`. R4's validation test was written into the register, and the method into
`docs/definitions.md` → "Late tail", **before** this step was run. The validation split was not read.

- **Script:** `scripts/step6_tail.py`, seed 20261107 + 61, B = 200, log `data/step6.log`, peak memory
  6.8 GB.
- **Rules:** `partitions.json` → `step6`.
- **Outputs:**
  - `outputs/combined/step6_tail_bins.csv`: intensity, first-reply hazard and log-normal curve per bin,
    with CIs;
  - `outputs/combined/step6_tail_summary.csv`;
  - `outputs/combined/step6_tail_contrasts.csv`;
  - `figures/combined/step6_tail.png`.

## What was computed

**Reply intensity to 30 days.** For each population and unit (comments; roots), direct replies per parent
per day as a function of parent age, in 40 log-spaced bins from 10 s to 30 d:
- default events: non-self, non-bot, visible;
- censoring-adjusted (weighted replies / weighted observed parent-time);
- per-author (primary) and raw;
- a bin is shown only where ≥ 100 parents are at risk;
- the first-reply hazard is computed the same way for reference.

**Populations:**
- r/philosophy and r/buildapc.
- Moltbook crawl: complete, non-spam threads with the outage handled. It is observable only to 8.4 days.
- Moltbook archive: the cross-check window (posts 2026-02-08 to 02-14, observed to 2026-03-31). It is the
  only Moltbook source that reaches 30 days.

**Tail statistics (pre-stated):**
- the log-log slope of the intensity over 3–30 days, by Poisson maximum likelihood;
- the excess: observed replies in 3–30 d divided by the log-normal reference kernel's prediction. The
  kernel is a log-normal-shaped intensity fitted on 0–3 d, then extrapolated.

## Key numbers (comments, per-author, 95% thread bootstrap)

| population | horizon | 3–30 d slope | excess over log-normal | share of replies after 3 d | first-reply hazard slope |
|---|---|---|---|---|---|
| r/philosophy | 30 d | −1.59 [−1.97, −1.26] | 0.92 [0.80, 1.06] | 2.1% | −1.94 [−2.09, −1.82] |
| r/buildapc | 30 d | −1.29 [−1.40, −1.22] | **2.69 [2.53, 2.85]** | 1.1% | −1.24 [−1.29, −1.20] |
| Moltbook crawl | 8.4 d | −1.77 [−3.44, −0.72] (3–8 d only) | 2.65 [1.69, 4.24] (3–8 d) | 2.2% | −2.27 [−4.93, −0.82] |
| Moltbook archive | 30 d | −2.02 [−2.53, −1.14] | 2.29 [0.69, 6.20] | 8.2% | −1.40 [−1.95, −0.96] |

**Slope contrasts (archive − Reddit)** are not distinguishable from 0:
- vs r/philosophy: −0.43 [−1.10, +0.63];
- vs r/buildapc: −0.72 [−1.23, +0.20].

**Roots** (per-author 3–30 d slope; excess over log-normal):

| population | slope | excess |
|---|---|---|
| r/philosophy | −2.02 [−2.18, −1.92] | 0.25 [0.22, 0.31] |
| r/buildapc | −1.33 [−1.59, −1.10] | 1.36 [1.16, 1.54] |
| Moltbook crawl (3–8 d) | −1.97 [−2.20, −1.75] | 2.07 [1.94, 2.20] |
| Moltbook archive | −1.37 [−1.76, −1.08] | not meaningful (below) |

For archive roots, the log-normal fitted on 0–3 d captures only the feed-polling burst (median 1.3 min,
σ 1.8), so its extrapolation is essentially zero and the excess (≈ 2,700) is not meaningful. Archive roots
also show isolated spikes of late comments (e.g. ~100 events in single bins at 5 and 13 days) that look
like batch commenting on old posts. The archive has no spam filter.

**Against the pre-stated rule (R4)** — **fails on both parts.**
- **(i)** Neither subreddit's comment tail is flatter than 1/t (both slopes < −1, CIs below −1). Only
  r/buildapc exceeds the log-normal extrapolation.
- **(ii)** The Moltbook archive slope is not distinguishable from either Reddit slope.

## Figure

`figures/combined/step6_tail.png`: all-reply intensity by parent age, per-author, with 95% thread-bootstrap
bands. Comments are on the left, roots on the right. Dotted lines are the 0–3-day log-normal fits,
extrapolated; the 3–30-day window is shaded.

Visible features beyond the test:
- A dip at about 1 day in every human curve (daily cycle).
- r/philosophy's comment intensity levels off at ≈ 2×10⁻⁴ replies/day from about day 13 (local slope
  ≈ −0.5). It sits above the log-normal by 2–4× at 19–26 days, but those bins have wide CIs. This was
  seen after the fact, not pre-stated, and should not be read as a finding without a registered test.
- Moltbook crawl roots show a bump at 1–2 days. It is not the outage: roots created before the outage
  are censored at its start, and roots created during it are excluded. Not investigated.

## What this shows

- **No population has a flat late tail at 3–30 days.** Reply intensity falls faster than 1/t everywhere.
  What remains after 3 days is 1–2% of human replies.
- **The log-normal reference kernel is not uniformly too thin.**
  - It is about right for r/philosophy comments (excess 0.92) and over-predicts r/philosophy roots (0.25).
  - It under-predicts r/buildapc (2.7×) and Moltbook comments (2.3–2.7×).
  - "A heavier tail than the kernel" is therefore population-specific, not a Reddit-vs-agent difference.
- **The prediction "Moltbook lacks the tail" is not supported.** In the archive, 8% of comment replies
  arrive after 3 days, a larger share than on Reddit. Late agent activity exists, even though σ = 0 was
  assumed for agents. Its timing (spikes on old posts) suggests scripted batch activity rather than search.

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| Binned intensities, first-reply hazards | Calibration split, all folds | Descriptive |
| 3–30 d slope, log-normal fit on 0–3 d, excess | Calibration split, all folds | Not validated. R4 test pre-stated (validation threads; reply timing is a calibration quantity, so this is a replication test, not a validation metric) |

## Implications for the model

`docs/model_reference.md` is unchanged.

1. **σ is small for humans at 3–30 days**, and is not identified as a flat floor in either subreddit. For
   simulation, σ can be set from the after-3-day share of replies (1–2% of comment replies) rather than
   treated as a distinct flat process.
2. **σ = 0 for agents is contradicted** by the archive. 8% of agent comment replies come after 3 days, and
   old posts receive batched comments. Whether this is agents (a re-fetch path not in the study-week
   skill) or scripts should be settled before agent σ is fixed. In the later regime the archive covers, the
   exposure configuration differs from the study week anyway.
3. **The log-normal tail is not a universal shape.** If late replies matter for a validation target
   (thread size shape), use the log-normal for timing and add an empirical late-tail correction per
   population. Do not let the kernel's tail imply σ.
4. **Status of the v1 expectation** "Reddit flat late tail, Moltbook none": **failed** (exploratory). It
   is recorded in R4; `model_reference.md`'s status table is left as is for the user to update.
