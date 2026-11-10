# Step 5b: Saturation identification

**Calibration split only; exploratory.** The tests (R23–R25) and the rules (`partitions.json` → `step5b`,
`docs/definitions.md` → "Saturation identification") were committed in `e97aa6e` **before** this step was
run. The validation split was not read.

- **Script:** `scripts/step5b_identification.py`, seed 20261107 + 52 (+1 for the placebo), B = 200.
- **Logs:** `data/step5b_dose.log` (peak memory 5.7 GB), `data/step5b_placebo.log` (0.5 GB).
- **Outputs:** `outputs/hn/step5b_dose.csv`, `step5b_show.csv`, `step5b_placebo.csv`,
  `step5b_placebo_summary.csv`.

**Intervals** are 95% thread-bootstrap intervals, per-author unless noted. For the placebo they are
2.5–97.5% ranges over 20 simulations.

## 5b.1 Dose-response (R23)

The Step 5 model with steps at the 1st, 2nd and 3rd top-level comments. Each event belongs to the state it
ends. Coefficients are log hazard changes at each comment.

| | at 1st comment | at 2nd | at 3rd |
|---|---|---|---|
| Ask HN | −0.29 [−0.37, −0.19] | −0.04 [−0.11, +0.02] | +0.95 [+0.88, +1.01] |
| HN link | +0.69 [+0.56, +0.82] | +1.07 [+0.97, +1.19] | +0.28 [+0.20, +0.34] |
| Show HN | +0.12 [+0.02, +0.23] | +0.81 [+0.70, +0.93] | +0.31 [+0.21, +0.38] |
| **Ask − link** | **−0.98 [−1.13, −0.85]** | **−1.11 [−1.25, −1.00]** | +0.67 [+0.58, +0.78] |
| Ask − Show | −0.42 [−0.54, −0.29] | −0.85 [−0.97, −0.72] | +0.64 [+0.54, +0.75] |

- Ask − link, step-1 contrast minus step-2 contrast: +0.14 [−0.08, +0.36].
- Ask − link, step-1 contrast minus step-3 contrast: −1.65 [−1.84, −1.48].

**Against the pre-stated rule — fails.** Relative to links, Ask HN is damped as much at the 2nd top-level
comment as at the 1st. Under the pre-stated reading this is general dampening or feed dynamics, not
saturation by the *answer*. Ask − Show tells the same story: the contrast is more negative at the 2nd
comment than at the 1st.

The large positive step at the 3rd comment in every population (and its Ask − link sign flip) is not
interpreted. The placebo below shows this estimator has bias from within-bin hazard shape, and the
multi-step version was not placebo-checked.

## 5b.2 Show HN as a second control (R24)

| | β after vs before | raw |
|---|---|---|
| Show HN (28,744 roots) | +0.73 [+0.64, +0.82] | +0.70 [+0.61, +0.77] |
| **Ask − Show** | **−0.72 [−0.85, −0.57]** | −0.85 [−0.97, −0.72] |
| Show − link | −0.83 [−0.95, −0.68] | −1.12 [−1.22, −1.04] |

Ask HN and HN link are reproduced from Step 5 (+0.01 and +1.56).

**Against the pre-stated rule — passes.** Ask HN is damped relative to Show HN too, though by half as much
as relative to links. Show HN sits between the two.

## 5b.3 Synthetic placebo (R25)

**Simulation.**
- Threads: 30,422 "Ask-like" and 150,000 "link-like" roots.
- Comment rate: A·a_g·g(t)·[1 + (M − 1)·spike], with no saturation and no popularity feedback.
  - g is a log-normal shape (median 2 h, σ 1.5).
  - a_g ~ Gamma(k).
  - A spike occurs with probability π, starts log-uniformly between 5 min and 12 h, and lasts 2 h.
- For each population, (A, k) are matched to its mean top-level comments per root and its share with ≥ 1
  comment (Ask 6.42 / 0.736; link 1.755 / 0.236). The simulated values match within 1%.
- The Step 5 estimator is run unchanged.

| π | M | placebo β Ask | placebo β link | **placebo Ask − link** |
|---|---|---|---|---|
| 0.25 | 3 | −0.49 | −0.71 | +0.22 [+0.19, +0.24] |
| 0.25 | 10 | −0.17 | −0.39 | +0.22 [+0.17, +0.26] |
| 0.25 | 30 | +0.59 | +0.29 | +0.30 [+0.24, +0.34] |
| 1.0 | 3 | −0.36 | −0.59 | +0.23 [+0.20, +0.27] |
| 1.0 | 10 | +0.35 | +0.06 | +0.28 [+0.24, +0.32] |
| 1.0 | 30 | +1.18 | +0.84 | +0.35 [+0.31, +0.39] |
| | | | **observed** | **−1.55 [−1.70, −1.44]** (per-author), −1.97 raw |

**Against the pre-stated rule — passes.** In every setting, exposure spikes without saturation give a
*positive* Ask − link contrast. The observed negative contrast lies far outside every placebo range.

**What the placebo also shows.** With nothing happening at the first comment, the Step 5 β ranges from
−0.71 to +1.18 depending on volume and spike setting. The estimator is biased: coarse root-age bins meet a
steep within-bin hazard, and the before/after split falls at systematically different points within a bin.
So:
- single β values (e.g. link HR 4.8) are not hazard ratios;
- the contrast is biased upward (+0.2 to +0.35) in these settings, i.e. against finding saturation.

## Reading

- Spikes in exposure cannot produce the Ask − link contrast (R25), and Show HN confirms the direction
  (R24).
- The dampening is not specific to the first (answer) comment (R23): it is as strong at the second.
- On calibration data, "Ask HN threads are damped relative to link and Show threads after early comments"
  is supported. "κ: saturation once the question is *answered*" is not distinguished from a broader
  question-thread dampening, or from feed dynamics that differ by post type.

Status line in `model_reference.md`, as set on confirmation: "Partly supported (exploratory): question
threads are damped after early top-level comments (survives the synthetic placebo and the Show HN control),
but not specifically at the first one."

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| Dose-response steps (Ask HN, HN link, Show HN) | Calibration split; link capped at 150,000 roots by thread hash | Not validated. R23 test pre-stated |
| Show HN β and contrasts | Calibration split | Not validated. R24 test pre-stated |
| Placebo β and contrasts | Simulated data matched to calibration volumes | Simulation only. R25 |

## Implications for the model

`docs/model_reference.md`: status line updated on confirmation; κ respecified as depletion (see the reference).

1. **κ as specified (a step at the answer) is not identified by these data.** A question-thread dampening
   term that does not single out the first comment fits the dose-response as well.
2. **The Step 5-type estimator needs finer root-age bins, or a placebo-calibrated correction,** before any
   single β is used as a parameter. The placebo machinery (`run_placebo`) can calibrate it.
