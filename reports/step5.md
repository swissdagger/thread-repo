# Step 5: Saturation without scores

**Calibration split only. The outcome is exploratory** and is recorded as R3 in
`docs/results_register.md`. R3's validation test was written into the register, and the method into
`docs/definitions.md` → "Saturation", **before** this step was run. The validation split was not read.

- **Script:** `scripts/step5_saturation.py`, seed 20261107 + 51, B = 200, log `data/step5.log`, peak
  memory 5.8 GB.
- **Rules:** `partitions.json` → `step5`.
- **Outputs:** `outputs/hn/step5_saturation.csv`, `outputs/hn/step5_saturation_bins.csv`,
  `figures/hn/step5_saturation.png`.
- **A first run crashed** on an indexing bug in the bootstrap bookkeeping, before writing anything; it was
  fixed and rerun (`data/step5_failed1.log`).

## What was computed

- **Treatment:** Ask HN roots (30,422 live roots, calibration split). **Control:** HN link roots (674,393
  in the population; the first 150,000 in thread-hash order are used).
- **Events:** top-level comments that are visible (not `dead`) and not written by the root's author, over
  3 days.
- **Candidate answer:** the first such comment. **Score is not used anywhere.**
- **Model:**
  - Each root's 3 days are split at the candidate answer into "before" (ending with that comment, its
    event) and "after" (the later top-level comments).
  - Cells are root × root-age bin (as in Step 4b.1) × state. The model is a conditional Poisson with
    **thread fixed effects**, root-age bin dummies and an "after" indicator.
  - With thread fixed effects, β = 0 under any thread-specific Poisson rate. Thread popularity alone
    cannot produce β ≠ 0; only a change in the rate at the first comment can.
- **Weights:** per-author by the root's author (primary) and raw.
- **Intervals:** 95% thread-bootstrap. Ask HN and link are bootstrapped independently, so the contrast's
  CI comes from independent replicates.

## Key numbers

| | roots | share with a top-level comment by 3 d | median time to first (min) | **β after vs before (per-author)** | HR | raw β |
|---|---|---|---|---|---|---|
| Ask HN | 30,422 | 0.74 | 25.3 | **+0.01 [−0.07, +0.09]** | 1.01 | −0.15 [−0.25, −0.06] |
| HN link | 150,000 | 0.24 | 57.3 | **+1.56 [+1.45, +1.67]** | 4.76 | +1.82 [+1.77, +1.86] |
| **Ask − link (saturation contrast)** | | | | **−1.55 [−1.70, −1.44]** | 0.21 | −1.97 [−2.09, −1.87] |

- **Within a link thread,** the rate of new top-level comments jumps almost 5× once the first one arrives.
- **Within an Ask HN thread,** it stays flat per-author (raw: falls 14%).
- **Relative to the control,** Ask HN's rate after the candidate answer is 0.21× what the link pattern
  would predict. **This passes the pre-stated test (R3) on the calibration split.**

**Figure:** `figures/hn/step5_saturation.png`. It shows the per-author top-level comment rate by root age,
before and after the first comment. The pooled curves mix threads: "after" is dominated by threads hot
enough to get a comment early, so their gap is larger than the within-thread β. The table, not the figure,
is the estimate.

## What this does and does not show

- **The pattern matches the v1 expectation:** saturation for questions, not for links.
- **It does not isolate κ.** The contrast assumes that what the feed does around the first comment is the
  same for both post types. On HN, a link post's first comment tends to coincide with it reaching the
  front page, so part of the link jump may be e_feed changing at that moment rather than popularity
  feedback in F. Ask HN posts are also ranked by votes, but whether their front-page arrival lines up with
  the first comment in the same way was not checked.
- **Two checks that would separate these** (not run; Step 5 was to stop here):
  1. Show HN as a second control.
  2. A dose-response check: does the Ask HN rate step down at the 2nd and 3rd top-level comments too, as
     saturation-by-answer predicts it should not?

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| β after vs before (Ask HN, HN link), saturation contrast | Calibration split, all folds; link capped at 150,000 roots by thread hash | Not validated. R3 test pre-stated: per-author contrast < 0 with CI excluding 0 on validation threads (a calibration quantity, reply timing; a replication test, not a validation metric) |
| Per-bin descriptive hazards | Calibration split | Descriptive only |

## Implications for the model

`docs/model_reference.md` is unchanged.

1. **Keep κ.** Questions stop attracting new top-level comments, relative to links, once a candidate
   answer arrives. This is the first v1 expectation the data support (exploratory).
2. **e_feed responds to activity.** The link jump (HR 4.8 within thread) is larger than any F-side
   popularity term fitted so far. A feed rank that rises with early comments or votes is the more likely
   carrier. The simulation needs e_feed(rank) to respond to early activity, or the saturation contrast
   will be misattributed to κ.
