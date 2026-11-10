# Step 4c: Reply-back, Show HN without the OP, matched r/Fantasy control

**Calibration split only. Every outcome is exploratory** and is registered as R20–R22 in
`docs/results_register.md`. Each validation test was written into the register **before** the run; the
calibration outcome is recorded against that pre-stated rule. The validation split was not read.

Script: `scripts/step4c_extensions.py` (seed 20261107 + 41/42/43, B = 200; log `data/step4c.log`, peak
memory 9.2 GB). Shared code: `forum_model/age_fe.py` (the 4b.1 cell design), and
`forum_model/fe_poisson.FEPoissonBoot`, which prepares a conditional Poisson once and refits it under
thread-bootstrap weights. Its unit-weight fit equals `fit_fe_poisson` to 1e-14.

**Intervals** are 95% thread-clustered bootstrap intervals (multinomial thread counts, B = 200), per
`CLAUDE.md`. Sandwich SEs are in the CSVs alongside.

**Per-author weighting is the primary version.** For the FE Poisson, the per-author fit solves the weighted
Poisson score equations: y → w·y, offset → offset + log w, with w = 1/(author's parents in the fold);
unknown authors are dropped. Raw is reported alongside.

---

## 4c.1 Reply-back and the comment-age effect

**Design.** This is the 4b.1 thread × time-bin fixed-effects model on the same capped parents: whole
threads in hash order, ≤ 400k parents per population. It is refitted with one change: replies written by
the author of the comment's parent are removed as events. A reply-back is "the person this comment
replied to, replying back". For top-level comments that person is the OP. Δ = age coefficient without
reply-backs − age coefficient with all replies, computed on the same bootstrap replicates.

| platform | population | reply-backs, share of replies | age, all replies | age, no reply-backs | **Δ (per-author)** | Δ (raw) |
|---|---|---|---|---|---|---|
| Reddit | r/philosophy | 39% | −0.41 [−0.44, −0.39] | +0.16 [+0.12, +0.19] | **+0.57 [+0.55, +0.59]** | +0.52 [+0.50, +0.54] |
| Reddit | r/buildapc | 69% | −0.81 [−0.88, −0.75] | −0.70 [−0.80, −0.58] | **+0.12 [+0.06, +0.18]** | +0.28 [+0.21, +0.34] |
| HN | Ask HN | 30% | −0.43 [−0.45, −0.41] | −0.34 [−0.36, −0.31] | **+0.09 [+0.08, +0.10]** | +0.11 [+0.11, +0.12] |
| HN | HN link | 22% | −0.52 [−0.55, −0.49] | −0.39 [−0.42, −0.36] | **+0.13 [+0.12, +0.14]** | +0.16 [+0.16, +0.17] |
| HN | Show HN | 40% | −0.63 [−0.64, −0.61] | −0.41 [−0.43, −0.38] | **+0.22 [+0.20, +0.24]** | +0.24 [+0.23, +0.26] |

Columns 4–5 are per-author, with 95% thread-bootstrap intervals. Raw age coefficients with all replies
reproduce Step 4b.1 exactly (−0.69, −0.75, −0.42, −0.56, −0.57).

**Against the pre-stated rule (R20).** The rule was: Δ > 0 in both subreddits, and each Reddit Δ above
every HN Δ.
- Per-author it **fails**: r/buildapc's +0.12 is below HN link (+0.13) and Show HN (+0.22).
- Raw it passes.
- Per `CLAUDE.md`, the per-author version is the comparison, so the notification-specific reading is not
  supported on calibration.

**What the numbers do show:**
- **Reply-backs carry part of the comment-age effect everywhere, including HN**, which has no reply
  notification (Δ +0.09 to +0.22). A channel shared by both platforms is needed. The natural candidate is
  e_own: people return to threads they have a stake in, here a conversation they are in.
- **r/philosophy is the outlier.** Removing reply-backs turns its within-thread age effect from −0.41 to
  +0.16. Among non-reply-back replies, an older comment is *not* disfavoured at a fixed thread age. In a
  discussion forum, the age effect is almost entirely the back-and-forth of conversations.
- **r/buildapc is the opposite.** 69% of its replies are reply-backs (asker and helper), yet removing them
  barely changes the per-author age effect (−0.81 → −0.70). Help replies of every kind go to fresh
  comments.
- **Without reply-backs, the residual age effect** (F's novelty/self-excitation term proper) is −0.34 to
  −0.70 per-author, except r/philosophy (≈ 0 or positive).

## 4c.2 Show HN without the OP's replies

**Design.** This is the 4b.2 proxy (b) model: first 30 minutes, arrival rank among early siblings,
piecewise-exponential first-reply hazard. The parents are the same as 4b.2 (2,658 comments, 986 threads).
First replies were recomputed after removing replies by the root's author (`forum_model.units.refirst`).

- **The OP writes 64% of the first replies** these early comments receive. Removing them leaves 230
  first-reply events, down from 620.

| model (per-author) | β, all replies | β, OP replies excluded | difference (all − excluded) |
|---|---|---|---|
| position only | +0.71 [+0.59, +0.90] | +0.43 [−0.03, +0.71] | +0.28 [−0.03, +0.82] |
| + lateness + depth | +0.27 [+0.07, +0.46] | +0.15 [−0.64, +0.62] | +0.12 [−0.31, +0.77] |

Raw-weighted results are similar (`outputs/combined/step4c_showhn_op.csv`).

**Against the pre-stated rule (R21)** — not passed. The excluded-β CI includes 0, but the difference CI
also includes 0. Removing the OP halves the point estimate, but with 230 events the data cannot
distinguish "OP sequencing explains it" from "it persists". **The pre-stated validation test is
underpowered:** the validation split holds about a quarter as many threads. One option is to pool Show HN
with Ask HN, where OPs also answer, to gain power. That would be a new test, registered before it is run.

## 4c.3 r/Fantasy contest threads vs matched non-contest threads

**Design.** This is the 4b.1 thread × time-bin fixed-effects model on top-level parents. Contest-mode
threads are compared with non-contest r/Fantasy threads with ≥ 20 comments (the 4b.2 control
definition, ≤ 400k parents).

| thread set | parents / threads | age, per-author | age, raw |
|---|---|---|---|
| contest | 15,751 / 743 | −0.26 [−0.39, −0.10] | −0.17 [−0.23, −0.09] |
| non-contest, 20+ comments | 399,998 / 15,587 | −0.15 [−0.19, −0.11] | −0.20 [−0.22, −0.17] |
| **contest − non-contest** | | **−0.11 [−0.23, +0.04]** | +0.03 [−0.03, +0.11] |

**Against the pre-stated rule (R22)** — not passed. The contest coefficient is no less negative than the
matched control's. Randomising display order does not detectably change the comment-age effect. That is
consistent with 4b.1, where adding early display position moved the age coefficient by ≤ 0.003.

---

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| 4c.1 age coefficients with / without reply-backs, Δ | Calibration split, all folds, ≤ 400k parents per population (same as 4b.1) | Not validated. R20 test pre-stated (replication on validation threads; calibration quantity) |
| 4c.2 Show HN position β with / without OP replies | Calibration inner fit fold (same parents as 4b.2) | Not validated. R21 test pre-stated; flagged underpowered |
| 4c.3 contest vs non-contest age coefficient | Calibration split, all folds | Not validated. R22 test pre-stated |

## Implications for the model

`docs/model_reference.md` is unchanged.

1. **The comment-age effect is partly dialogue reply-back on every platform, not only Reddit's inbox.** In
   λ = A·E·F, a reply-back should come through the exposure of the person replied to:
   - e_reply where notifications exist;
   - e_own for a conversation the user is in, on HN, which has no notifications.

   It should not come through F's novelty term. Fitting exp(−age/τ) on all replies overstates novelty
   decay by 0.09–0.22 on HN and by 0.12–0.57 on Reddit (per-author).
2. **F's novelty term proper is population-dependent.** It is ≈ 0 in r/philosophy once reply-backs are
   removed, and −0.34 to −0.70 elsewhere. A single τ across populations would misfit discussion forums.
3. **Display order does not produce the age effect** (4b.1 position control; 4c.3 contest vs matched
   control). This is consistent with H1's human-side evidence that position does not absorb time decay.
4. **Show HN's positive position effect stays unexplained.** It is consistent with OP sequencing, but not
   established. e_thread for Show HN should not be read from proxy (b) until this is resolved.
