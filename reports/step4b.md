# Step 4b: Age within thread, position with controls, root timing, speed reconciliation

**Calibration split only. Every outcome below is exploratory** and is registered as R16–R19 (with updates to
R2, R6 and R11) in `docs/results_register.md`, each with a pre-stated validation-split test. The validation
split was not read.

Script: `scripts/step4b_extensions.py` (subcommands `age_fe`, `position_b`, `roots_early_late`, `speed`,
`ties`). Rules for each run are recorded in `partitions.json` → `step4b`. Seed 20261107. Logs are in
`data/step4b_*.log`. Peak memory was 6.4–8.9 GB per run.

## Decisions applied

1. **Proxy (c)** is a placebo check only. It is used below as a placebo in 4b.2 and as one population in
   4b.1. No contest-mode data was added.
2. **Proxy (a)** was refitted with random tie-breaking. It is not materially different from oldest-first in
   either subreddit, so it is kept for both, labelled look-ahead-affected (below).
3. **μ horizon:** 3 days primary, 7 days reported (`docs/definitions.md`, `partitions.json` → `mu_horizon`).
4. **Register renamed** H1–H15 → R1–R15, with a "design hypothesis (spec H0–H5)" column. It reads
   *pending* for every row: the specification's H0–H5 text is not in the repo (see the end of this report).

---

## 4b.1 Comment age with thread × time-bin fixed effects

**Design.** Each calibration-split parent comment is followed over thread-age bins (0–10 min, 10–30 min,
30 min–1 h, 1–2 h, 2–4 h, 4–8 h, 8–16 h, 16–32 h, 32–72 h; 3-day horizon). One row is one
(comment, bin) cell: y = direct replies in the bin (default filters from `docs/definitions.md`), offset =
log of the comment's exposure in the bin. Covariates:

- log(1 + comment age in minutes) at the cell midpoint;
- depth dummies;
- OP comment.

Three models: no fixed effects (common intercept + bin dummies), thread FE (+ bin dummies), and thread ×
time-bin FE. The thread × time-bin model is a conditional likelihood, so all thread-level exposure (feed
rank, popularity, thread age) cancels within each cell. Display position is added where it is known: log
arrival rank among siblings, active only in the first-30-minute bins. Up to 400k parents per population,
taken as whole threads in hash order. CIs are 95%, thread-clustered sandwich.

**Coefficient on log(1 + comment age), with 95% CI:**

| population | parents / threads | no FE | thread FE | thread × time-bin FE | + early position |
|---|---|---|---|---|---|
| r/philosophy | 399,702 / 11,604 | −1.02 [−1.03, −1.00] | −0.86 [−0.87, −0.84] | **−0.69 [−0.71, −0.67]** | −0.69 |
| r/buildapc | 399,989 / 50,088 | −2.34 [−2.37, −2.32] | −1.19 [−1.21, −1.16] | **−0.75 [−0.79, −0.71]** | −0.75 |
| Ask HN | 399,843 / 22,544 | −0.92 [−0.94, −0.90] | −0.71 [−0.73, −0.70] | **−0.42 [−0.44, −0.40]** | −0.42 |
| HN link | 399,927 / 10,599 | −1.40 [−1.43, −1.36] | −0.73 [−0.74, −0.71] | **−0.56 [−0.58, −0.55]** | −0.56 |
| Show HN | 180,393 / 15,294 | −1.04 [−1.06, −1.03] | −0.83 [−0.84, −0.82] | **−0.57 [−0.59, −0.56]** | −0.57 |
| r/Fantasy contest, top-level | 15,751 / 743 | −0.27 [−0.35, −0.18] | −0.25 [−0.32, −0.18] | **−0.17 [−0.23, −0.10]** | – (order random) |

- **Fixed effects absorb 30–70% of the raw age slope** (r/philosophy 32%, Show HN 45%, Ask HN 54%, HN
  link 60%, r/buildapc 68%). That share is thread-level: the thread's audience decays, and every comment
  in it ages with it.
- **The rest is a comment-age effect.** At a fixed thread age, within the same thread, a comment 10× older
  gets 0.38–0.56× the reply rate (10^−0.42 to 10^−0.75), since age enters as log(1 + minutes). This is the
  self-excitation kernel separated from thread novelty.
- **Early display position does not absorb it.** Adding log arrival rank in the first 30 minutes changes
  the age coefficient by ≤ 0.003. The position term itself is: r/philosophy −0.15 [−0.23, −0.07],
  r/buildapc −0.11 [−0.14, −0.08], Ask HN −0.16 [−0.23, −0.10], HN link −0.23 [−0.31, −0.15], Show HN
  +0.10 [−0.01, +0.21].
- **r/Fantasy contest threads** have the smallest age effect (−0.17). These are social or promotional
  threads, and no matched non-contest series was fitted in 4b.1, so the smaller value is not attributed to
  display.
- **Depth and OP terms** (`outputs/combined/step4b_age_fe.csv`):
  - Within thread × time-bin, OP comments are −0.19 to −0.36 on HN (link, Show HN, Ask HN), 0.00 on
    r/buildapc and +0.05 on r/philosophy.
  - Relative to top-level comments, deeper comments get fewer replies on HN link and Show HN (−0.45 to
    −0.58) and at depth 2–3 on r/buildapc (−0.25 to −0.29). Elsewhere depth effects are small or mixed in
    sign (−0.12 to +0.12).

---

## 4b.2 Position-bias exponent under proxy (b), with controls

**Design.** This is the Step 4 proxy (b) set-up with controls added:

- **Parents:** comments created in the first 30 minutes of the thread, with ≥ 2 such siblings.
- **Position:** arrival rank among those siblings.
- **Model:** piecewise-exponential first-reply hazard over 0–30 min (13 log-spaced bins), censored at
  thread start + 30 min. Each bin effect α_k is profiled out; the β terms are fitted by Newton.
- **Controls:** log(1 + minutes since thread start) and depth dummies. A depth dummy is used only if it has
  ≥ 10 first-reply events; Show HN depth ≥ 3 had none, and its coefficient diverged in a first run.
- **Uncertainty:** thread-clustered sandwich SEs.
- **Weighting:** per-author (reported) and raw.

Without controls, the model reproduces the Step 4 proxy (b) β to three decimals.

**β on log arrival rank, per-author (hazard ∝ rank^β), with 95% CI:**

| population | parents / threads | β, position only | β, + lateness + depth | lateness coefficient | top-level only, + controls |
|---|---|---|---|---|---|
| r/philosophy | 7,237 / 2,678 | +0.04 [−0.06, +0.13] | **+0.00 [−0.11, +0.12]** | +0.12 [+0.02, +0.21] | +0.03 [−0.09, +0.14] |
| r/buildapc | 1,145,824 / 380,895 | −0.32 [−0.35, −0.29] | **−0.18 [−0.22, −0.13]** | −0.12 [−0.14, −0.11] | −0.29 [−0.33, −0.25] |
| Ask HN | 13,571 / 3,750 | −0.27 [−0.38, −0.17] | **−0.32 [−0.42, −0.21]** | +0.35 [+0.23, +0.46] | −0.31 [−0.42, −0.20] |
| HN link | 79,682 / 13,132 | +0.06 [+0.03, +0.09] | **−0.06 [−0.09, −0.03]** | +0.48 [+0.42, +0.54] | −0.01 [−0.05, +0.02] |
| Show HN | 2,658 / 986 | +0.71 [+0.55, +0.87] | **+0.27 [+0.09, +0.44]** | +0.62 [+0.48, +0.76] | +0.31 [+0.12, +0.50] |
| **Placebo:** r/Fantasy contest, top-level | 813 / 283 | +0.25 [−0.15, +0.64] | **+0.30 [−0.12, +0.73]** | −0.18 [−0.49, +0.14] | (top-level) |
| r/Fantasy control (non-contest, 20+ comments), top-level | 39,352 / 10,454 | −0.26 [−0.32, −0.21] | **−0.19 [−0.26, −0.12]** | −0.14 [−0.20, −0.07] | (top-level) |

Raw-weighted values are close (`outputs/combined/step4b_position_b.csv`).

- **The lateness control matters, and in both directions.**
  - On HN, comments arriving later in the first 30 minutes are replied to *faster* (lateness +0.35 to
    +0.62): the thread's audience is still growing. Controlling for this makes β more negative (HN link
    +0.06 → −0.06; Ask HN −0.27 → −0.32).
  - On r/buildapc, lateness is negative, and the control halves β (−0.32 → −0.18).
  - Step 4's uncontrolled β therefore mixed position with thread growth or decay.
- **With controls, the position exponent splits by population type.**
  - Help and Q&A (r/buildapc, Ask HN): −0.18 to −0.32, i.e. 0.80–0.88× per doubling of rank.
  - Discussion (r/philosophy, HN link): 0 to −0.06. r/philosophy has only 7k early parents.
  - Show HN: positive (+0.27). A display mechanism cannot produce this. A candidate is the OP answering
    commenters in sequence, but that was not tested.
- **Placebo.** In contest-mode threads (random display order), early arrival rank has no detectable effect.
  - Per-author: +0.30 [−0.12, +0.73]. The CI is wide: 813 parents in 283 threads.
  - Contest minus control: +0.49 [+0.06, +0.93] per-author, +0.23 [−0.07, +0.52] raw.
  - The per-author contrast points to display order as the source of the arrival-rank effect in ordinary
    r/Fantasy threads. The raw contrast does not exclude zero, and contest threads are social or
    promotional threads, so this is suggestive, not established.
- **This differs from Step 4 proxy (c)**, where contest β was −0.22 [−0.51, −0.08] using arrival rank over
  the whole thread life, without a lateness control. The 30-minute window with a lateness control removes
  that effect. The two designs are reported side by side in R11.

---

## 4b.3 Early/late split for roots

Root posts observed ≥ 1 day, default filters (live roots; Moltbook complete, non-spam threads, outage
handled). The table gives the share of roots whose first reply falls in each window. CIs are 95%, thread
bootstrap, B = 200 (`outputs/combined/step4b_roots_early_late.csv`).

| population | roots | < 10 min (per-author) | 10–60 min | 1 h–1 d | ratio to Moltbook < 10 min | 10–60 min | 1 h–1 d |
|---|---|---|---|---|---|---|---|
| Moltbook | 70,728 | **0.879** [0.875, 0.884] | 0.027 | 0.033 | – | – | – |
| r/philosophy | 29,570 | 0.103 | 0.105 | 0.078 | 0.12 [0.11, 0.12] | 3.9 [3.5, 4.3] | 2.3 [2.1, 2.5] |
| r/buildapc | 940,721 | 0.602 | 0.226 | 0.059 | 0.68 [0.68, 0.69] | 8.4 [7.7, 9.1] | 1.8 [1.7, 1.9] |
| Ask HN | 30,422 | 0.172 | 0.339 | 0.206 | 0.20 [0.19, 0.20] | 12.6 [11.6, 13.8] | 6.2 [5.8, 6.6] |
| HN link | 674,393 | 0.030 | 0.096 | 0.094 | 0.03 [0.03, 0.03] | 3.6 [3.3, 3.9] | 2.8 [2.6, 3.0] |
| Show HN | 28,744 | 0.037 | 0.103 | 0.182 | 0.04 [0.04, 0.05] | 3.8 [3.5, 4.2] | 5.5 [5.0, 5.8] |

Raw shares are within 0.02 of per-author shares, except r/philosophy 1 h–1 d (0.107 raw).

**Roots run opposite to comments.** For comments (Step 3b, audit), Moltbook was 1.5–4× slower than humans
in the first 10 minutes and 9–18× slower after an hour. For roots:

- Moltbook beats every human population in the first 10 minutes, by 1.5× (r/buildapc) to 30× (HN link).
- It falls 2–13× below humans from 10 minutes on.

This is the feed-polling signature from Step 2b: agents read the new-posts feed, not threads. Root and
comment arrivals need separate exposure channels.

---

## 4b.4 Speed reconciliation

The full table and the four choices that define a speed number are now in `docs/definitions.md` → "Speed
reconciliation" (`outputs/combined/step4b_speed.csv`, `step4b_speed_naive.csv`). Main points:

- **The published 0.8-minute Moltbook half-life is a root quantity.** It is matched only by:
  - the KM median first reply to a post: 0.50 min (crawl), 0.47 min (archive);
  - the archive root Weibull half-life: 0.45 min;
  - the naive median of all replies to posts: 1.3–2.8 min.
- **Moltbook comments are not that fast.** The KM median first reply to a comment is:
  - 9.0 min by default;
  - 7.9 min with spam posts included, 11.7 min with incomplete threads, 8.8 min with no outage handling;
  - 1.3 min with self-replies included. Self-replies are the only filter that moves it by more than
    3 minutes.
- **Like-for-like comparisons.**
  - Comments: Moltbook 9.0 min against 9.2 (r/buildapc) to 71 min (Ask HN) for humans, i.e. 1–8× faster.
  - Roots: Moltbook 0.5 min against 5–106 min for humans, i.e. 10–210× faster.
  - The published ~195× (0.8 min vs 2.6 h) compares a Moltbook root quantity with a pooled human one.
    2.6 h is within the range of naive pooled parent-to-child medians on slow populations (r/philosophy
    1.9 h; HN 1.4 h; replies to posts 3–4.7 h), but depends on a subreddit mix we cannot reproduce.
- **Statistic choice.**
  - Exponential half-lives (24–804 min) are tail-dominated and misleading; every Weibull shape is
    0.25–0.61.
  - Weibull half-life > log-normal median > KM median in almost every row.
  - For Moltbook roots, the Weibull half-life (18.7 min) is 37× the KM median (0.50 min): 97% of roots get
    a reply, mostly in seconds, and a long tail pulls the Weibull. The KM median is the robust summary.
  - The published Weibull shape 0.11 is not reproduced. The lowest shape is 0.25, for Moltbook roots with
    every filter removed, and that fit is unstable (Weibull half-life 1,622 min against a KM median of
    1.1 min).
- **Archive vs crawl.** The archive cross-check window (posts 2026-02-08 to 02-14) is a different regime:
  - 1.05M posts, but comments on only 6% of them;
  - stored comments are 98% of the count the posts report, so this is real sparsity, not truncation;
  - root first-reply speed agrees with the crawl (0.47 vs 0.50 min); comment speed is slower (17.1 vs
    9.0 min).
- **Human filters.**
  - Bots and self-replies change human comment medians by < 1.5 min.
  - OP self-comments dominate root speed where OPs comment first: r/philosophy 26.4 → 6.9 min (abstract
    rule), Show HN 105.5 → 7.4 min, HN link 57.3 → 40.5 min.

---

## Proxy (a) tie-break sensitivity (decision 2)

Proxy (a) was refitted with ties in final score broken at random rather than oldest-first. The bootstrap was
B = 200, as in Step 4. The materiality rule is in `partitions.json`, and results are in
`outputs/combined/step4b_proxy_a_ties.csv`.

| subreddit | weighting | β oldest-first | β random | flattening oldest | flattening random | material? |
|---|---|---|---|---|---|---|
| r/philosophy | per-author | −0.362 [−0.384, −0.337] | −0.357 [−0.379, −0.332] | +0.021 | +0.020 | no |
| r/philosophy | raw | −0.363 | −0.359 | +0.019 | +0.018 | no |
| r/buildapc | per-author | −0.268 [−0.372, −0.142] | −0.260 [−0.365, −0.132] | +0.016 | +0.015 | no |
| r/buildapc | raw | −0.254 | −0.236 | +0.009 | +0.008 | no |

Across all four rows, \|Δβ\| ≤ 0.019 (tolerance ≥ 0.05) and \|Δflattening\| ≤ 0.0011 (tolerance 0.01).
The flattening CIs fall on the same side of 0. **Proxy (a) is kept for both subreddits**, still labelled
look-ahead-affected (it uses final score).

---

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| 4b.1 age, depth, OP and position coefficients (FE Poisson) | Calibration split, all folds, ≤ 400k parents per population | Not validated. Registered as R16, a replication test on validation threads (calibration quantity: reply timing) |
| 4b.2 position exponent, lateness and depth coefficients (PEM) | Calibration inner **fit** fold (cap 2M parents) | Not validated. R11 updated, with a replication test |
| 4b.3 root early/late shares and ratios | Calibration split, all folds (descriptive) | Not validated. R17 |
| 4b.4 half-lives, log-normal medians, Weibull shapes | Calibration inner **fit** fold | **Never a validation metric** (CLAUDE.md). R18 is a replication check only |
| 4b.4 KM medians, p_reply, naive medians | Calibration split, all folds (descriptive) | As above |
| Proxy (a), random ties | Calibration inner fit fold, held-out fold for held-out gain (as in Step 4) | R19 |

The 20% validation split was not read by any 4b script. All five validation-list quantities (depth, size
shape, reciprocity, OP return, bimodality) are untouched.

## Changes to shared code

- **`forum_model/fe_poisson.py`**
  - The Newton solver now uses step-halving on the conditional log-likelihood. Full Newton steps
    overshot to |β| ~ 10^15 on the (comment, bin) design.
  - It drops columns absorbed by the fixed effects, reporting them as NaN.
  - It accepts a separate clustering variable.
  - The converged optimum is unchanged. Step 3b's K4 results were not recomputed: those fits converged,
    and the change only alters the iteration path.
- **`forum_model/pem.py`:** `fit_multi` handles several static covariates with a thread-clustered
  sandwich. Checked by simulation: it recovers −0.256 / 0.297 against true −0.25 / 0.3.
- **`forum_model/units.py`:**
  - variant switches for the speed table: `include_self`, `include_bots`, `include_spam`, `segments`;
  - `moltbook_archive_population`.
- **`scripts/step4_position.py`:** a `ties` option for proxy (a).
- **Bug caught before reporting.** In the first 4b.2 run, a tag name collided with the thread count and
  overwrote it, so the contest and control labels were lost. A degenerate depth dummy also appeared for
  Show HN. Both were fixed and 4b.2 was rerun. log-position estimates moved by ≤ 0.007.

## Implications for the model

`docs/model_reference.md` is unchanged. These are proposals.

1. **Two time effects, both needed, in different terms** (corrected on confirmation). The share of the age
   slope removed by thread × time-bin fixed effects (30–70%) is the thread's audience decaying. That is
   feed-level exposure: E, through e_feed, not F. The within-thread comment-age effect that survives the
   fixed effects (elasticity −0.4 to −0.75) is F's novelty/self-excitation term. Position in the first 30
   minutes does not substitute for either.
2. **Position term.** e_thread(pos) ∝ pos^β with β ≈ −0.2 to −0.3 for help and Q&A populations and ≈ 0
   for discussion populations, once a lateness control is included. The single cross-population curve from
   Step 4 (0.73–0.86× per doubling) overstated the effect where thread growth or decay was mixed in. Show
   HN needs an OP-sequencing term before its β can be read as exposure.
3. **Roots need a feed-exposure channel separate from thread exposure.** On Moltbook it is fast polling
   (88% of roots get a reply within 10 minutes); for humans it is the front page or sort. Using one kernel
   for roots and comments would mix a 0.5-minute and a 9-minute process.
4. **Speed numbers carry their definition.** Comparisons with Eziz 2026 use the root quantity, against
   which our crawl and archive agree (0.47–0.50 min). The 0.11 Weibull shape prior should be dropped;
   calibration shapes are 0.36–0.42 on Moltbook under default filters.

## Open items needing a decision

1. **Design hypotheses H0–H5.** The specification text is not in the repo (`docs/model_reference.md` and
   `docs/brief.md` do not contain it), so every R entry's mapping reads *pending*. Please add the H0–H5
   text (or the file it lives in), and I will fill in the one-line mappings.
2. **Show HN positive position effect.** One option is to test whether excluding OP replies as events
   removes it (one extra PEM fit).
3. **4b.1 for r/Fantasy.** Optionally, add non-contest r/Fantasy threads to 4b.1 as the matched comparison
   for the contest age coefficient.

Stopping here as instructed. Nothing from Step 4b has been committed.
