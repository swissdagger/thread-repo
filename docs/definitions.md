# Definitions

These definitions are binding for every step from the audit onward. Where an earlier step used a
different quantity under a similar name, the audit tables (`reports/audit.md`,
`outputs/audit/mu_definitions.csv`) show both side by side.

## Units

- **Comment (parent).** A non-root item in a calibration-split thread that reaches the root (not an orphan),
  is not written by an explicit-list bot (`config/reddit_bot_authors.txt`), and is visible to readers.
  HN `dead` items are excluded (audit check 6). Removed and deleted comments are kept (check 5).
- **Direct reply (event).** A child of the parent that is not a self-reply (same author as the parent), is
  not written by an explicit-list bot, is visible, and arrives before the parent's censoring time c.
- **Censoring time c.** Observation end minus the comment's creation time:
  - Reddit and HN: the data end in `partitions.json`.
  - Moltbook: the thread's last crawl time, and for comments created in contiguous segment 1 also the start
    of the comment outage.
- **Root.** The post (submission or story). Root-level quantities carry the subscript 0 (μ₀) and are kept
  separate from comment-level ones.

## Quantities

**μ (branching ratio).** The mean number of non-self, non-bot direct replies per comment,
censoring-adjusted.

- It is estimated by the Nelson–Aalen **mean cumulative function** (MCF) for recurrent events:
  MCF(H) = Σ over replies r with delay ≤ H of 1 / Y(r), where Y(t) is the number of comments still observed
  at delay t. With no censoring this is the plain mean count.
- **μ is reported at horizon H = 3 days**, with **7 days** as a check. On these data the 3-to-7-day increase
  is 0.3–1.5% for human non-root comments and 3% for Moltbook (`reports/audit.md`).
- **Per-author μ** weights each comment by 1/(its author's comments in the population, within the inner
  fold), giving unknown authors weight 0. It is the version used for cross-population comparison.
- μ > 1 would be supercritical. **Distance to criticality is 1 − μ.**

**μ_first (first-reply integral).** The kernel integral from fitting time to *first* reply with a
defective-survival hazard μ·g(t) (Step 3). It equals μ only if replies to a comment form a Poisson process
around the kernel. When they don't, it estimates −log(1 − p_reply(∞)): the propensity to start a
conversation, not the mean offspring. **It is no longer called the branching ratio.**

**p_reply.** The probability that a comment receives at least one direct reply by horizon H = 3 days,
estimated by Kaplan–Meier on time to first reply.

**Kernel shape and half-life.** From the first-reply fit μ_first·g(t) on the calibration inner fit fold:

- **Weibull shape** (reference kernel): shape < 1 means a decreasing hazard.
- **Log-normal median and σ** (best held-out family since Step 3b).
- **Half-life**: the time by which half of the kernel's mass has arrived, G(t) = 0.5. It is a property of
  g alone.

**Median delay.** The censoring-adjusted median time to first reply among comments that receive a reply by
3 days: the t at which the Kaplan–Meier cumulative incidence reaches p_reply(3 d) / 2.

**Empirical mean replies (Step 2).** All children per comment, including self-replies, with no censoring
adjustment and bots removed (Step 2, variant `base`). It is descriptive only.

**Top-1% share.** The share of comments (or replies) written by the population's top 1% most active authors.
It is a population descriptor, not a robustness variant.

## Uncertainty

All intervals are 95% thread-clustered bootstrap intervals:

- threads are resampled with replacement (multinomial counts), with the seed from `partitions.json`;
- 200 replicates, or 100 for refitted kernels.

Step 4 standard errors are thread-clustered (sandwich on per-thread efficient scores), alongside the
bootstrap.

## Splits

- **Validation split:** 20% of threads, stratified by month. It is untouched by every analysis so far.
- **Inner folds:** the calibration split is divided 80/20 into "fit" and "held-out" by hash of thread id.
  Model choice uses held-out likelihood.
- **Everything reported so far is exploratory**, on the calibration split. See `docs/results_register.md`.

## Speed reconciliation (Step 4b.4)

"How fast do replies come" has several answers depending on four choices. Every number below is from the
calibration split, 3-day horizon, `outputs/combined/step4b_speed.csv` and `step4b_speed_naive.csv`.

**The four choices**

1. **Unit.** First reply to a *root* (post) or to a *comment* (non-root parent). On Moltbook these are
   different processes: agents poll the new-posts feed, and the study-week skill never fetches comments
   (Step 2b).
2. **Which replies count.** Defaults: no self-replies, no explicit-list bots, Moltbook spam posts excluded,
   only completely stored Moltbook threads, Moltbook outage handled by censoring segment 1 at the outage.
3. **Source.** Moltbook crawl (primary, study week) or SimulaMet archive (cross-check, posts created
   2026-02-08 to 02-14, observed to 2026-03-31).
4. **Statistic.**
   - *Exponential half-life*: ln 2 / rate of an exponential first-reply kernel. It is badly misspecified
     here (all Weibull shapes are 0.25–0.61), so it is dominated by the tail and reported only for
     completeness.
   - *Weibull half-life*: G(t) = 0.5 for the fitted Weibull kernel (reference kernel).
   - *Log-normal median*: the median of the fitted log-normal kernel (best held-out family). Same
     definition as the half-life.
   - *KM median delay*: the censoring-adjusted median time to first reply among parents that get one by
     3 days (see "Median delay" above). Model-free.
   - *Naive median*: median parent-to-child delay over all stored comments, with no filters and no
     censoring (`step4b_speed_naive.csv`).

**Main table (minutes; default filters unless stated)**

| population | unit | exp. half-life | Weibull half-life (shape) | log-normal median | KM median | p_reply (3 d) |
|---|---|---|---|---|---|---|
| Moltbook crawl | root | 81 | 18.7 (0.42) | 5.9 | **0.50** | 0.97 |
| Moltbook crawl | comment | 204 | 20.4 (0.37) | 14.8 | 9.0 | 0.057 |
| Moltbook crawl | comment, + self-replies | 141 | 3.5 (0.28) | 2.3 | 1.3 | 0.082 |
| Moltbook archive | root | 24 | **0.45** (0.41) | 0.43 | **0.47** | 0.042 |
| Moltbook archive | comment | 598 | 33.8 (0.36) | 24.7 | 17.1 | 0.030 |
| r/philosophy | root | 388 | 49.7 (0.43) | 33.8 | 26.4 | 0.33 |
| r/philosophy | comment | 356 | 92.4 (0.52) | 65.2 | 37.5 | 0.52 |
| r/buildapc | root | 790 | 48.3 (0.47) | 16.8 | 5.1 | 0.88 |
| r/buildapc | comment | 804 | 29.7 (0.42) | 20.0 | 9.2 | 0.51 |
| Ask HN | root | 161 | 86.8 (0.61) | 58.5 | 25.3 | 0.74 |
| Ask HN | comment | 366 | 141.7 (0.54) | 103.0 | 71.2 | 0.32 |
| HN link | root | 246 | 102.9 (0.58) | 79.4 | 57.3 | 0.24 |
| HN link | comment | 199 | 94.5 (0.59) | 69.0 | 43.9 | 0.47 |
| Show HN | root | 372 | 188.4 (0.59) | 137.9 | 105.5 | 0.33 |
| Show HN | comment | 275 | 116.5 (0.56) | 83.6 | 53.2 | 0.44 |

**Sensitivity of Moltbook comment speed (KM median, min):** default 9.0; + self-replies 1.3; + spam posts
7.9; + incomplete threads 11.7; no outage handling 8.8; all four together 4.0. Self-replies are the only
choice that moves it by more than 3 minutes. Roots barely move (0.48–0.57 min), except that ignoring the
outage drops p_reply from 0.97 to 0.81: the outage hides replies that were never stored.

**Human sensitivity:** self-replies and bots change comment medians by < 1.5 min. For roots, counting the
OP's own comments matters where the OP routinely comments first: r/philosophy (the rule requiring an
abstract) 26.4 → 6.9 min, Show HN 105.5 → 7.4 min, HN link 57.3 → 40.5 min.

**Archive vs crawl.** The archive window has 1.05M posts but only 190k comments (98% of the 194k the posts
report). Only 6% of posts have any comment, against 97% of complete crawl posts in the study week, so the
archive describes a different regime of the platform rather than a truncated copy of it. Root first-reply
speed agrees (KM median 0.47 vs 0.50 min); comment speed is slower in the archive (17.1 vs 9.0 min).

**The published 0.8-minute half-life** (Moltbook) vs 2.6 hours (Reddit) (`docs/model_reference.md`, from
Eziz 2026):

- 0.8 min matches only **root** quantities: the KM median first reply to a post (0.47–0.50 min, crawl and
  archive), the archive root Weibull half-life (0.45 min), and the naive median of all replies to posts
  (1.3–2.8 min, `naive_median_all_replies_min`). No comment-level estimate under any filter is below
  1.3 min, and the default comment-level estimates are 9–20 min.
- 2.6 h is in the range of naive pooled parent-to-child medians on slower human populations (r/philosophy
  all comments 1.9 h, replies to posts 4.7 h; HN 1.4 h and 3.1 h), and well above r/buildapc (15 min).
  It depends on the subreddit mix, which we cannot reproduce.
- The **published ratio (~195×) compares a Moltbook root quantity with a human pooled quantity**. Like for
  like (comments, no self-replies, KM median), Moltbook is 9.0 min against 9.2–71 min for humans: 1–8×
  faster, not 195×. Like for like on roots, Moltbook is 10–210× faster (0.5 min vs 5–106 min), which is the
  feed-polling effect.
- The published Weibull shape 0.11 is not reproduced by any variant here (lowest 0.25, Moltbook roots with
  every filter removed).

**Rule from here on:** any speed number names its unit (root or comment), its reply filter, its source and
its statistic. Speed numbers are calibration quantities and are never validation metrics.
