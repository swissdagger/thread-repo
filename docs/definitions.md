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
