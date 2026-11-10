# Brief: first model fit (two-clock intensity model)

Goal: fit one interpretable parameter set per population, check it predicts what it was never fitted
to, and hand a generator to the simulator. Read `docs/definitions.md` (binding), `docs/model_reference.md`
and `docs/design_hypotheses.md` first.

**Work in four stages. Stop and report after each one.** Calibration split only until Stage 4, and Stage 4
runs only after explicit confirmation. Every pre-stated test is committed before the stage that runs it.

## Populations

- Humans: r/philosophy, r/buildapc, Ask HN, HN link, Show HN (calibration split; inner fit fold for
  estimation, inner held-out fold for comparison).
- Agents: Moltbook crawl, complete non-spam threads, self-replies excluded, accounts in
  `config/moltbook_scaffolding_accounts.txt` excluded, outage handled as in earlier steps. Threads above
  the 100-comment cap are excluded by the completeness rule; state that the fitness distribution is
  therefore conditional on that selection.
- HN is fitted jointly across Ask / Show / link where a post-type contrast is needed (κ, below), and
  per post type otherwise.

## The model

Three event streams per thread T, all sharing one thread fitness θ_T.

**1. Comment stream** (replies to non-root comments, excluding reply-backs and self-replies):

    log λ_c(t) = θ_T + log a(τ_T(t)) + log k(u_c(t)) + log c(h(t)) + β · x_c(t)

- τ_T: thread age. u_c: comment age. h: hour of day (UTC, per population).
- a(·), the thread clock (audience): log-normal shape in thread age, i.e. log τ and (log τ)² terms.
- k(·), the comment clock: log-normal shape in comment age, i.e. log u and (log u)² terms.
- c(·): daily rhythm, hour-of-day effects, shared within population.
- x_c(t): depth dummies (1, 2, 3+), OP-comment indicator, log(1 + direct replies so far), and early
  position: log arrival rank among siblings × 1[τ_T ≤ 30 min] (lateness is carried by a(·)).

Because both clock shapes are log-normal, the model is log-linear in log age and (log age)², so it is a
Poisson GLM on cells.

**2. Root stream** (top-level comments on the post, excluding the OP's own):

    log λ_R(t) = b · θ_T + log a_R(τ_T(t)) + log c(h(t)) + (γ − κ · 1[question]) · n_top(t)

- a_R(·): log-normal shape in thread age. b: loading of thread fitness on the root stream.
- n_top(t): top-level comments so far. γ is popularity feedback; κ is question depletion.
- κ is identified only from the post-type contrast, so it is estimated in the joint HN fit with
  1[question] = Ask HN. For other populations fit γ alone and report it as net.

**3. Reply-back stream.** For a comment c replying to p, the rate at which author(p) replies back to c:

    λ_back,c(t) = ρ · g_back(u_c(t)) · exp(β_back · x_c)

- g_back: log-normal kernel in comment age, normalised to integrate to 1, so ρ is the expected number of
  reply-backs per comment.
- x_c: depth, whether p is the root (top-level comment, so the replier is the OP).
- Reported per platform: this is the e_reply / e_own channel (Reddit inbox vs HN stake-only).

**Fitness:** θ_T = log of a Gamma(r, 1/r) variable (mean 1). Estimate comment- and root-stream
parameters with thread fixed effects (conditional Poisson, as in Step 4b), then estimate r and b by
maximum likelihood on thread totals given the fitted expected counts (negative binomial). As a check, also
fit a comment-level gamma effect to the comment stream and report whether it improves held-out likelihood
(HN link was overdispersed).

**Cells:** each unit (comment, root, or reply-back opportunity) × fine log-spaced time bins, 10 s to
3 days, at least 30 bins; the 3-day horizon and censoring rules as in `docs/definitions.md`. Hour of day
is the bin midpoint's. Use the existing fixed-effects Poisson code (step-halving, clustered errors).

**Derived quantities from the fit:** μ (mean offspring, integrated comment + reply-back streams) and
μ₀ (root stream), per population, both compared with the audit's direct estimates; distance to
criticality; per-author versions.

## Baselines (nested, fitted on the same fold)

- **B1, single-clock Hawkes:** drop a(·) from the comment stream (comment kernel + fitness only), the
  Medvedev / Eziz form.
- **B2, Gómez parent choice:** no time structure; popularity, recency rank and root bias in a
  conditional logit over the comments present at each reply. Evaluated on parent choice only.
- **B3, two-clock without the reply-back stream:** all replies pooled in one comment stream. Tests the
  fitting-order rule.

## Stage 1: estimator recovery (synthetic only). Stop and report.

Write the generator first (it is reused in Stage 3 and by the simulator): given a thread's real root
arrival time and post type, draw θ_T, then simulate the three streams by thinning (Ogata) to 3 days,
with authors assigned as in Stage 3.

- Simulate threads from known parameters set near plausible values (use the audit's μ and the Step 3
  kernels as guides), at the sizes of each real population, including populations where many comments
  arrive in a burst (thread age and comment age nearly collinear).
- Fit the model and B1 to the synthetic threads.
- Pre-stated pass rule: every true parameter inside its 95% thread-bootstrap interval in ≥ 90% of 20
  replicate datasets, and the two clock shapes recovered in the burst-heavy setting. If a parameter fails,
  report which and why before Stage 2.

## Stage 2: fit and compare. Stop and report.

- Fit the full model and B1–B3 per population on the inner fit fold. 95% thread-bootstrap intervals
  (B = 100 for refitted models).
- Compare on the inner held-out fold:
  - timing log-likelihood per event (model vs B1, B3);
  - parent-choice log-likelihood: the model's implied P(next reply goes to c) = λ_c / Σ λ over the
    thread's comments at that moment, vs B2;
  - report differences with bootstrap intervals.
- Pre-stated pass rule: the full model beats B1 and B3 on timing and B2 on parent choice, with intervals
  excluding 0, in at least 4 of the 5 human populations.
- Report all parameters per population with intervals, μ and μ₀ against the audit values, κ with its
  interval, ρ and g_back by platform, and the fitness shape r.

## Stage 3: forward prediction on the held-out fold. Stop and report.

For each thread in the inner held-out fold, keep its real root arrival time and post type, and simulate
200 replicate threads from the fitted parameters (fitness drawn fresh). Do the same with B1 and B3.

**Author model** (needed for reciprocity, OP return, distinct commenters), fitted on the inner fit fold:

- reply-back events: author fixed (author of the parent);
- other events: with probability π_prev the author is a previous participant in the thread (chosen in
  proportion to their comments so far), otherwise a new author; π_prev fitted per population, optionally
  depending on thread size so far.

**Validation-list statistics**, simulated vs observed on held-out threads:

- depth distribution (p50, p90, p99 and the CCDF);
- thread-size shape (CCDF, and size given fitness quantile);
- distinct commenters per thread (threads with 20+ comments);
- dyadic reciprocity and OP return.

Flag reciprocity and OP return as **weak tests**: they sit close to the fitted reply-back stream and the
author model. Depth, size shape and distinct commenters are the strong tests.

**Pre-stated pass rule:**

- coverage: the observed value lies inside the simulated 95% band for depth p50/p90/p99 and size
  p50/p90/p99 in at least 4 of 5 human populations;
- comparison: the full model's simulated depth and size CCDFs are closer to observed than B1's
  (Wasserstein distance on log scale, interval of the difference excluding 0) in at least 4 of 5.

Report Moltbook separately; its pass rule is coverage only.

**What a miss means:** Stages 1–3 are exploratory. A miss can lead to a revised model, but every revision
is logged in `docs/results_register.md` with its reason, and the stage is rerun.

## Stage 4: validation split (only after explicit confirmation)

Before reading the validation split:

- freeze the model and code (tag the commit);
- write the Stage 2 and Stage 3 comparisons, unchanged, as validation tests in the register, plus the
  existing R-entry tests that the fit can now address;
- commit.

Then run once and report as is. No refitting after the validation split is read.

## Out of scope

Voting; the dialogue-act classifier (the OP-comment indicator stands in); cross-platform transfer;
LLM simulations; Lemmy; Show HN position (parked, R21).

## Reporting

As in the exploration: one report per stage in `reports/fit1_stageN.md`, every number with a 95%
thread-bootstrap interval, the fitted-on / validated-on table, and "Implications for the model".
