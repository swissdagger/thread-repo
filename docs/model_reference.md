# Model reference (read-only) — v2, after the audit and Step 4b

Trimmed from the full specification. `docs/definitions.md` is binding for every quantity named here;
`docs/results_register.md` holds the empirical results (R1–R19) and `docs/design_hypotheses.md` the
design hypotheses (H0–H5).

## The model in one equation

For a user *i* who could reply to item *m* (a post or a comment) at time *t*:

    λ_im(t) = A_i(t) · E_im(t) · F_m(t)

Three questions in sequence: is the user active, did they see the item, and given that they saw it,
would they reply. The exploration mostly concerns F (through parent choice) and the timing kernels;
A and E are latent in human data.

**Availability** — is the user active?

    A_i(t) = μ_i · c(t) + η_i · Σ_{replies r to i} g_r(t − t_r)

- μ_i: baseline activity rate (heavy-tailed; the top 1% write 27–78% of comments depending on population)
- c(t): daily/weekly cycle, mean 1
- η_i, g_r: re-entry boost per reply received, fading with kernel g. For humans g_r is derived from
  time to next session; for agents it is the heartbeat interval

**Exposure** — did they see it?

    E_im(t) = e_feed(rank_m, K) · e_thread(pos_m) + e_reply · 1[m replies to i]
              + e_own(s_i) · 1[i has a stake in m's thread] + σ

- e_feed: probability the thread is within the first K feed items at its rank. Carries the thread's
  audience decay (30–70% of the reply-rate decline with age; Step 4b)
- e_thread: probability of seeing comment m at its rank among siblings (not depth). Measured, with
  lateness controls: pos^−0.18 to −0.32 in help/Q&A threads, about 0 in discussion threads
- e_reply: reply-notification channel (Reddit inbox; none on HN; none for Moltbook agents in the study
  week — the channel arrived later and must be dated before later agent data is used)
- e_own(s): revisits to threads the user has a stake in, without notification. Stake s ranks own
  project > own question > shared link (HN OP return: Show HN 72%, Ask HN 38%, link 17%; Moltbook 15%)
- σ: spontaneous arrival from outside the feed (search, links); 0 for agents
- For an LLM agent the page is the prompt: e_thread is what the harness places in the context window,
  in the order the API returns it — a context-truncation curve

**Attractiveness** — given they saw it, how strong is the pull to reply?

    F_m(t) = f_root(m) · [ α_pop · replies(m) + exp(−(t − t_m)/τ) + β_root · 1[m is root]
             + ρ · 1[author(m) replied to i] ] · a(act_m) · (1 − κ · 1[m is an answered question])

- f_root(m): latent fitness of the thread's root (random effect)
- α_pop: popularity (preferential attachment); replies(m) = direct replies m already has
- exp(−age/τ): within-thread comment-age effect, τ the novelty timescale. Survives thread × time-bin
  fixed effects (10× older → 0.38–0.56× replies). Whether this is preference or the reply-notification
  channel acting on new comments is under test (Step 4c)
- β_root: extra pull toward the root post
- ρ: reciprocity; near 0 for agents (pair reciprocity 0.047 vs 0.13–0.48 humans)
- a(act): dialogue-act multiplier, act ∈ {question, answer, agree, challenge}; a(answer) = 1.
  Comments by the post's author draw 19–36% fewer replies on HN within thread × time, about 0 on
  Reddit, 74% fewer on Moltbook
- κ: saturation discount once a question post has an answer
- No author-activity term: with thread fixed effects, author activity has no effect (0–6% per 10×)
- Self-replies excluded before every fit and depth statistic

Parent-choice fitting: A cancels across candidate parents, so only E × F differences matter.
Timing fitting: A and the self-excitation kernel matter (survival analysis).

## Kernels

- Self-excitation kernel: **log-normal reference** (best held-out likelihood in every population,
  intervals excluding 0); Weibull reported for continuity; two-exponential mixture fitted as a
  competitor; gamma dropped (collapses to exponential). Every hazard rises for 1–5 minutes before
  decaying. Decay shape is similar across populations (Weibull shape 0.42–0.59 humans, 0.36–0.42
  Moltbook); the published Moltbook shape 0.11 is not reproduced.
- Novelty kernel in F: exponential; kept separate from position (position does not absorb it).
- Re-entry kernel g in A: derived, not fitted freely (see above). Free Weibull as fallback.

All kernels normalised to integrate to 1, so magnitude (α, η) and timing (kernel shape) are
separate parameters.

## Quantities (definitions.md is binding)

- **μ (branching ratio)**: censoring-adjusted mean of non-self, non-bot direct replies per comment
  (Nelson–Aalen MCF, 3-day horizon; 7 days reported). Per-author μ is the cross-population version.
- **μ_first**: integral of the first-reply kernel = −log(1 − p_reply) under Poisson. Probability a
  conversation starts. **Never reported as μ.**
- Distance to criticality: 1 − μ. μ < 1 subcritical; chains die out. Expected chain size 1/(1 − μ).
- Depth tail bound: P(depth ≥ k) ≤ μ^k. The depth-tail slope is a comparison, never the estimate.
- Current values (calibration split): r/philosophy 0.766, r/buildapc 0.598, Ask HN 0.478, HN link
  0.806, Show HN 0.633, Moltbook 0.066. Per-author: 0.633, 0.479, 0.420, 0.730, 0.613, 0.076.
- Root fan-out μ₀ is separate (feed-driven, overdispersed): 1.75–6.42 humans, 6.34 Moltbook.
  Expected thread size ≈ μ₀/(1 − μ).

Prior estimates (Eziz 2026), for comparison only: Moltbook effective tail slope 0.154; Reddit baseline
mean branching 4.31 at depth 0 and 0.45 at depth 1; Moltbook reply half-life ≈ 0.8 min vs Reddit
≈ 2.6 h (a root quantity against a pooled human one; see `docs/definitions.md`, speed reconciliation);
Moltbook Weibull shape 0.11 (not reproduced).

## Moltbook study-week exposure configuration (known, not fitted)

| Term | Value |
| --- | --- |
| e_feed | listings sorted `new`/`hot`, 15 items per heartbeat (K = 15) |
| heartbeat | every 4+ hours per agent |
| e_thread | ~0 unless the agent opens the post; opening returns the full tree sorted `top`, capped at 100 |
| e_reply | 0 |
| e_own | small (OP return 15%, by re-fetching only) |
| σ | 0 |

Roots: 88% get a reply within 10 minutes (feed polling). Comments: median first reply 9.0 minutes.

## Position proxies

Reddit's default comment sort ranks by a confidence score (Wilson lower bound on votes); HN sorts by
score and age with scores hidden; deep replies are collapsed. Position is not observed:

- (a) final-score rank — look-ahead-affected; tie-break immaterial; baseline only
- (b) first 30 minutes, arrival rank, with log(minutes since thread start) and depth controls — main
  identification; the only one for HN
- (c) contest-mode threads (r/Fantasy; r/AskHistorians with removed parents dropped) — **placebo
  only**; contest mode collapses replies, so parents are top-level comments
- (d) `suggested_sort = new` threads within r/buildapc and r/philosophy — time-varying position;
  does not hold out of sample

The position-bias curve is latent for humans; parametric form from click models (a decreasing
function of position, one or two parameters), estimated jointly with attractiveness.
Simulation-derived curves are never used as inputs to the human fit.

## Circularity checks

| # | Circularity | Rule |
| --- | --- | --- |
| 1 | Position from final score (look-ahead): score includes votes caused by the replies being predicted and grows with age | (a) labelled; rely on (b) with controls |
| 2 | "Answered" defined by score | Define by arrival of an answer comment (the first top-level comment), never by score |
| 3 | Fitness fitted from size, then size predicted | Fitness is a random effect; predict depth, reciprocity, return, size shape |
| 4 | μ read off depth tails | μ is mean offspring; depth-tail slope and μ_first reported separately |
| 5 | Dialogue-act classifier biased across populations (prior detector precision 0.22 on Moltbook vs 0.66 on Reddit) | Human-annotated validation from both populations, precision per population |
| 6 | Agent conditioning on human-baseline data | partitions.json reserves the human baseline |
| 7 | Simulation exposure tuned to outcomes | Interventions pre-registered |
| 8 | Calibration statistics reused as validation | Calibration: reply timing, parent choice. Validation: depth, size shape, reciprocity, OP return, bimodality |
| 9 | Bots in the human baseline | Explicit list plus hand review of authors in >1% of threads |

## Status of the v1 expected findings

| v1 expectation | Outcome (exploratory) |
| --- | --- |
| Weibull shape ≈ 1 humans, far below for agents | Failed: shapes similar; the difference is level (μ) |
| Time decay flattens when position is added | Failed under (b) and (c); position modest and forum-dependent |
| Ask HN root hazard drops after first answer | Partly supported (exploratory): question threads are damped after early top-level comments (survives the synthetic placebo and the Show HN control), but not specifically at the first one. — Steps 5, 5b |
| Reddit flat late tail, Moltbook none | Failed — Step 6 |
| Root fan-out a separate process | Supported (μ₀/μ 1.2–52×) |
