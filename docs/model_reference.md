# Model reference (read-only)

Trimmed from the full specification. Contains only what the exploration needs: the equation,
what each term means, the kernel choices, and the circularity checks.

## The model in one equation

For a user *i* who could reply to item *m* (a post or a comment) at time *t*:

    λ_im(t) = A_i(t) · E_im(t) · F_m(t)

Three questions in sequence: is the user active, did they see the item, and given that they
saw it, would they reply. The exploration mostly concerns F (through parent choice) and the
timing kernels; A and E are latent in human data.

**Availability** — is the user active?

    A_i(t) = μ_i · c(t) + η_i · Σ_{replies r to i} g_r(t − t_r)

- μ_i: user's baseline activity rate (heavy-tailed across users)
- c(t): daily/weekly cycle, mean 1
- η_i, g_r: re-entry boost per reply received, fading with kernel g

**Exposure** — did they see it?

    E_im(t) = e_feed(rank_m, K) · e_thread(pos_m) + e_reply · 1[m replies to i] + σ

- e_feed: probability the thread is within the first K feed items at its rank
- e_thread: probability of seeing comment m at its page position (position-bias curve)
- e_reply: platform's reply-notification channel (Reddit inbox ~high, HN none, Moltbook none)
- σ: spontaneous arrival from outside the feed (search, links)

**Attractiveness** — given they saw it, how strong is the pull to reply?

    F_m(t) = f_root(m) · [ α_pop · replies(m) + exp(−(t − t_m)/τ) + β_root · 1[m is root]
             + ρ · 1[author(m) replied to i] ] · a(act_m) · (1 − κ · 1[m is an answered question])

- f_root(m): latent fitness of the thread's root post (random effect)
- α_pop: popularity (preferential attachment); replies(m) = direct replies m already has
- τ: novelty timescale
- β_root: extra pull toward the root post
- ρ: reciprocity, extra pull if m's author previously replied to i
- a(act): dialogue-act multiplier, act ∈ {question, answer, agree, challenge}; a(answer) = 1
- κ: saturation discount once a question post has an answer
- Self-replies excluded

Parent-choice fitting: A cancels across candidate parents, so only E × F differences matter.
Timing fitting: A and the self-excitation kernel matter (survival analysis).

## Derived quantities

- Non-root branching ratio μ = integral of the fitted self-excitation kernel = expected direct
  replies per comment. μ < 1 subcritical; chains die out. Expected chain size 1/(1 − μ).
- Depth tail bound: P(depth ≥ k) ≤ μ^k.
- Root fan-out μ_0 is a separate quantity (feed-driven); expected thread size ≈ μ_0/(1 − μ).
- Distance to criticality = 1 − μ, the headline number per population.

Prior estimates to compare against: Moltbook effective tail slope 0.154 (Eziz 2026);
Reddit baseline mean branching 4.31 at depth 0 and 0.45 at depth 1 (same study);
Moltbook reply half-life ≈ 0.8 min vs Reddit ≈ 2.6 h; Moltbook Weibull shape 0.11.

## Kernels

Three time kernels, not one:

1. Self-excitation kernel (how a comment's rate of attracting replies fades). Default Weibull
   (nests exponential at shape 1); fit the two-exponential mixture as a competitor; choose on
   held-out likelihood. Integral of this kernel is μ.
2. Novelty kernel in F (exp(−age/τ)). Keep exponential. Prediction: τ shrinks once page
   position enters exposure, because position was absorbing the time decay.
3. Re-entry kernel g in A. Not fitted freely: for humans, derive it from the time to next
   session (μ_i, c(t)); for agents it is the heartbeat interval. Free Weibull as fallback.

All kernels normalised to integrate to 1, so magnitude (α, η) and timing (kernel shape) are
separate parameters.

## Position proxies (within-thread exposure)

Reddit's default comment sort ranks by a confidence score (Wilson lower bound on votes); HN
sorts by score and age; deep replies are collapsed. Position is not observed; three proxies:

- (a) confidence sort from final score — look-ahead-affected (see circularity 1)
- (b) early-thread arrival order, first 30 minutes — position set by timing, not quality
- (c) contest-mode threads — order randomized by the platform

The position-bias curve is latent for humans; parametric form from click models (a decreasing
function of position, one or two parameters), estimated jointly with attractiveness.
Simulation-derived curves are never used as inputs to the human fit.

## Circularity checks

| # | Circularity | Fix in this exploration |
| --- | --- | --- |
| 1 | Position proxied by final score: score includes votes caused by the replies being predicted and grows with age, so position would absorb time decay mechanically | Report proxy (a) as look-ahead-affected; rely on (b) early-thread and (c) contest-mode; use HN with hidden scores and age explicit |
| 2 | "Answered question" defined by a high-score answer | Define by arrival of the first top-level (candidate answer) comment, never by score |
| 3 | Latent fitness fitted from the size distribution, then size "predicted" | Fitness is a random effect on held-out threads; validation targets are depth, reciprocity, return, size shape — not size |
| 4 | Branching ratio read off the depth tail, then depth "predicted" by μ^k | μ from the kernel integral; depth-tail slope reported as comparison only |
| 5 | Dialogue-act classifier biased across populations (prior detector precision 0.22 on Moltbook vs 0.66 on Reddit) | Not run in this exploration; when added, human-annotated validation from both populations, precision per population |
| 6 | Agents later conditioned on the same data used for the human fit | `partitions.json` reserves these subreddits and windows; never reuse for agent conditioning |
| 7 | Simulation exposure tuned until agents look human | Not applicable here; interventions pre-registered later |
| 8 | Calibration statistics reused as validation | Calibration: reply timing, parent choices. Validation: depth, size shape, reciprocity, return, bimodality |
| 9 | Human baseline contains bots | 2018–2021 window; drop Pushshift-era bot lists; sensitivity to removing top 1% most active authors |

## Expected findings if the model is on the right track

- Weibull shape near 1 for humans, far below 1 for Moltbook.
- Time decay in the reply hazard flattens under proxies (b) and (c) when position is added.
- Reply hazard to Ask HN roots drops after the first top-level comment; link posts do not.
- Reddit shows a flat late-reply tail (days out); Moltbook does not.
- Root fan-out behaves as a different process from non-root replies in all three datasets.
