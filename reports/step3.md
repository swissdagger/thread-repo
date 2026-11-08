# Step 3: Reply-delay hazard and kernel

> **Correction (Step 3b):** per-author fits gave each comment by a deleted or unknown author weight 1.
> Corrected per-author μ: r/philosophy 0.538, r/buildapc 0.518, Ask HN 0.330, **HN link 0.509 (was 0.241)**,
> Show HN 0.563, Moltbook 0.068. **K4 is superseded:** with thread fixed effects, author activity does not
> predict replies (`reports/step3b.md` §3). The per-author rows in `outputs/*/step3_*` were replaced. All
> raw results below stand.
>
> **Audit:** HN outputs were recomputed with HN dead items excluded (μ changes ≤ 0.5%). The "branching
> ratio" is now defined as the censoring-adjusted mean offspring μ (MCF). The first-reply integral below is
> **μ_first**: it understates μ on HN and roots, and overstates it on r/buildapc (`reports/audit.md` §2,
> `docs/definitions.md`).


Status: done. Waiting for confirmation before Step 4.

Script: `scripts/step3_kernels.py`. Shared code: `forum_model/hazard.py`.

- Tables:
  - `outputs/<dataset>/step3_fits.csv`: every fit, with its parameters.
  - `outputs/<dataset>/step3_hazard.csv`: the life tables.
  - `outputs/<dataset>/step3_summary.csv`: headline numbers per community and unit.
  - `outputs/combined/step3_key_numbers.csv`: the summaries across datasets.
- Figures:
  - `figures/combined/step3_nonroot_hazard.png`
  - `figures/<dataset>/step3_hazard_fits.png`
  - `figures/moltbook_crawl/step3_depth_kernels.png`

Also in this round:

- Step 2b: `reports/step2b.md`.
- Proxy (c) preparation: below.
- Reddit now runs one subreddit at a time. Peak memory: Step 1 r/buildapc rebuild 8.6 GB (it previously ran out at about 15 GB); Step 3 Reddit 6.7 GB; HN 7.2 GB. The rebuilt r/buildapc tables match the committed Step 1 outputs exactly.

## Model updates in force (from the Step 2 confirmation; `model_reference.md` unchanged)

1. The human non-root branching prior is **0.6–0.8**.
2. The position-bias curve is over **page position among siblings**, not depth.
3. `no_top1` is no longer a robustness variant. The share of comments by the top 1% of authors is reported as a **population descriptor**, and **per-author normalised** statistics are the comparable version across populations.

## What was computed

**Units.**

- Non-root comments are the parents of the self-excitation kernel.
- Root posts (live roots only) are fitted separately for contrast.
- On Moltbook, depth-1 and depth-≥2 comments are also fitted separately, as a check.

**Event.** A parent's first direct reply that is not a self-reply and not by an author on the explicit bot
list, and that arrives before the parent's censoring time. Parents authored by listed bots are excluded.

**Censoring.**

| dataset | parent censored at | why |
|---|---|---|
| Reddit | data end, 2023-01-31 | Archiving is not a usable cap: 0.34% of r/buildapc comments arrive more than 180 days after the post. |
| HN | data end, 2022-11-16 | Replies more than 14 days after the story are rare (0.025%) but present. |
| Moltbook | the thread's last crawl time | Comments created in contiguous segment 1 are also censored at the start of the outage (2026-01-31 10:39). Only parent comments created in the two contiguous segments are used. |

**Moltbook threads.** The primary fit uses completely stored, non-spam threads. A sensitivity fit adds the
incomplete threads, each censored at its last stored comment; that assumes the stored comments are the
earliest ones, which is undocumented (Step 2b).

**Model.** Replies to a comment arrive as a Poisson process with intensity μ·g(t), where g is a kernel that
integrates to 1:

- exponential;
- Weibull (nests the exponential at shape 1);
- two-exponential mixture.

**Fits.** All three kernels are fitted to the first-reply data (the brief's specification), by maximum
likelihood with censoring. In this model, the first-reply hazard is μ·g(t) and the survival function is
exp(−μG(t)): the curve is defective because a comment never gets a reply with probability exp(−μ). μ is
profiled out in closed form.

The same kernels are also fitted to **all** direct replies. Under the Poisson assumption both give the same
μ; the gap between them measures over- or under-dispersion.

The per-author version weights each comment by 1/(number of comments by its author).

**Non-parametric hazard.** An actuarial life table: one bin from 0 to 10 s, then 25 log-spaced bins from
10 s to 10 days.

**Branching ratio.**

- (a) **The estimate** is the kernel integral μ from the family that wins on held-out likelihood.
- (b) **For comparison only**, exp(slope) of log P(depth ≥ k) against k. The fit runs from k = 2 to the
  last k with at least 100 comments, on calibration comments with bots removed. Reddit and HN are reported
  full and size-matched (threads under 100 comments); Moltbook uses complete threads.

**Model comparison.** Held-out log-likelihood per comment, on the inner held-out fold.

## Key numbers

### Non-root branching ratio and kernel

| | parents (fit fold) | best kernel (held-out) | **μ, kernel integral** | μ, all replies | μ, per-author | Weibull shape | half-life of fitted kernel | median delay to first reply | top-1% share of parents / of replies |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| r/philosophy | 419,898 | Weibull | **0.750** | 0.781 | 0.556 | 0.52 | 1.54 h | 38 min | 23% / 25% |
| r/buildapc | 2,000,000 † | Weibull | **0.732** | 0.604 | 0.513 | 0.42 | 0.49 h | 9 min | 45% / 39% |
| Ask HN | 334,108 | mixture | **0.396** | 0.488 | 0.287 | 0.54 | 1.58 h | 75 min | 29% / 35% |
| HN link | 2,000,000 † | mixture | **0.643** | 0.816 | **0.241** | 0.59 | 1.09 h | 44 min | 47% / 49% |
| Show HN | 144,598 | mixture | **0.588** | 0.637 | 0.510 | 0.56 | 1.30 h | 55 min | 26% / 29% |
| **Moltbook** (complete threads) | 471,467 | mixture ‡ | **0.060** | 0.067 | 0.067 | 0.37 | 0.20 h | 9 min | **78% / 53%** |
| Moltbook, depth 1 | 422,532 | mixture | 0.066 | 0.073 | 0.078 | 0.37 | 0.19 h | 9 min | 79% / 53% |
| Moltbook, depth ≥ 2 | 48,935 | mixture | **0.0099** | 0.013 | 0.013 | 0.37 | 0.33 h | 22 min | 69% / 58% |
| Moltbook, with incomplete threads | 629,722 | mixture | 0.058 | 0.065 | 0.071 | 0.38 | 0.24 h | 11 min | 75% / 49% |

† Capped at 2,000,000 parents by a deterministic hash. The held-out fold is not capped (1.55M and 1.40M
parents).

‡ The mixture beats the Weibull on held-out log-likelihood by only 0.0007 per comment on Moltbook and
0.006–0.02 on HN, effectively a tie. On Reddit the Weibull wins by 0.03–0.07. The exponential loses
everywhere, by 0.1–1.4.

Distance to criticality, 1 − μ: **0.25–0.27** for Reddit, **0.36–0.60** for HN, **0.94** for Moltbook.

### Branching ratio two ways: kernel (the estimate) vs depth tail (comparison)

| | μ, kernel | depth-tail slope, full (k range, R²) | depth-tail slope, < 100 comments (k range, R²) | gap (tail − kernel), full / < 100 |
|---|---:|---|---|---|
| r/philosophy | 0.750 | 0.919 (2–84, 0.88) | 0.825 (2–34, 0.96) | +0.17 / +0.08 |
| r/buildapc | 0.732 | 0.965 (2–179, 0.63) | 0.846 (2–57, 0.95) | +0.23 / +0.11 |
| Ask HN | 0.396 | 0.528 (2–14, 1.00) | 0.453 (2–10, 1.00) | +0.13 / +0.06 |
| HN link | 0.643 | 0.685 (2–32, 0.96) | 0.589 (2–21, 0.99) | +0.04 / −0.05 |
| Show HN | 0.588 | 0.522 (2–13, 1.00) | 0.483 (2–11, 1.00) | −0.07 / −0.11 |
| Moltbook | 0.060 | 0.014 (k = 2–3 only; 2 points) | – | −0.05 |

### Root posts (live), for contrast

| | μ₀, kernel from first reply | μ₀, all replies | variance of replies per root | Weibull shape | median delay to first reply |
|---|---:|---:|---:|---:|---:|
| r/philosophy | 0.41 | 3.28 | 272 | 0.43 | 28 min |
| r/buildapc | 2.38 | 2.70 | 8,941 | 0.47 | 5 min |
| Ask HN | 1.39 | 6.90 | 1,106 | 0.61 | 25 min |
| HN link | 0.28 | 1.85 | 49 | 0.57 | 56 min |
| Show HN | 0.42 | 1.88 | 42 | 0.59 | 106 min |
| Moltbook | 3.69 | 6.62 | 33 | 0.42 | **30 s** |

### Shape of the empirical hazard

`figures/combined/step3_nonroot_hazard.png`

- **Every population's first-reply hazard rises before it decays.** It peaks at about 1 min (r/buildapc,
  Moltbook), 3 min (r/philosophy) or 5 min (HN), then falls roughly as a power law out to 1 day.
- Neither competitor can rise. A Weibull with shape below 1 overshoots near zero; the mixture puts a kink
  near 1 hour. Both under-fit the tail beyond a day on HN and Moltbook.
- **Moltbook's hazard runs about 10× below the human curves from 1 minute to 1 day, roughly parallel to
  them.** The populations differ mainly in level (μ), not in timing shape.

## Fitted-on / validated-on

| quantity | fitted on | compared or validated on |
|---|---|---|
| Kernel parameters, μ (all families, raw and per-author) | Calibration split, **inner fit fold** (80% of calibration threads, chosen by hash; capped at 2M parents for r/buildapc and HN link). Moltbook: complete non-spam threads, contiguous segments. | Family choice: **inner held-out fold** (the other 20% of calibration threads), by held-out log-likelihood per comment |
| Non-parametric hazard, half-life, median first-reply delay | Inner fit fold | Not validated; these are calibration quantities |
| Depth-tail slope (comparison only) | Depths of calibration-split comments | Compared with kernel μ; not used as an estimate |

**The 20% validation split was not used** for any fit, comparison or descriptive number. Reply half-life
is reported as a calibration quantity and is not a validation metric. No validation-list quantity (depth,
size shape, reciprocity, OP return, bimodality) was computed in this step. The depth-tail slope uses depth
only as a comparison against the kernel, on calibration threads.

## Proxy (c) data preparation, ahead of Step 4

**What contest mode does.** Reddit's description of the setting (as reproduced in PRAW 7.7.1,
`SubmissionModeration.contest_mode`): "The comment thread will default to being sorted randomly. Replies to
top-level comments will be hidden behind '[show replies]' buttons. Scores will be hidden from
non-moderators." So you were right that child comments are collapsed by default. **Proxy (c) is restricted
to replies whose parent is a top-level comment**, recorded in `partitions.json`.

I couldn't reach Reddit's own help pages: praw.readthedocs.io and the r/modnews mirrors are outside this
session's allowed network domains. The docstring was read from PRAW's source package on PyPI instead.

**Data.** r/Fantasy (contest mode, primary) and r/AskHistorians (random sort, secondary) were added to
`partitions.json`, reserved for the human baseline, and reconstructed through Step 1.

| | window threads | comments | proxy threads | comments in proxy threads |
|---|---:|---:|---:|---:|
| r/Fantasy | 84,811 | 2,215,103 | **996 contest mode** (977 with comments) | 45,785 |
| r/AskHistorians | 254,598 | 895,550 | **736 random sort** (703 with comments) | 45,129 |

**S10 bot review** (authors in more than 1% of threads, reviewed by hand;
`outputs/reddit/step2_proxy_c_author_review.csv`):

- **Added to the explicit list:**
  - `RedditFantasyBot`: 21,354 comments in 14% of r/Fantasy threads, never top-level, about 5 min after
    its parent.
  - `subredditsummarybot`: 136 comments on r/AskHistorians, 39 on r/philosophy.
  - The flags were refreshed. The 39 r/philosophy comments are too few to move the Step 2 r/philosophy
    numbers.
- **Name-rule hits judged human:** `lurkmode_off`, `get_in_the_robot`, `robotreader`, `robothelvete`,
  `CommodoreBelmont`.
- **Kept, but flagged for concentration:**
  - `GarrickWinter` comments in 36% of r/Fantasy contest threads; apparently the moderator who runs them.
  - `Gankom` wrote **14,476 of the 45,129 comments (32%) in r/AskHistorians random-sort threads**, with a
    75-second median reply delay, i.e. chat in recurring threads.
- AutoModerator is already listed. It writes 18% of all r/AskHistorians comments, one on 75% of threads.

**Removal and deletion rates** (`outputs/reddit/step2_proxy_c_deletion_rates.csv`):

| | author deleted | body `[removed]` | body `[deleted]` |
|---|---:|---:|---:|
| r/AskHistorians, all comments | 37.4% | **35.3%** | 1.7% |
| r/AskHistorians, top-level comments | 38.6% | 37.1% | 1.3% |
| **r/AskHistorians random-sort threads, top-level comments** | 46.8% | **43.9%** | 2.4% |
| r/AskHistorians random-sort threads, replies | 18.5% | 17.0% | 1.1% |
| r/Fantasy, all comments | 5.0% | 1.2% | 2.4% |
| r/Fantasy contest threads, top-level comments | 3.8% | 0.6% | 2.5% |

**What these threads are.**

- r/AskHistorians random-sort threads are mostly the weekly "Sunday Digest" plus a few meta and AMA posts.
- r/Fantasy contest threads are self-promotion threads, monthly general-discussion and Friday social
  threads, polls and giveaways.

Neither is a typical thread.

**Proxy (c) caveat for Step 4.** On r/AskHistorians, 44% of the top-level comments in the proxy threads
(the parents in proxy c) were removed by moderators, at unknown times. A removed parent may have been
visible only briefly, so its exposure is unknown. **r/Fantasy should carry proxy (c). r/AskHistorians
should be used only with removed parents dropped, and reported as a secondary check.**

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`.

- **K1. The Weibull shape is not "near 1 for humans, far below 1 for Moltbook".**
  - Human shapes are **0.42–0.59**; Moltbook's is **0.37**. Per-author and all-replies fits agree.
  - A Weibull fitted only to comments that got a reply gives 0.39–0.55 for humans and 0.37 for Moltbook.
  - The prior Moltbook shape of 0.11 is not reproduced here (self-replies and bots excluded, complete
    threads).
  - The human/agent contrast is in **magnitude**: μ is 0.06 vs 0.40–0.75, about 10× lower across the whole
    1 min–1 day range. It is not in the decay shape. The reference's first expected finding fails as
    stated.
- **K2. The kernel family needs an onset.** Every population's hazard rises for 1–5 minutes before it
  decays, which is reading-and-typing time for humans and generation-plus-heartbeat latency for agents.
  The Weibull (shape below 1) and the exponential mixture are both monotone and cannot represent it. I
  suggest adding a log-normal or a gamma-with-onset kernel as a third competitor before Step 4. The
  current μ estimates barely depend on it, because little mass lies in the first minutes.
- **K3. The human non-root branching ratio matches the updated prior on Reddit, not on HN.**
  - Reddit: μ = 0.73–0.75, inside 0.6–0.8.
  - HN: 0.40 for Ask HN, 0.59 for Show HN, 0.64 for HN link. The prior should be per platform, or 0.4–0.8
    overall.
  - Per-author (the comparable version): 0.51–0.56 for Reddit, 0.24–0.51 for HN, 0.067 for Moltbook.
  - Moltbook's 0.060 is well under the expected 0.2.
- **K4. On HN link stories, replies follow the prolific authors.** Per-author μ is 0.24 against 0.64 raw:
  comments by heavy commenters attract far more replies than one-off commenters' comments. A per-author
  activity term (μ_i in A, or the ρ reciprocity term) has to carry this; a single comment-level μ cannot.
  Reddit shows the same effect, more weakly (0.51–0.56 against 0.73–0.75).
- **K5. Reply counts are not Poisson around the kernel.**
  - μ from all replies differs from μ from the first reply: r/buildapc 0.60 vs 0.73 (too many comments get
    exactly one reply, from help dialogues); HN link 0.82 vs 0.64 (overdispersed); Ask HN 0.49 vs 0.40.
  - r/philosophy and Moltbook are close.
  - So comment-level heterogeneity (F_m) is needed for counts. μ from the first reply is the probability
    of starting a conversation, not the mean offspring, where the two differ.
- **K6. The depth-tail slope is not a substitute for the kernel (circularity 4 confirmed).**
  - It overstates μ on Reddit by 0.17–0.23 in full threads (0.08–0.11 size-matched): deep chains are
    correlated dialogue, not independent branching.
  - On HN it scatters around the kernel (−0.11 to +0.13).
  - On Moltbook it rests on two depth points.
  - Depth must stay a validation target, predicted from the kernel μ, never fitted to it.
- **K7. On Moltbook, depth changes the magnitude, not the timing.** Depth-1 μ = 0.066 against depth-≥2
  μ = 0.010, with the same Weibull shape (0.37) and similar half-lives (12 vs 20 min). Together with Step 2b
  (comments are reached only through an optional thread fetch, with no reply notification), this supports
  **one timing kernel per population**, with a depth- or exposure-dependent multiplier on μ, not separate
  kernels per depth. Excluding incomplete threads barely matters (0.060 vs 0.058).
- **K8. Root fan-out is not a Poisson kernel process.** μ₀ from the first reply is far below μ₀ from all
  replies (r/philosophy 0.41 vs 3.28; Ask HN 1.39 vs 6.90), and the variance of replies per root runs to the
  thousands. That is the feed-driven, fitness-heterogeneous root process the reference keeps separate
  (μ₀, f_root). Moltbook roots get their first reply after a median of **30 s**, which suggests agents
  reading the `new` feed.
- **K9. Half-lives against the reference's priors.** The fitted kernel half-life is 0.5–1.5 h on Reddit
  (prior 2.6 h), 1.1–1.6 h on HN, and **12 min on Moltbook (prior 0.8 min)**; median delays to first reply
  are 9–75 min (humans) and 9 min (Moltbook). Agents are faster, but not by the factor of about 200 the
  priors imply. The priors probably include self-replies, which follow their parent after a median of
  1.6 min (Step 2b).
