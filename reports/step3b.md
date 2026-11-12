# Step 3b: Onset kernels, the per-author correction, the K4 check, early vs late replies

Status: done. These additions were requested after Step 3. Nothing here touches the 20% validation split.

Script: `scripts/step3b_extensions.py`. Shared code: `forum_model/hazard.py` (log-normal and gamma
kernels), `forum_model/fe_poisson.py` (Poisson with thread fixed effects) and `forum_model/units.py` (the
Step 3 populations, now shared).

Tables:

- `outputs/<dataset>/step3b_{competitors,k4,early_late}.csv`, combined in `outputs/combined/step3b_*.csv`.
- Corrected per-author rows are in the `step3_*` files (see section 2).

Populations, parents, events, censoring and inner folds are exactly those of Step 3. After the refactor,
Step 3 reproduces its Moltbook results to within 0.0.

## Model updates in force

These are reported here; `model_reference.md` is unchanged.

1. **The human non-root branching prior is per platform:** Reddit 0.6–0.8, HN 0.4–0.65.
2. **For cross-population comparison, use per-author μ.**
3. **The position-bias curve is over page position among siblings.**
4. **The top-1% share is a population descriptor.** `no_top1` is not a robustness variant.

## 1. Kernel competitors with an onset

The Weibull is the reference. Both new kernels are fitted to first replies on the inner fit fold and
compared on held-out log-likelihood per comment.

### Non-root comments

| | μ, Weibull | μ, log-normal (change) | μ, gamma (change) | held-out: log-normal − Weibull | held-out: log-normal − best Step 3 family | log-normal median / σ / mode |
|---|---:|---:|---:|---:|---:|---|
| r/philosophy | 0.750 | 0.741 (−1.2%) | 0.896 (+19%) | **+0.065** | +0.065 | 1.09 h / 1.92 / 99 s |
| r/buildapc | 0.732 | 0.729 (−0.4%) | 0.987 (+35%) | **+0.124** | +0.124 | 0.33 h / 2.01 / 21 s |
| Ask HN | 0.396 | 0.394 (−0.6%) | 0.423 (+7%) | +0.030 | +0.023 | 1.71 h / 1.92 / 155 s |
| HN link | 0.645 | 0.638 (−1.1%) | 0.710 (+10%) | +0.051 | +0.030 | 1.13 h / 1.74 / 195 s |
| Show HN | 0.587 | 0.581 (−1.0%) | 0.644 (+10%) | +0.038 | +0.028 | 1.38 h / 1.89 / 141 s |
| Moltbook | 0.0594 | 0.0597 (+0.5%) | 0.0607 (+2.3%) | +0.008 | +0.008 | 0.25 h / 2.60 / ~1 s |
| Moltbook, depth 1 | 0.0653 | 0.0656 (+0.4%) | 0.0669 (+2.5%) | +0.009 | +0.009 | 0.24 h / 2.59 / ~1 s |
| Moltbook, depth ≥ 2 | 0.0100 | 0.0104 (+3.5%) | 0.0100 (−0.6%) | +0.001 | −0.0001 (tie) | 0.66 h / 3.12 / ~0 s |
| Moltbook, with incomplete threads | 0.0577 | 0.0579 (+0.3%) | 0.0605 (+4.8%) | +0.007 | +0.006 | 0.31 h / 2.61 / ~1 s |

**Log-normal**

- It is the best kernel on held-out likelihood for non-root comments in every population. It ties on
  Moltbook depth ≥ 2.
- **μ barely moves: −1.2% to +0.5% for non-root comments.** The branching ratios in Step 3 stand.
- For humans its mode is 20 s to 3 min, which captures the onset.
- For Moltbook it is very broad (σ = 2.6) with a mode at about 1 s. It fits the long decay, not the
  1-minute hump.

**Gamma (shape > 1)**

- In every unit it ran to its boundary, shape → 1, which is the exponential. It loses to the Weibull by
  0.01–1.44 per comment.
- The data reject an onset with an exponential tail. Its μ is inflated (+2% to +35% for non-root
  comments), so it is not usable as a μ estimate.

**Roots.** Log-normal μ₀ moves −0.3% to −12% from the Weibull (r/buildapc −10%, Moltbook −12%). It beats
the Step 3 mixture on r/buildapc (+0.08) and Ask HN (+0.03), ties on HN link, and loses on Moltbook roots
(−0.08). Root fan-out stays a separate, non-Poisson process (Step 3, K8).

## 2. Correction: per-author weights (supersedes Step 3's per-author μ and K4)

**The bug.** Step 3 gave each comment by a deleted or unknown author weight 1, as if it were its own
author. A known author's comments share a total weight of 1. So deleted comments dominated the per-author
fits wherever they were common:

| | unknown-author share of parent comments | their share of per-author weight | reply rate, unknown | reply rate, known |
|---|---:|---:|---:|---:|
| r/philosophy | 20.3% | 48.8% | 0.43 | 0.55 |
| r/buildapc | 4.9% | 31.4% | 0.39 | 0.52 |
| Ask HN | 2.7% | 11.9% | 0.02 | 0.33 |
| HN link | 2.6% | **46.7%** | 0.005 | 0.48 |
| Show HN | 2.1% | 7.2% | 0.006 | 0.45 |
| Moltbook | 0.1% | 2.5% | 0.02 | 0.06 |

**The fix.** Unknown-author comments now get **weight 0** in every per-author version: Step 3 fits and
life tables, the Step 3b early/late split, Step 4, and Step 2's thread-level per-author statistics. Step
2's comment-level per-author statistics already excluded them. The per-author rows in
`outputs/*/step3_{fits,hazard,summary}.csv` were replaced (`step3_kernels.py per_author_refresh`). Raw
results are unchanged.

**Corrected per-author μ (Weibull), the comparable cross-population version:**

| | r/philosophy | r/buildapc | Ask HN | HN link | Show HN | Moltbook | Moltbook depth 1 / ≥ 2 |
|---|---|---|---|---|---|---|---|
| Step 3 (wrong) | 0.556 | 0.513 | 0.287 | **0.241** | 0.510 | 0.067 | 0.078 / 0.013 |
| **corrected** | **0.538** | **0.518** | **0.330** | **0.509** | **0.563** | **0.068** | 0.079 / 0.013 |
| raw μ, for comparison | 0.750 | 0.732 | 0.396 | 0.645 | 0.587 | 0.060 | 0.066 / 0.010 |

The per-author Weibull shapes are 0.37–0.57, essentially as before.

**Step 2 thread-level per-author statistics** gave deleted-root threads weight 1 in the same way. Corrected
values, using r/philosophy and r/buildapc as examples:

- mean thread size: 3.84 → **5.07** and 5.94 → **6.67**;
- mean max depth: 2.89 → 3.21 and 3.29 → 3.41.

HN and Moltbook are unchanged: deleted HN roots already form their own class, and Moltbook root authors
are almost all known. Updated tables are in `outputs/*/step2_*` and `outputs/combined/step2_key_numbers.csv`.

## 3. K4 check: does author activity predict replies within a thread?

**Model.** A Poisson regression with **thread fixed effects**, estimated through its conditional
likelihood. Within a thread, given the thread's total replies, the counts across comments are
multinomial. That is the conditional-logit form of parent choice, so thread exposure, feed rank and
popularity all cancel.

- Outcome: direct (non-self, non-bot) replies a comment receives before censoring.
- Offset: log G(c) from the Step 3 Weibull kernel.
- Covariates: log(1 + the author's comments in **other** calibration threads), depth dummies, log(1 +
  minutes from thread start to the comment), and whether the comment is by the post's author.
- Data: the inner fit fold; comments by unknown authors excluded. Standard errors clustered by thread.
- The fixed-effects estimator was checked on simulated data: with a true within-thread effect of 0.30 it
  recovers 0.30, while a pooled model confounded by thread popularity gives 0.78.

| | threads (FE) | **author activity, within thread** (SE) | rate ratio per 10× activity | activity, pooled | minutes after thread start, within thread | comment by the post's author, within thread |
|---|---:|---|---:|---:|---:|---:|
| r/philosophy | 5,516 | **+0.017** (0.002) | 1.04 | +0.025 | −0.39 | −0.48 |
| r/buildapc | 299,469 | **−0.001** (0.006) | 1.00 | +0.001 | −0.23 | +0.07 (0.05) |
| Ask HN | 10,397 | **+0.008** (0.003) | 1.02 | −0.003 | −0.37 | −0.23 |
| HN link | 61,447 | **+0.018** (0.001) | 1.04 | +0.026 | −0.45 | −0.48 |
| Show HN | 6,181 | **+0.032** (0.003) | 1.08 | +0.035 | −0.31 | −0.49 |
| Moltbook | 10,857 | **+0.000** (0.004) | 1.00 | −0.019 | −0.31 | −1.36 |

**Result.**

- **Author activity has essentially no effect on replies, within threads or pooled:** 0–8% more replies
  per 10× activity in every population.
- Step 3's K4 ("replies follow prolific authors"; HN link per-author 0.24 vs raw 0.64) was the
  weighting artifact in section 2.
- The smaller per-author-versus-raw gap that remains (Reddit about 0.2, HN link 0.14, Ask HN 0.07) is
  **compositional**. Comments by the post's author get 21–39% fewer replies within the same thread on
  Reddit and HN, except r/buildapc, where there is no effect (Moltbook 74%), and posters are mostly occasional users. Late comments in a thread get
  fewer replies too.
- These within-thread **lateness effects** (coefficients −0.23 to −0.45 on log minutes) are the target of
  the Step 4 position check.

## 4. Early vs late replies: the descriptive check

**What is measured.** For non-root comments in the calibration split, observed for at least 1 day, the
probability that the first (non-self, non-bot) reply arrives within 10 minutes, from 10 to 60 minutes, or
from 1 hour to 1 day. The same is computed among comments that got a reply within a day. A 3-day horizon
is in the CSV as a sensitivity check and agrees.

| | < 10 min, raw | 10–60 min, raw | 1 h–1 d, raw | < 10 min, per-author | 10–60 min, per-author | 1 h–1 d, per-author | among replied: < 10 min, raw / per-author |
|---|---:|---:|---:|---:|---:|---:|---:|
| r/philosophy | 0.121 | 0.184 | 0.201 | 0.079 | 0.135 | 0.184 | 24% / 20% |
| r/buildapc | 0.264 | 0.141 | 0.100 | 0.183 | 0.114 | 0.095 | 52% / 47% |
| Ask HN | 0.046 | 0.103 | 0.154 | 0.041 | 0.087 | 0.129 | 15% / 16% |
| HN link | 0.089 | 0.177 | 0.191 | 0.078 | 0.141 | 0.158 | 19% / 21% |
| Show HN | 0.080 | 0.148 | 0.192 | 0.084 | 0.141 | 0.179 | 19% / 21% |
| **Moltbook** | **0.030** | **0.013** | **0.011** | **0.019** | **0.018** | **0.020** | **55% / 34%** |
| human ÷ Moltbook | 1.5–8.8× | 7.9–14× | 9.1–18× | 2.2–9.6× | 4.8–7.8× | 4.8–9.2× | |

**Reading.**

- **Raw (per comment), the Moltbook deficit widens with delay:** 1.5–9× within 10 minutes (Ask HN only
  1.5×) and 9–18× after an hour. Over half of Moltbook's replied comments are answered within 10 minutes.
- **Per-author, the deficit also widens with delay, but less.** *(Corrected in the audit.)* I first wrote
  "roughly constant at about 5–10×". That came from comparing ranges pooled across populations. Within each
  population, with thread-bootstrap intervals (`reports/audit.md`), the per-author gap goes from 4.1× to
  9.3× on r/philosophy, from 2.1× to 6.7× on Ask HN, from 4.1× to 8.6× on HN link and from 4.4× to 9.3× on
  Show HN, between the < 10 min and 1 h–1 d windows. Only r/buildapc narrows, from 9.5× to 4.8×. A typical
  agent's comments draw a reply with about the same small probability (about 0.02) in every window; human
  per-author reply chances grow over the first day.
- So "Moltbook matches humans early and falls 10× below later" holds per comment (1.5–4× early, except
  r/buildapc; 9–18× late) and, more weakly, per author (2–4× early; 7–9× late) in 4 of 5 human
  populations. The raw near-match in the first 10 minutes is strengthened by the most active agents' comments
  (78% of parent comments are by the top 1%), which get fast replies.

## Fitted-on / validated-on

| quantity | fitted on | compared or validated on |
|---|---|---|
| Log-normal and gamma kernels; corrected per-author Weibull | Calibration inner fit fold (capped at 2M parents for r/buildapc and HN link) | Inner held-out fold (held-out log-likelihood per comment) |
| K4 fixed-effects and pooled Poisson | Calibration inner fit fold | Not validated (descriptive test of a modelling assumption) |
| Early/late shares | Whole calibration split (descriptive) | – |

The 20% validation split was not used.

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`.

- **The self-excitation kernel should be log-normal.** It wins on held-out likelihood for non-root comments
  in every population and changes μ by at most 1.2%. Keep the Weibull as the reference for continuity.
  Drop the gamma: its tail is too light.
- **μ_i (author activity) does not carry reply attractiveness.** With thread fixed effects, a comment's
  pull on replies is unrelated to how active its author is. The reciprocity term ρ and the post's-author
  effect matter instead: comments by the post's author get fewer replies, not more. A per-author activity
  multiplier in F is not supported.
- **Per-author μ is the comparable number:** Reddit 0.52–0.54, HN 0.33–0.56, Moltbook 0.068, about 8×
  below humans.
- **The human/agent gap widens with delay, per comment and (less) per author.** *(Corrected in the audit.)*
  The kernel *shapes* are similar (Step 3, K1). The widening comes from human reply chances continuing to
  accumulate over the first day while agents' do not. That fits the agent exposure gate (Step 2b) better
  than a different decay shape.
