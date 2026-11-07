# Brief: exploratory analysis before simulation

Goal: check the model choices in `model_reference.md` against three public datasets.
Work step by step. Stop and report after Step 1 before fitting anything.

## Datasets

1. **Reddit** — `HuggingFaceGECLM/REDDIT_comments` on Hugging Face. Start with r/philosophy
   and r/buildapc. Window 2018-01-01 to 2021-12-31. Fields: id, link_id, parent_id,
   created_utc, retrieved_on, author, score, body, and any submission-level `contest_mode`
   flag available.
2. **Hacker News** — BigQuery `bigquery-public-data.hacker_news.full`, or the Hugging Face
   snapshot `jkeisling/hacker-news-corpus-2007-2022` if BigQuery is not set up. Window
   2019–2021. Separate "Ask HN" posts from link posts.
3. **Moltbook** — `SimulaMet/moltbook-observatory-archive` on Hugging Face (first week,
   2026-01-28 to 2026-02-04). Later `giordano-dm/moltbook-crawl`. Note: the archive has a
   41.7-hour coverage gap; run periodicity-sensitive analyses on contiguous segments.

## Step 1 — Thread reconstruction. Stop and report.

Build reply trees from parent_id (strip `t1_`/`t3_` prefixes on Reddit). Per comment
compute: depth, number of direct replies, time since parent, time since root, whether the
author is the root author.

Report per dataset:
- threads, comments, orphans, missing parents, deleted or removed authors
- contest-mode threads per subreddit
- distribution of `retrieved_on − created_utc` (how stale the score snapshot is)

## Step 2 — Descriptive structure

Per dataset:
- distributions of thread size and max depth (log axes)
- mean direct replies by depth: root fan-out versus non-root
- share of comments with at least one reply
- original-poster return rate
- dyadic reciprocity: fraction of author pairs with replies in both directions
- distribution of distinct commenters per thread, threads with 20+ comments (bimodality check)

## Step 3 — Reply-delay hazard and kernel

For each comment: time to first direct reply, or censoring at data end.

- Non-parametric hazard in log-spaced bins, 10 seconds to 10 days.
- Fit exponential, Weibull, and two-exponential mixture by maximum likelihood with censoring,
  on the calibration split. Compare on held-out log-likelihood. Report parameters and Weibull
  shape per dataset; overlay fits on the empirical hazard.
- Branching ratio two ways, report both and the gap:
  (a) integral of the fitted self-excitation kernel — this is the estimate;
  (b) depth-tail slope from log P(depth ≥ k) against k — comparison only.

Expected ballpark: Reddit non-root branching near 0.45; Moltbook well under 0.2.

## Step 4 — The position check (Reddit and HN)

Three position proxies, all reported, clearly labelled:
- (a) confidence-sort position from final score — labelled look-ahead-affected
- (b) early-thread only: replies within the first 30 minutes of a thread, position by
  arrival order
- (c) contest-mode threads only, where order was randomized

Refit the reply hazard with position as a covariate under each proxy. Report whether time
decay flattens under (b) and (c), not just (a). On Hacker News use its score-and-age sort,
with age as the known term and scores hidden.

## Step 5 — Saturation without scores

In Ask HN threads, mark the first top-level comment as the candidate answer. Compare the
reply hazard to the root before versus after its arrival. Repeat for link posts as the
control. Do not use scores to define "answered".

## Step 6 — External-exposure tail

For Reddit and Moltbook, plot the reply hazard out to 30 days. Report whether Reddit shows a
flat late tail that Moltbook lacks.

## Reporting

One markdown summary per step in `reports/`, each with: what was computed, the key numbers,
the figures, the fitted-on / validated-on statement, and a short "Implications for the model"
section if anything contradicts `model_reference.md`.
