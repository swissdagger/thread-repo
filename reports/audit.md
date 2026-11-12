# Audit (before Step 4b)

Status: done. No new analysis was started. Everything here is **exploratory, on the calibration split**;
the 20% validation split was not read.

Code:

- `scripts/checks.py`: the 10 data checks. Records are in `outputs/audit/checks/*.json`; tables in
  `outputs/audit/check*.csv`.
- `scripts/audit_uncertainty.py`: μ by definition, plus bootstrap intervals for Steps 2–3b. Output in
  `outputs/audit/*.csv`.
- `scripts/step4_position.py`: thread-clustered SEs and bootstrap intervals for Step 4.
  - Summaries: `outputs/combined/step4_*_summary.csv`.
  - Replicates: `outputs/combined/step4_*_bootstrap.parquet`.

New documents: `docs/definitions.md` and `docs/results_register.md`.

## 1. Data checks

| # | check | threshold | result | status |
|---|---|---|---|---|
| 1 | GECLM completeness: stored comments vs submission `num_comments`, by subreddit and year | every subreddit-year with ≥ 100 threads (num_comments ≥ 10): median stored/claimed in [0.90, 1.15]; ≤ 5% of threads below 0.5 | 16 subreddit-years. Median ratio 1.00–1.054. At most 0.17% of threads below 0.5; at most 1.3% above 1.5. | **PASS** |
| 2 | Negative or zero parent→child delays; timestamp units and zones | negative < 0.01% of comments per dataset; epoch seconds in UTC within the expected range | Negative: r/philosophy 2, r/Fantasy 2, r/AskHistorians 57 (0.006%), HN 593 (0.006%), r/buildapc 0, Moltbook 0. Zero delays: r/AskHistorians 10.5% (AutoModerator, same-second), HN 0.19%, r/Fantasy 0.18%, r/buildapc 0.05%. All timestamps inside each window plus follow-up. | **PASS** |
| 3 | 2M-parent cap: how sampled; reconfirm Steps 3–4 | an independent random sample reproduces μ within 1%, Weibull shape within 0.01, Step 4 (a) β within 4 SE, flattening within 0.01 | The cap is a deterministic hash-based simple random sample of comments (sha256 of seed and comment id), independent of covariates. numpy-random resample: μ −0.13% (r/buildapc), +0.06% (HN link); shape Δ 0.002 and 0.001; Step 4 (a) β −0.2685 → −0.2675; flattening 0.0158 → 0.0157. | **PASS** |
| 4 | Per-author weights across inner folds | known authors' weights sum to 1 within each fold; unknown authors weight 0 | max \|sum − 1\| = 0 and unknown weight = 0 in all 6 populations; weights never mix folds | **PASS** |
| 5 | Removed-comment handling | removed or deleted parents handled like others; excluding them moves μ (MCF, 3 d) by < 5% | They are kept as parents, and replies to them count as events (16.6% of r/philosophy replies). Excluding them: r/philosophy +4.2%, HN +2.2–2.7%, r/buildapc +1.0%, Moltbook 0.03%. | **PASS** |
| 6 | HN dead/flagged items excluded from parents, events and sibling positions | none present | **They were present** (1.2–1.8% of HN comments, used as parents, events and siblings). **Fixed**: `hn_population(exclude_dead=True)` by default; Step 4 siblings skip them. Impact on μ (MCF): Ask HN +0.03%, HN link −0.46%, Show HN +0.12%. Step 3, 3b and 4 HN outputs recomputed. | **FAIL → fixed** |
| 7 | Moltbook `created_at` semantics; 30-second root reply median | server UTC timestamps; < 0.01% of comments created after they were crawled | ISO 8601 with microseconds and +00:00 on 100% of rows; 0 comments created after crawl; 102 comments share a timestamp. The root first-reply median is 30 s (the 10% quantile is 4 s): see the interpretation below. | **PASS** |
| 8 | Stability of `contest_mode` / `suggested_sort` (proxy c and d threads) | ≥ 90% of r/Fantasy contest threads show score obscuring at comment retrieval; recurring series consistent | r/Fantasy: **100%** of contest threads have ≥ 90% of comments with the API's obscured score of 1 (median retrieval 93 days after posting), against 29% in control threads. So contest mode was still on at retrieval. Proxy (d) series are consistent (r/buildapc daily "Simple Questions" 96–99% `new` every year; r/philosophy weekly "Open Discussion" 100%). **r/AskHistorians series switch over time** (see below). | **PASS** with a caveat |
| 9 | Sort assumptions for proxies (a) and (d) | documented; unverifiable parts flagged | 54% of r/buildapc sibling groups have tied top final scores, and 69% of comments end at score 1, so (a)'s tie-break (oldest first) matters. No stickied flag, no vote split, no reader-side sort in the data. | **FLAG** (unverifiable) |
| 10 | HN snapshot completeness against id ranges; title-prefix classification | every 1M-id block in the window's comment id range ≥ 99% present; < 1% ambiguous titles | Ids 18,797,959–29,753,871: **100%** present in every block. Ambiguous Ask/Show titles 0.004% (34 with "Ask HN" not at the start, 12 "Ask YC", 2 with a leading space). 265 Ask HN posts have a url and stay Ask HN. | **PASS** |

A first version of check 10 measured monthly coverage by timestamp and reported 9%. That came from HN
re-timestamping second-chance stories, which inflates a month's id span; it was not missing data. The check
was redefined on comment id ranges (comments are never re-dated).

### Interpretations the checks call for

- **Moltbook's 30-second root replies (check 7).**
  - 50% of posts get their first comment within 30 s and 12% within 5 s.
  - The ten most frequent first repliers write 36% of first replies. Eight of them have median delays of
    10–40 s (`botcrong`, `Starclawd-1`, `alignbot`, `ClaudeOpenBot`, `WarrenBuffer`, `TipJarBot`,
    `XiaoWang_Assistant`, `Jorday`), consistent with agents polling the `new` listing.
  - Excluding those ten, the median is still 33 s. Fast first replies are widespread: whichever agent
    polls `new` reaches a fresh post first.
  - The two largest first repliers (`Stromfee`, `FinallyOffline`) have median first-reply delays of about
    2.5 days: they sweep posts nobody has replied to.
  - The fast root replies therefore fit the Step 2b reading: the root is in feed listings, comments are not.
- **r/AskHistorians sort modes change over time (check 8).**
  - "Friday Free-for-All" was non-random in 2018–2020 (1 and 5 random threads in 2019–20) and random in
    all of 2021.
  - "Thursday Reading" used `new` in 2018–2020 and random in 2021.
  - **Step 4's "matched series" control is therefore a 2021-vs-2018–20 comparison, partly against
    newest-first threads.** It is confounded by time and display order and should not be read as a
    matched control.
  - r/Fantasy's recurring threads are in contest mode almost every week (general discussion 118/119,
    Friday social 90/92, self-promotion 58/68), so they have no within-series control either.

## 2. μ under the new definition

μ = censoring-adjusted mean of non-self, non-bot direct replies per comment (Nelson–Aalen mean cumulative
function, horizon 3 days). p_reply and the median delay use Kaplan–Meier. 95% thread-clustered bootstrap
intervals, 200 replicates. Full table: `outputs/audit/mu_definitions.csv`.

### Non-root comments

| | **μ (MCF, 3 d)** [95% CI] | μ, 7 d | **μ per-author** [CI] | p_reply (3 d) [CI] | median delay [CI] | Step 3 first-reply integral (best family) | Step 3 all-replies kernel | Step 2 empirical mean |
|---|---|---:|---|---|---|---:|---:|---:|
| r/philosophy | **0.766** [0.760, 0.772] | 0.773 | **0.633** [0.624, 0.641] | 0.516 [0.511, 0.521] | 37.5 min [36.1, 38.7] | 0.750 (Weibull) | 0.781 | 0.781 |
| r/buildapc | **0.598** [0.585, 0.607] | 0.600 | **0.479** [0.415, 0.526] | 0.510 [0.499, 0.518] | 9.2 min [9.2, 9.3] | 0.732 (Weibull) | 0.604 | 0.610 |
| Ask HN | **0.478** [0.463, 0.491] | 0.485 | **0.420** [0.402, 0.440] | 0.320 [0.312, 0.328] | 71 min [69, 73] | 0.398 (mixture) | 0.488 | 0.493 |
| HN link | **0.806** [0.805, 0.807] | 0.811 | **0.730** [0.726, 0.734] | 0.468 [0.468, 0.469] | 44 min [44, 44] | 0.646 (mixture) | 0.812 | 0.817 |
| Show HN | **0.633** [0.628, 0.639] | 0.639 | **0.613** [0.604, 0.620] | 0.437 [0.434, 0.441] | 53 min [52, 55] | 0.592 (mixture) | 0.638 | 0.645 |
| **Moltbook** | **0.066** [0.065, 0.068] | 0.068 | **0.076** [0.072, 0.081] | 0.057 [0.056, 0.059] | 9.0 min [8.4, 9.6] | 0.060 (mixture) | 0.067 | 0.105 |
| Moltbook, depth 1 | 0.073 [0.071, 0.075] | 0.075 | 0.087 [0.082, 0.092] | 0.063 [0.061, 0.064] | 8.8 min [8.2, 9.4] | 0.066 | 0.073 | – |
| Moltbook, depth ≥ 2 | 0.012 [0.010, 0.014] | 0.012 | 0.013 [0.010, 0.016] | 0.010 [0.009, 0.011] | 26 min [14, 44] | 0.010 | 0.013 | – |

### Root posts (μ₀)

| | μ₀ (MCF, 3 d) [CI] | per-author [CI] | p_reply [CI] | median delay | Step 3 first-reply integral | Step 2 live-root fan-out |
|---|---|---|---|---|---:|---:|
| r/philosophy | 3.16 [2.96, 3.33] | 2.19 [2.01, 2.35] | 0.335 | 26 min | 0.41 | 3.44 |
| r/buildapc | 2.64 [2.51, 2.87] | 2.54 [2.47, 2.62] | 0.881 | 5.0 min | 2.38 | 2.91 |
| Ask HN | 6.42 [6.05, 6.76] | 5.38 [5.17, 5.63] | 0.736 | 25 min | 1.37 | 6.85 |
| HN link | 1.75 [1.74, 1.77] | 1.73 [1.70, 1.76] | 0.235 | 57 min | 0.27 | 1.90 |
| Show HN | 1.81 [1.76, 1.90] | 1.93 [1.84, 2.02] | 0.333 | 106 min | 0.41 | 2.24 |
| Moltbook | 6.34 [6.30, 6.37] | 6.75 [6.69, 6.82] | 0.970 | **0.5 min** | 3.69 | 5.41 |

### How the three earlier quantities relate to μ

- **Step 3's all-replies kernel μ matches μ (MCF) to within 2%** everywhere: both are censoring-adjusted
  means.
- **Step 3's first-reply integral is not μ.** It equals −log(1 − p_reply) under a Poisson assumption:
  - it **understates** μ on HN link (0.646 vs 0.806), Ask HN (0.398 vs 0.478) and every root (r/philosophy
    roots 0.41 vs 3.16), where reply counts are overdispersed;
  - it **overstates** μ on r/buildapc (0.732 vs 0.598), where too many comments get exactly one reply.
  - It is renamed **μ_first** (`docs/definitions.md`); "branching ratio" now means μ (MCF).
- **Step 2's empirical mean** includes self-replies and has no censoring adjustment. It agrees with μ for
  humans (within 2–3%) but not for Moltbook (0.105 vs 0.066). There, self-replies are 36% of depth-2
  comments, and Step 2 also used the whole window rather than the contiguous segments.
- **Horizon:** μ(7 d) − μ(3 d) is +0.3% to +1.5% for human non-root comments and +3% for Moltbook.

**Cross-population comparison (per-author μ):** humans 0.42–0.73, Moltbook 0.076, a ratio of 5.5–9.6.

## 3. Uncertainty for Steps 2–4

### Step 2 headline numbers

Bots removed, full population (Moltbook: completely stored threads). 200 thread-bootstrap replicates. The
size-matched (< 100 comments) intervals are in `outputs/audit/step2_bootstrap.csv`.

| metric | r/philosophy | r/buildapc | Ask HN | HN link | Show HN | Moltbook |
|---|---|---|---|---|---|---|
| mean thread size | 8.91 [8.37, 9.42] | 7.13 [6.99, 7.36] | 12.74 [12.22, 13.37] | 8.61 [8.53, 8.72] | 5.55 [5.33, 5.77] | 6.04 [6.00, 6.09] |
| mean thread size, per-author | 5.07 [4.58, 5.72] | 6.67 [6.57, 6.80] | 11.28 [10.64, 11.94] | 6.96 [6.76, 7.15] | 6.10 [5.80, 6.39] | 5.94 [5.88, 6.02] |
| thread size p99 | 240 [218, 256] | 38 [38, 39] | 232 [213, 256] | 210 [207, 213] | 101 [95, 106] | 31 [30, 32] |
| max depth p99 | 29 [28, 30] | 13 [13, 14] | 9 [9, 10] | 14 [14, 14] | 9 [9, 10] | 2 [2, 2] |
| root fan-out (live roots) | 3.44 [3.27, 3.61] | 2.91 [2.77, 3.16] | 6.85 [6.56, 7.26] | 1.90 [1.88, 1.91] | 2.24 [2.17, 2.31] | 5.41 [5.37, 5.45] |
| non-root mean replies | 0.781 [0.777, 0.787] | 0.610 [0.591, 0.620] | 0.493 [0.481, 0.506] | 0.817 [0.816, 0.818] | 0.645 [0.639, 0.651] | 0.105 [0.103, 0.107] |
| non-root mean replies, per-author | 0.650 [0.640, 0.659] | 0.488 [0.403, 0.545] | 0.421 [0.403, 0.440] | 0.666 [0.662, 0.670] | 0.600 [0.591, 0.608] | 0.072 [0.067, 0.078] |
| share of comments with ≥ 1 reply | 0.523 [0.519, 0.528] | 0.517 [0.501, 0.526] | 0.329 [0.322, 0.336] | 0.471 [0.471, 0.472] | 0.444 [0.441, 0.447] | 0.082 [0.081, 0.084] |
| fan-out at depth 1 | 0.778 [0.763, 0.791] | 0.576 [0.531, 0.602] | 0.438 [0.422, 0.455] | 1.275 [1.270, 1.280] | 0.835 [0.824, 0.846] | 0.116 [0.114, 0.118] |
| fan-out at depth 2 | 0.806 [0.795, 0.818] | 0.636 [0.627, 0.643] | 0.596 [0.585, 0.607] | 0.843 [0.840, 0.845] | 0.552 [0.542, 0.562] | 0.012 [0.011, 0.014] |
| OP return (complete authorship) | 0.410 [0.397, 0.424] | 0.725 [0.724, 0.726] | 0.381 [0.375, 0.389] | 0.168 [0.166, 0.170] | 0.721 [0.713, 0.731] | 0.146 [0.144, 0.149] |
| OP return (all threads, lower bound) | 0.326 [0.318, 0.334] | 0.613 [0.612, 0.614] | 0.400 [0.395, 0.407] | 0.193 [0.191, 0.195] | 0.744 [0.738, 0.754] | 0.146 [0.144, 0.149] |
| reciprocity (complete authorship) | 0.241 [0.229, 0.249] | 0.483 [0.482, 0.484] | 0.174 [0.169, 0.177] | 0.134 [0.133, 0.135] | 0.366 [0.357, 0.376] | 0.047 [0.046, 0.049] |
| reciprocity (all threads) | 0.263 [0.258, 0.268] | 0.393 [0.369, 0.408] | 0.135 [0.131, 0.138] | 0.160 [0.159, 0.160] | 0.296 [0.290, 0.303] | 0.047 [0.046, 0.048] |
| Sarle BC, log distinct commenters | 0.398 [0.374, 0.422] | 0.535 [0.519, 0.546] | 0.497 [0.478, 0.517] | 0.419 [0.411, 0.426] | 0.430 [0.411, 0.449] | 0.278 [0.265, 0.292] |

### Step 3 kernels

Weibull first-reply fit on the capped fit fold. 100 thread-bootstrap replicates, warm-started.

| | μ_first (Weibull) [95% CI] | Weibull shape [CI] | kernel half-life [CI] | threads (capped fit fold) |
|---|---|---|---|---:|
| r/philosophy | 0.750 [0.738, 0.762] | 0.523 [0.517, 0.529] | 1.54 h [1.49, 1.58] | 11,867 |
| r/buildapc | 0.732 [0.702, 0.754] | 0.421 [0.419, 0.424] | 0.49 h [0.48, 0.50] | 579,011 |
| Ask HN | 0.398 [0.385, 0.410] | 0.543 [0.540, 0.547] | 2.36 h [2.30, 2.43] | 18,331 |
| HN link | 0.648 [0.646, 0.649] | 0.592 [0.590, 0.593] | 1.57 h [1.57, 1.58] | 102,021 |
| Show HN | 0.591 [0.585, 0.598] | 0.559 [0.554, 0.564] | 1.94 h [1.89, 1.99] | 12,287 |
| Moltbook | 0.059 [0.058, 0.061] | 0.374 [0.368, 0.379] | 0.34 h [0.31, 0.36] | 67,014 |
| Moltbook, depth1 | 0.065 [0.064, 0.067] | 0.374 [0.369, 0.379] | 0.34 h [0.31, 0.36] | 67,014 |
| Moltbook, depth2plus | 0.010 [0.008, 0.011] | 0.369 [0.351, 0.395] | 0.82 h [0.50, 1.37] | 17,541 |

The log-normal's held-out advantage over the Weibull (log-likelihood per comment, held-out fold, 200
replicates), using the Step 3b parameters:

| | r/philosophy | r/buildapc | Ask HN | HN link | Show HN | Moltbook |
|---|---|---|---|---|---|---|
| log-normal − Weibull | +0.065 [+0.060, +0.071] | +0.124 [+0.121, +0.127] | +0.029 [+0.028, +0.032] | +0.051 [+0.050, +0.051] | +0.038 [+0.035, +0.041] | +0.008 [+0.008, +0.009] |

Every interval excludes 0.

### Step 3b

**K4 (author activity within threads).** These already used thread-clustered SEs. After the HN
recomputation without dead items:

| | r/philosophy | r/buildapc | Ask HN | HN link | Show HN | Moltbook |
|---|---|---|---|---|---|---|
| coefficient (SE) | +0.017 (0.002) | −0.001 (0.006) | +0.003 (0.003) | +0.011 (0.001) | +0.026 (0.003) | +0.000 (0.004) |

That is 0–6% more replies per 10× activity.

**Early/late split.** Share of comments whose first reply falls in each window (1-day horizon), with the
human ÷ Moltbook ratio; 95% CIs from independent thread bootstraps:

| | < 10 min, per-author | 1 h–1 d, per-author | ratio < 10 min, per-author | ratio 1 h–1 d, per-author | ratio < 10 min, raw | ratio 1 h–1 d, raw |
|---|---|---|---|---|---|---|
| r/philosophy | 0.079 [0.076, 0.081] | 0.184 [0.181, 0.188] | 4.1 [3.8, 4.4] | 9.3 [8.1, 10.6] | 4.1 [3.9, 4.2] | 17.6 [16.7, 18.6] |
| r/buildapc | 0.183 [0.158, 0.204] | 0.095 [0.082, 0.106] | 9.5 [8.0, 11.0] | 4.8 [4.0, 5.7] | 8.9 [8.6, 9.2] | 8.8 [8.3, 9.4] |
| Ask HN | 0.041 [0.039, 0.043] | 0.132 [0.127, 0.137] | 2.1 [1.9, 2.3] | 6.7 [5.8, 7.7] | 1.5 [1.5, 1.6] | 13.6 [12.8, 14.4] |
| HN link | 0.079 [0.078, 0.080] | 0.170 [0.169, 0.171] | 4.1 [3.8, 4.5] | 8.6 [7.5, 9.8] | 3.0 [2.9, 3.0] | 16.9 [16.0, 17.8] |
| Show HN | 0.085 [0.082, 0.088] | 0.183 [0.180, 0.187] | 4.4 [4.0, 4.8] | 9.3 [8.1, 10.5] | 2.7 [2.6, 2.8] | 17.0 [16.1, 17.9] |
| Moltbook | 0.019 [0.018, 0.021] | 0.020 [0.017, 0.023] | – | – | – | – |

**Correction to Step 3b.** I wrote that per author the Moltbook gap is "roughly constant at about 5–10×". With
intervals and population-by-population ratios, **it widens with delay in 4 of 5 human populations** (for
example r/philosophy 4.1× → 9.3×) and narrows only on r/buildapc. `reports/step3b.md` and the register are
corrected.

### Step 4: thread-clustered SEs and intervals

Per-author weighted. 200 thread-bootstrap replicates; the fit and held-out folds are resampled
independently.

| proxy | population | threads (fit) | β [95% CI] | SE: model / **thread-clustered** | flattening [95% CI] | held-out gain [95% CI] |
|---|---|---:|---|---|---|---|
| (a) | r/philosophy | 9,037 | −0.362 [−0.384, −0.337] | 0.004 / **0.013** | +0.021 [+0.018, +0.023] | +0.046 [+0.038, +0.054] |
| (a) | r/buildapc | 497,408 | −0.268 [−0.372, −0.142] | 0.0015 / **0.060** | +0.016 [+0.003, +0.032] | +0.036 [+0.008, +0.074] |
| (b) | r/philosophy | 2,678 | +0.037 [−0.075, +0.131] | 0.051 / 0.048 | +0.004 [−0.007, +0.016] | +0.000 [−0.001, +0.001] |
| (b) | r/buildapc | 380,895 | −0.320 [−0.352, −0.293] | 0.009 / 0.015 | −0.012 [−0.014, −0.011] | +0.005 [+0.004, +0.008] |
| (b) | Ask HN | 3,750 | −0.273 [−0.367, −0.161] | 0.036 / 0.053 | −0.042 [−0.054, −0.028] | +0.003 [−0.002, +0.010] |
| (b) | HN link | 13,132 | +0.056 [+0.025, +0.086] | 0.014 / 0.015 | +0.008 [+0.004, +0.014] | −0.000 [−0.002, +0.001] |
| (b) | Show HN | 986 | +0.707 [+0.586, +0.901] | 0.069 / 0.082 | **+0.115** [+0.095, +0.144] | −0.002 [−0.038, +0.027] |
| (c) | r/Fantasy contest | 580 | −0.219 [−0.506, −0.082] | 0.016 / **0.117** | +0.010 [+0.002, +0.029] | +0.025 [+0.007, +0.063] |
| (c) | r/Fantasy control (20+ comments) | 17,074 | −0.459 [−0.484, −0.426] | 0.004 / 0.015 | +0.030 [+0.027, +0.032] | +0.071 [+0.059, +0.081] |
| (c) | r/AskHistorians random (removed dropped) | 414 | −0.141 [−0.223, +0.039] | 0.023 / 0.052 | +0.004 [+0.000, +0.010] | −0.013 [−0.043, +0.008] |
| (c) | r/AskHistorians control (20+ comments) | 278 | −0.233 [−0.314, −0.119] | 0.017 / 0.046 | +0.006 [+0.001, +0.010] | +0.026 [+0.013, +0.041] |
| (d) | r/philosophy | 130 | −0.096 [−0.200, +0.009] | 0.058 / 0.059 | +0.018 [−0.002, +0.039] | −0.001 [−0.007, +0.002] |
| (d) | r/buildapc | 919 | −0.262 [−0.452, −0.010] | 0.006 / **0.143** | +0.151 [+0.005, +0.280] | −0.005 [−0.008, +0.001] |

Contrasts in proxy (c), from paired independent bootstraps:

| | difference in β [95% CI] | difference in decay slope M0 [95% CI] |
|---|---|---|
| r/Fantasy contest − control | [−0.04, +0.38] | [−0.03, +0.15] |
| r/AskHistorians random − control | [−0.02, +0.28] | [−0.31, −0.16] (different thread types) |
| r/AskHistorians "matched series" | [−0.14, +0.36] | [−0.11, +0.19] (time- and sort-confounded; check 8) |

**What changes in Step 4.**

1. **Model-based SEs understated uncertainty by up to 95×** where few large threads dominate:
   - (a) r/buildapc: 0.0015 vs 0.060 clustered;
   - (d) r/buildapc: 0.006 vs 0.143;
   - (c) r/Fantasy contest: 0.016 vs 0.117.

   Thread-clustered SEs replace them in every Step 4 table.
2. **"About half of the arrival-rank effect is display position" is not established.** The r/Fantasy
   contest-vs-control difference in β has a 95% CI of [−0.04, +0.38]: compatible with no display effect
   and with all of it. Within contest threads arrival rank still matters (β −0.22 [−0.51, −0.08]).
3. **The flattening conclusions hold:**
   - (b) is within ±0.05 everywhere except Show HN (+0.115, 986 threads, no held-out gain);
   - (c) shows no detectable difference in decay;
   - (d) r/buildapc's flattening has a wide interval (+0.005 to +0.28) and no held-out gain.
4. **The r/AskHistorians matched-series control is withdrawn as a control** (check 8).

## 4. Changes made during the audit

| change | files | effect on earlier results |
|---|---|---|
| HN dead items excluded as parents, events and siblings | `forum_model/units.py`, `scripts/step4_position.py` | HN μ within 0.5%; Step 3 HN μ_first: HN link 0.645 → 0.646, per-author 0.509 → 0.545; K4 HN link 0.018 → 0.011; Step 4 (b) HN β: Ask HN −0.260 → −0.273, HN link +0.065 → +0.056, Show HN +0.663 → +0.707. Step 3, 3b and 4 HN outputs recomputed. |
| Step 4: thread-clustered SEs and bootstrap intervals | `forum_model/pem.py`, `scripts/step4_position.py` | Point estimates unchanged; SEs and the (c) contrast as above |
| Kernel competitors, MCF and KM estimators, bootstrap helpers | `forum_model/hazard.py` | – |
| Shared population definitions | `forum_model/units.py` | Step 3 reproduces its Moltbook summary exactly (max difference 0.0) |
| Bootstrap of per-author comment-level means: weights fixed at 1/(author's comments in the full sample) | `scripts/audit_uncertainty.py` | A first run recomputed author means inside each replicate; that biased the replicates (HN link estimate 0.666 fell outside its interval [0.683, 0.689]). After the fix every estimate in the audit lies inside its interval. Point estimates unchanged. |
| Step 3b early/late claim corrected | `reports/step3b.md`, `docs/results_register.md` | Per-author gap widens with delay in 4 of 5 human populations |
| μ redefined as censoring-adjusted mean offspring; first-reply integral renamed μ_first | `docs/definitions.md` | See §2 |

## 5. Open items needing a decision

1. **Proxy (c) has no clean within-series control for either subreddit** (check 8). r/Fantasy rests on the
   placebo against ordinary threads; r/AskHistorians' series changed sort over time. Options: accept proxy
   (c) as a placebo check only, or look for contest-mode toggles in other GECLM subreddits (census in Step 2).
2. **Proxy (a) tie-breaking (check 9):** 54% of r/buildapc sibling groups tie on final score. Report (a)
   with ties broken at random as a sensitivity, or drop (a)? It is look-ahead-affected anyway.
3. **μ horizon:** 3 days (primary) vs 7 days. Moltbook differs by 3%. Keep 3 days?
4. **Step 4b is not defined in the brief or in our decisions so far.** Please say what it should cover
   (for example, estimating the position-bias curve with lateness controls). I have stopped here.

## Fitted-on / validated-on

| quantity | data |
|---|---|
| Checks | Whole processed datasets (description only, no fitting), except check 3 (inner fit fold) |
| μ, p_reply, median delay, early/late, Step 2 intervals | Calibration split (both inner folds), descriptive |
| Weibull bootstrap; Step 4 estimates and intervals | Calibration inner fit fold; held-out gains on the inner held-out fold |
| Log-normal − Weibull | Inner held-out fold |

**The 20% validation split was not used anywhere in the audit.**
