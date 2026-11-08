# Step 2: Descriptive structure

Status: done. **Nothing has been fitted.** Waiting for confirmation before Step 3.

Script: `scripts/step2_structure.py`, with subcommands `census`, `reddit`, `hn`, `moltbook_crawl`,
`figures` and `keynumbers`. Shared code: `forum_model/structure.py`, `spam.py`, `census.py`.

Tables are in `outputs/<dataset>/step2_*.csv`:

- `step2_summary_long.csv`: every metric, in long format.
- `step2_summary_wide.csv`: the same metrics, wide.
- `step2_fanout_by_depth.csv`
- `step2_ccdf.csv`: the distributions behind the figures.

All headline numbers are collected in `outputs/combined/step2_key_numbers.csv`. Figures are in
`figures/combined/step2_*.png` and `figures/<dataset>/step2_fanout_variants.png`.

## Decisions applied (from the Step 1 confirmation)

- **Moltbook source:** `giordano-dm/moltbook-crawl` is primary; the archive is a cross-check only, from
  2026-02-08 on.
  - The archive cross-check was not run here. Step 2's window (2026-01-28 to 02-04) does not overlap the
    archive's usable period.
  - Moltbook structure uses the whole thread window. The two contiguous segments are recorded in
    `partitions.json` and reserved for Steps 3 and 6.
- **Truncation:**
  - Moltbook thread size uses the API-reported `comment_count` for every non-spam post.
  - Moltbook shape statistics use only completely stored threads (stored ≥ reported):
    97,996 calibration threads, 65.5% of non-spam calibration posts. In the whole window, only 1 complete
    non-spam thread has 100 or more comments. So "complete" and "complete, under 100" are effectively the
    same population.
  - Reddit and HN shape statistics are reported for the full population and size-matched (threads under
    100 comments), side by side.
- **Spam filter (De Marzo et al.):**
  - Rule: a post with 5+ stored comments is excluded if fewer than 50% of its comments have unique
    content or fewer than 20% have unique authors.
  - Result: 2,822 of the 189,961 window posts are excluded (2,177 on content, 1,463 on authors).
  - Those posts hold 131,649 stored comments (11%) but **992,068 API-reported comments (32% of all
    reported comments)**. Spam is concentrated in the mega-threads.
  - Table: `outputs/moltbook_crawl/step2_spam_filter.csv`.
- **Bots:** the primary filter is the explicit list. The name rule is reported only as the `namerule`
  sensitivity variant, and its numbers are almost identical to the primary filter's in every table. The
  380k blocklist is not used.
  - **One addition:** BernardJOrtcutt, the r/philosophy moderation bot, was not on the list. It posts a
    rules reminder in 78% of r/philosophy threads (42,622 comments) and was distorting every r/philosophy
    statistic. It counts as a known utility bot, so it is now in `config/reddit_bot_authors.txt`. The
    Reddit bot flags were refreshed in place (`step1_reconstruct.py refresh_reddit_bot_flags`) and the
    Step 1 report row was corrected.
  - **Check for others:** I reviewed every author who comments in more than 1% of a subreddit's threads.
    Nobody else in either subreddit is automated. `-UserRemoved-` (41,754 r/buildapc comments) looks like
    a human account despite its name; it is kept.
  - On HN, `dang` is the human moderator and is kept.
- **Position proxies:** proxy (b) is the main identification. Proxy (d), threads with
  `suggested_sort = new`, is added. HN uses proxy (b) only. All recorded in `partitions.json`; nothing
  about position is computed in Step 2.
- **Scope:** every number below comes from the **calibration split only** (80% of threads). Validation
  threads have not been touched.

### Filter variants and populations

| variant | what is removed |
|---|---|
| `base` | authors on the explicit bot list. Nothing on HN or Moltbook. |
| `no_top1` | `base` plus the top 1% most active authors. |
| `namerule` | Reddit only: `base` plus authors matching the bot-name rule. |

Removing an author drops their comments; the children of those comments keep their original depth.

| population | threads included |
|---|---|
| Reddit/HN `full` | all calibration threads |
| Reddit/HN `lt100` | calibration threads with fewer than 100 stored comments |
| Moltbook `claimed` | thread size only, from the API-reported count, unfiltered |
| Moltbook `complete` | completely stored threads |

### How each statistic is defined

- **Raw vs per-author.** Comment-level statistics are averaged within each author, then across authors.
  Thread-level statistics weight each thread by 1 / (number of threads by that root author).
- **OP return.** Computed over threads with at least one non-OP comment. Three versions:
  - `complete`: root author and all comment authors known.
  - `root_known`: root author known.
  - `all`: unknown root author counts as "did not return", so this is a lower bound.
- **Reciprocity.** For each unordered pair of authors, does a reply go in both directions? Pairs are
  counted within each thread; a global version pools all threads. Self-replies and unknown authors are
  excluded. Reported over all threads and over threads with complete authorship.
- **Bimodality.** Threads with 20+ comments. I use Sarle's bimodality coefficient (above 0.555 suggests
  bimodality) on log(distinct commenters) and on distinct ÷ size, plus Hartigan's dip statistic. With
  thousands of threads every dip p-value is about 0, so the dip *statistic* is the informative number.

## Key numbers (variant `base` unless stated)

### Thread size

`figures/combined/step2_size_ccdf.png`

| | threads | median | mean | p99 | ≥ 20 comments | mean, per-author | mean, `no_top1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| r/philosophy | 58,242 | 0 | 8.9 | 240 | 4.8% | 3.8 | 6.9 |
| r/buildapc | 1,086,593 | 4 | 7.1 | 38 | 3.7% | 5.9 | 3.9 |
| Ask HN | 32,391 | 2 | 12.7 | 232 | 9.5% | 11.3 | 9.0 |
| HN link | 824,983 | 0 | 8.6 | 210 | 6.7% | 7.0 | 4.6 |
| Show HN | 33,105 | 0 | 5.5 | 101 | 6.4% | 6.1 | 4.1 |
| **Moltbook, API-reported count** (non-spam) | 149,706 | 4 | 10.3 | 35 | 4.4% | 10.1 | n/a (unfiltered) |
| Moltbook, stored, complete threads | 97,996 | 4 | 6.0 | 31 | 4.2% | 5.9 | 1.3 |

Moltbook's reported sizes have the heaviest far tail: the largest non-spam post reports 158,067 comments,
and it lies above every human curve past about 2,000 comments.

### Max depth (threads with ≥ 1 comment), full | size-matched

`figures/combined/step2_maxdepth_ccdf.png`

| | median | p90 | p99 | max |
|---|---|---|---|---|
| r/philosophy | 1 \| 1 | 11 \| 7 | 29 \| 16 | 127 \| 66 |
| r/buildapc | 3 \| 3 | 7 \| 7 | 13 \| 13 | 271 \| 81 |
| Ask HN | 2 \| 2 | 5 \| 4 | 9 \| 7 | 25 \| 25 |
| HN link | 2 \| 1 | 9 \| 6 | 14 \| 11 | 64 \| 37 |
| Show HN | 1 \| 1 | 5 \| 5 | 9 \| 8 | 17 \| 17 |
| **Moltbook** (complete) | **1** | **2** | **2** | 16 |

Moltbook's shallowness is not a size artefact. Even size-matched human threads reach a p99 depth of
7–16; Moltbook's is 2.

### Fan-out: root vs non-root

`figures/combined/step2_fanout_by_depth.png`

Mean direct replies, full | size-matched (Moltbook: complete threads):

| | root, all roots | root, live roots | depth 1 | depth 2 | all non-root | non-root, per-author | share of comments with ≥ 1 reply | root / non-root, `no_top1` |
|---|---|---|---|---|---|---|---|---|
| r/philosophy | 1.93 \| 0.76 | 3.44 \| 1.23 | 0.78 \| 0.56 | 0.81 \| 0.61 | 0.78 \| 0.62 | 0.65 \| 0.54 | 0.52 \| 0.47 | 1.67 / 0.59 |
| r/buildapc | 2.78 \| 2.19 | 2.91 \| 2.26 | 0.58 \| 0.57 | 0.64 \| 0.64 | 0.61 \| 0.61 | 0.49 \| 0.54 | 0.52 \| 0.54 | 1.25 / 0.29 |
| Ask HN | 6.46 \| 3.42 | 6.85 \| 3.63 | 0.44 \| 0.41 | 0.60 \| 0.48 | 0.49 \| 0.42 | 0.42 \| 0.42 | 0.33 \| 0.33 | 4.90 / 0.31 |
| HN link | 1.58 \| 0.80 | 1.90 \| 0.95 | 1.28 \| 0.93 | 0.84 \| 0.67 | 0.82 \| 0.70 | 0.67 \| 0.58 | 0.47 \| 0.44 | 1.02 / 0.43 |
| Show HN | 1.97 \| 1.55 | 2.24 \| 1.76 | 0.84 \| 0.75 | 0.55 \| 0.49 | 0.65 \| 0.59 | 0.60 \| 0.57 | 0.44 \| 0.43 | 1.58 / 0.45 |
| **Moltbook** | **5.41** | 5.41 | **0.116** | **0.012** | **0.105** | 0.072 | **0.082** | 1.13 / 0.051 |

- "All roots" includes roots with no comments: 75% of HN link stories, and removed Reddit posts. "Live"
  excludes removed, deleted or dead roots.
- Below depth 1, human fan-out is roughly flat with depth, at 0.5–0.8.
- Moltbook's falls by an order of magnitude from depth 1 to depth 2. 89.5% of its stored comments are
  direct replies to the post.
- These are **empirical mean offspring, not branching-ratio estimates**. The branching ratio comes from
  the kernel integral in Step 3.

### Original-poster return

Base variant. Full-population values; size-matched values are within 0.02 of these except where noted.

| | complete authorship (n threads) | all threads, lower bound (n) | per-author (complete) | `no_top1`, complete / all |
|---|---|---|---|---|
| r/philosophy | 0.41 (4,325) | 0.33 (13,560) | 0.37 | 0.39 / 0.26 |
| r/buildapc | 0.73 (675,200) | 0.61 (944,600) | 0.72 | 0.74 / 0.57 |
| Ask HN | 0.38 (19,410) | 0.40 (23,000) | 0.39 | 0.39 / 0.41 |
| HN link | 0.17 (119,000) | 0.19 (173,600) | 0.26 | 0.19 / 0.21 |
| Show HN | 0.72 (8,712) | 0.74 (10,780) | 0.72 | 0.73 / 0.75 |
| **Moltbook** | **0.15** (83,532) | 0.15 (83,666) | 0.18 | 0.20 / 0.20 |

On r/philosophy, only 32% of the threads in the lower-bound set have complete authorship, because 18.7%
of comment authors are deleted.

### Dyadic reciprocity

Share of author pairs with replies in both directions, counted within threads.

| | complete authorship, full \| size-matched | all threads, full \| size-matched | per-author (all) | global pairs (all) | `no_top1` (all) |
|---|---|---|---|---|---|
| r/philosophy | 0.24 \| 0.24 | 0.26 \| 0.26 | 0.24 | 0.27 | 0.25 |
| r/buildapc | 0.48 \| 0.48 | 0.39 \| 0.46 | 0.38 | 0.40 | 0.29 |
| Ask HN | 0.17 \| 0.18 | 0.13 \| 0.16 | 0.12 | 0.14 | 0.11 |
| HN link | 0.13 \| 0.13 | 0.16 \| 0.14 | 0.12 | 0.18 | 0.12 |
| Show HN | 0.37 \| 0.37 | 0.30 \| 0.33 | 0.30 | 0.30 | 0.31 |
| **Moltbook** | **0.047** | **0.047** | 0.046 | 0.071 | 0.065 |

### Distinct commenters per thread (threads with ≥ 20 comments): bimodality check

`figures/combined/step2_distinct_commenters.png`

Values are full | size-matched (20–99 comments).

| | threads | median distinct | Sarle BC, log(distinct) | Sarle BC, distinct ÷ size | dip statistic |
|---|---|---|---|---|---|
| r/philosophy | 2,809 \| 1,587 | 30 \| 15 | 0.40 \| 0.34 | 0.36 \| 0.42 | 0.015 \| 0.026 |
| r/buildapc | 39,710 \| 35,440 | 9 \| 8 | **0.53** \| 0.40 | **0.53** \| **0.53** | **0.042** \| **0.047** |
| Ask HN | 3,061 \| 2,251 | 37 \| 27 | 0.50 \| 0.41 | 0.40 \| 0.44 | 0.015 \| 0.021 |
| HN link | 55,270 \| 33,430 | 50 \| 32 | 0.42 \| 0.42 | 0.33 \| 0.35 | 0.009 \| 0.015 |
| Show HN | 2,119 \| 1,781 | 29 \| 25 | 0.43 \| 0.43 | 0.36 \| 0.39 | 0.017 \| 0.020 |
| **Moltbook** (complete) | 4,132 | 14 | 0.28 | 0.40 | 0.042 |

- No population clears the 0.555 threshold on log(distinct commenters).
- r/buildapc comes closest. It has a visible mode at 3–6 distinct commenters (OP-and-helper exchanges)
  plus a broad crowd mode.
- Moltbook's 20+ comment threads cluster narrowly around 14 distinct agents.
- Moltbook `no_top1` reaches a ratio-BC of 0.56, but that rests on only 327 threads.

### Sort-mode census, all 50 GECLM subreddits

(`outputs/reddit/step2_sort_census.csv`; submissions created 2018–2021; comment counts are the
submission's `num_comments` snapshot.)

Totals across the 50 subreddits:

| | threads | comments |
|---|---:|---:|
| `contest_mode` true | 3,992 | about 230k |
| `suggested_sort = random` (also a randomized order) | 885 | – |
| `suggested_sort = new` | 19,332 | about 5.0M |
| `suggested_sort = qa` | 139,526 | – (almost all r/changemyview) |

| subreddit | contest-mode threads (with ≥ 20 comments) | contest comments | `random` | `new` threads (≥ 20 comments) | `new` comments |
|---|---:|---:|---:|---:|---:|
| WritingPrompts | 2,209 (42) | 7,066 | 0 | 0 | 0 |
| **Fantasy** | **996 (618)** | 44,858 | 2 | 817 (701) | 38,343 |
| gaming | 191 (190) | 15,085 | 0 | 149 | 15,477 |
| Fitness | 171 (55) | 5,333 | 0 | **2,488 (2,161)** | **1,641,471** |
| podcasts | 167 (150) | 44,035 | 0 | 0 | 0 |
| SkincareAddiction | 60 (20) | 872 | 0 | 1,454 (1,451) | 367,774 |
| malefashionadvice | 52 (35) | 3,091 | 0 | **3,259 (2,957)** | 696,419 |
| AskHistorians | 48 (5) | 749 | **736** | 432 | 24,963 |
| boardgames | 29 (28) | 31,497 | 0 | 1,853 (1,792) | 254,427 |
| books | 7 (6) | 410 | **146** | 260 | 62,024 |
| femalefashionadvice | 1 | 255 | 0 | 2,289 (2,141) | 399,859 |
| bodyweightfitness | 3 | 211 | 0 | 1,771 (1,606) | 132,813 |
| buildapc | 5 (5) | 39,934 | 0 | 1,441 (1,336) | 703,656 |
| philosophy | 2 (0) | 17 | 0 | 211 (205) | 17,414 |

Caveat: `suggested_sort` records a per-post override. A subreddit-wide default sort is stored on the
post only in some subreddits (r/buildapc, r/science and r/explainlikeimfive record `confidence` on almost
every post; most others record nothing). Both `new` and contest mode therefore mostly pick out stickied,
daily or mega threads, which are not typical threads.

## Fitted-on / validated-on

Nothing was fitted in Step 2.

Every statistic above is descriptive and computed on the **calibration split** of each dataset: Reddit
r/philosophy and r/buildapc 2018–2021; HN 2019–2021; Moltbook crawl 2026-01-28 to 02-04, non-spam. The
validation split (20% of threads) has not been read.

Depth, size shape, reciprocity, OP return and bimodality are on the **validation list**. Looking at them
here on calibration threads is descriptive only; none of them will be used as fit targets. Fan-out by
depth and the share of comments with a reply describe reply volume. They are not timing quantities and
do not replace the kernel-based branching ratio.

## Decisions needed before Step 3

1. **Proxy (c) data.** Two options:
   - Add r/Fantasy, which has the most usable contest-mode threads (618 with 20+ comments), possibly
     with r/gaming and r/podcasts (190 and 150 such threads).
   - Or use `suggested_sort = random` threads, which are also randomized: r/AskHistorians (736) and
     r/books (146).

   Each addition must go into `partitions.json` (reserved for the human baseline) before any fit.
2. **Proxy (d) data.** In r/buildapc and r/philosophy there are 1,652 `new` threads with about 720k
   comments, mostly daily or mega threads. Should proxy (d) stay within the two study subreddits, or
   also take in the large `new` sources (Fitness, malefashionadvice, femalefashionadvice, boardgames)?
3. **Moltbook depth in the Step 3 kernel.** The offspring pattern below suggests fitting depth-1 and
   depth-≥2 non-root kernels separately, at least as a check. Confirm whether to add that comparison.

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`.

- **S1. Root fan-out is a different process in every dataset, and most of all on Moltbook.** The ratio of
  root to non-root mean offspring is:

  | | full population | size-matched |
  |---|---|---|
  | r/philosophy | 2.5× | 1.2× |
  | r/buildapc | 4.6× | 3.6× |
  | HN link | 1.9× | 1.1× |
  | Ask HN | 13× | 8× |
  | Moltbook | **52×** | – |

  This supports a separate μ_0, as the reference expects.
- **S2. On Moltbook, offspring depends on depth, so one μ won't fit.** Depth-1 comments get 0.116
  replies and depth-2 comments get 0.012. A single depth-independent self-excitation kernel would
  average two very different processes. Humans are roughly flat at 0.5–0.8 below depth 1; the exception
  is HN link, where depth-1 comments get more replies (1.28) than deeper ones.
- **S3. Human mean offspring is above the reference's 0.45 prior.** Empirical non-root mean offspring is
  0.61 on r/buildapc and 0.78 on r/philosophy (0.40–0.57 at depth 1 when size-matched). The kernel
  integral in Step 3 is the estimate, but under a subcritical model it should land close to these means.
  If it comes out near 0.45, the kernel is missing mass, probably in the late tail.
- **S4. Removing the top 1% is a different population, not a robustness check.**
  - It removes 23% of comments on r/philosophy, about 45% on r/buildapc and HN link, and 78% on
    Moltbook.
  - It changes structure, not just scale. Moltbook root fan-out falls from 5.4 to 1.1. r/buildapc
    fan-out starts alternating by depth parity (`figures/reddit/step2_fanout_variants.png`), because deep
    r/buildapc chains are OP-and-helper ping-pong and removing the helpers deletes every other level.
  - Heavy-tailed μ_i is part of the mechanism. Results without the top 1% should be read as a different
    population, not as "the same answer, cleaner".
- **S5. OP return on HN needs an exposure channel the model doesn't have.** HN has no reply
  notifications (e_reply = none in the reference), yet Show HN OPs return 72% of the time and Ask HN OPs
  38%. They must be checking their own threads. As written, E_im would leave OPs with only σ, so OP
  return would be under-predicted on HN. An own-thread exposure term is needed. Moltbook OPs, also
  without notifications, return 15%.
- **S6. Agents almost never reciprocate.** Pair reciprocity is 0.047 on Moltbook against 0.13–0.48 for
  humans, and removing the top 1% doesn't change that (0.065). So ρ should be close to 0 for agents. The
  r/buildapc value (0.48) is driven by help dialogues.
- **S7. Bimodality in distinct commenters appears only on r/buildapc, and it is weak.** That subreddit
  sits near the threshold with a dialogue mode at 3–6 distinct commenters. A two-type thread mixture may
  be needed for help forums; nowhere else needs one. As a validation target, compare distributions, not
  dip p-values: at these sample sizes every p-value is about 0.
- **S8. "Complete authorship" is a size filter.** Any deleted author excludes a thread, and large threads
  almost always contain one. Complete-authorship statistics therefore come from small threads; on
  r/philosophy only 4,325 of 13,560 threads qualify. Read them next to the size-matched versions, as the
  tables above do.
- **S9. Moltbook shape statistics are conditional on small, complete threads.** The excluded mega-threads
  carry most reported comments: 65% of reported non-spam comments (1.36M of 2.10M) are in threads that are not
  completely stored. The model is validated on how agents behave in ordinary threads only. Censoring stored size at
  100 in Step 3 fits handles the trees but not that selection.
- **S10. Subreddit moderation bots must be on the bot list.** One unlisted moderation bot moved
  r/philosophy's OP return from 0.15 to 0.41 and its reciprocity from 0.11 to 0.24. Any subreddit added
  for proxy (c) or (d) needs the same check (authors in more than 1% of threads, reviewed by hand) before
  its statistics are used.
