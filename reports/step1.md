# Step 1: Thread reconstruction

Status: done. **Nothing has been fitted.** Waiting for confirmation before Step 2.

Script: `scripts/step1_reconstruct.py` (shared code in `forum_model/trees.py`, `split.py`, `authors.py`).
Windows, sources, seed and split rules: `partitions.json`. Tables: `outputs/<dataset>/step1_*.csv`.
Per-comment features (depth, direct replies, time since parent, time since root, is-root-author,
self-reply, author flags, split): `data/processed/<dataset>/items.parquet` (gitignored, regenerable).

## What was computed

For every item in in-window threads:

- **depth**: root = 0, top-level comment = 1.
- **direct replies**: number of children in the data. Bots and deleted authors are included;
  filters are applied at analysis time.
- **time since parent** and **time since root**, in seconds.
- **is_root_author**: the comment's author is the root's author. Null if either author is unknown.
- **self_reply**: the comment's author is the parent's author.

Trees are built by pointer jumping on `parent_id` (Reddit `t1_`/`t3_` prefixes stripped). Definitions:

| term | meaning |
|---|---|
| missing parent | `parent_id` matches no item in the data |
| orphan | a comment whose chain does not reach the root (the missing-parent comment plus all its descendants) |
| thread in window | root created inside the window. All of its comments are kept, even those created after the window ends, so reply delays are not cut off at the boundary. |

Data sources, all pinned to dataset revisions in `partitions.json`:

| dataset | source | window (root creation) |
|---|---|---|
| Reddit | `HuggingFaceGECLM/REDDIT_comments`, plus `HuggingFaceGECLM/REDDIT_submissions` for the root author, root time and `contest_mode` (the comments set has no submissions) | r/philosophy, r/buildapc, 2018-01-01 – 2021-12-31 |
| HN | `jkeisling/hacker-news-corpus-2007-2022`. BigQuery has no credentials here; this snapshot ends 2022-11-16. | 2019-01-01 – 2021-12-31 |
| Moltbook (archive) | `SimulaMet/moltbook-observatory-archive` | 2026-01-28 – 2026-02-04 |
| Moltbook (crawl) | `giordano-dm/moltbook-crawl`. **Added in this step; see finding M1.** | 2026-01-28 – 2026-02-04 |

Data hygiene set up in this step:

- **Calibration / validation split:** 20% of threads go to validation, split by thread and stratified
  by month (and by subreddit or HN post class). Assignment is deterministic: rank by
  sha256(seed:dataset:thread_id), seed 20261107. The validation share is within 0.0005 of 0.20 in
  every stratum with 50 or more threads.
- **Reddit bot filters:** an explicit list (`config/reddit_bot_authors.txt`) and a name rule, reported
  separately.
- **Top 1% most active authors:** flagged per subreddit for Reddit, and over the whole dataset for HN
  and Moltbook.

## Key numbers

### Integrity per dataset

| | r/philosophy | r/buildapc | HN (all classes) | Moltbook archive | Moltbook crawl |
|---|---:|---:|---:|---:|---:|
| threads in window | 72,797 | 1,358,245 | 1,187,341 | 187,362 | 189,961 |
| threads with ≥1 comment | 37,265 | 1,188,451 | 315,885 | **3** | 166,430 |
| comments | 698,507 | 9,693,736 | 9,765,245 | **1,127** | 1,181,994 |
| comments / platform-reported count | 1.06 | 1.04 | 1.04–1.13 | **0.0004** | **0.38** |
| missing parents | 156 | 23 | 0 † | 0 | 0 |
| orphans (incl. missing-parent heads) | 508 | 60 | 0 † | 0 | 0 |
| cross-thread parents / cycles | 0 / 0 | 0 / 0 | – / 0 | 0 / 0 | 0 / 0 |
| comments with negative time since parent | 2 | 0 | 593 | 0 | 0 |
| comments with zero time since parent | 0 | 4,643 ‡ | 18,182 | 0 | 0 |
| comment author deleted / unknown | 130,820 (18.7%) | 475,790 (4.9%) | 255,094 (2.6%) | 0 | 1,055 (0.1%) |
| ‑ body `[deleted]` | 30,803 | 311,354 | 255,053 (deleted) | 0 | 0 |
| ‑ body `[removed]` / HN `dead` | 86,823 | 53,519 | 180,619 (dead) | – | – |
| root author deleted / unknown | 18,111 (24.9%) | 257,639 (19.0%) | 45,362 § | 15 | 4 |
| comments by root author | 23,517 (3.4%) | 2,149,675 (22.2%) | 231,721 (2.4%) | 1 | 87,266 (7.4%) |
| comments where root-author status is undetermined | 165,846 | 1,459,759 | 255,449 | 0 | 1,056 |
| self-replies (author = parent author) | 6,607 | 132,722 | 111,757 | 1 | 68,778 (5.8%) |
| distinct comment authors | 124,370 | 855,506 | – ¶ | 385 | 15,841 |
| max depth | 127 | 271 | 64 | 2 | 32 |
| top-1% authors: count, share of comments | 1,425 / **27.4%** | 9,630 / **45.4%** | – / **44.9%** | 17 / 16.2% | 332 / **77.0%** |
| comments on the bot list | 42,664 ※ | 8,797 | n/a | n/a | n/a |
| comments caught by the bot-name rule | 953 | 7,839 | n/a | n/a | n/a |
| comments left after bot list + top 1% removed | 507,397 | 5,289,247 | 5,382,364 | 944 | 271,512 |
| threads, calibration / validation | 58,242 / 14,555 | 1,086,593 / 271,652 | 949,877 / 237,464 | 149,890 / 37,472 | 151,969 / 37,992 |

※ Updated in Step 2. BernardJOrtcutt, r/philosophy's moderation bot (42,622 comments), was added to the
explicit list. The Step 1 figure was 42. The counts after removing bots and the top 1% are unchanged,
because it was already a top-1% author.
† HN: inside in-window threads there are no missing parents. A further 3,239 comments descend from
1,553 parents absent from the dump, so their thread can't be determined; they are excluded. 45 comments
belong to threads that started before the window, and 3.5M belong to threads that started after it;
both are excluded by design.
‡ 4,568 of the 4,643 same-second replies are AutoModerator.
§ HN: 45,340 of these are deleted stories. They have no title or url, so they fall into `text_other`.
Only 986 of them have comments.
¶ HN distinct authors are counted per post class (Ask 84,063; link 211,513; Show 52,058). The top-1%
ranking is over all HN commenters.

HN by post class (the brief asks for Ask HN to be separated from link posts):

| class | threads | with comments | comments | dead roots | comments by root author | max depth |
|---|---:|---:|---:|---:|---:|---:|
| ask_hn | 40,486 | 29,193 | 514,039 | 2,466 | 28,410 (5.5%) | 25 |
| link | 1,031,228 | 255,745 | 8,912,677 | 187,847 | 134,533 (1.5%) | 64 |
| show_hn | 41,380 | 19,963 | 231,029 | 5,440 | 52,929 (22.9%) | 20 |
| text_other | 74,125 | 10,893 | 104,820 | 14,865 (+45,340 deleted) | 15,781 | 19 |
| poll | 122 | 91 | 2,680 | 5 (+11 deleted) | 68 | 14 |

Checks that the reconstruction is correct:

- **Moltbook crawl:** the crawler stores its own `depth`. My reconstructed depth disagrees with it for
  only 3 of 1,181,994 comments.
- **Reddit:** every `t3_` parent equals the comment's own `link_id` (0 violations). There are no
  duplicate comment or submission ids within the window.

### Contest-mode threads per subreddit (`outputs/reddit/step1_contest_mode*.csv`)

| | threads | contest mode | comments in them | `suggested_sort = new` (comments) |
|---|---:|---:|---:|---:|
| r/philosophy | 72,797 | **2** | 17 | 211 (18,666) |
| r/buildapc | 1,358,245 | **5** | 43,350 | 1,441 (739,961) |

Two r/buildapc megathreads (19,350 and 14,640 comments) make up 78% of all contest-mode comments.
No `random` suggested sort appears anywhere. HN and Moltbook have no contest mode.

### Score-snapshot staleness, `retrieved_on − created_utc`

(`outputs/*/step1_staleness_*.csv`; figures below)

| | median | 5%–95% | share < 1 h | share < 1 day | missing |
|---|---|---|---|---|---|
| r/philosophy comments | 74 d | 12.7 d – 218 d | 0 | 0.004% | 70,122 (10%, all 2021) |
| r/buildapc comments | 96 d | 17.7 d – 215 d | 0 | 0.009% | 916,719 (9.5%, all 2021) |
| Reddit roots | 60–62 d | 8.7 d – 303 d | 0.3–0.5% | 0.8–1.1% | 34–40% (2020–21 only) |
| HN | **not available**: the dump has no per-item retrieval time | | | | |
| Moltbook crawl comments (`crawled_at`) | **9.7 h** | 50 min – 1.8 d | 6.4% | 79% | 0 |
| Moltbook crawl roots | 7.4 d | 4.8 d – 9.6 d | 0.06% | 0.4% | 0 |
| Moltbook archive (latest snapshot; mostly the dump-date upper bound) | 10–12 h | – | 3–5% | 100% | 0 |

Reddit comment staleness grows by year: the r/buildapc median is 31 d in 2018, 93 d in 2019,
103 d in 2020 and 156 d in 2021. `retrieved_on` is missing for comments made January–June 2021 (the
gap in the monthly figure).

### Moltbook coverage

**Archive** (`figures/moltbook/step1_coverage.png`): the comment stream barely exists before
2026-02-08. Only 228 comments were created inside the window, and 154,820 were created on 02-08 alone.
For first-week threads the archive holds 1,127 comments; the posts report 2.86M.

**Crawl** (`figures/moltbook_crawl/step1_coverage.png`): comments stop for **41.6 h, from
2026-01-31 10:39 to 2026-02-02 04:16 UTC**. This is the brief's 41.7 h gap, which the crawl's README
explains as a platform outage that disabled commenting while posts continued. There is also a 17.6 h
quiet stretch on 01-28 at launch, when posts stop too.

Contiguous comment segments inside the window (`outputs/moltbook_crawl/step1_contiguous_comment_segments.csv`):

| segment | duration | comments |
|---|---|---:|
| 01-28 19:24 → 01-31 10:39 | 63.2 h | 195,512 |
| 02-02 04:16 → 02-04 23:59 | 67.7 h | 862,757 |

**Crawl truncation** (`outputs/moltbook_crawl/step1_comment_cap.csv`). The crawl keeps about 100
comments per post.

- Threads reporting 100 or fewer comments (99.5% of threads): 89% of reported comments are stored.
- Threads reporting more than 100 comments (913 threads): they account for **62% of all reported
  comments** (1.91M of 3.10M), and only **6.4%** of those are stored.
- The largest stored thread has 1,017 comments, so the cap is not a strict 100.

### Figures

- `figures/reddit/step1_staleness.png`: distribution of comment and root staleness (log axes).
- `figures/reddit/step1_staleness_by_month.png`: median and 10–90% band by month.
- `figures/moltbook_crawl/step1_staleness.png`, `figures/moltbook/step1_staleness.png`
- `figures/moltbook_crawl/step1_coverage.png`, `figures/moltbook/step1_coverage.png`: posts and
  comments per hour (zero-filled), with the thread window shaded.

## Fitted-on / validated-on

Nothing was fitted in Step 1. The calibration/validation split was created
(`data/processed/<dataset>/split.parquet`; counts in `outputs/<dataset>/step1_split_counts.csv`). No
quantity from either side has been used for anything yet. All counts above are descriptive and cover
both splits.

## Decisions needed before Step 2

1. **Moltbook source.** The archive cannot support the first-week analysis (M1). I propose using
   `moltbook-crawl` as the primary Moltbook source, with the same window and the two contiguous
   segments above. The archive would be kept only as a cross-check for the period after 02-08.
2. **Moltbook truncation.** Choose one way to handle the crawl's per-post cap (M2):
   - restrict size- and depth-shape validation to threads that are stored completely
     (stored ≥ reported; 65.5% of threads), or
   - report the targets with the truncation stated, and treat thread size as censored above about 100.
3. **Contest mode (proxy c).** It is nearly empty (R1). The options are:
   - add subreddits that use contest mode routinely (they must be recorded in `partitions.json` before
     any fit), or
   - accept that (c) is a 7-thread case study, and rest the position check on (b) and HN.
4. **Bot list.** I could not find a single standard "Pushshift-era" list. The provisional list is
   `config/reddit_bot_authors.txt`: AutoModerator, a public bot list, and well-known utility bots.
   The name rule mixes real bots (UnitConvertBot, AmputatorBot, CakeDay--Bot, TitleLinkHelperBot) with
   probable humans (JohnCabot, RagnarLothbot, NotObviouslyARobot). Its total volume is small (953 and
   7,839 comments). Please send the list you consider standard. Otherwise I'll promote only the
   clearly automated name-rule hits into the explicit list and drop the rule.

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`.

- **R1. Proxy (c) is nearly unusable in these partitions.** There are 7 contest-mode threads, and two
  megathreads hold 78% of their comments. The Step 4 claim that time decay flattens under (c) cannot be
  tested meaningfully on r/philosophy and r/buildapc as they stand.
  `suggested_sort = new` threads (1,652, with 759k comments) are not a substitute. There, display
  position is the reverse of arrival order, so position and age are perfectly collinear.
- **R2. Proxy (a) uses effectively final scores, and the look-ahead varies by year.** No Reddit comment
  score was captured within 1 hour. The median capture lag is 2.5–3 months and doubles from 2018 to
  2021. Two consequences:
  - The circularity-1 warning applies at full strength.
  - Any year-to-year comparison under proxy (a) mixes different amounts of post-hoc voting.
- **R3. Cross-population look-ahead is not like-for-like.** Moltbook crawl comment scores were captured
  about 10 h after posting; Reddit's were captured months later. Proxy (a) means something different
  on each platform and should not be compared across them.
- **M1. Moltbook source.** The archive named in the brief has essentially no comments for the first
  week. The 41.7 h gap and the first-week framing in the brief match `moltbook-crawl`. The prior
  estimates in the reference (Eziz 2026: tail slope 0.154, half-life ≈ 0.8 min, Weibull shape 0.11)
  should be checked for which source they came from before comparing against them.
- **M2. Moltbook thread size is censored by the crawler.** 62% of reported comments sit in 913 large
  threads that are stored only to about 6%. This hits three validation targets: size shape, depth, and
  bimodality of distinct-commenter counts. If left uncorrected, Moltbook will look smaller and
  shallower than it is, and the human-vs-agent contrast in branching (expected μ well under 0.2) would
  be partly a crawler artefact.
- **M3. Moltbook censoring is per thread.** The crawl ends at different times for different threads,
  and the comment stream tails off after about 02-08 as the crawler falls behind. For the Step 3
  survival fits, each thread's observation end should be its crawl time, not a single data end.
  Fits that depend on periodicity must use the two contiguous segments.
- **A1. Activity is much more concentrated in some populations.** The top 1% of authors write 27% of
  comments on r/philosophy, about 45% on r/buildapc and HN, and 77% on Moltbook. A single agent writes
  11.4% of all Moltbook comments. This fits a heavy-tailed μ_i, but the "without top 1%" sensitivity
  analysis removes very different fractions of the data in each population, so results with and
  without it are not directly comparable across datasets.
- **A2. Original-poster return has a large undefined share.** The root author is unknown for 19–25% of
  Reddit threads (deleted accounts). Return rate and β_root can only be measured on threads with a
  known root author, which is a selected subset.
- **H1. HN post classes need a deleted/dead class.** Deleted HN stories lose their title and url and
  land in `text_other`. 18% of link stories are `dead`, and most of these have no comments. For the
  Step 5 Ask-HN versus link comparison, deleted and dead roots should be excluded or kept as their own
  class.
