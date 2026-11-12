# Step 6b: Moltbook late-reply checks

**Calibration split only; exploratory.** The tests and readings (R26, R27) and the rules
(`partitions.json` → `step6b`, `docs/definitions.md` → "Sweeper bump and late-reply authorship") were
committed in `e97aa6e` **before** this step was run.

- **Script:** `scripts/step6b_agents_tail.py`, seed 20261107 + 62, B = 200, log `data/step6b.log`, peak
  memory 2.8 GB.
- **Outputs:**
  - `outputs/moltbook_crawl/step6b_sweepers.csv`, `step6b_sweepers_bins.csv`,
    `figures/moltbook_crawl/step6b_sweepers.png`;
  - `outputs/moltbook_archive/step6b_late_authors.csv`, `step6b_late_authors_top10.csv`,
    `step6b_spike_bins.csv`.

## 6b.1 Crawl root bump and the two sweepers (R26)

`Stromfee` and `FinallyOffline` are the sweepers found in audit check 7. They reply to posts nobody has
answered, with median delays of about 2.5 days. Their replies were removed as events, and the crawl root
all-reply intensity (Step 6 definition) was recomputed.

- **They write 19.3% of all root replies, and 68% of root replies in the 1–2-day window.**
- **Bump ratio R** compares replies in the 1–2-day window with a log-log fit to the neighbouring bins
  (0.25–0.9 d and 2.2–6 d):

| | R (per-author) | R (raw) |
|---|---|---|
| all replies | 2.74 [2.64, 2.86] | 2.57 [2.52, 2.63] |
| sweepers excluded | **1.39 [1.29, 1.49]** | 1.46 [1.40, 1.52] |
| difference | 1.35 [1.28, 1.42] | 1.11 [1.06, 1.15] |

**Against the pre-stated rule — fails.** The bump shrinks from +174% to +39% excess (the sweepers explain
about 80% of it), but it does not disappear.

The figure (`figures/moltbook_crawl/step6b_sweepers.png`) also shows the sweepers raising root intensity
from under an hour on, not only at 1–2 days. The residual bump is not explained here.

## 6b.2 Who writes the archive's late replies (R27)

Archive cross-check window, default events. Late = reply delay 3–30 d; early = ≤ 3 d. Concentration among
reply authors, with 95% thread-bootstrap CIs:

| unit | window | replies | distinct authors | top-1 share | top-5 share | HHI |
|---|---|---|---|---|---|---|
| roots | late | 1,903 | 744 | 0.02 [0.02, 0.16] | **0.10 [0.08, 0.46]** | 0.005 [0.004, 0.058] |
| roots | early | 141,942 | 5,267 | 0.18 [0.18, 0.18] | 0.40 [0.39, 0.40] | 0.048 |
| comments | late | 243 | 65 | 0.15 [0.14, 0.30] | **0.54 [0.49, 0.69]** | 0.073 [0.064, 0.14] |
| comments | early | 5,806 | 863 | 0.03 [0.02, 0.05] | 0.11 [0.10, 0.17] | 0.007 |

**Reading (pre-stated: concentrated if top-5 share > 0.5):**

- **Late replies to posts are not concentrated across authors.** They are *less* concentrated than early
  replies, whose top-5 share is dominated by fast pollers.
  - They are concentrated *within threads*. Six of the top ten late authors put all their late replies
    into one or two posts: `AlphaSwarm_SAGE` 41, `AlphaSwarm_CONTRARIAN` 41, `AlphaSwarm_ALPHA` 22,
    `lyralink` 32, `MarvinMSPN` 31, `Only` 29.
  - Others spread one reply per post across 21–25 posts (`cybercentry`, `alphaA`, `sola4x`): sweeper
    behaviour.
  - The wide upper CIs come from those single-thread bursts.
- **Late replies to comments are concentrated.** The top-5 share is 0.54, and its CI lower bound is 0.49.
  - `ordinals` (37 replies, 34 threads) and `garymetaz` (37 replies, 20 threads) write 30% of them.
  - Most top late repliers write no early replies at all (late fraction of their own replies = 1.0).
- **No 3–30-day bin exceeds 2× the fitted power law in raw counts.** The spikes visible in the Step 6
  figure come from per-author weighting: posts by one-post authors carry full weight, so one burst on
  such a post stands out. They are not platform-wide events.

Author IDs and names are in `outputs/moltbook_archive/step6b_late_authors_top10.csv`.

## Fitted-on / validated-on

| quantity | fitted on | validated on |
|---|---|---|
| Bump ratio R with and without sweepers | Calibration split (crawl, complete non-spam threads) | Not validated. R26 test pre-stated |
| Late-reply author concentration | Calibration split of the archive cross-check window | Descriptive. No validation test (archive is cross-check only) |

## Implications for the model

`docs/model_reference.md` unchanged apart from the status table, as instructed.

1. **Agent late replies come from identifiable behaviours, not a diffuse σ:**
   - sweepers that reply to unanswered posts days later;
   - multi-agent "swarms" that pile dozens of replies into one old post;
   - a few accounts that reply to old comments and never early.

   Agent σ = 0 should be replaced by explicit sweeper and burst agents, or those accounts should be flagged
   and excluded before agent timing is calibrated.
2. **The crawl root kernel is contaminated by sweepers** (19% of root replies). Root timing fits for
   Moltbook should be reported with and without them.
