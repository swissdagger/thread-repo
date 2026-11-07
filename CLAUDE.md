# CLAUDE.md — conventions for the forum-model exploration

This repo runs exploratory analysis for an exposure-first model of forum participation
(humans on Reddit and Hacker News, LLM agents on Moltbook). The task is in `docs/brief.md`.
The model and its traps are in `docs/model_reference.md`. Read both before writing code.

## Working rules

- Work step by step as laid out in `docs/brief.md`. After Step 1, stop and write the report;
  do not fit anything until Step 1 has been confirmed.
- One script per step in `scripts/` (`step1_reconstruct.py`, `step2_structure.py`, ...).
  Shared functions in `forum_model/` (tree building, hazard estimation, kernel fits, sorts).
- Outputs: CSV tables in `outputs/<dataset>/`, PNG plots in `figures/<dataset>/`, and one
  markdown summary per step in `reports/stepN.md`.
- Use polars or DuckDB. Stream; never load a full dump into memory.
- Pin dependency versions in `requirements.txt`. Fix and log random seeds.
- `docs/model_reference.md` is read-only. If a finding changes a modelling decision, write it
  in the step report under "Implications for the model" rather than editing the reference.

## Data hygiene (every step)

- `partitions.json` records exactly which subreddits, time windows and HN date ranges each fit
  uses. Every script reads it. These partitions are reserved for the human baseline and must
  never be reused later for agent conditioning.
- Drop authors on the standard Pushshift-era bot lists. Run every analysis with and without
  the top 1% most active authors; report both.
- Hold out 20% of threads from every fit (split by thread, stratified by month) for validation.
- Two fixed lists, nothing on both:
  - Calibration: reply timing, parent choices.
  - Validation: depth, size shape, reciprocity, original-poster return, bimodality of
    distinct-commenter counts.
- In every step report, state for each fitted quantity which data it was fitted on and which
  it was validated on.

## Things that will silently produce wrong answers

- Final score is look-ahead-affected: it includes votes caused by the replies we are
  predicting, and it grows with age. Any position proxy built from it must be labelled as such
  and paired with the early-thread and contest-mode versions.
- Never define "answered question" by score. Use the arrival of an answer comment.
- The branching ratio is the integral of the fitted self-excitation kernel. The depth-tail
  slope is a comparison, not the estimate.
- Reply half-life is a calibration quantity; it is never a validation metric.
