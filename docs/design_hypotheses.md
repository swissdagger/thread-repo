# Design hypotheses (spec v1)
Tested mainly by simulation experiments. Register entries R1–R19 are
empirical results on human and Moltbook data; each maps to at most a
few of these.

H0 (sanity check) The hybrid reproduces real timing statistics.
   Expected by construction; not a finding.
H1 (mechanism) A substantial share of fitted root bias and popularity
   disappears when within-thread display order is randomized.
   Test: randomized-order simulation. Human-side evidence (R2, R11):
   display position is modest and population-dependent and does not
   absorb time decay.
H2 (decomposition) Visibility kernels and root fan-out vary across
   platforms, not populations; reciprocity and challenge propensity
   vary across populations, not platforms.
   Test: platform x population matrix.
H3 (intrinsic difference) Under exposure-matched replay, agents'
   conversion and reciprocity stay below human values. Test: replay.
H4 (intervention) Adding reply exposure and a thread-opened path raises
   agents' non-root mu from the Moltbook baseline (0.066; per-author
   0.076) toward the human range (per-author 0.42-0.73), and the depth
   distribution shifts as the fitted model predicts.
   Test: exposure interventions. (Revised from spec v1's 0.15 / 0.45.)
H5 (dialogue) Thread depth increases with challenge propensity, in
   humans and agents. Test: challenge sweep; cross-community human data.

Mapping guide: R2, R11 → H1; R12, R17, R10 → H2; R13 → H3; R6, R8 → H4; R1 → H0; R3, R4, R9, R15, R16 → model terms (no H); R5, R7, R14 → validation targets; R18, R19 → definitions and proxies.
