# Ingest Report: Time-Indexed Foundation Models for Imputation (TabPFN-TS / MoTM benchmark)

Source: "Are Time-Indexed Foundation Models the Future of Time Series Imputation?" (Le Naour, Nabil, Petralia, Agoua; EDF R&D; TMLR 01/2026, arXiv:2511.05980v2). Raw: `raw/2511.05980.pdf`.

Note: the user listed this as "TabPFN-TS (TMLR 2026)". The actual TMLR 2026 paper is the **benchmark/survey** that evaluates TabPFN-TS and MoTM and rates TabPFN-TS best (NMAE 0.293) — so it is ingested as a benchmark source with TabPFN-TS and MoTM as the headline entities.

## Created
- [[source-time-indexed-imputation]] — WHY: Source summary for the TMLR 2026 benchmark (required one-per-raw-file); holds the full NMAE comparison incl. NuwaTS.
- [[tabpfn-ts]] — WHY: The headline model (benchmark-best zero-shot imputer); Fourier features + TabPFN in-context regression.
- [[motm]] — WHY: The second time-indexed FM (scalable, ~100× faster); modulated-INR basis + ridge — a distinct named model worth its own entity.
- [[time-indexed-foundation-model]] — WHY: The paper's central concept (continuous-time H(t)→x(t) zero-shot imputation); a reusable hub distinguishing these from patch-based forecasters and PLM-repurposing (NuwaTS).

## Modified
- [[nuwats]] — WHY: This is an independent third-party benchmark of NuwaTS — it lags behind TabPFN-TS (all settings) and MoTM (10/11). Added an "外部评估" warning callout for balance. source_count 2→3.
- [[csdi]] — WHY: CSDI benchmarked at NMAE 0.664 (zero-shot/cross-domain), sometimes below local heuristics — added the finding. source_count 7→8.
- [[index]] — WHY: Added 4 new pages to Sources, Entities, Concepts.
- [[log]] — WHY: Recorded ingest activity per wiki protocol.

## New Cross-Links
- [[tabpfn-ts]] ↔ [[motm]] (inverse designs; accuracy vs speed)
- [[tabpfn-ts]] ↔ [[time-indexed-foundation-model]]
- [[motm]] ↔ [[time-indexed-foundation-model]]
- [[time-indexed-foundation-model]] ↔ [[nuwats]] (time-indexed beats PLM-repurposing zero-shot)
- [[tabpfn-ts]] ↔ [[nuwats]] (benchmark: NuwaTS lags)
- [[time-indexed-foundation-model]] ↔ [[csdi]] / [[missing-not-at-random]]
