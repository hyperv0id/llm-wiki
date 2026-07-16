# Ingest Report: TiDE (arXiv:2304.08424)

## Created
- wiki/source-tide.md — WHY: source-summary for Das et al. 2024, TiDE residual-MLP encoder–decoder for LTSF with covariates (arXiv:2304.08424)
- wiki/tide.md — WHY: entity page for TiDE architecture, covering problem setting, encoding/decoding, theory (LDS near-optimality), empirical results, historical position, and limitations
- wiki/temporal-decoder.md — WHY: technique page for TiDE's per-horizon-step residual fusion head, a key architectural innovation enabling direct covariate highways

## Modified
- wiki/index.md — WHY: added [[source-tide]] to Sources, [[tide]] to Entities, [[temporal-decoder]] to Techniques
- wiki/log.md — WHY: recorded 2026-07-13 ingest operation
- wiki/source-exost.md — WHY: added TiDE as baseline context for exogenous variable modeling
- wiki/source-exotst.md — WHY: added TiDE comparison (ExoTST outperforms TiDE by 8–12% on carbon flux)
- wiki/source-timexer.md — WHY: added TiDE as related MLP/covariate baseline
- wiki/source-crosslinear.md — WHY: added TiDE as exogenous baseline reference
- wiki/source-exollm.md — WHY: added TiDE as non-LLM exogenous baseline
- wiki/heterogeneous-covariates.md — WHY: added TiDE as exemplar of residual-MLP covariate handling
- wiki/source-tide.md — WHY: 2026-07-16 maintenance pass — confidence→medium on tide.md entity page, added cross-ref backlinks
- wiki/tide.md — WHY: 2026-07-16 maintenance pass — confidence high→medium (source_count:1 lint fix), added [[deepar]] backlink

## New Cross-Links
- [[source-tide]] ↔ [[tide]]
- [[tide]] ↔ [[temporal-decoder]]
- [[tide]] ↔ [[channel-independence]]
- [[tide]] ↔ [[direct-forecast]]
- [[tide]] ↔ [[lstf]]
- [[tide]] ↔ [[ltsf-linear]]
- [[tide]] ↔ [[patchtst]]
- [[tide]] ↔ [[nbeatsx]]
- [[tide]] ↔ [[tft]]
- [[tide]] ↔ [[deepar]]
- [[tide]] ↔ [[source-exost]]
- [[tide]] ↔ [[source-exotst]]
- [[tide]] ↔ [[source-timexer]]
