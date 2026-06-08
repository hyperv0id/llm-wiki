# Ingest Report: HEPHAESTUS (ICLR 2026, under review)

## Created
- **wiki/source-hephestus.md** — WHY: Source summary for the HEPHAESTUS paper; 300-500 words covering the three core innovations (AMS-MoE, PTA, HSA), SOTA results on 6 traffic benchmarks, and caveats about under-review status.
- **wiki/hephestus.md** — WHY: Entity page for HEPHAESTUS as a unified traffic forecasting model; includes full architecture pipeline, component relationships, performance tables, efficiency analysis, case study of dynamic scale selection, and a comparison table to related models (TimeMixer, PHAT, PatchTST, PathFormer, GWNet).
- **wiki/ams-moe.md** — WHY: Technique page for Adaptive Multi-Scale Mixture of Experts — the core routing innovation. Documents Moving-Patch (boundary replication + overlapping stride-1 + linear projection), temporal-aware routing with noise-injected Top-K gating, auxiliary load balancing loss, and comparison to fixed-scale alternatives (TimeMixer, PathFormer).
- **wiki/periodic-temporal-attention.md** — WHY: Technique page for PTA — time-aware cross-attention using learnable daily/weekly periodic embedding matrices as queries. Distinct from standard self-attention in using absolute time position encodings as query source. Includes comparison to PHAT, Autoformer, CycleNet, TimeMixer periodicity approaches.
- **wiki/heterogeneous-spatial-attention.md** — WHY: Technique page for HSA — low-rank pattern library decomposition enabling per-node spatial transformation at O(Nr + rCD) cost, with gated fusion balancing global (Common Linear) and local (Specific Linear) spatial patterns.

## Modified
- **wiki/phat.md** — WHY: Added HEPHAESTUS as a complementary ICLR 2026 temporal heterogeneity model in the Connections table.
- **wiki/timemixer.md** — WHY: Added HEPHAESTUS as successor work replacing fixed down-sampling with MoE routing in the 与其他模型的关系 section.
- **wiki/index.md** — WHY: Added source-hephestus (Sources), hephestus (Entities), ams-moe, periodic-temporal-attention, heterogeneous-spatial-attention (Techniques).
- **wiki/log.md** — WHY: Chronological ingest record.

## New Cross-Links
- [[hephestus]] ↔ [[phat]] (both ICLR 2026, temporal heterogeneity approaches)
- [[hephestus]] ↔ [[timemixer]] (multi-scale mixing, MoE routing replaces fixed decomposition)
- [[hephestus]] ↔ [[ams-moe]], [[periodic-temporal-attention]], [[heterogeneous-spatial-attention]] (components)
- [[ams-moe]] ↔ [[timemixer]], [[pathformer]] (multi-scale comparison)
- [[heterogeneous-spatial-attention]] ↔ [[gwnet]], [[dgcrnn]] (spatial modeling comparison)
- [[periodic-temporal-attention]] ↔ [[phat]], [[cyclenet]], [[autoformer]] (periodicity comparison)
