# Ingest Report: MetaDG — Meta Dynamic Graph for Traffic Flow Prediction (AAAI 2026)

## Created
- **wiki/source-metadg.md** — WHY: Source-summary covering MetaDG's three modules (DNG, STCE, DGQ), Meta-DGCRU, SOTA results on PEMS03/04/07/08, and ablation evidence. Required per ingest workflow.
- **wiki/metadg.md** — WHY: Entity page for the MetaDG model. Novel GCRU-based framework that extends dynamics from adjacency matrices to meta-parameters, unifying ST-isolated modeling. Significant new architecture with SOTA results.
- **wiki/meta-dynamic-graph.md** — WHY: Concept page for the core design principle: dynamics should govern not just spatial topology but also model parameters and intermediates. Represents a conceptual shift from "dynamics as an add-on" to "dynamics as the organizing principle."
- **wiki/st-unification.md** — WHY: Concept page for the ST-isolated → ST-unification spectrum introduced by MetaDG. Provides a clear framing that situates existing work (STGCN, GWNet, DCRNN, DGCRN) and explains the paper's contribution in a structured taxonomy.
- **wiki/dynamic-graph-qualification.md** — WHY: Technique page for the DGQ module. Novel approach to qualifying message-passing reliability via cross-time-step similarity with adaptive scaling coefficients. Ablation confirms consistent (if modest) gains.

## Modified
- **wiki/gwnet.md** — WHY: Added MetaDG and st-unification to Related Pages section. MetaDG cites GWNet as a key ST-isolated baseline; MetaDG's ST-unification framing directly critiques GWNet's separate temporal+spatial processing.
- **wiki/stgcn.md** — WHY: Added MetaDG and st-unification cross-references. STGCN is the canonical ST-isolated model; MetaDG's framing provides historical context for STGCN's architecture choice.
- **wiki/dcrnn.md** — WHY: Added MetaDG and st-unification cross-references. DCRNN's DCGRU is the direct predecessor of MetaDG's Meta-DGCRU; MetaDG extends dynamics to meta-parameters whereas DCRNN only models spatial diffusion dynamics.
- **wiki/traffic-forecasting.md** — WHY: Added "Dynamic Graph with Meta-Parameters" subsection introducing MetaDG paradigm. source_count updated 29→30. New footnote [^src-metadg] added.
- **wiki/index.md** — WHY: Added all 5 new pages to appropriate categories (Sources, Entities, Concepts, Techniques).
- **wiki/log.md** — WHY: Mandatory ingest log entry with paper summary, created pages, and updated pages.

## New Cross-Links
- [[metadg]] ↔ [[gwnet]]
- [[metadg]] ↔ [[stgcn]]
- [[metadg]] ↔ [[dcrnn]]
- [[metadg]] ↔ [[traffic-forecasting]]
- [[meta-dynamic-graph]] ↔ [[st-unification]]
- [[meta-dynamic-graph]] ↔ [[metadg]]
- [[st-unification]] ↔ [[metadg]]
- [[dynamic-graph-qualification]] ↔ [[metadg]]

## Pages NOT Created
- Dynamic Node Generation (DNG) technique page — considered but too tightly coupled to MetaDG's specific embodiment; covered adequately in metadg.md entity page
- Spatio-Temporal Correlation Enhancement (STCE) technique page — considered but SCE (cross-attention) and TCE (GRU gate smoothing) are individually known techniques; the combination is novel but best documented within metadg.md
- Spatio-Temporal Heterogeneity concept page — covered within meta-dynamic-graph.md and st-unification.md as secondary concepts
