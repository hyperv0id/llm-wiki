# Ingest Report: PatchSTG (KDD 2025)

## Created
- `wiki/source-patchstg.md` — WHY: source-summary required for every ingested `raw/` file; covers PatchSTG's four-component architecture, LargeST results, 10× speedup claim, and ablation findings
- `wiki/patchstg.md` — WHY: entity page for the model, positioned as efficient dynamic spatial modeling paradigm distinct from linear (BigST) and low-rank (STWave) approaches
- `wiki/irregular-spatial-patching.md` — WHY: technique page documenting the three-stage spatial partitioning pipeline (leaf KDTree → padding → backtracking), which is PatchSTG's core algorithmic contribution bridging KDTree and Transformer patching
- `wiki/leaf-kdtree.md` — WHY: technique page for the novel tree algorithm variant; critical enough as the most ablated component to deserve its own page separate from the full patching pipeline

## Modified
- `wiki/traffic-forecasting.md` — WHY: added "Spatial Patching / Efficient Dynamic Spatial Modeling" section positioning PatchSTG against linear-based and low-rank-based alternatives; source_count 33 → 34
- `wiki/large-scale-spatial-temporal-graph.md` — WHY: expanded the one-liner "图划分 (PatchSTG): 分块处理" into a full description with complexity, dual attention mechanism, and performance claims; source_count 3 → 4; added cross-links to [[patchstg]], [[leaf-kdtree]], [[irregular-spatial-patching]]
- `wiki/index.md` — WHY: added source, entity, and two technique entries
- `wiki/log.md` — WHY: chronological activity record

## New Cross-Links
- [[patchstg]] ↔ [[traffic-forecasting]]
- [[patchstg]] ↔ [[large-scale-spatial-temporal-graph]]
- [[patchstg]] ↔ [[irregular-spatial-patching]]
- [[patchstg]] ↔ [[leaf-kdtree]]
- [[irregular-spatial-patching]] ↔ [[leaf-kdtree]]
- [[patchstg]] ↔ [[ragc]] (both large-scale efficiency methods, complementary approaches)
- [[patchstg]] ↔ [[specstg]] (both KDD-era efficient STG methods, patching vs spectral)
