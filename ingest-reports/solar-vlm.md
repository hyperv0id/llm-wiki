# Ingest Report: Solar-VLM

## Created
- [[source-solar-vlm]] — WHY: Source summary page for arXiv:2604.04145, a unified multimodal VLM framework for solar power spatiotemporal forecasting

## Modified
- [[index]] — WHY: Added [[source-solar-vlm]] to the Sources section
- [[log]] — WHY: Recorded ingest activity per wiki protocol

## New Cross-Links
- [[source-solar-vlm]] ↔ [[source-st-vision-llm]]
- [[source-solar-vlm]] ↔ [[source-gpt4mts]] (unresolved — page does not exist yet)

## WHY This Paper Matters

Solar-VLM is among the first works to propose a **truly unified multimodal framework** (time-series + satellite imagery + text) for **multi-site PV forecasting**, combining:
1. A frozen Qwen-VLM backbone for visual and text encoding
2. **Graph Attention Network** over a KNN spatial graph for cross-site dependency modeling
3. **Cross-site attention** for adaptive multimodal information sharing across stations
4. **Retrieval-augmented dual-memory** for long-term historical information

The key architectural insight is a **two-stage cross-site strategy**: GNN on time-series features (physically comparable across sites) followed by attention on fused multimodal features (heterogeneous, less suitable for graph propagation). This design principle may be applicable to other multi-site spatiotemporal forecasting problems.
