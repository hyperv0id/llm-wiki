# Ingest Report: VoT — Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting

**Date**: 2026-05-03
**Source**: arXiv:2603.15452 (ICLR 2026)
**Authors**: Siyuan Wang, Peng Chen, Yihang Wang, Wanghui Qiu, Chenjuan Guo, Bin Yang, Yang Shu (ECNU)

## Created

- **wiki/source-event-driven-ts-forecasting.md** — WHY: Source-summary page for the VoT paper, covering the dual-branch architecture, event-driven reasoning with HIC, multi-level alignment (ETA + AFF), and experimental results on 10 datasets.
- **wiki/vot.md** — WHY: Entity page for the VoT model. VoT is the first multimodal TS forecasting method to use LLMs for both feature extraction and reasoning while supporting both exogenous and endogenous text. Same lab (ECNU) as MindTS.
- **wiki/event-driven-reasoning.md** — WHY: Concept page for the event-driven reasoning paradigm. This is a novel approach in multimodal TS forecasting — using LLMs to reason over exogenous text (news, policy documents) rather than just extracting features. Includes the three-step generative pipeline (template → summary → reasoning).
- **wiki/historical-in-context-learning.md** — WHY: Technique page for HIC, which retrieves corrected historical reasoning examples as error-informed guidance during inference. Unique in using corrected reasoning (not raw data) as retrieval targets, enabling error-informed learning without fine-tuning.
- **wiki/multi-level-alignment.md** — WHY: Concept page for the multi-level alignment approach. Covers both representation-level (ETA) and prediction-level (AFF) alignment. Includes comparison table with MindTS's fine-grained alignment — both from ECNU but decompose along different axes (trend/seasonal vs. endo/exo).
- **wiki/endogenous-text-alignment.md** — WHY: Technique page for ETA, which uses decomposed pattern extraction (dual-query attention for trend/seasonal) and decomposed contrastive learning. Comparison with MindTS's alignment highlights the different decomposition strategies.
- **wiki/adaptive-frequency-fusion.md** — WHY: Technique page for AFF, which performs frequency-domain fusion with learnable per-band weights. Unique among multimodal TS fusion techniques (UniCA uses time-domain attention, MoST uses SNR-based modality selection).

## Modified

- **wiki/multimodal-time-series-forecasting.md** — WHY: Added VoT as a new multimodal forecasting method with comparison table vs. ChannelMTS. Added VoT to related pages and citation. Updated source_count from 3 to 4.
- **wiki/fine-grained-time-text-semantic-alignment.md** — WHY: Added cross-references to VoT's ETA technique and VoT model in related pages. Both use contrastive learning for text-TS alignment but decompose along different axes.
- **wiki/mindts.md** — WHY: Added cross-references to VoT (same lab: ECNU) and ETA technique in related pages. Both models come from the same research group and share the idea of decomposing text into complementary views.
- **wiki/content-condenser-reconstruction.md** — WHY: Added HIC to the comparison table of related techniques. Both are retrieval-based techniques but serve different purposes (HIC for error-informed reasoning, Content Condenser for redundancy filtering).
- **wiki/index.md** — WHY: Added all 7 new pages to their respective categories (Sources, Entities, Concepts, Techniques).
- **wiki/log.md** — WHY: Appended ingest entry documenting all created and updated pages.

## New Cross-Links

- [[vot]] ↔ [[mindts]] (same lab: ECNU)
- [[endogenous-text-alignment]] ↔ [[fine-grained-time-text-semantic-alignment]] (complementary alignment techniques)
- [[historical-in-context-learning]] ↔ [[content-condenser-reconstruction]] (retrieval-based techniques)
- [[vot]] ↔ [[multimodal-time-series-forecasting]] (model ↔ task)
- [[adaptive-frequency-fusion]] ↔ [[covariate-fusion-module]] (fusion techniques)
- [[event-driven-reasoning]] ↔ [[multimodal-time-series-forecasting]] (paradigm ↔ task)

## Key Insights

1. **Same lab, different tasks**: VoT (forecasting) and MindTS (anomaly detection) both come from ECNU's Decision Intelligence group. Both decompose text into complementary views for alignment, but VoT extends alignment to the prediction level via frequency-domain fusion.

2. **LLM reasoning is novel in TS forecasting**: VoT is the first method to use LLMs for reasoning (not just feature extraction) in multimodal time series forecasting, as shown in Table 1 of the paper.

3. **Complementary alignment strategies**: ETA decomposes along trend/seasonal (time series intrinsic properties), while MindTS decomposes along endogenous/exogenous (text source properties). These are complementary rather than competing approaches.