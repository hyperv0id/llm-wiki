# Ingest Report: Language in the Flow of Time (TaTS)

**Source**: Zihao Li et al., "Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative", ICLR 2026, arXiv:2502.08942
**Date**: 2026-05-03

## Created

- `wiki/source-language-in-the-flow-of-time.md` — WHY: Source summary for the TaTS paper, covering CTR phenomenon, TT-Wasserstein metric, TaTS framework, and experimental results across 18 datasets and 9 models.
- `wiki/tats.md` — WHY: Entity page for the TaTS (Texts as Time Series) framework. TaTS is a unique plug-and-play approach that treats text as auxiliary variables, contrasting with VoT's LLM reasoning, UniCA's covariate adaptation, and Aurora's generative foundation model approach.
- `wiki/chronological-textual-resonance.md` — WHY: Concept page for CTR, the key phenomenon discovered in this paper — that time-series-paired texts naturally exhibit periodic properties mirroring the original time series. This is a novel concept distinct from engineered alignment techniques (ETA, MindTS alignment).
- `wiki/tt-wasserstein.md` — WHY: Technique page for the TT-Wasserstein metric, which quantifies CTR level via Wasserstein distance between normalized spectra. Useful as a dataset quality gauge and predictor of TaTS effectiveness.
- `wiki/texts-as-auxiliary-variables.md` — WHY: Concept page for the core design principle behind TaTS — treating encoded text as additional time series variables via concatenation. This is the simplest multimodal fusion strategy, contrasting with attention-based, reasoning-based, and homogenization-based approaches.

## Modified

- `wiki/multimodal-time-series-forecasting.md` — WHY: Added TaTS to the comparison table (new column) and added a dedicated section describing TaTS's plug-and-play approach. Updated source_count from 5 to 6.
- `wiki/vot.md` — WHY: Added TaTS to the comparison table and added a paragraph contrasting VoT's LLM reasoning approach with TaTS's simpler plug-and-play approach. Added cross-reference to TaTS.
- `wiki/chronos.md` — WHY: Added cross-reference to TaTS, noting the contrast between Chronos's numerical-only tokenization and TaTS's text-as-auxiliary-variables approach.
- `wiki/endogenous-text-alignment.md` — WHY: Added a comparison table between VoT's ETA (decomposed contrastive learning) and TaTS's approach (no explicit alignment, concatenation-based). Added cross-references.
- `wiki/fine-grained-time-text-semantic-alignment.md` — WHY: Added cross-references to TaTS and CTR, noting that CTR is an observed phenomenon rather than an engineered alignment technique.
- `wiki/aurora.md` — WHY: Added cross-reference to TaTS, contrasting Aurora's generative foundation model approach with TaTS's lightweight plug-in approach.
- `wiki/index.md` — WHY: Added all 5 new pages to their respective categories (Sources, Entities, Concepts, Techniques).
- `wiki/log.md` — WHY: Appended ingest entry with summary of key innovations, created/updated pages.

## New Cross-links

- [[tats]] ↔ [[chronological-textual-resonance]] ↔ [[tt-wasserstein]] ↔ [[texts-as-auxiliary-variables]] (core cluster)
- [[tats]] ↔ [[vot]] (plug-and-play vs. LLM reasoning comparison)
- [[tats]] ↔ [[unica]] (auxiliary variables vs. covariate homogenization)
- [[tats]] ↔ [[aurora]] (lightweight plugin vs. generative foundation model)
- [[tats]] ↔ [[chronos]] (text handling vs. numerical tokenization)
- [[chronological-textual-resonance]] ↔ [[endogenous-text-alignment]] (observed phenomenon vs. engineered alignment)
- [[chronological-textual-resonance]] ↔ [[fine-grained-time-text-semantic-alignment]] (frequency-domain vs. contrastive alignment)
- [[texts-as-auxiliary-variables]] ↔ [[covariate-homogenization]] (alternative multimodal fusion strategies)
- [[texts-as-auxiliary-variables]] ↔ [[channel-independence]] (related TS design principle)
- [[multimodal-time-series-forecasting]] ↔ [[tats]] (task concept ↔ model)

## Key Insights from Comparison

1. **TaTS vs. VoT**: TaTS is simpler (MLP + concat, no architecture change) while VoT achieves deeper semantic integration through LLM reasoning and multi-level alignment. TaTS is more broadly compatible; VoT is more powerful for reasoning-heavy tasks.

2. **TaTS vs. UniCA**: Both are plug-in frameworks, but TaTS handles only text (via concatenation) while UniCA handles categorical/image/text (via homogenization + attention fusion). TaTS requires no TSFM-specific adaptation.

3. **TaTS vs. Aurora**: TaTS is a lightweight framework; Aurora is a full generative foundation model. TaTS is deterministic; Aurora supports probabilistic forecasting via Flow Matching.

4. **CTR vs. ETA/MindTS alignment**: CTR is an **observed natural phenomenon** (frequency-domain periodicity alignment), while ETA and MindTS's alignment are **engineered techniques** (contrastive learning). CTR provides theoretical justification for treating text as auxiliary variables.

5. **TT-Wasserstein** is a novel contribution — the first metric specifically designed to quantify text-TS alignment quality in the frequency domain. It can serve as a dataset quality filter and predict TaTS effectiveness.
