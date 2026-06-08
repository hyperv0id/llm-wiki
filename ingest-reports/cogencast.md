# Ingest Report: CoGenCast (ICML 2026)

## Created
- wiki/source-cogencast.md — WHY: First source-summary for this paper; captures 3 core innovations + experimental results
- wiki/cogencast.md — WHY: Entity page for the CoGenCast model itself; the first hybrid LLM+FM forecasting model in the wiki
- wiki/hybrid-llm-flow-matching-forecasting.md — WHY: New paradigm concept bridging LLM and FM literature; provides contrast table vs Time-LLM, Sundial, FlowTS, Aurora
- wiki/one-step-flow-generation.md — WHY: Key technique enabling NFE=1 generation; connects to InstaFlow, Rectified Flow, Consistency Models
- wiki/average-velocity-modeling.md — WHY: Technical foundation of one-step generation; JVP-corrected loss is a novel FM training approach

## Modified
- wiki/flow-matching.md — WHY: Added CoGenCast + one-step-flow + average-velocity cross-links in Related Pages
- wiki/generative-time-series-forecasting.md — WHY: Added CoGenCast row to methods comparison table + related page links
- wiki/sundial.md — WHY: Added CoGenCast cross-links; both are FM-based forecasting but differ in architecture (custom Transformer vs LLM encoder-decoder)
- wiki/time-llm.md — WHY: Added CoGenCast as evolution-link in Connections; CoGenCast extends Time-LLM's LLM+TS vision with encoder-decoder + FM
- wiki/index.md — WHY: Added all 5 new pages to Source/Entity/Concept/Technique sections
- wiki/log.md — WHY: Chronological ingest record

## New Cross-Links
- [[cogencast]] ↔ [[sundial]] — Both FM-based generative forecasting (ICML 2025 vs ICML 2026)
- [[cogencast]] ↔ [[time-llm]] — CoGenCast extends Time-LLM's LLM+TS paradigm
- [[cogencast]] ↔ [[flowts]] — Both use FM for TS, different approaches (pure FM vs LLM+FM)
- [[hybrid-llm-flow-matching-forecasting]] ↔ [[generative-time-series-forecasting]] — New sub-paradigm of generative forecasting
- [[one-step-flow-generation]] ↔ [[flow-matching]] — Connects one-step efficiency to FM theory
- [[one-step-flow-generation]] ↔ [[rectified-flow]] — Both aim for few/one-step generation
- [[average-velocity-modeling]] ↔ [[flow-matching]] — JVP-corrected loss extends FM training
