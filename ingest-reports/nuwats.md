# Ingest Report: NuwaTS

Source: NuwaTS — a Foundation Model Mending Every Incomplete Time Series (Cheng et al., Sichuan University / HKUST-GZ / Squirrel Ai, arXiv:2405.15317v3, Oct 2024). Raw: `raw/2405.15317.pdf`.

## Created
- [[source-nuwats]] — WHY: Source summary for arXiv 2405.15317 (required one-per-raw-file).
- [[nuwats]] — WHY: Core entity for the NuwaTS model — architecture, four versions, experiments, ablations.
- [[variable-wise-partitioning]] — WHY: Genuinely novel benchmarking concept (partition along variable dimension, not time) for cross-variable/cross-domain generalization — reusable beyond NuwaTS, no prior page.
- [[plug-and-play-prefix-tuning]] — WHY: Distinct PEFT technique (P-tuning-v2-style removable domain prefix, <100KB) not previously represented; complements model-reprogramming and prompt-as-prefix.

## Modified
- [[contrastive-learning]] — WHY: New application axis — NuwaTS's mask-invariant patch representations (same patch under different mask ratios as positives); bumped source_count 1→2.
- [[channel-independence]] — WHY: NuwaTS uses CI to enable cross-variable/cross-domain zero-shot (different variable counts); added section, source_count 7→8.
- [[time-llm]] — WHY: NuwaTS gives a direct counter-finding (text-alignment underperforms linear embedding for incomplete series); resolves prior confidence:high/source_count:1 lint flag (now 2).
- [[patch-reprogramming]] — WHY: NuwaTS Table 14 shows this technique is worse than linear embedding for high-missing patches — bounds its applicability; source_count 1→2.
- [[model-reprogramming]] — WHY: NuwaTS is an adjacent-but-distinct PLM-for-TS approach (partial backbone fine-tuning + frozen-backbone prefix); source_count 1→2.
- [[imputeformer]] — WHY: NuwaTS realizes ImputeFormer's stated future-work (cross-domain imputation foundation model + multi-task + mask-invariant representations); source_count 1→2.
- [[instance-normalization]] — WHY: NuwaTS applies RevIN (missing-as-zero) for cross-domain distribution alignment; source_count 3→4.
- [[csdi]] — WHY: NuwaTS is a CSDI baseline; both use mask-based SSL but diffusion vs PLM paradigms — cross-paradigm link; source_count 4→5.
- [[index]] — WHY: Added 4 new pages to Sources, Entities, Concepts, Techniques.
- [[log]] — WHY: Recorded ingest activity per wiki protocol.

## New Cross-Links
- [[nuwats]] ↔ [[variable-wise-partitioning]]
- [[nuwats]] ↔ [[plug-and-play-prefix-tuning]]
- [[nuwats]] ↔ [[contrastive-learning]]
- [[nuwats]] ↔ [[channel-independence]]
- [[nuwats]] ↔ [[instance-normalization]]
- [[nuwats]] ↔ [[time-llm]] (反例 / counter-finding)
- [[nuwats]] ↔ [[patch-reprogramming]] (反例 / counter-finding)
- [[nuwats]] ↔ [[model-reprogramming]]
- [[nuwats]] ↔ [[imputeformer]] (realizes future work)
- [[nuwats]] ↔ [[csdi]] (cross-paradigm imputation)
- [[nuwats]] ↔ [[chronos]] (reuse NLP weights vs train-from-scratch)
- [[variable-wise-partitioning]] ↔ [[channel-independence]]
- [[plug-and-play-prefix-tuning]] ↔ [[prompt-as-prefix]]
