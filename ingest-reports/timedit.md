# Ingest Report: TimeDiT (KDD 2025)

## Source
- `raw/2409.02322.pdf` — TimeDiT: General-purpose Diffusion Transformers for Time Series Foundation Model (Cao, Ye, Zhang & Liu, USC, KDD 2025)

## Created
- `wiki/source-timedit.md` — WHY: source-summary for the TimeDiT paper, 300-500 word summary of DiT-for-time-series architecture, unified masking, physics-informed sampling, and experimental results
- `wiki/timedit.md` — WHY: entity page for TimeDiT as a proto-foundation model, distinguishing it from TimesFM/Chronos/Moirai by its diffusion-based probabilistic approach and multi-task capability
- `wiki/timedit-masking.md` — WHY: technique page for the unified masking mechanism (random/block/stride/reconstruction), the core innovation enabling multi-task training from a single model
- `wiki/timedit-physics-informed.md` — WHY: technique page for the finetuning-free PDE-guided Langevin dynamics sampling, with Theorem 3.1 closed-form solution and energy-based optimization framework

## Modified
- `wiki/diffusion-models.md` — WHY: added TimeDiT as the first DiT+Diffusion time series foundation model in the time series diffusion section
- `wiki/dit.md` — WHY: added TimeDiT (KDD 2025) as a Subsequent Impact entry, adapting DiT backbone to time series
- `wiki/timesfm.md` — WHY: added cross-reference to TimeDiT as a competing foundation model with broader task coverage (four tasks vs forecasting-only)
- `wiki/index.md` — WHY: added all 4 new pages to their respective categories (Sources, Entities, Techniques)
- `wiki/log.md` — WHY: chronological ingest record

## New Cross-Links
- [[timedit]] ↔ [[dit]] (DiT backbone adaptation)
- [[timedit]] ↔ [[diffusion-models]] (diffusion foundation)
- [[timedit]] ↔ [[timesfm]] (competing TS foundation models)
- [[timedit]] ↔ [[chronos]] (Chronos pre-training dataset and competing model)
- [[timedit]] ↔ [[csdi]] (prior TS diffusion work)
- [[timedit-masking]] ↔ [[mae]] / [[videomae]] (masking paradigm comparison)
- [[timedit-physics-informed]] ↔ [[energy-based-model]] (EBM foundation)
- [[timedit-physics-informed]] ↔ [[langevin-dynamics]] (sampling method)
- [[timedit-physics-informed]] ↔ [[dyffusion]] (another physics-informed diffusion approach)

## Notes
- Code not yet public; pre-trained checkpoint promised in paper but unreleased as of Feb 2025
- Paper characterized as "proto-foundation model" — acknowledges sequence length and channel limitations not yet fully addressed
- Stride masking identified as most critical component (ablation: MSE 0.424→0.862 on Solar when removed)
