# Ingest Report: NsDiff

## Created
- wiki/source-nsdiff.md — WHY: source-summary for ICML 2025 Spotlight paper; covers LSNM+UANS innovations, SOTA results across 9 datasets, and key contributions
- wiki/nsdiff.md — WHY: entity page for the NsDiff model; core method architecture, performance, comparisons to TimeGrad/TMDM, and limitations
- wiki/location-scale-noise-model.md — WHY: concept page for LSNM; the paper's key theoretical contribution that generalizes DDPM's fixed-unit-variance assumption; distinct enough from existing [[diffusion-model]] page to warrant its own concept
- wiki/uncertainty-aware-noise-schedule.md — WHY: technique page for UANS; the noise scheduling mechanism that enables the diffusion process to adapt to data uncertainty levels; a concrete technique distinct from general diffusion concepts

## Modified
- wiki/diffusion-models.md — WHY: added NsDiff as a new bullet point under "挑战与未来方向" section; NsDiff represents a significant advancement in applying diffusion models to non-stationary time series with its LSNM framework
- wiki/timegrad.md — WHY: added NsDiff as a cross-reference under "关联页面"; NsDiff explicitly benchmarks against and outperforms TimeGrad, and its LSNM framework generalizes TimeGrad's approach
- wiki/index.md — WHY: added all four new pages to Sources, Entities, Concepts, and Techniques categories respectively
- wiki/log.md — WHY: recorded this ingest with full description following the established format

## New Cross-links
- [[nsdiff]] ↔ [[location-scale-noise-model]] (LSNM is NsDiff's core concept)
- [[nsdiff]] ↔ [[uncertainty-aware-noise-schedule]] (UANS is NsDiff's core technique)
- [[nsdiff]] ↔ [[diffusion-models]] (NsDiff advances diffusion model application)
- [[nsdiff]] ↔ [[timegrad]] (NsDiff benchmarks against and outperforms TimeGrad)
- [[location-scale-noise-model]] ↔ [[diffusion-models]] (LSNM generalizes DDPM's variance assumption)
- [[location-scale-noise-model]] ↔ [[uncertainty-aware-noise-schedule]] (complementary design)
- [[uncertainty-aware-noise-schedule]] ↔ [[ddpm]] (UANS extends DDPM's fixed β_t I noise)
