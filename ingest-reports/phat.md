# Ingest Report: PHAT (ICLR 2026)

## Source
- **Title**: PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting
- **Authors**: Jiaming Ma, Qihe Huang, Haofeng Ma et al. (USTC)
- **Venue**: ICLR 2026
- **PDF**: Zotero storage P7S74LHI

## Created
- [[phat]] — entity page for the Period Heterogeneity-Aware Transformer. WHY: first model explicitly designed for period heterogeneity; core technique (PNA) is a novel attention mechanism with positive-negative decomposition and modulation terms; significant performance on 14 datasets.
- [[source-phat]] — source-summary page. WHY: ICLR 2026 paper with substantial theoretical grounding (stick-breaking, variance reduction); 300-500 word summary covering method, results, and connections.

## Modified
- None (new ingest, no existing pages require update beyond index/log).

## Cross-Links Established
- [[phat]] ↔ [[autoformer]] — PHAT cites Autoformer's Auto-Correlation as predecessor; contrasts pooling vs. bucket-based heterogeneity handling
- [[phat]] ↔ [[fedformer]] — Both use frequency-domain analysis but for different purposes
- [[phat]] ↔ [[timesnet]] — Both fold 1D→2D; PHAT uses X-shaped attention vs. TimesNet's CNN
- [[phat]] ↔ [[patchtst]] — PHAT's folding preserves temporal resolution vs. PatchTST's down-sampling aggregation
- [[phat]] ↔ [[itransformer]] — Orthogonal approaches: variate-level attention vs. period-bucket attention
- [[phat]] ↔ [[cyclenet]] — Both model periodicity; CycleNet: single learnable cycle per channel; PHAT: heterogeneous period detection + grouping
- [[phat]] ↔ [[sparsetsf]] — SparseTSF: fixed-period down-sampling; PHAT: detected-period folding

## Recommendations for Future Work
- Add PHAT to [[periodicity-modeling-in-time-series]] analysis page as a new "Route 6: Period Heterogeneity" or extend existing Periodic Bucket route
- Consider cross-reference from [[crossformer]] (both address cross-variate dependency; PHAT does so within period-homogeneous groups)
