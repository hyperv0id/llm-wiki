# Ingest Report: HyperD — Hybrid Periodicity Decoupling Framework for Traffic Forecasting

## Created
- `wiki/source-hyperd-hybrid-periodicity-decoupling.md` — WHY: source-summary page required for every raw/ source
- `wiki/hyperd.md` — WHY: central entity page for the framework; cross-links to all component pages
- `wiki/hybrid-periodicity-decoupling.md` — WHY: the core conceptual contribution; distinct from prior uniform-frequency or implicit-periodicity approaches
- `wiki/traffic-forecasting.md` — WHY: foundational domain concept; needed for context and will be referenced by future traffic-forecasting sources
- `wiki/frequency-aware-residual-representation.md` — WHY: the most critical architectural component (ablation: +3.2% MAE when removed)
- `wiki/spatial-temporal-attentive-encoder.md` — WHY: dual-pathway encoder is the architectural centerpiece
- `wiki/dual-view-alignment-loss.md` — WHY: novel loss function that enables complementary dual-pathway learning
- `wiki/demlp-decoder.md` — WHY: two-stage decoder with trend-decoupling is a distinct technical contribution

## Modified
- `wiki/index.md` — WHY: all new pages added to catalog
- `wiki/log.md` — WHY: ingest event recorded

## New cross-links
- [[hyperd]] ↔ [[frequency-aware-residual-representation]]
- [[hyperd]] ↔ [[spatial-temporal-attentive-encoder]]
- [[hyperd]] ↔ [[dual-view-alignment-loss]]
- [[hyperd]] ↔ [[demlp-decoder]]
- [[hyperd]] ↔ [[hybrid-periodicity-decoupling]]
- [[hybrid-periodicity-decoupling]] ↔ [[frequency-aware-residual-representation]]
- [[spatial-temporal-attentive-encoder]] ↔ [[dual-view-alignment-loss]]
- [[spatial-temporal-attentive-encoder]] ↔ [[frequency-aware-residual-representation]]
- [[demlp-decoder]] ↔ [[spatial-temporal-attentive-encoder]]
- [[traffic-forecasting]] ↔ [[hyperd]]
