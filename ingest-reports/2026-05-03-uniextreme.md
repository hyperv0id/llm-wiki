# Ingest Report: UniExtreme

## Created

- wiki/source-uniextreme.md — WHY: Source summary for the UniExtreme paper, documenting core findings (spectral right-shift, hierarchical extreme diversity), two key modules (AFM, EPA), and experimental results
- wiki/uniextreme.md — WHY: Entity page for the UniExtreme model, providing architectural overview, performance comparison table, and cross-references to related foundation models
- wiki/extreme-weather-forecasting.md — WHY: Concept page capturing the general problem of extreme weather forecasting, its three core challenges (spectral disparity, hierarchical drivers, data scarcity), and method evolution from single-type to universal models
- wiki/adaptive-frequency-modulation.md — WHY: Technique page for AFM, UniExtreme's key spectral innovation — details Beta filtering design (mode/spread parameterization), logarithmic band partitioning, and spatiotemporal band aggregation with comparison to existing frequency-domain approaches
- wiki/event-prior-augmentation.md — WHY: Technique page for EPA, UniExtreme's key memory innovation — details categorized memory construction from real extreme events, dual-level attention fusion (intra-type + inter-type), and comparison with existing memory/prompt architectures

## Modified

- wiki/aurora.md — WHY: Added cross-reference to UniExtreme as a domain-specific (weather) foundation model contrast (Aurora: general TS multimodal generative; UniExtreme: weather extreme event discriminative); updated source_count to 5, last_updated to 2026-05-03
- wiki/simdiff.md — WHY: Added cross-reference to UniExtreme as another frequency-aware approach (Beta filters vs Diffusion; weather vs TS); updated source_count to 2, last_updated to 2026-05-03
- wiki/timesfm.md — WHY: Added cross-reference to UniExtreme as a domain-specific foundation model (TimesFM: general TS; UniExtreme: weather extremes); updated source_count to 2, last_updated to 2026-05-03
- wiki/index.md — WHY: Added 5 new pages across Sources, Entities, Concepts, and Techniques categories
- wiki/log.md — WHY: Appended ingest entry with complete page list

## New Cross-Links

- [[uniextreme]] ↔ [[aurora]] — both foundation models, different domains and approaches (weather discriminative vs TS generative)
- [[uniextreme]] ↔ [[simdiff]] — both frequency-aware, different techniques (Beta filtering vs Diffusion)
- [[uniextreme]] ↔ [[timesfm]] — both foundation models, different domains (weather vs general TS)
- [[uniextreme]] ↔ [[most]] — both region-specific foundation models, different domains (weather vs traffic)
- [[adaptive-frequency-modulation]] ↔ [[fedformer]] — both frequency-domain methods, different approach (Beta filtering vs frequency decomposition)
- [[adaptive-frequency-modulation]] ↔ [[frequency-aware-residual-representation]] — related spectral techniques
- [[event-prior-augmentation]] ↔ [[mixture-of-experts]] — attention routing vs MoE routing
- [[event-prior-augmentation]] ↔ [[most]] — event memory vs modality selection
- [[extreme-weather-forecasting]] ↔ [[traffic-forecasting]] — both face rare-event modeling challenges
- [[extreme-weather-forecasting]] ↔ [[multimodal-time-series-forecasting]] — different augmentation strategies
- [[extreme-weather-forecasting]] ↔ [[generative-time-series-forecasting]] — discriminative vs generative for extremes
