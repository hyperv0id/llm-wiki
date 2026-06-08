# Ingest Report: RSTIB-MLP (ICML 2025)

## Created
- [[source-rstib-mlp]] — WHY: Source-summary for the RSTIB-MLP paper (ICML 2025), covering dual noise effect, RSTIB principle, MLP instantiation with analytical regularization bounds, knowledge distillation module with noise impact indicator, and 6-dataset robustness evaluation
- [[rstib-mlp]] — WHY: Entity page documenting RSTIB-MLP's architecture (3 RSTIB regularizations + KD module + spatial-temporal prompts), benchmark results under noisy conditions, efficiency comparison, ablation studies, and connections to related models
- [[rstib]] — WHY: Concept page for the Robust Spatial-Temporal Information Bottleneck principle — genuinely novel theoretical extension of IB/RGIB to dual-noise STF with lifted Markov assumption, interaction information decomposition, and comparison to prior IB variants (IB, DVIB, GIB, RGIB)
- [[noise-impact-indicator]] — WHY: Technique for per-time-series noise quantification via teacher model prediction error (softmax-normalized), enabling dynamic regularization balancing in RSTIB-MLP's learning objective

## Modified
- [[ltsf-linear]] — WHY: Added cross-reference to RSTIB-MLP: both demonstrate MLP viability for time series; LTSF-Linear on clean data, RSTIB-MLP extends lineage to robust prediction under noise
- [[timemixer]] — WHY: Added cross-reference comparing the two MLP-based models: TimeMixer focuses on multi-scale mixing for prediction accuracy, RSTIB-MLP on IB-guided robust representation learning against noise
- [[index]] — WHY: Added source-summary, entity, concept, and technique page entries
- [[log]] — WHY: Recorded ingest activity

## New Cross-Links
- [[rstib-mlp]] ↔ [[rstib]] — Model instantiates the principle
- [[rstib-mlp]] ↔ [[noise-impact-indicator]] — Noise impact indicator is the key technique in RSTIB-MLP's training regime
- [[rstib-mlp]] ↔ [[ltsf-linear]] — Both MLP-based models for time series, complementary focus (clean vs robust)
- [[rstib-mlp]] ↔ [[timemixer]] — Both MLP-based time series models, different design goals
- [[rstib-mlp]] ↔ [[stid]] — Inherits spatial-temporal prompts from STID
- [[rstib-mlp]] ↔ [[frets]] — FreTS uses frequency-domain implicit noise filtering; RSTIB-MLP uses explicit IB-guided noise regularization
- [[rstib]] ↔ [[information-bottleneck]] — RSTIB extends the IB framework
