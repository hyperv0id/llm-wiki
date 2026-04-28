# Ingest Report: ConFormer

**Date**: 2026-04-28  
**Source**: Wang et al. - "Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting" (KDD 2026)

## Created Pages

- **wiki/source-conformer.md** — Source summary for the KDD 2026 paper, covering ConFormer's architecture, innovations (accident-aware graph propagation, Guided Layer Normalization), datasets, and performance results.

- **wiki/conformer.md** — Entity page for the ConFormer model, describing its architecture and linking to related concepts.

- **wiki/guided-layer-normalization.md** — Technique page explaining GLN's mechanism, mathematical reformulation, and extensibility to other models.

- **wiki/accident-aware-traffic-forecasting.md** — Concept page defining the problem domain of accident-aware traffic forecasting and ConFormer's approach.

## Modified Pages

- **wiki/traffic-forecasting.md** — Added new "Accident-Aware" subsection describing the problem and ConFormer's solution. Updated source_count from 6 to 7.

- **wiki/index.md** — Added entries for source-conformer, conformer, guided-layer-normalization, and accident-aware-traffic-forecasting in respective categories.

- **wiki/log.md** — Added ingest record.

## New Cross-Links

- [[traffic-forecasting]] → [[accident-aware-traffic-forecasting]] (new subsection)
- [[traffic-forecasting]] → [[source-conformer]] (new reference)
- [[conformer]] → [[guided-layer-normalization]]
- [[conformer]] → [[accident-aware-traffic-forecasting]]
- [[conformer]] → [[staeformer]] (previous SOTA)
- [[guided-layer-normalization]] → [[instance-normalization]] (related technique)

## Summary

ConFormer addresses a critical gap in traffic forecasting: modeling the disruptive impact of accidents. The paper introduces two key innovations (accident-aware graph propagation and Guided Layer Normalization) and contributes two enriched benchmark datasets (Tokyo and California highways with accident annotations). Performance improvements of up to 10.7% in accident scenarios demonstrate the value of explicitly modeling accident propagation.