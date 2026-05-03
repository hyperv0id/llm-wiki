---
title: "Cross-View Text Fusion"
type: technique
tags:
  - text-fusion
  - cross-view-attention
  - multimodal-time-series
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Cross-View Text Fusion

Cross-view text fusion is a technique introduced in [[mindts|MindTS]] (ICLR 2026) that decomposes time-series-paired text into two complementary views — endogenous and exogenous — and fuses them via cross-view attention before aligning with time series[^src-multimodal-ts-anomaly-detection].

## Two Views

- **Endogenous text**: Per-patch statistical descriptions (mean, max, trend) that are temporally aligned with time series patches
- **Exogenous text**: Shared background context (news, reports) that provides broader semantic information but lacks direct temporal correspondence

The endogenous text serves as **query** (time-specific), while exogenous text serves as **key/value** (background knowledge). This ensures the fused representation is both contextually rich and temporally precise[^src-multimodal-ts-anomaly-detection].

## Related

- [[fine-grained-time-text-semantic-alignment]] — the alignment technique that uses this fusion output
- [[mindts]] — model that introduced cross-view text fusion
- [[endogenous-text-alignment]] — VoT's different approach to endogenous/exogenous text

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
