---
title: "Periodic Temporal Attention (PTA)"
type: technique
tags:
  - attention
  - periodic-modeling
  - temporal-attention
  - traffic-forecasting
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Periodic Temporal Attention (PTA)

Periodic Temporal Attention (PTA) is the temporal encoding component of [[hephestus|HEPHAESTUS]], designed to explicitly capture daily and weekly periodic patterns in traffic data via parameterized periodic embedding matrices used as queries in a cross-attention mechanism[^src-hephestus].

## Mechanism

### Learnable Periodic Embeddings

Two embedding matrices are maintained and learned end-to-end[^src-hephestus]:

- **PD ∈ R^(LD × dtid)**: Daily periodic embedding, LD = 288 (5-minute intervals per day)
- **PW ∈ R^(LW × dtiw)**: Weekly periodic embedding, LW = 288×7 = 2016 (5-minute intervals per week)

### Time-Aware Query Generation

For each time step t, embeddings are retrieved by modulo indexing[^src-hephestus]:

1. **Retrieve**: PD[t mod LD, :] and PW[t mod LW, :]
2. **Concatenate**: Pe = [PD_indices ∥ PW_indices] ∈ R^(H × (dtid+dtiw))
3. **Project**: Pe' = Pe·Wqt ∈ R^(H×D)
4. **Broadcast**: Qt = Broadcast(Pe') across all N spatial nodes → R^(N×H×D)

### Cross-Attention Computation

Key Kt and value Vt are standard learned projections of the input H[^src-hephestus]:
- Kt = H·Wkt, Vt = H·Wvt
- Attention: At = Softmax(Qt·Kt^T / √D)
- Output: Zt = At·Vt

Layer normalization, residual connections, and multi-head mechanism are applied as in standard Transformer blocks[^src-hephestus].

## Design Rationale

Unlike standard temporal self-attention where queries are derived from the input sequence itself, PTA uses **learned periodic queries** — the query encodes absolute time position (which day of week, which 5-minute slot), making the attention pattern explicitly periodic rather than purely data-driven. This allows the model to learn "at 8:00 AM on Monday, what should traffic look like?" without needing to infer the pattern from noisy observations[^src-hephestus].

## Ablation Impact

Removing PTA degrades performance across all metrics[^src-hephestus]:
- METR-LA: MAE 3.36→3.45 (+2.7%), RMSE 7.06→7.19 (+1.8%)
- PEMS08: MAE 13.56→13.98 (+3.1%)

The impact is smaller than AMS-MoE removal but still significant, confirming that explicit periodicity modeling provides complementary value beyond multi-scale routing[^src-hephestus].

## Relationship to Other Approaches

| Method | Periodicity handling |
|--------|---------------------|
| [[phat|PHAT]] | FFT period detection + periodic folding + PNA attention within buckets |
| [[autoformer|Autoformer]] | Auto-Correlation via time-delay aggregation |
| [[cyclenet|CycleNet]] | Single learnable recurrent cycle per channel |
| [[timemixer|TimeMixer]] | Implicit via seasonal mixing (bottom-up) |
| [[hephestus|PTA]] | Explicit learnable daily/weekly embedding queries |

PTA is closest to ASTGCN's multi-component temporal attention but more parameterized and integrated within a Transformer rather than separate CNN+attention branches[^src-hephestus].

[^src-hephestus]: [[source-hephestus]]
