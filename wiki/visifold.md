---
title: "VisiFold"
type: entity
tags:
  - traffic-forecasting
  - spatial-temporal
  - transformer
  - long-term-forecasting
  - tokenization
  - efficient-ml
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# VisiFold

VisiFold is an efficient long-term traffic forecasting framework that introduces the [[temporal-folding-graph|Temporal Folding Graph (TFG)]] and [[node-visibility|Node Visibility]] to break resource constraints in both temporal and spatial dimensions[^src-visifold].

## Core Problem

Long-term traffic forecasting (24–48 steps) faces two critical challenges that conventional spatial-temporal graphs cannot address[^src-visifold]:

1. **Snapshot-stacking inflation** — GPU memory and runtime grow rapidly with the forecast horizon, as each time step requires a separate snapshot in the spatial-temporal graph
2. **Cross-step fragmentation** — temporal dependencies are partitioned across separate snapshots, conveyed only through multiple intermediate representations, degrading long-term prediction quality

## Architecture

VisiFold's pipeline consists of four stages[^src-visifold]:

### 1. Temporal Folding Graph (TFG)
All attributes across a sequence of T snapshots are embedded into a single enriched node attribute vector (TF-token). The N×T×C input is squeezed to N×T, avoiding spatial-temporal decoupling. Information propagates on a single graph, with temporal dynamics compressed within nodes and spatial dependencies exchanged across nodes. See [[temporal-folding-graph]].

### 2. Embedding Fusion
TF-tokens undergo linear projection to token embeddings $E_x \in \mathbb{R}^{N \times d}$, then concatenated with:
- **Spatial embeddings** $E_s \in \mathbb{R}^{N \times d}$ — learnable per-node identity matrix
- **Time-of-day embeddings** $E_{tod}$ — shared across all nodes, derived from the last timestamp
- **Day-of-week embeddings** $E_{dow}$ — shared across all nodes

Final embedding: $E = E_x \| E_s \| E_{tod} \| E_{dow} \in \mathbb{R}^{N \times 4d}$[^src-visifold].

### 3. Node Visibility
Applied during training only (not inference). See [[node-visibility]] for details[^src-visifold]:
- **Node-level masking** — randomly removes a proportion r of nodes from the encoder (following MAE design), directly reducing input size
- **Subgraph sampling** — randomly partitions remaining (1−r)N nodes into subgraphs of size s, increasing parallelism

### 4. Transformer Encoder + MLP Head
The refined representations are processed by L layers of multi-head self-attention and feed-forward network blocks with layer normalization and skip connections. An MLP prediction head with GELU activation produces the final output. Huber loss is used for training[^src-visifold].

## Complexity

The TFG eliminates the temporal module, reducing token count from N×T to N. After node visibility, complexity drops to $O((1-r)N s + ps)$, where s is the constant subgraph size and p is padding[^src-visifold].

## Performance

Evaluated on PEMS04, PEMS08, and SEATTLE with 24/36/48-step horizons vs. 12 baselines (HA, VAR, DCRNN, GWNet, GMAN, AGCRN, DMSTGCN, SSTBAN, STID, STAEformer, STPGNN, STDN)[^src-visifold]:

- **Accuracy**: SOTA across all prediction horizons and datasets
- **Training speed**: ~7× faster than STAEformer (17.8× at 48 steps), up to 52.2× vs. SSTBAN
- **GPU memory**: ~4× less than STAEformer (15.7× at 48 steps), increasing only marginally with longer horizons
- **Inference**: <1 second, suitable for real-time deployment

## Key Ablation Findings

- **Spatial embeddings** are the dominant accuracy driver; removing them causes the largest degradation[^src-visifold]
- **Node visibility** significantly improves robustness and accuracy while reducing memory by 10–12%[^src-visifold]
- **Mask ratio 0.8** (80% nodes masked) still yields performance gains over the baseline, revealing substantial data redundancy[^src-visifold]
- **Node-level masking** outperforms alternative strategies (AllZero, PartialZero, RandomValue) due to train-test gap avoidance[^src-visifold]
- **TFG >> SF** (spatial folding): folding along temporal dimension is significantly better than spatial folding because spatial embeddings cannot be seamlessly integrated[^src-visifold]
- Node-specific temporal embeddings and subgraph interaction schemes (leader tokens) provide no additional benefit[^src-visifold]

## Limitations

Purely data-driven — cannot handle emergent events (accidents, closures). The authors suggest multi-modal metadata integration and temporal module incorporation as future work[^src-visifold].

## Related Pages

- [[temporal-folding-graph]] — the core tokenization innovation
- [[node-visibility]] — masks + subgraph sampling mechanism
- [[ragc|RAGC]] — another efficient large-scale traffic forecasting approach
- [[traffic-forecasting]] — broader context
- [[source-visifold]] — source summary

[^src-visifold]: [[source-visifold]]