---
title: "Spatio-Temporal Reasoning"
type: concept
tags:
  - spatio-temporal
  - reasoning
  - llm
  - time-series
  - graph
created: 2026-06-04
last_updated: 2026-06-04
source_count: 1
confidence: medium
status: active
---

# Spatio-Temporal Reasoning

**Spatio-temporal reasoning in time series** is the ability to answer natural-language queries that require explicit reasoning over temporal dynamics and spatial dependencies in systems where state evolves over time and is coupled across space (e.g., a graph)[^src-streasoner]. Formally defined by Ni et al. (2026): given a graph G, time series T, and query Q, the model must generate an intermediate reasoning chain R and final answer A[^src-streasoner].

## Differences from Forecasting

| Dimension | Spatio-Temporal Forecasting | Spatio-Temporal Reasoning |
|-----------|---------------------------|--------------------------|
| Output | Numerical predictions | Text-based answers with reasoning chains |
| Spatial use | Implicit (GNN aggregation) | Explicit (trace paths, identify propagation delays) |
| Task type | Regression | QA, classification, causal inference |
| Example | "Predict traffic at Node 5 at T+6" | "Which source node caused the congestion at Node 2 at 9:00?" |
| Evaluation | MAE, RMSE, MAPE | Accuracy, reasoning quality |

Traditional ST forecasting methods (e.g., DCRNN, STGCN) only answer "what will happen"; ST reasoning answers "what happened, where, when, and why"[^src-streasoner].

## Key Capabilities

Spatio-temporal reasoning requires[^src-streasoner]:

1. **Spatial localization**: Identify which nodes are involved
2. **Temporal tracing**: Track propagation through time, accounting for delays τᵢⱼ
3. **Causal attribution**: Determine which source node caused a downstream effect
4. **Multi-hop reasoning**: Follow influence across multiple edges in the graph
5. **Integration of modalities**: Combine time series numerical data, graph topology, and text semantics

## Core Challenges

1. **Data scarcity**: Existing ST datasets rarely provide paired natural language descriptions of spatial entities, dependencies, or temporal dynamics[^src-streasoner]
2. **Evaluation gaps**: No standardized multi-dimensional benchmarks decomposing ST reasoning into distinct tasks[^src-streasoner]
3. **Modeling limitations**: Unclear how to fuse time series, graph structure, and textual information without sacrificing numerical precision or global context[^src-streasoner]
4. **Spatial grounding**: Models may exploit superficial temporal patterns rather than perform genuine spatial attribution[^src-streasoner]

## ST-Bench

The first benchmark for spatio-temporal reasoning, introduced with [[streasoner|STReasoner]], consisting of four tasks[^src-streasoner]:

| Task | Description | Requires |
|------|------------|----------|
| T1: Etiological | Infer global system dynamics from observations | Understanding demand/propagation semantics |
| T2: Entity | Recognize semantic roles of nodes | Distinguishing source vs propagation nodes |
| T3: Correlation | Causal reasoning over spatial structure | Multi-hop influence path tracing |
| T4: In-context Forecasting | Predict future under spatial dependencies | Joint spatio-temporal modeling |

Data is generated via a network SDE-based multi-agent pipeline that ensures aligned text-data pairs[^src-streasoner].

## S-GRPO: Spatial-Aware Training

**S-GRPO** (Spatial-Aware Group Relative Policy Optimization) is a RL algorithm proposed alongside STReasoner to explicitly incentivize spatial reasoning[^src-streasoner]. It generates two groups of responses — with and without spatial structure — and only grants bonus reward (α) when spatial information improves accuracy[^src-streasoner]. This contrastive design is analogous to how humans learn spatial reasoning: by comparing what can and cannot be explained without spatial context.

## Existing Models

| Model | Spatial | Reasoning | Paradigm |
|-------|:--:|:--:|------|
| [[streasoner|STReasoner]] | ✓ | ✓ Multi-step CoT | Align+SFT+S-GRPO |
| [[vot|VoT]] | ✗ | ✓ LLM reasoning | Event-driven dual-branch |
| Time-R1 | ✗ | ✓ Multi-step | SFT+GRPO (text only) |
| ChatTS | ✗ | ✓ Basic QA | SFT |
| [[time-llm|Time-LLM]] | ✗ | ✗ | Prompt reprogramming |
| [[conformer|ConFormer]] | ✓ | ✗ | Accident-informed prediction |

As of 2026, STReasoner is the only model that explicitly performs spatio-temporal reasoning with both spatial graph awareness and multi-step chain-of-thought[^src-streasoner].

## Related Pages

- [[streasoner]] — STReasoner model entity
- [[source-streasoner]] — source summary
- [[spatio-temporal-foundation-model]] — ST foundation model concept
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[multimodal-spatial-reasoning]] — broader MLLM spatial reasoning framework (2D/3D/embodied)
- [[time-llm]] — Time-LLM, predecessor TS-LLM
- [[vot]] — VoT, event-driven reasoning model

[^src-streasoner]: [[source-streasoner]]
