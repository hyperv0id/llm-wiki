---
title: "Backdoor Adjustment"
type: concept
tags:
  - causal-inference
  - do-calculus
  - structural-causal-model
  - spatio-temporal
created: 2026-07-17
last_updated: 2026-07-17
source_count: 2
confidence: high
status: active
---

# Backdoor Adjustment

**Backdoor adjustment** is a causal inference technique derived from Pearl's [[do-calculus]] that estimates the causal effect of a treatment $X$ on an outcome $Y$ by conditioning on a set of confounders $S$ that block all "backdoor paths" — non-causal paths between $X$ and $Y$ that pass through common causes[^src-e2-cstp][^src-cast].

## The Backdoor Criterion

A set of variables $S$ satisfies the backdoor criterion relative to $(X, Y)$ if[^src-e2-cstp]:

1. No element of $S$ is a descendant of $X$
2. $S$ blocks every path between $X$ and $Y$ that contains an arrow into $X$ (i.e., all backdoor paths)

When the criterion holds, the interventional distribution can be estimated from observational data:

$$P(Y \mid \text{do}(X=x)) = \int_S P(Y \mid X=x, S=s) \cdot P(S=s) \, dS$$

## Application in Spatio-Temporal Prediction

Backdoor adjustment has been adopted by two recent methods to address confounding in spatio-temporal forecasting:

### E²-CSTP: Multi-Modal Confounding

[[e2-cstp|E²-CSTP]] (NeurIPS 2025) identifies the backdoor path $X_{st} \leftarrow S \rightarrow Y_{st}$, where an unobserved confounder $S$ simultaneously affects both the spatio-temporal signal and the prediction target[^src-e2-cstp]. The framework integrates over $S$ conditioned on auxiliary modalities (environmental images $E$, event text $C$):

$$P(Y_{st} \mid \text{do}(X_{st}=x), E, C) = \int_S P(Y_{st} \mid X_{st}=x, S=s_i, E, C) \cdot P(S=s_i \mid E, C) \, dS$$

The intervention is realized by adjusting each input $x$ to $x̂$ via adversarial training that drives $\partial x̂/\partial S \to 0$, ensuring $x̂ \perp S \mid E, C$[^src-e2-cstp].

### CaST: Temporal + Spatial Backdoor Paths

CaST (NeurIPS 2023) identifies two backdoor paths in STG data: $X \leftarrow E \rightarrow Y$ (temporal environment $E$ as confounder) and $X \leftarrow C \rightarrow Y$ (spatial context $C$ as confounder)[^src-cast]. It applies backdoor adjustment by stratifying temporal environments via a Vector Quantization codebook, and uses front-door adjustment with Hodge-Laplacian for dynamic spatial causation[^src-cast].

## Comparison with Front-Door Adjustment

| Criterion | Backdoor | Front-Door |
|-----------|----------|------------|
| Mechanism | Block confounders (condition on $S$) | Isolate mediator (condition on $M$) |
| Data requirement | Need measured confounders or proxies | Need mediator between $X$ and $Y$ |
| Use in ST | E²-CSTP, CaST (temporal) | CaST (spatial) |

## Related Pages

- [[do-calculus]] — foundational framework
- [[e2-cstp]] — multi-modal causal ST prediction using backdoor adjustment
- [[causal-time-series-forecasting]] — broader causal TS paradigm
- [[structural-causal-model]] — SCM formalization underlying backdoor paths

[^src-e2-cstp]: [[source-e2-cstp]]
[^src-cast]: [[source-cast]]
