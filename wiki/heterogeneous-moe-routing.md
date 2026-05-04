---
title: "异质性感知 MoE 路由"
type: technique
tags:
  - mixture-of-experts
  - routing
  - heterogeneity
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# 异质性感知 MoE 路由

**异质性感知 MoE 路由 (Heterogeneity-Aware MoE Routing)** 是 [[mixture-of-experts|FaST]] 框架（KDD 2026）中提出的动态专家路由机制。它解决了标准 Top-K MoE 路由中 expert 极化（少数 expert 承担大部分负载）的问题，通过设计异质性感知的门控网络使路由决策适配节点多样性，使各 expert 获得平衡的梯度更新。

## Related Pages
- [[mixture-of-experts]]
- [[gated-linear-units]]
- [[source-fast-long-horizon-forecasting]]