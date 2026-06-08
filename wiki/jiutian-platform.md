---
title: "Jiutian Platform"
type: entity
tags:
  - china-mobile
  - ai-platform
  - network-simulation
  - network-optimization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Jiutian Platform

The **Jiutian platform** is an AI platform developed by China Mobile, featuring scenario construction, network simulation, optimization strategy formulation, and performance evaluation capabilities[^src-uomo]. It has been fully deployed within China Mobile, supporting network development across **31 provinces** in China[^src-uomo].

## Capabilities

- **Full-element network simulation**: Efficiently simulates interactions between communication systems and user behavior
- **Custom algorithm development**: Supports operators in developing customized algorithms and applications
- **Production deployment**: Algorithms can be deployed into production environments and validated with real network data
- **Optimization pipeline**: Scenario construction -> network simulation -> strategy formulation -> performance evaluation

## UoMo Integration

The [[uomo|UoMo]] model is deployed on the Jiutian platform in the mobile traffic module, with its predictions feeding into the optimization selection module[^src-uomo]. The platform's optimization workflow:

1. UoMo generates traffic data (prediction or generation based on task)
2. Platform formulates and solves network optimization/planning problems
3. Optimal strategies are validated using real live network traffic data

## Deployed Optimizations

Currently live in Nanning, Guangxi Province[^src-uomo]:

- **BS deployment**: Grid-level base station placement optimization maximizing served users while minimizing operation costs and capacity shortfalls. UoMo-based strategy outperforms POI-based and resident-based approaches
- **BS sleep control**: C-RAN RRU activation/sleep scheduling based on UoMo's long-term traffic predictions, balancing QoS, equipment depreciation, and energy consumption

## Hardware

UoMo training on Jiutian uses 4x NVIDIA A100 GPUs (80GB each) with PyTorch 2.0.1[^src-uomo].

[^src-uomo]: [[source-uomo]]
