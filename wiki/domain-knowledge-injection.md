---
title: "Domain Knowledge Injection"
type: technique
tags:
  - llm
  - fine-tuning
  - prompt-engineering
  - spatio-temporal
  - few-shot-learning
  - calibration
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# Domain Knowledge Injection

**Domain Knowledge Injection** is a technique for calibrating numerical time-series predictions with human-like contextual reasoning, introduced by [[fstllm|FSTLLM]] (ICML 2025)[^src-fstllm]. It fine-tunes a large language model on carefully structured multi-component prompts to integrate domain knowledge, node-specific patterns, and backbone predictions into a unified reasoning step.

## Motivation

Prior LLM-for-time-series methods either fine-tune the LLM on purely numerical inputs (GPT4TS, LLM4TS) or prepend generic task descriptions as prompts ([[time-llm|Time-LLM]])[^src-fstllm]. These approaches underuse the LLM's capacity for contextual reasoning. FSTLLM argues that injecting node-specific domain knowledge — spatial context, temporal usage patterns, real-world constraints — enables human-like reasoning that is especially valuable in data-scarce regimes[^src-fstllm].

## Prompt Structure

Each training prompt consists of six components, designed to give the LLM a comprehensive view of the forecasting task[^src-fstllm]:

| Component | Content | Source |
|-----------|---------|--------|
| **Task Instruction** | Domain, time frequency, historical/forecast steps | Hand-crafted |
| **Node Description** | LLM-summarized node documents + user reviews | ChatGPT-4o |
| **Node Pattern** | Daily/weekly trends, peak/off-peak periods, typical values | ChatGPT-4o from training data |
| **Historical Input** | Raw time series $X_{ij} \in \mathbb{R}^T$ | Dataset |
| **Numerical Prediction Token** | STGNN backbone predictions $C_{ij} \in \mathbb{R}^T$ | STGNN output |
| **Future Token** | Ground truth (training only; omitted at inference) | Dataset |

The LLM is asked to output calibrated numerical predictions enclosed in brackets, with the future token serving as the supervised target during training[^src-fstllm].

## Training

A LLaMA-2-7B model is supervised-fine-tuned (SFT) with **QLoRA** (4-bit quantization, LoRA rank 64, learning rate 2e-4, 2 epochs) on a single NVIDIA A6000 GPU[^src-fstllm]. The node-by-node fine-tuning means node count does not affect per-node computational cost, though inference latency remains a bottleneck on single-GPU setups[^src-fstllm].

## Behavior at Inference

At inference, the fine-tuned LLM receives all prompt components except the future token and outputs context-aware predictions. FSTLLM reports that the LLM learns to respect real-world constraints — for example, capping parking availability predictions at a lot's maximum capacity of 512 spaces, and adjusting predictions based on weekday/weekend patterns and peak/off-peak timing[^src-fstllm].

## Effectiveness

Ablation studies show that removing Domain Knowledge Injection causes the largest performance degradation among FSTLLM's components (MAE 21.0 → 25.1 on Nottingham), confirming it is the framework's most critical module[^src-fstllm].

## Related

- [[fstllm]] — the originating method
- [[llm-enhanced-graph-construction]] — the companion graph module
- [[time-llm]] — generic prompt-as-prefix approach (contrast)
- [[few-shot-traffic-forecasting]] — the problem setting

[^src-fstllm]: [[source-fstllm]]
