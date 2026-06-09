---
title: "DoFlow"
type: entity
tags:
  - time-series
  - causal-inference
  - continuous-normalizing-flow
  - flow-matching
  - counterfactual
  - interventional
  - generative-model
  - anomaly-detection
  - iclr
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# DoFlow

**DoFlow** is a flow-based generative model defined over a causal directed acyclic graph (DAG) that unifies **observational, interventional, and counterfactual** time-series forecasting on multivariate dynamical systems[^src-doflow]. It was proposed by Dongze Wu, Yao Xie (Georgia Tech) and Feng Qiu (Northwestern–Argonne) and published at ICLR 2026[^src-doflow]. The name evokes Pearl's *do*-operator combined with normalizing **flow**.

## Motivation

Most modern forecasters are purely observational—they learn correlations and extrapolate, but cannot answer causal "what-if" queries[^src-doflow]. DoFlow targets two causal queries: an **interventional** query ("how will the forecast change under a planned modification of certain variables?") and a **counterfactual** query ("what would this specific observed trajectory have looked like had we intervened differently?")[^src-doflow]. The authors state that, to their knowledge, no general framework for counterfactual time-series forecasting existed at the time of writing[^src-doflow].

## Problem Setup

DoFlow models a $K$-dimensional series over a topologically-sorted causal DAG with structural causal model (SCM) $X_{i,t} := f_i(X_{i,t^-}, X_{\text{pa}(i),t^-}, U_{i,t})$, where $U_{i,t}$ is exogenous noise independent across nodes and time, and causal influences occur with at least a one-time-step lag (no within-time-step effects)[^src-doflow]. Each sequence is split into a context window $\{X_1,\dots,X_\tau\}$ (never intervened on) and a forecasting window $\{X_{\tau+1},\dots,X_T\}$ where interventions may be applied[^src-doflow]. An intervention schedule $I \subseteq [K]\times\{\tau+1,\dots,T\}$ sets $\text{do}(X_{i,t}:=\gamma_{i,t})$; with $I=\emptyset$ the model reduces to standard observational forecasting[^src-doflow].

## Architecture

DoFlow learns a **separate time-conditioned [[continuous-normalizing-flow|CNF]] per node** $i$, shared across time steps, conditioned on a hidden state that summarizes histories[^src-doflow]:

- **RNN history encoder.** An LSTM or GRU summarizes node $i$'s past via $h_{i,t}=\text{RNN}(x_{i,t}, h_{i,t-1})$; the CNF conditioning state concatenates the node and its parents: $H_{i,t-1}:=\text{concat}(h_{i,t-1}, h_{\text{pa}(i),t-1})$[^src-doflow].
- **Per-node CNF.** A [[neural-ordinary-differential-equation|Neural ODE]] $\frac{dx_{i,t}(s)}{ds}=v_i(x_{i,t}(s), s; H_{i,t-1})$ over ODE-time $s\in[0,1]$ connects the data distribution ($s=0$) to a base $\mathcal{N}(0,1)$ ($s=1$); $s$ is distinct from the time-series index $t$[^src-doflow]. The velocity field $v_i$ is a small MLP (3 layers, hidden width 64 in experiments)[^src-doflow].
- **Training.** Conditional Flow Matching ([[flow-matching|CFM]]) regresses $v_i$ onto a straight-line reference velocity $\partial_s\phi = z - x_{i,t}$ from a linear interpolant $\phi=(1-s)x_{i,t}+sz$; the loss is summed over the whole forecasting window with hidden states updated autoregressively from observed values during training[^src-doflow].

## Forward and Reverse Processes

DoFlow exploits CNF invertibility[^src-doflow]:

- **Forward (encoding)** $z_{i,t}^F := \Phi_\theta(x_{i,t}^F; H_{i,t-1}^F)$ integrates the ODE from $s=0$ to $s=1$, mapping a factual observation to a latent embedding conditioned on the factual hidden state[^src-doflow].
- **Reverse (decoding)** $\hat{x}_{i,t} := \Phi_\theta^{-1}(z_{i,t}; \hat{H}_{i,t-1})$ integrates from $s=1$ to $s=0$; the latent $z_{i,t}$ is either sampled from $\mathcal{N}(0,1)$ or obtained by encoding a factual sample[^src-doflow].

## Three Prediction Modes

| Mode | How latents are obtained | Output |
|------|--------------------------|--------|
| Observational | $z\sim\mathcal{N}(0,1)$, all nodes generated | probabilistic forecast |
| Interventional | $z\sim\mathcal{N}(0,1)$ for non-intervened nodes; intervened nodes fixed to $\gamma_{i,t}$ | probabilistic forecast under $\text{do}(\cdot)$ |
| Counterfactual | factual values **encoded** then decoded under the counterfactual state | single deterministic trajectory |

Forecasting proceeds in **topological order** (parents before children), with hidden states $\hat{H}_{i,t}$ autoregressively updated from generated/intervened values so interventions propagate through the DAG[^src-doflow]. Counterfactual generation follows the **abduction–action–prediction** procedure (see [[causal-counterfactual-recovery]])[^src-doflow]. Because counterfactuals fix the abducted exogenous noise, they yield a single deterministic trajectory rather than a distribution[^src-doflow].

## Likelihood-based Anomaly Detection

Via the change-of-variables / Liouville continuity equation, DoFlow assigns an explicit conditional log-density to a generated forecast trajectory: $\log p_\theta(\hat{x}_{\tau+1:T}\mid\hat{H}_\tau)=\sum_t [\log q(z_t) + \int_0^1 \nabla\!\cdot v_\theta(\cdot)\,ds]$ (Proposition 3.1)[^src-doflow]. Anomalous contexts get lower self-assigned density, enabling early outage detection: on real hydropower outages, DoFlow's log-probability becomes abnormal as early as 20 minutes before the outage[^src-doflow].

## Experiments

- **Synthetic DAGs** (Tree, Diamond, FC-Layer, Chain; additive and nonlinear-non-additive SCMs), evaluated with RMSE, MMD, CRPS[^src-doflow]. Interventions invert the root-node sinusoidal cycle by a half-period (antiphase) to create a challenging regime[^src-doflow]. DoFlow consistently beats adapted observational baselines (GRU, TFT, TiDE, TSMixer, DeepVAR, MQF2) on observational and interventional forecasting and uniquely supports counterfactuals (baselines report NA)[^src-doflow].
- **Hydropower system** (Argonne National Laboratory): an 8-node natural DAG (turbine → generator → transformer, with M&C unit); interventional forecasting under 12 real outages plus likelihood-based anomaly detection[^src-doflow].
- **Cancer treatment** (Bica et al. 2020a): treatments (chemo/radio assignment + dosage) are causal parents of tumor volume; only the outcome node is modeled with a CNF while treatments are observed[^src-doflow]. DoFlow's normalized RMSE substantially beats CRN, RMSN, and MSM for causal treatment-effect estimation[^src-doflow].

## Limitations

- **Known DAG required.** DoFlow assumes a known, correctly-specified causal DAG and **causal sufficiency** (no unobserved confounders); a natural extension is to couple it with time-series causal discovery or deconfounding generative models (e.g., DeCaFlow, Neural Causal Models)[^src-doflow].
- **Per-node flows.** A separate flow is trained per node, though each network is shallow and total model size / training time are comparable to modern baselines[^src-doflow].

## Relation to Other Models

DoFlow is a causal generative forecaster, distinct from observational flow/diffusion forecasters such as [[tsflow|TSFlow]], [[flowts|FlowTS]], and [[sundial|Sundial]], which lack interventional/counterfactual capability[^src-doflow]. Its [[continuous-normalizing-flow|CNF]] backbone is the temporal extension of static causal normalizing flows; [[e2-cstp|E²-CSTP]] addresses causality in spatio-temporal prediction via a complementary confounder-blocking route.

## Links

- [[source-doflow]] — source summary
- [[causal-counterfactual-recovery]] — abduction–action–prediction & counterfactual recovery theory
- [[continuous-normalizing-flow]] — CNF backbone
- [[flow-matching]] — CFM training objective
- [[neural-ordinary-differential-equation]] — Neural ODE foundation
- [[probability-flow-ode]] — related deterministic ODE sampling with exact likelihood
- [[normalizing-flow]] — invertible generative models
- [[e2-cstp]] — causal spatio-temporal prediction (deconfounding route)

[^src-doflow]: [[source-doflow]]
