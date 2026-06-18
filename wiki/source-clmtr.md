---
title: "CLMTR: Contrastive Learning for Multi-modal Trajectory Representation"
type: source-summary
tags:
  - trajectory
  - contrastive-learning
  - multi-modal
  - representation-learning
  - self-supervised
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# CLMTR: Contrastive Learning for Multi-modal Trajectory Representation

CLMTR proposes a generic self-supervised framework for learning multi-modal trajectory representations that combine spatial, temporal, and textual features [^src-clmtr].

## Core Architecture

The framework consists of three main components. **Multi-modal feature embedding** uses Node2Vec for spatial location embeddings (capturing spatial proximity among prominent grid cells), sine-based learnable functions for time embeddings (capturing periodicity), and pre-trained BERT for text embeddings. An attention-based fusion mechanism computes cross-modal interaction scores and produces enhanced fused embeddings via weighted combination [^src-clmtr].

**Intra-trajectory contrastive learning** contrasts different modal features within the same trajectory. The default view pairs textual features against fused spatio-temporal features, enabling cross-modal learning where one modality informs another. Trajectories serve as positive pairs when they represent different views of the same trajectory, and negative pairs otherwise [^src-clmtr].

**Inter-trajectory contrastive learning** compares different trajectories using a similarity-based strategy: nearest-neighbor trajectories (via Fréchet distance for spatial/temporal and Edit distance for textual) form positive pairs, while trajectories beyond k-nearest neighbors are negatives. Four data augmentation strategies are employed: downsampling, distorting (adding noise), trimming (creating sub-trajectories), and simplification (Douglas-Peucker algorithm). The InfoNCE loss is used with a momentum-updated encoder and a negative sample queue [^src-clmtr].

## Empirical Results

Experiments on Geolife+ and T-Drive+ datasets (enriched with POI data via AMAP API) evaluate three downstream tasks. CLMTR outperforms six baselines (At2vec, At2vec-attn, E2DTC, ST2Vec, CL-TSim, TrajCL) on trajectory similarity search, clustering, and travel time estimation across nearly all metrics. Ablation studies confirm the effectiveness of each component: Node2Vec-based location embedding, sine-based time embedding, BERT text embedding, and attention-based fusion all outperform alternatives. The intra- and inter-trajectory contrastive components both contribute significantly, with downsampling and trimming identified as the most effective augmentation strategies [^src-clmtr].

The framework is efficient, training in about 70 minutes on two RTX 3090 GPUs, with Transformer-based inference outperforming RNN-based methods. Limitations include the 200-point maximum trajectory length and the focus on Euclidean space rather than road networks [^src-clmtr].

[^src-clmtr]: [[source-clmtr]]