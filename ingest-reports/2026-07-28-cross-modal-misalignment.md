# Ingest 报告：on-the-value-of-cross-modal-misalignment-in-multimodal-representation-learning

## 创建
- wiki/source-cross-modal-misalignment.md — WHY：arXiv:2504.10143 NeurIPS 2025 源摘要（LVM、selection/perturbation、Thm.4.1、Cor.4.1/4.2、OpenCLIP）
- wiki/cross-modal-misalignment.md — WHY：概念页（缓解 vs 利用、形式化、与 TS–VL/CFA/Time-MMD 关系）

## 修改
- wiki/ts-vl-alignment.md — WHY：外生含义表补 MMCL 理论行；脚注/相关；source_count 2→3
- wiki/constrained-text-fusion.md — WHY：谱系挂 misalignment；脚注/相关；source_count 1→2
- wiki/contrastive-learning.md — WHY：MMCL misalignment 专节；source_count 5→6
- wiki/multimodal-time-series-forecasting.md — WHY：专节「缓解 vs 利用」+ 相关/脚注；source_count 16→17
- wiki/time-mmd.md — WHY：谱系与相关挂 misalignment；source_count 2→3
- wiki/index.md — WHY：登记 source-cross-modal-misalignment / cross-modal-misalignment
- wiki/log.md — WHY：记录 ingest

## 新建交叉链接
- [[cross-modal-misalignment]] ↔ [[source-cross-modal-misalignment]]
- [[cross-modal-misalignment]] ↔ [[ts-vl-alignment]] / [[constrained-text-fusion]] / [[contrastive-learning]] / [[multimodal-time-series-forecasting]] / [[time-mmd]]

## 源文件
- raw/on-the-value-of-cross-modal-misalignment-in-multimodal-representation-learning.pdf（只读）

## 自检
- 脚注 `[^src-cross-modal-misalignment]` → `[[source-cross-modal-misalignment]]`；新/改 md 无逃逸 wikilink 竖线
- 核心 claim：misalignment ≠ 纯噪声；MMCL block-id 无偏语义；预训练缓解 / OOD 可利用
- 未 git / 未改 raw
