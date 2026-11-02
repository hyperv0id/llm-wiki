---
title: "VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection"
type: source-summary
tags:
  - time-series
  - anomaly-detection
  - zero-shot
  - multimodal
  - vision-transformer
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection

## 核心论点

论文提出 VETime 框架，针对零样本时序异常检测中 1D 数值模型因滑动窗口感受野局限难以捕捉长程上下文异常、2D 视觉与 VLM 模型因非可逆映射或粗粒度窗口导致时间索引丢失且推理延迟昂贵的两难困境，提出通过跨模态补丁级有序对齐与动态专家融合实现双模态感知互补 [^src-2602-16681-vetime]。

## 核心贡献

1. **可逆视觉转换与补丁级时间对齐**：提出可追溯图像转换（[[traceable-image-conversion|TIC]]）与补丁级时序对齐（[[patch-level-temporal-alignment|PTA]]）。TIC 通过季节-趋势分解与主周期自适应折叠将 1D 序列重构为 RGB 图像流形；PTA 逆转折叠几何并通过重采样、池化与自注意力机制，将冻结预训练 ViT 提取的 2D 视觉补丁精确投影回 1D 共享时间轴 [^src-2602-16681-vetime]。
2. **多尺度对比与动态门控融合**：设计异常窗口对比学习（[[anomaly-window-contrastive-learning|AWCL]]），分别以窗内与窗间对比拉近短程点异常与长程上下文异常的跨模态表征；提出任务自适应多模态融合（[[task-adaptive-multimodal-fusion|TMF]]），配合序列重构辅助任务与路由负熵正则化实现动态专家路由 [^src-2602-16681-vetime]。
3. **严格零样本实证表现**：在仅基于 0.5B 数学方程合成数据预训练的严格零样本协议下，VETime 在 TSB-AD 26 个数据集（11 单变量、15 多变量）评测中，单变量 44 项对比中取得 24 项第一、平均排名 2.05，多变量 60 项对比中取得 34 项第一、平均排名 1.58（表 1）；较视觉大模型基线 VLM4TS 实现约 100 倍推理加速（单序列耗时 0.02–0.04 秒 vs 1.72–9.27 秒，表 2、表 11）[^src-2602-16681-vetime]。

## 局限与边界

论文自述局限包括：对高频不规则噪声敏感（依赖季节-趋势分解）；固定 $224 \times 224$ 画布在序列长 5000 下单点单像素物理上限约 10 个变量，高维场景存在空间拥挤（§5、附录 C.3）；在非周期信号（MGAB、Stock）与周期尖峰异常（Power）上存在负迁移风险（附录 D.5）；引入视觉主干使模型参数量增至 157.96M（附录 C.5），且多模态时序表征仍缺乏严密数理理论解释（§5）[^src-2602-16681-vetime]。

## 元信息

- **作者与机构**：Yingyuan Yang$^1$, Tian Lan$^1$, Yifei Gao$^1$, Yimeng Lu$^1$, Xuming An$^1$, Wenjun He$^2$, Meng Wang$^2$, Chenghao Liu$^3$, Chen Zhang$^1$（$^1$Department of Industrial Engineering, Tsinghua University, Beijing, China; $^2$2012 Lab, Huawei Technologies Ltd, Shenzhen, China; $^3$Datadog AI Research）
- **出版状态**：arXiv:2602.16681v3 [cs.CV] 29 Oct 2026，NeurIPS 2026 投稿格式（40th Conference on Neural Information Processing Systems (NeurIPS 2026)）

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
