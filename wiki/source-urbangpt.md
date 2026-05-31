---
title: "UrbanGPT: Spatio-Temporal Large Language Models (KDD 2024)"
type: source-summary
tags:
  - spatial-temporal
  - large-language-model
  - instruction-tuning
  - zero-shot
  - traffic-forecasting
  - crime-prediction
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# UrbanGPT: Spatio-Temporal Large Language Models

**Authors**: Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu (HKU / SCUT), Lei Shi, Long Xia, Dawei Yin (Baidu Inc.), Chao Huang* (HKU)
**Venue**: KDD 2024 (arXiv:2403.00813, May 19, 2024)
**Project**: https://urban-gpt.github.io/ | **Code**: https://github.com/HKUDS/UrbanGPT

## 核心贡献

UrbanGPT 是首个将大语言模型（LLM）应用于时空预测的工作，目标是构建一个能够在数据稀缺场景下展示出色零样本泛化能力的时空大语言模型[^src-urbangpt]。论文核心贡献包括：（1）首次提出时空大语言模型框架，能跨不同数据集预测多样城市现象，尤其在标注数据有限条件下；（2）通过时空依赖编码器与指令微调范式的无缝集成，使 LLM 理解跨时空的复杂相互依赖性；（3）在 4 个基准数据集上的零样本和监督实验中全面超越 10 个基线模型[^src-urbangpt]。

## 架构设计

UrbanGPT 是一个四组件流水线[^src-urbangpt]：

**（1）时空依赖编码器**：多级门控膨胀卷积网络（无图结构），公式为 Ψ_r^{(l)} = (W̄_k * E'_r + b̄_k) ⊙ δ(W̄_g * E'_r + b̄_g) + E'_r，其中 W̄_k, W̄_g 是 1D 膨胀卷积核。多级关联注入层 S_r^{(l)} = (W_s * Ψ_r^{(l)} + b_s) + S_r^{(l-1)} 通过残差累加保留不同时间粒度的模式。关键设计决策：编码器不依赖图结构（GCN/GAT），因为在零样本场景下目标区域的空间关系未知，无图设计更通用[^src-urbangpt]。

**（2）时空-文本对齐模块**：轻量投影 H = Ψ̃ · W_p + b_p（d=64 → d_L=4096，Vicuna-7b 隐维度）。对齐后的表征被表示为特殊 token 嵌入 <ST_start>, <ST_HIS>, ..., <ST_HIS>, <ST_end>，使 LLM 在自注意力中将时空信息作为"特殊含义的词"处理[^src-urbangpt]。

**（3）时空 prompt 指令**：三个维度的自然语言编码——时间信息（"2020年1月7日星期二08:30"）、空间信息（"史坦顿岛行政区，POI 包括公共安全、教育设施、住宅类"）、任务描述（预测未来 12 步出租车流入流出）。这些文本利用 LLM 内化的世界知识推断时空模式[^src-urbangpt]。

**（4）回归预测层**：Ŷ_{r,f} = W_3 [σ(W_1 · H_{r,f}), σ(W_2 · Γ_{r,f})]，其中 H_{r,f} 是时空编码器输出，Γ_{r,f} 是 LLM 输出预测 token 的隐层表示。核心精妙：LLM 不直接输出数值（LLM 的多分类损失 vs 回归需求结构不匹配），而是输出富含时空推理信息的隐向量，由回归层负责精确数值映射[^src-urbangpt]。

## 训练与优化

多任务联合优化：回归任务 L_r = (1/N) Σ|y_i - ŷ_i|（MAE），分类任务 L_c = -(1/N) Σ[δ(y_i)·log(ŷ_i) + (1-δ(y_i))·log(1-ŷ_i)]（二元交叉熵），总损失 L = L_LLMs + L_r + L_c[^src-urbangpt]。LLM 骨干为 Vicuna-7b，历史/预测步数 H=P=12。训练数据来自 NYC 的 taxi（2017Q1）、bike（2017Q2）、crime（2016-2018）三个数据集各 80 个区域[^src-urbangpt]。

## 实验结果

**零样本跨区域**（NYC-taxi/-bike/-crime 未见区域）：所有 3 个数据集所有指标全面超越 10 个基线。NYC-taxi inflow MAE=6.16 vs 最佳基线 ASTGCN=9.75（↓36.8%），NYC-crime burglary Macro-F1=0.67（传统模型 Recall≈0，UrbanGPT Recall=0.34）。**跨城市**（CHI-taxi，完全未训练）：所有 12 个预测步均超越基线，优势不随时间步衰减。**监督预测**（2017 年训练 → 2021 年测试）：UrbanGPT 保持竞争力，说明 LLM 文本知识不引入噪声[^src-urbangpt]。

**消融实验**：贡献排序为时空编码器(-STE) ≈ 回归层(-T2P) > 时空上下文(-STC) > 多数据集(-Multi)。去除回归层（强制 LLM 直接输出文本数值）性能退化最严重[^src-urbangpt]。

## 局限性与批评

（1）高计算成本：7B 参数，单传感器推理耗时 174s，无法扩展到大规模传感器网络[^src-urbangpt]。（2）对 LLM 骨干高度依赖，未做 LLM 规模的消融实验。（3）无图设计在已知路网场景下浪费拓扑信息。（4）Prompt 模板工程复杂，依赖统一的 POI 标注体系。（5）预测范围固定 H=P=12，未探索通过修改指令灵活改变预测步数的可能性。（6）消融实验（-STE 编码器不是 -E，确认）和贡献排序图 5 的定量数值在提取中仅粗略读取，以原文为准。

[^src-urbangpt]: [[source-urbangpt]]
