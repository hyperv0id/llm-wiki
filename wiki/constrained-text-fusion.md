---
title: "Constrained Text Fusion (CFA)"
type: entity
tags:
  - multimodal-time-series
  - text-fusion
  - constrained-fusion
  - plug-in-adapter
  - time-mmd
  - kdd-2026
created: 2026-07-28
last_updated: 2026-08-01
source_count: 3
confidence: high
status: active
---

# Constrained Text Fusion / CFA

**Constrained Text Fusion** 指 Lee et al.（LG AI Research, KDD ’26 MILETS / arXiv:2603.22372）对多模态时序预测中 **文本辅助模态** 的融合原则：**辅助信息必须受控注入，不能默认 add/concat**。作者提出并验证 **Controlled Fusion Adapter (CFA)**——低秩瓶颈残差的 plug-in 适配器——并与 Gating / FiLM / Orthogonal 等 constrained 族对照[^src-constrained-text-fusion]。

## 核心主张

1. 时序是**主模态**，文本是**辅助上下文**；文本可含无关或冲突信号。  
2. **Naive fusion**（first/middle/last × additive/concat）在广泛设定上常 **低于 unimodal TS**。  
3. **Constrained fusion** 通过门控、特征调制、正交分量或低秩子空间，**过滤后再融合**，系统优于 naive。  
4. **CFA** 同时满足：任意 TS 骨干 **plug-in** + **constrained**（Table 1 中唯一双满足项）[^src-constrained-text-fusion]。

## 方法族

| 方法 | 融合形式 | 约束含义 |
|------|----------|----------|
| Naive Add / Concat | 直接 \(F(Z_{\mathrm{TS}}, Z_{\mathrm{Text}})\) | 无过滤 |
| Gating | \(z_{\mathrm{TS}}+g\odot z_{\mathrm{Text}}\) | 逐步相关门 |
| FiLM | \(\gamma\odot z_{\mathrm{TS}}+\beta\)（由文本产） | 保结构、调尺度偏置 |
| Orthogonal | \(z_{\mathrm{TS}}+z_{\mathrm{Text}}^\perp\) | 不覆盖 TS 子空间 |
| **CFA** | \(z_{\mathrm{TS}}+W_{\mathrm{up}}\phi(W_{\mathrm{down}}z_{\mathrm{Text}})\) | 低秩残差 + 近零 init |

CFA：\(W_{\mathrm{down}}:D\to D/r\)，ReLU∘LN，\(W_{\mathrm{up}}\) 回 \(D\)；默认 \(r=8\)；注入各 encoder 层。目标不是“新 adapter 品牌”，而是**把文本信号容量压到低维子空间**[^src-constrained-text-fusion]。

## 数据与规模

评测锚定 **[[time-mmd|Time-MMD]]** 9 域 × 4 视界 × **14 TS** × **4 文本编码器（冻结）** × 10 融合策略，全文 **>20K** 实验量级；图级 “~2K settings” 聚合 naive vs constrained[^src-constrained-text-fusion]。

## 关键数字（可操作）

- CFA 相对 unimodal：**9/9 域更好**，**7/9 rank-1**；**13/14 骨干**更好（标准 Transformer 除外）。  
- Table 4 示意 win rate：CFA 常 **88.9%+**；naive 可 **0–44%** 且出现 Div.。  
- Toy 文本：bottleneck 对 **irrelevant** 增益最大（~+20% MSE）；matching 注入更强于 contradicting。  
- 开销：参数 **+0.61%**、FLOPs **+0.04%**[^src-constrained-text-fusion]。

## 谱系位置

| 路线 | 关系 |
|------|------|
| [[time-mmd|Time-MMD]] / MM-TSFlib | 数据与 naive 投影融合床；本文在同九域上系统证伪“无控多模态默认赢” |
| [[tats|TaTS]] | Plug-in 但 **first-add naive**；本文基线 |
| [[unica|UniCA]] / [[cora-tsfm|CoRA]] | TSFM 协变量适配；CFA 是**任意 TS 骨干**的融合层约束，非 foundation 适配框架 |
| [[vot|VoT]] | Late / multi-level 融合 + LLM 推理；仍属任务侧融合，非低秩 residual 族 |
| [[timi|TiMi]] / [[non-fusion-guidance|Non-Fusion Guidance]] | **放弃特征融合**，LLM 知识 → MoE 路由；与 CFA 同认“乱融有害”，解法正交（不融合 vs 约束融合） |
| [[tess|TESS]] | 半合成证据（FNSPID + GPT-5.2 生成文本、token 级标注）定位两个机制级瓶颈：冗余 token 分散注意力（R<sub>t</sub><0）、删冗余后语义仍难解码为数值；解法比 CFA 更彻底——不做特征层受控融合，把文本压成 4 个离散时序原语 + 置信门控后注入 PatchTST（语义瓶颈 vs 低秩特征残差）；与 TiMi 的完全不融合构成「约束注入→语义瓶颈→不融合」谱系[^src-tess] |
| [[time-vlm|Time-VLM]] | 架构特定 VLM 桥接 + 门控；文内归 architecture-specific；门控与 constrained 思想可对照 |
| [[ts-vl-alignment|TS–VL Alignment]] | 表示空间诊断：TS–TXT 难对齐 → 与“文本不可无控注入”互补 |
| [[cross-modal-misalignment|Cross-modal misalignment]] | 预训练 MMCL：省略/扰动语义进不了对比表示；CFA 是**任务侧**再过滤，理论层解释“为何文本容量应压低”[^src-cross-modal-misalignment] |

## 局限

仅文本；理论浅；未系统扩 vision/tabular；部分 constrained 仍可能偶发低于 unimodal，故 CFA 强调更强过滤[^src-constrained-text-fusion]。

## 相关页面

- [[source-constrained-text-fusion]] — 源摘要  
- [[time-mmd]] · [[multimodal-time-series-forecasting]] · [[non-fusion-guidance]] · [[timi]] · [[vot]] · [[time-vlm]] · [[ts-vl-alignment]] · [[tats]] · [[cross-modal-misalignment]] · [[source-cross-modal-misalignment]] · [[tess]]

[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
[^src-cross-modal-misalignment]: [[source-cross-modal-misalignment]]
[^src-tess]: [[source-tess]]
