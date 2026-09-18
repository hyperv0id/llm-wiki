---
title: "InterpGN: Shedding Light on Time Series Classification Using Interpretability Gated Networks (Wen et al., ICLR 2025)"
type: source-summary
tags:
  - time-series-classification
  - mixture-of-experts
  - interpretability
  - shapelets
  - gating
  - iclr-2025
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# InterpGN: Shedding Light on Time Series Classification Using Interpretability Gated Networks

**Authors**: Yunshi Wen¹, Tengfei Ma²（共同一作）, Ronny Luss, Debarun Bhattacharjya, Achille Fokoue, Agung Julius。¹Rensselaer Polytechnic Institute，²Stony Brook University，其余 IBM Research。[^src-interpgn]
**Venue**: ICLR 2025（Published as a conference paper）。[^src-interpgn]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/FZY9GLUQ/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

可解释模型（shapelet 类）可解释但表达力弱，常输给深度模型；直接用阈值距离构造逻辑谓词会降低 shapelet 质量且跨数据集语义不稳。SBM 里 RBF 谓词的 max 算子提升可解释性，却丢失 shapelet 出现次数、时间戳信息，也无法捕获频域等其它域的判别特征。[^src-interpgn]

## 方法

InterpGN = 可解释 expert（SBM）+ 深度 expert（FCN），MoE 门控决定二者的混合比例，插在 shapelet transform 之后、分类输出之前：[^src-interpgn]

1. **逻辑谓词（§4.1，Eq 1-2）**：对每个通道 m、每长度 l 学习 K 个 shapelet，L={⌈δT⌉ | δ=0.05,0.1,0.2,0.3,0.5,0.8}；距离取滑动窗口最小欧氏距离（Eq 1），再用 RBF（Eq 2）把距离转为「shapelet 存在性」概率谓词 p_i∈R^{MK|L|}，论文自称相比阈值型谓词更可解释。[^src-interpgn]
2. **SBM 可解释 expert（§4.2，Eq 3-4）**：谓词直接接线性分类器 r_i,c = Σ w_{c,m,k,l}·p_{i,k,l}，权重即 shapelet 对类别的贡献，支持局部（样本含哪些 shapelet）与全局（w>0/=0/<0 对应类内应含/无关/不应含）两种解释。训练损失 L_int = L_ce + λ_div·L_div + λ_reg·L_reg：L_div（Eq 4）抑制 shapelet 冗余，L1 正则促使权重稀疏。max 算子梯度用 straight-through 的 softmax 梯度替代。[^src-interpgn]
3. **MoE 门控（§4.4，Eq 5-6）**：路由输入信号是 SBM 自身输出的置信度——η(x_i)=(Σ_c(r̂_i,c)²−1/C)/(1−1/C) 是修改版 Gini Index，SBM 输出越接近 one-hot 越自信。混合输出 h_i = r_i·η(x_i) + z_i·(1−η(x_i))：越不自信越倚重 DNN。**只有 2 个 expert（SBM 与 DNN），不做 top-k 稀疏激活，无负载均衡损失**；推理时若 η(x_i)>阈值 η 则完全弃用 DNN。默认超参（Table 3，附录 A.1）：K=10、ε=1、λ_div=λ_reg=0.1、β 为常数调度、**推理门控值 η=1**（即主实验不弃用 DNN）、Adam lr 0.005、weight decay 0、batch size 32。[^src-interpgn]
4. **训练（§4.5）**：L_hybrid = β·L_int + L̄_ce，L̄_ce 作用于混合输出 h_i；β 可为常数或余弦衰减，防止模型塌缩成纯 DNN。整体端到端可微。[^src-interpgn]

## 实验

- **Table 1（UEA 30 数据集）**：InterpGN avg acc 0.760 / avg rank 3.500 / 8 个 Top-1；SBM 单独使用 0.726（rank 5.733）；对照 FCN 0.746、可解释基线 RLPAM 0.740、ShapeConv 0.743、ShapeNet 0.697。[^src-interpgn]
- **Table 2（MIMIC-III 院内死亡早预测）**：InterpGN accuracy 0.703、precision 0.784、ROC-AUC 0.703，优于 STRF（0.653/0.666/0.653）、FCN（0.639/0.734/0.698）与 SBM（0.675/0.659/0.658）。[^src-interpgn]
- **§5.3 门控阈值 η（Figure 21）**：η=0.5 与 η=1 的准确率差 0.0035；SBM utility rate 越大准确率越高。ε≥5 显著降低准确率，λ_reg 在 0-1 范围内影响不大。[^src-interpgn]

## 局限

论文 §6 自述两点：其一，某类的重要 shapelet 未必与该类子序列相似，解释时需借负规则（「不应包含某 shapelet」）；其二，框架目前只含一个可解释 expert 与一个 DNN，未来计划扩展为多个 expert 的完整 MoE 组合。[^src-interpgn]

## 与其它 MoE 方法的差异

论文自述（§4.4）：IME（Ismail et al. 2023）把简单模型（线性回归、软决策树）当作可解释 gating network 去调度专家；InterpGN 反其道，让可解释 expert（SBM）本身充当 gating network，从而减少参数并提升可解释性。与常规 top-k 稀疏 MoE 不同，本方法按样本连续加权两个 expert，并在推理时按置信度硬性跳过 DNN。[^src-interpgn]

[^src-interpgn]: [[source-interpgn]]
