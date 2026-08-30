---
title: "Attention Distance Preference (三角级数距离偏好)"
type: concept
tags:
  - llm
  - attention
  - rope
  - positional-encoding
  - kv-cache
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Attention Distance Preference（三角级数距离偏好）

Distance preference 指：注意力头对特定 Q-K 距离 $\Delta=p_q-p_k$ 上的 key 给出更高注意力，且这一偏好可以由 Q/K 的几何中心预测。这是 TriAttention 论文的核心理论结果，连接 [[rope]] 的旋转结构与 [[qk-concentration]] 的经验现象[^src-triattention]。

## 从 RoPE logit 到三角级数

对 query $q$（位置 $p_q$）与 key $k$（位置 $p_k$），RoPE 注意力的 pre-softmax logit 可精确写为逐频带复数旋转的实部之和（TriAttention 附录 B）[^src-triattention]：

$$\text{logit}(q,k)=\sum_f \|q_f\|\|k_f\|\cos(\omega_f\Delta+\phi_f),\quad \phi_f=\arg(q_f)-\arg(k_f)$$

其中 $q_f,k_f\in\mathbb{C}$ 是频带 $f$ 的 pre-RoPE 分量，$\omega_f=\theta^{-2f/d}$。当 [[qk-concentration]] 成立时，用中心 $\bar q_f,\bar k_f$ 替换 $q_f,k_f$，logit 近似为只依赖距离的三角级数（式 3/17）[^src-triattention]：

$$\text{logit}(\Delta)\approx\sum_f\left[a_f\cos(\omega_f\Delta)+b_f\sin(\omega_f\Delta)\right]$$

系数 $a_f=\|\bar q_f\|\|\bar k_f\|\cos\bar\phi_f$、$b_f=-\|\bar q_f\|\|\bar k_f\|\sin\bar\phi_f$ 由中心决定。RoPE 频率是几何级数（$\theta^{-2f/d}$）而非经典 Fourier 级数的调和级数，论文称原理类似 Fourier synthesis：学习到的中心决定系数，系数决定 attention-vs-distance 曲线的形状——有的头在近距离成峰（局部注意力），有的在远距离成峰（attention sink 型模式）[^src-triattention]。

## 实验验证

论文从校准数据计算 $\mathbb{E}[q_f],\mathbb{E}[k_f]$ 代入预测式（式 4），在 Qwen3-8B 全部 1152 个头、约 10K token 序列上，将对数间隔距离 $\Delta\in\{1,2,4,8,\dots\}$ 处的预测 logit 与实际 logit 做 Pearson 相关（Reconstruction Correlation $\bar r$，逐 query 平均）[^src-triattention]：

- 第 1 层第 1 个头的 $\bar r=0.72$（论文称选它以避免 cherry-picking）；
- 三个模型族（Qwen3、Qwen2.5、Llama3）的逐头 $\bar r$ 分布右偏、峰值约 0.6–0.9，均值均 >0.5[^src-triattention]。

论文将此作为"Q/K 集中 → 距离偏好可预测"因果链的证据：偏好不需观察实际注意力即可从中心算出[^src-triattention]。

## 含义与边界

含义：重要性估计可以脱离 post-RoPE 注意力观察——[[kv-cache-compression]] 中 post-RoPE 方法受限于可用的近期 query 数量，而距离偏好由稳定的中心编码，可对任意未来位置离线评估。[[triattention]] 的 $S_{\text{trig}}$ 即该式的逐 key 评分版本[^src-triattention]。

边界：重构是近似而非精确——约半数头的 $\bar r$ 在 0.5 附近或以下（Qwen3-8B 上 $r>0.70$ 的头仅 13.0%），论文的回应是对低集中头叠加范数分数并用 $(1-R_f)$ 加权，而非声称级数处处成立；本条为对论文数字的归纳，论文未逐一讨论低相关头的成因[^src-triattention]。

## 相关页面

- [[qk-concentration]] — 级数近似成立的前提
- [[triattention]] — 将级数用作 key 重要性评分
- [[rope]] — logit 三角形式的来源
- [[attention-sink]] — 远距离成峰的一种表现
- [[source-triattention]] — 源摘要

[^src-triattention]: [[source-triattention]]
