---
title: "KITE 三件套：历史条件流形 × 知识引导条件化 × 无分类器引导"
type: analysis
tags:
  - flow-matching
  - manifold
  - classifier-free-guidance
  - knowledge-guided
  - exogenous
  - probabilistic-forecasting
  - analysis
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# KITE 三件套：历史条件流形 × 知识引导条件化 × 无分类器引导

本文把 [[kite|KITE]] 的三块积木——[[history-conditional-manifold|HCM]]、[[knowledge-guided-conditioning|KGC]]、[[classifier-free-guidance|CFG]]——串成一条因果链：先改**起点**，再改**路上听谁的**，最后用**有/无条件差**拧旋钮。[^src-kite]

## 一张图看清分工

```
历史内生 X_endo
      |
      v
 [HCM]  学 (μ_hist, σ_hist, δ_hist) → 源 Y0 ≈ 目标邻域
      |
      |   历史/未来外生 X_exo, Y_exo + 统计先验 S
      |              |
      |              v
      |         [KGC] 双线性注意：W1 + s_ij W2
      |              |
      v              v
  Y_s = s Y_endo + (1-s) Y0   →  速度网 v_θ(Y_s, s, c̃)
                                      |
                         训练: c̃ 以 p_con 变为 ∅   [CFG 联合训练]
                         推理: v̂ = (1+γ)v(c) − γ v(∅)
                                      |
                                      v
                         预测分布 p_θ(Y_endo | ·)
```

| 瓶颈 | 症状 | 模块 | 出手点 |
|------|------|------|--------|
| 拓扑落差 | 从 \(N(0,I)\) 走到局域协变量条件分布，路径长、难学 | HCM | **源** \(Y_0\) |
| 伪相关放大 | 迭代生成把虚假内外生相关越滚越大 | KGC | **条件通路**注意力 |
| 条件强度不可控 | 只能全有或全无外生，难适配场景 | CFG | **速度场外推** |

[^src-kite]

## 1. 流形先把起点搬近：HCM

Flow Matching 学的是从源到目标的直线路径速度 \(Y_{\text{endo}}-Y_0\)。若 \(Y_0\) 是无信息高斯，回归目标大、路径管粗，向量场局部 Jacobian 预算高——误差更容易被放大。[^src-kite]

HCM 用历史合成

\[
Y_0=\mu_{\text{hist}}+\sigma_{\text{hist}}\delta_{\text{hist}},
\]

质心贴近历史依赖中心，尺度由覆盖约束校准，扰动落在可学低秩流形与各向同性噪声的混合上。结果是：**匹配目标尺度下降 + 路径更规则**（Prop. 1–2）。[^src-kite]

这与 [[gaussian-process-prior-flow-matching|TSFlow 的 GP 源]]同属「信息源」路线，但 HCM 是可学历史条件流形，而不是固定核规则。[^src-kite]

**链上角色**：HCM **不**读外生；它只保证生成从「靠谱邻域」出发。外生的对错，交给下一环。

## 2. 条件化用先验拧注意力：KGC

生成每一步的状态 \(Y_s\) 要吃外生。若注意力纯数据驱动，训练集会的假相关会写进 \(v_\theta\)；多步积分后假相关变成系统性偏置。[^src-kite]

KGC 不把先验当加性 bias 糊在输入上，而是改查询投影：

\[
q_i(W_1+s_{ij}W_2)k_j^\top,
\]

并在实现里用 \(A_b+\tilde S\odot A_g+\log(\tilde S+\delta)\)——先验为零则门控掐断。历史/未来外生顺序注入，未来缺失可跳过。[^src-kite]

先验语义还可分阶段：历史侧 Granger、未来侧 Pearson，与「过去是否驱动 / 同期是否共变」对齐。[^src-kite]

**链上角色**：KGC 定义**有条件模式** \(v_\theta(\cdot,c)\) 里「\(c\) 如何进入网络」。它回答的是 *how to condition*，不是 *whether / how hard to condition*。

## 3. CFG 在两种模式之间拧旋钮

仅有 KGC 仍只有一条条件通路。[[classifier-free-guidance|CFG]] 在训练时以概率 \(p_{\text{con}}\) 把 \(c=\{X_{\text{exo}},Y_{\text{exo}}\}\) 换成 \(\varnothing\)，让同一网络同时逼近：[^src-kite]

- 协变量条件速度场 \(v_\theta(Y_s,s,c)\)
- 协变量无关速度场 \(v_\theta(Y_s,s,\varnothing)\)

推理外推：

\[
\hat v_s=(1+\gamma)v_\theta(Y_s,s,c)-\gamma\,v_\theta(Y_s,s,\varnothing).
\]

\(\gamma=0\) 退回条件生成；\(\gamma>0\) 把轨迹从「无外生平均行为」推开，更贴当前协变量驱动。KITE 经验上 \(\gamma\in\{1.2,1.4\}\) 最稳，说明该任务对外生依赖强，但过强引导会伤分布校准。[^src-kite]

**与图像 CFG 的同构**：形式同 Ho & Salimans；语义从「类别/文本遵循」换成「外生驱动强度」。条件丢弃的对象是协变量集合，不是类标。[^src-kite]

**链上角色**：CFG 假设 KGC 已经把「有 \(c\)」和「无 \(c\)」两种前向都学好；它不修注意力结构，只在输出速度场做线性外推。

## 4. 为什么必须三件一起

消融（确定性 NP / Sdwpfh1，概率 ETTm2 / Weather）给出协同结构：[^src-kite]

| HCM | KGC | CFG | 现象（相对裸 FM） |
|-----|-----|-----|-------------------|
| ✓ | | | 源变好，传输负担下降 |
| | ✓ | | 条件更干净，伪相关减轻 |
| ✓ | ✓ | | 进一步增益——好源 + 好条件 |
| | ✓ | ✓ | 无好源时 CFG 仍有用，但上限较低 |
| ✓ | ✓ | ✓ | 全任务最优 |

解读：

1. **HCM 与 KGC 正交**：一个缩路径长度/曲率，一个正条件方向；合用不是重复劳动。
2. **CFG 放大的是 KGC 学到的条件差**：若条件通路本身被伪相关污染，加大 \(\gamma\) 等于放大脏信号。所以 KGC 是 CFG 的前提，不是可选项。
3. **HCM 降低 CFG 的操作难度**：起点已在目标邻域时，有/无条件速度场的差更局部、更可外推；从远高斯出发时，两种模式的轨迹分叉更大、外推更不稳。

一句话：**流形管几何，知识管结构，CFG 管剂量。**[^src-kite]

## 5. 端到端数据流（训练 / 推理）

**训练一步**（概念序）：[^src-kite]

1. 由 \(X_{\text{endo}}\) 经 HCM 采样 \(Y_0\)；随机 \(s\)，构 \(Y_s\)；
2. 抽先验 \(S\)，KGC 将 \(X_{\text{exo}},Y_{\text{exo}}\) 注入网络（或按 \(p_{\text{con}}\) 置空 \(c\)）；
3. 回归 \(v_\theta(Y_s,s,\tilde c)\approx Y_{\text{endo}}-Y_0\)；叠加覆盖损失 \(L_{CC}\)。

**推理**：从 HCM 的 \(Y_0\) 出发，每步用 KGC 条件化的 \(v(c)\) 与 \(v(\varnothing)\) 做 CFG 混合，积分到 \(s=1\)。[^src-kite]

## 6. 对外生预测谱系的位置

- **确定性外生融合**（[[source-timexer|TimeXer]]、[[source-exotst|ExoTST]]、[[source-exost|ExoST]]）：解决「怎么把外生拼进点预测」；无分布、无源几何、无引导剂量。
- **概率无显式未来外生**（[[timegrad|TimeGrad]]、[[csdi|CSDI]]、[[tsflow|TSFlow]]）：有分布；TSFlow 有信息源，但未来协变量与统计知识注意力不是一等公民。
- **KITE**：信息源（HCM）+ 知识门控条件（KGC）+ 剂量控制（CFG），专打「未来外生已知的概率预测」。[^src-kite]

边界：未来外生噪声/缺失/不确定时，当前 KGC/CFG 假设 \(Y_{\text{exo}}\) 可信；这是作者自陈的下一战场，也是与鲁棒外生建模（ExoTST 缺失实验、ExoST select-then-balance）的自然接口。[^src-kite]

## 7. 可带走的设计清单

若你要在别的生成式预报里复用这套串法：[^src-kite]

1. **先换源，再加条件**：否则条件网络在学「从垃圾起点硬扛」。
2. **先验进投影，不进拼接**：让 \(s_{ij}\) 改子空间，而不是当第 \(d+1\) 个特征。
3. **CFG 只外推已学好的两种模式**：条件通路要干净；脏条件 + 大 \(\gamma\) = 自信地错。
4. **分阶段先验**：历史偏因果、未来偏相关——比全局单一统计量更贴时间语义。
5. **覆盖约束保探索**：可学 \(\sigma\) 无 \(L_{CC}\) 容易塌，源又退回「过度自信的点」。

## 相关页面

- [[kite]] / [[source-kite]]
- [[history-conditional-manifold]]
- [[knowledge-guided-conditioning]]
- [[classifier-free-guidance]]
- [[flow-matching]] / [[flow-matching-forecasting]]
- [[gaussian-process-prior-flow-matching]] / [[tsflow]]
- [[source-timexer]] / [[source-exotst]] / [[source-exost]]
- [[multimodal-exogenous-guided-long-term-st-forecasting]]

[^src-kite]: [[source-kite]]
