---
title: "残差 Interaction 分解（Residual Interaction Decomposition）"
type: technique
tags:
  - residual-networks
  - mobius-inversion
  - interaction-decomposition
  - interpretability
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# 残差 Interaction 分解

把训练好的 ResNet 读成**残差支路掩码上的集合函数**，用布尔格 Möbius 反演拆出 shortcut 基线、一阶修正与高阶 interaction，每项含义唯一、求和严格等于原输出[^src-resnets-route-without-routers]。这是 [[implicit-functional-routing|隐性功能路由]] 与 [[implicit-residual-experts|隐式残差专家]] 的测量基础。

## 定义与记号

$L$ 个可掩码残差块的网络（ResNet-18 的 `[2,2,2,2]` BasicBlock 组给 $L=8$，ResNet-34 的 `[3,4,6,3]` 给 $L=16$）。掩码 $m\in\{0,1\}^L$ 下第 $l$ 块计算

$$x_{l+1}(m)=\phi_l\!\left(s_l(x_l(m))+m_l F_l(x_l(m))\right)$$

$s_l$ 为 identity 或投影 shortcut，$\phi_l$ 为加法后的激活，$m_l=0$ 只关残差支路[^src-resnets-route-without-routers]。固定输入 $x$ 给出残差掩码响应 $h_x:\{0,1\}^L\to\mathbb{R}^{C}$；标量读数 $v_x=\rho_x\circ h_x$，$\rho_x$ 在所有掩码间固定（如取全掩码预测类那一维）[^src-resnets-route-without-routers]。

**维度符号**：$B$ 图像数；$L$ 残差支路数（8 / 16）；$C=1000$ ImageNet logits；掩码轴宽 $2^L$（256 / 65,536）；非空 interaction 字典 $M=2^L-1$（255 / 65,535）。

## 数据流与维度

### Stage 1 · 枚举掩码前向

对每张图把 $2^L$ 个掩码各跑一遍完整前向，shortcut 始终在：

$$\begin{aligned}
h_x(m) &= \text{ResNet}\big(x;\ m\big) && \text{// } [2^L, C]\ \text{——掩码轴是干预，不是 batch；每行是一套支路开关下的 logits}\\
H &= \operatorname{stack}_{x\in\mathcal{B}}\big(h_x\big) && \text{// } [B, 2^L, C]\ \text{——后续所有统计都在第 2 轴（掩码）上做}
\end{aligned}$$

成本是普通推理的 $2^L$ 倍，故 ResNet-34 只取 $N=1{,}000$、ResNet-18 取 $N=10{,}000$[^src-resnets-route-without-routers]。

### Stage 2 · Möbius 反演

对每个子集 $S\subseteq[L]$ 取 $S$ 上的一阶差分（inclusion–exclusion），消去真子集已解释的部分：

$$\begin{aligned}
\Delta_S h_x &= \sum_{T\subseteq S}(-1)^{|S|-|T|}h_x(\mathbf{1}_T) && \text{// } [B, 2^L, C]\to[B, 2^L, C]\ \text{——沿 mask 轴做子集差分，$C$ 维逐坐标独立}\\
\Delta_\emptyset h_x &= h_x(\mathbf{0}) && \text{// } [B, C]\ \text{——全支路关闭、只走 shortcut 的基线}\\
\sum_{S\subseteq[L]}\Delta_S h_x &= h_x(\mathbf{1}_{[L]}) && \text{// } [B, C]\ \text{——含空集的 256 / 65,536 项之和严格还原全掩码 logits}
\end{aligned}$$

该关系即 Lemma 2.2（Möbius 逆关系），证明见附录 C[^src-resnets-route-without-routers]。一阶 $\Delta_{\{l\}}v_x=v_x(\mathbf{1}_{\{l\}})-v_x(\mathbf{0})$ 是单支路相对 shortcut 基线的效应；二阶 $\Delta_{\{i,j\}}v_x=v_x(\mathbf{1}_{\{i,j\}})-v_x(\mathbf{1}_{\{i\}})-v_x(\mathbf{1}_{\{j\}})+v_x(\mathbf{0})$ 是两路联用的非可加量，无法由各自单独效应相加解释[^src-resnets-route-without-routers]。

### Stage 3 · 读出与幅值

$$\begin{aligned}
v_x(m) &= h_{x,\hat y(x)}(m),\quad \hat y(x)=\arg\max_c h_{x,c}(\mathbf{1}_{[L]}) && \text{// } [B, 2^L, \bcancel{C}]\to[B, 2^L]\ \text{——沿 $C$ 维取全掩码预测类，下标在所有掩码间冻结}\\
a^h_S(x) &= \lVert \Delta_S h_x\rVert_2 && \text{// } [B, 2^L, \bcancel{C}]\xrightarrow{\text{沿 } C \text{ 维取 } \ell_2} [B, 2^L]\ \text{——向量幅值；标量读数另用 } a^v_S=|\Delta_S v_x|
\end{aligned}$$

标量读数是 decision-conditioned 的（坐标取决于该图全掩码预测），向量读数不含标签信息；论文两者都报，主检验用向量读数[^src-resnets-route-without-routers]。

### Stage 4 · 汇合到顺序统计与 Top-$K$

$$\begin{aligned}
E_k(x) &= \sum_{S:\,|S|=k} a_S(x)^2 && \text{// } [B, 2^L]\xrightarrow{\text{按 } |S| \text{ 分箱求和}} [B, L]\ \text{——第 $k$ 阶的平方系数质量}\\
\mathcal{A}_K(x) &= \operatorname{TopK}_{S\neq\emptyset}\, a_S(x) && \text{// } [B, 2^L]\xrightarrow{\text{逐样本取 Top-}K} [B, K]\ \text{——每张图自己的主导 interaction 下标}
\end{aligned}$$

$E_k$ 归一化后即 [[interaction-order-spectrum|interaction 阶谱]]；$\mathcal{A}_K$ 是 [[implicit-residual-experts|隐式残差专家]] 的支撑集[^src-resnets-route-without-routers]。

## 朴素聚合为什么不成立

把掩码子网络的输出当作「路径贡献」直接相加是 path-ensemble 视角的自然延伸，但 Table 1 否掉了它[^src-resnets-route-without-routers]：

| 模型 | 聚合方式 | Norm ratio | 相对误差 | 最优缩放后误差 | 余弦 | Top-1 一致率 |
|---|---|---|---|---|---|---|
| ResNet-18 | 直接求和 $A^{\mathrm{raw}}$ | 79.400 | 79.200 | 0.961 | 0.222 | 0.006 |
| ResNet-18 | 平均 $A^{\mathrm{mean}}$ | 0.310 | 0.977 | 0.961 | 0.222 | 0.006 |
| ResNet-18 | 基线居中 $A^{\mathrm{ctr}}$ | 77.400 | 77.200 | 0.959 | 0.229 | 0.006 |
| ResNet-18 | Möbius | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| ResNet-34 | 直接求和 | 18,852.327 | 18,851.887 | 0.889 | 0.440 | 0.025 |
| ResNet-34 | 平均 | 0.288 | 0.910 | 0.889 | 0.440 | 0.025 |
| ResNet-34 | 基线居中 | 17,750.219 | 17,749.760 | 0.879 | 0.459 | 0.024 |
| ResNet-34 | Möbius | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |

（Table 1；ResNet-18 $N=10{,}000$、ResNet-34 $N=1{,}000$；Möbius 行取整前误差 ResNet-18 $8.51\times10^{-7}$、ResNet-34 $1.76\times10^{-16}$。）最优缩放用最小二乘标量 $\alpha_x^\star=\langle h_x^{\mathrm{full}},A_x\rangle/\lVert A_x\rVert_2^2$ 单独拟合量级，误差仍停在 0.88–0.96，说明失败来自方向而非尺度[^src-resnets-route-without-routers]。路径数因此不等于可加项数，唯一可加单元是 $\Delta_S h_x$。

## 使用边界

- 系数一般不正交，$E_k/Z$ 只能读成系数质量的分布，不能读成「第 $k$ 阶解释了多少 logit 范数」[^src-resnets-route-without-routers]。
- 掩码是**解析干预**，普通推理仍 dense；Top-$K$ 支撑是事后从完整分解里挑的，不代表可执行稀疏路径[^src-resnets-route-without-routers]。
- 穷举成本 $O(2^L)$，$L=16$ 已降到 $N=1{,}000$；更深网络需要高阶 interaction 的高效估计器（结论段自述）[^src-resnets-route-without-routers]。

## 相关页面

- [[interaction-order-spectrum]] — 从 $\Delta_S$ 得到的阶谱、有效阶 $\kappa$ 与残差缩放命题
- [[implicit-residual-experts]] — Top-$K$ 支撑的集中性与长尾
- [[implicit-functional-routing]] — 支撑集的类结构与输入依赖
- [[mixture-of-experts]] — 显式 router 的对照对象

[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
