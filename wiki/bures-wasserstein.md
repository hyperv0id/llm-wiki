---
title: "Bures-Wasserstein Discrepancy"
type: technique
tags:
  - optimal-transport
  - wasserstein
  - distribution-alignment
  - gaussian
created: 2026-07-17
last_updated: 2026-11-09
source_count: 3
confidence: high
status: active
---

# Bures–Wasserstein 差异

**定义**：两个多元高斯分布之间平方 2-Wasserstein 距离的闭式表达。对 $\mathcal{N}(\mu_1,\Sigma_1)$ 与 $\mathcal{N}(\mu_2,\Sigma_2)$，$W_2^2$ 写成均值项与协方差项之和[^src-distdf]：

$$
\mathrm{BW}(\mu_1,\mu_2,\Sigma_1,\Sigma_2)=\lVert\mu_1-\mu_2\rVert_2^2+\mathcal{B}(\Sigma_1,\Sigma_2),\qquad
\mathcal{B}(\Sigma_1,\Sigma_2)=\operatorname{Tr}\!\left(\Sigma_1+\Sigma_2-2\left(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\right)^{1/2}\right),
$$

其中 $\mathcal{B}$ 称为 Bures 度量，$\operatorname{Tr}(\cdot)$ 为矩阵迹。[[distdf|DistDF]] 以 Lemma 3.5 引用该式，证明指向 Peyré & Cuturi（2019）*Computational Optimal Transport* 的 Remark 2.31[^src-distdf]。

## 两项各管什么

**均值项** $\lVert\mu_1-\mu_2\rVert_2^2$ 只比较两个分布的中心位置，与维度上的尺度无关。

**协方差项** $\mathcal{B}(\Sigma_1,\Sigma_2)$ 比较二阶形状。中间的 $\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}$ 先把 $\Sigma_2$ 表示在 $\Sigma_1$ 的白化坐标系里，再取矩阵平方根求迹；$\mathcal{B}=0$ 当且仅当 $\Sigma_1=\Sigma_2$。它与 Frobenius 范数 $\lVert\Sigma_1-\Sigma_2\rVert_F$ 不同：后者只是坐标下的逐元素平方差，不对应任何传输代价；$\mathcal{B}$ 由高斯间最优传输映射的代价推出，在正定锥上诱导出与 $W_2$ 一致的几何。

两项合起来是 $\mathcal{N}(\mu_1,\Sigma_1)$ 与 $\mathcal{N}(\mu_2,\Sigma_2)$ 之间 $W_2^2$ 的精确值，因此 BW 的平方根是高斯分布空间上的度量；这也解释了为什么它可以直接当训练损失——最优值 $0$ 只在两个高斯完全相同时取到。

## 在 DistDF 中的用法

DistDF 把历史段与未来段沿时间轴拼成一个联合向量，用 batch 维当作联合分布的样本维：$Z=[X,Y]\in\mathbb{R}^{B\times(H+T)}$、$\hat{Z}=[X,\hat{Y}]$，在 batch 维上估计 $\mu_Z,\Sigma_Z$ 与 $\mu_{\hat{Z}},\Sigma_{\hat{Z}}$，再算 $\mathcal{L}_{\mathrm{Dist}}=\mathrm{BW}(\mu_Z,\mu_{\hat{Z}},\Sigma_Z,\Sigma_{\hat{Z}})$，与 MSE 混合[^src-distdf]：

$$
\mathcal{L}_{\mathrm{DistDF}}=\gamma\,\mathcal{L}_{\mathrm{Dist}}+(1-\gamma)\,\lVert Y-\hat{Y}\rVert_2^2 .
$$

由于协方差在拼接向量上计算，$\Sigma_Z$ 的尺寸为 $(H+T)\times(H+T)$，其非对角块同时包含未来步之间的相关（标签自相关）与历史—未来之间的相关。MSE 只逐点比较未来段，$\mathcal{B}$ 则把整条 $(H+T)$ 序列的二阶结构一起对齐，这是它作为正则项能改变训练结果的原因[^src-distdf]。

**消融证据**：Table 2 显示单独对齐均值项（$\mathrm{DistDF}^\dagger$）或单独对齐协方差项（$\mathrm{DistDF}^\ddagger$）均优于纯 MSE，两项合用时在 ETTm1、ETTh1、ECL、Weather 四个数据集上均取得最低误差（逐数据集数值见 [[distdf]] Table 2）[^src-distdf]。

**训练开销**：额外计算集中在 $\Sigma^{1/2}$ 与迹运算，需要对 $(H+T)\times(H+T)$ 矩阵做特征分解。论文在 $B=128$、$D=21$ 下测得单步前向 $0.74\text{--}0.80$ ms、反向 $0.58\text{--}0.64$ ms，$T$ 增至 $1024$ 时仍低于 $1$ ms（Appendix D.5、Fig. 7）；推理阶段不执行 BW，开销为零[^src-distdf]。

**收敛**：论文报告 BW 目标在训练中单调下降、若干 epoch 后进入平台，并观察到验证集 MAE/MSE 与 BW 曲线同向变化，据此认为压低 BW 会带动预测指标（Appendix D.7、Fig. 8）[^src-distdf]。

## 在 Gaussian Schrödinger Bridge 中的角色

当 Schrödinger bridge 的两个端点分布都是高斯时，最优耦合的交叉协方差为[^src-schrodinger-bridges-generative-modeling]

$$
C_\sigma=\tfrac12\left(\Sigma_0^{1/2}D_\sigma\Sigma_0^{1/2}-\sigma^2 I\right),\qquad
D_\sigma=\left(4\Sigma_0^{1/2}\Sigma_T\Sigma_0^{1/2}+\sigma^4 I\right)^{1/2},
$$

$\sigma\to0$ 时 $C_\sigma$ 收敛到经典 Bures–Wasserstein 最优传输耦合。动态一侧，协方差轨迹 $\Sigma_t$ 的最优性问题写作

$$
\inf_{\Sigma_t}\int_0^T\left[\tfrac12\lVert\dot{\Sigma}_t\rVert_{\Sigma_t}^2+U_\sigma(\Sigma_t)\right]dt,
$$

其中 $\lVert\dot{\Sigma}_t\rVert_{\Sigma_t}^2$ 就是正定矩阵上的 Bures–Wasserstein 动能，熵正则势 $U_\sigma(\Sigma_t)=\frac{\sigma_t^4}{8}\operatorname{Tr}(\Sigma_t^{-1})$ 惩罚小特征值；$\sigma_t\to0$ 时 $U_\sigma$ 消失，轨迹退化为 Bures–Wasserstein 测地线[^src-schrodinger-bridges-generative-modeling]。

同一个度量在两处承担不同角色：DistDF 用它做两个高斯之间的差异值，Gaussian SB 用它定义正定矩阵空间上的路径长度。

## 与 Sliced Wasserstein 的分工

[[sliced-wasserstein-distance|Sliced Wasserstein Distance (SWD)]] 把高维分布投到 $K$ 个随机一维方向，排序后逐点比较，靠 Cramér–Wold 定理保证所有方向一致即分布一致，代价 $O(KN\log N)$，不假设高斯性，能约束任意形状的完整边缘分布[^src-visreg]。

BW 与它构成一对取舍：BW 有闭式解、只对齐一阶与二阶矩，非高斯或重尾数据的高阶结构不在约束范围内[^src-distdf]；SWD 不假设分布族、能匹配完整形状，但没有闭式表达、对投影方向数 $K$ 敏感，且最优精度要求 $K$ 超过嵌入维度一个因子[^src-visreg]。论文选 BW 的依据是高斯假设下可得可微的闭式解，且 $T$ 增至 $1024$ 时额外开销仍低于 $1$ ms；把高阶统计量纳入同时保持可计算，被列为后续方向[^src-distdf]。

## 相关页面

- [[distdf]] — 使用 BW 的完整框架
- [[joint-distribution-wasserstein-alignment]] — BW 支撑的上界与对齐定理
- [[gaussian-schrodinger-bridge]] — BW 几何下的闭式 SB
- [[sliced-wasserstein-distance]] — 不假设高斯性的替代度量
- [[optimal-transport]] — Wasserstein 距离的一般定义与求解

---

[^src-distdf]: [[source-distdf]]
[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
[^src-visreg]: [[source-visreg]]
