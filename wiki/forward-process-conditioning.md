---
title: "Forward Process Conditioning（前向过程条件化）"
type: technique
tags:
  - diffusion-models
  - conditional-diffusion
  - spatio-temporal-imputation
created: 2026-08-29
last_updated: 2026-08-29
source_count: 3
confidence: medium
status: active
---

# Forward Process Conditioning（前向过程条件化）

**前向过程条件化**指把观测值 $z_0^c$ 写入扩散模型的**前向转移**，而不只在反向去噪网络的输入端注入观测条件。这一用法出自 [[rdpi|RDPI]]：论文把"将观测值纳入前向过程"（incorporated observed values into the forward process）列为自述贡献之一，并称据此推导了含前向条件的新 ELBO[^src-rdpi]。

## 动机：条件注入位置的缺口

[[csdi|CSDI]] 一系条件扩散插补把观测值与加噪目标拼接后送入去噪网络，前向过程本身保持无条件[^src-csdi]；[[pristi|PriSTI]] 的条件使用方式（先验引导注意力）同样只作用于反向过程，其正向扩散仅作用于插补目标[^src-pristi]。RDPI 论文认为这种"仅反向条件化"使模型在前向与插补过程中忽略观测与缺失数据的依赖关系，影响实际插补表现（论文表述，Introduction 与 Related Work）[^src-rdpi]。

## RDPI 的参数化（论文口径）

设扩散目标是初始估计与真值之间的残差 $z_0^m = f_\theta(x_0^c) - x_0^m$。论文给出的前向转移以观测值为条件（Eq 3）：

$$q(z_t^m \mid z_{t-1}^m, z_0^c) := \mathcal{N}\big(z_t^m;\ \sqrt{\alpha_t}\, z_{t-1}^m + \sqrt{\alpha_t}\, z_0^c,\ \beta_t I\big),\quad \alpha_t := 1-\beta_t$$

（原文均值项写作 $\sqrt{1-\beta_t}\, z_{t-1}^m + \sqrt{1-\beta_t}\, z_0^c$，本页按 $\alpha_t := 1-\beta_t$ 等价改写；观测项 $z_0^c$ 与残差项同系数，是 RDPI 前向过程区别于标准 DDPM 之处。）

并给出边际形式（Eq 4）与重参数化形式（Eq 5，均按论文原文抄录）：

$$q(z_t^m \mid z_0^m, z_0^c) = \mathcal{N}\big(z_t^m;\ \sqrt{\bar\alpha_t}\,(z_0^m + z_0^c),\ (1-\bar\alpha_t)I\big)$$

$$z_t^m = \sqrt{\bar\alpha_t}\, z_0^m + \sqrt{\bar\alpha_t}\, z_0^c + \sqrt{1-\bar\alpha_t}\,\epsilon_t$$

作者据此推导含前向条件的 ELBO（Eq 6，推导见论文附录），得到 ε-预测训练目标 $\|\epsilon_t - \epsilon_\theta(z_t^m, z_0^c, t)\|$（Eq 11-13；原文记法的去噪网络实参中 $z_0^c$ 出现两次，本页取简写）[^src-rdpi]。反向均值与采样方程（Eq 11、Algorithm 2）中观测项的系数随时间步变化，训练与采样均在每步显式使用 $z_0^c$[^src-rdpi]。

## 消融支持

RDPI 消融中 w/o cond-forw（前向不用观测条件）在 AQI36 In-sample 上 MAE 9.25±0.32 / MSE 310.44±10.22，劣于完整 RDPI 的 7.98±0.24 / 238.25±13.22（Table 5）；作者将其解释为前向无条件时模型无法利用观测-缺失关系（消融分析 (1)，论文口径）[^src-rdpi]。

## 与相关条件化方式的对照

| 方法 | 观测条件的注入位置 |
|------|-------------------|
| [[csdi\|CSDI]] | 反向去噪网络输入（拼接 + 条件掩码）[^src-csdi] |
| [[pristi\|PriSTI]] | 反向注意力的 Q/K（先验引导，前向无条件）[^src-pristi] |
| [[rdpi\|RDPI]] | 前向转移 + 反向去噪网络（论文口径）[^src-rdpi] |

## 范围

单一来源（RDPI 论文）；机制细节以论文原文为准，Eq 3 的逐步转移与 Eq 4 的边际形式均按原文抄录、其衔接推导不在本页展开。RDPI 的"两阶段"是框架级的确定性初值 + 残差扩散，与 [[two-stage-imputation]] 所记网络级双阶段不同义，见该页辨析。

[^src-rdpi]: [[source-rdpi]]
[^src-csdi]: [[source-csdi]]
[^src-pristi]: [[source-pristi]]
