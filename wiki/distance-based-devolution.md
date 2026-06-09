---
title: "Distance-Based Devolution"
type: technique
tags:
  - diffusion-models
  - time-series
  - forecasting
  - denoising-network
  - linear-model
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Distance-Based Devolution

**Distance-Based Devolution**（基于距离的去演化）是 [[armd|ARMD]] 反向过程所用的去噪（"去演化"）方法：用一个**线性骨干**网络预测中间态到目标序列的"距离"，再以随扩散步自适应的权重把距离与中间态组合成预测，替代传统扩散去噪网络常用的 t-embedding 方案[^src-armd]。

## 机制

给定中间态 $X^t_{1-t:T-t}$ 与扩散步 $t$，去演化网络 $R(\cdot)$ 预测未来序列 $\hat X^0(X^t,t,\theta)$[^src-armd]：

1. **距离预测**：线性模块给出中间态到目标的距离 $D=\mathrm{Linear}(X^t_{1-t:T-t})$[^src-armd]。
2. **自适应平衡**：用随 $t$ 递减的权重 $W(t)\in[0,1]$ 平衡输入与距离：

$$\hat X^0(X^t,t,\theta)=\frac{W(t)\cdot X^t_{1-t:T-t}+(1-bW(t))\cdot D}{(1+cW(t))d},$$

其中 $W(t)$ 以 DDPM 的 $\bar\alpha_t$ 初始化、随线性模块参数一起训练，$b,c,d$ 为逐数据集 grid search 的平衡超参[^src-armd]。

直觉：$t$ 越大、中间态离目标越远，越依赖距离 $D$；$t$ 越小、中间态越接近目标 $X^0_{1:T}$，输出应越接近输入、越少依赖 $D$[^src-armd]。随后由 $\hat X^0$ 闭式算出预测演化趋势 $\hat z(t,\theta)$，训练目标为 $L_\theta=\mathbb{E}_t[|z^t-\hat z(t,\theta)|]$（L1 损失）[^src-armd]。

## 训练中的 deviation

为增加样本多样性、防过拟合，训练时向 $R(\cdot)$ 输入加微小 deviation $\eta_{0:t}\cdot\epsilon$（$\eta_{0:t}$ 取 DDPM 的 $\bar\alpha_t$，越接近历史序列扰动越小）[^src-armd]。消融显示移除该 deviation 会损害性能[^src-armd]。

## 为何用线性骨干

许多扩散类 TSF 模型（Diffusion-TS、CSDI、D³VAE）在去噪网络用 Transformer 骨干；ARMD 改用**线性骨干**，既贴合 ARMA 的"线性组合前 $k$ 步"假设，又大幅加速训练与采样[^src-armd]。消融中线性骨干在 14 个设置的 11 个上优于 Transformer 骨干[^src-armd]。

## 消融证据

将基于距离的方法替换为常规 **t-embedding** 去噪方案（先基于中间态出初步预测，再把 $t$ 嵌入并与之结合）后，性能在全部数据集下降，表明距离法优于 t-embedding 法[^src-armd]。

[^src-armd]: [[source-armd]]
