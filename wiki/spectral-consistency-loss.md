---
title: "Spectral Consistency Loss"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - loss-function
  - spectral-analysis
  - regularization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Spectral Consistency Loss ($L_{\text{SCons}}$)

**频谱一致性损失** ($L_{\text{SCons}}$) 是 [[lscd|LSCD]] 中引入的训练损失项，用于强制扩散模型生成的插补信号在频域上与观测信号保持一致[^src-lscd]。它属于训练后期精调阶段的损失，在标准得分匹配完成后应用。

## 定义

$$L_{\text{SCons}} = \left\| \text{LS}(x_0^{\text{co}}) - \text{LS}(\hat{x}_0^{\text{co}}) \right\|_2^2$$

其中[^src-lscd]：
- $\text{LS}(\cdot)$ 是 [[lomb-scargle-periodogram|Lomb–Scargle 周期图]]算子（在 LSCD 中为可微实现）
- $x_0^{\text{co}}$ 是原始观测条件值
- $\hat{x}_0^{\text{co}} = \hat{x}_0 \odot m^{\text{co}}$ 是从扩散去噪过程重建的完整时间序列在条件掩码位置的值

该损失惩罚原始信号和重建信号的 Lomb–Scargle 周期图之间的差异，强制插补后的信号保留与观测信号一致的基本频率结构[^src-lscd]。

## 在训练流程中的位置

LSCD 采用**两阶段训练**策略[^src-lscd]：

### 阶段 1：得分匹配

标准扩散训练目标：

$$L(\theta) = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}, \text{LS}(x_0^{\text{co}}))\|^2\right]$$

此阶段学习近似完整数据分布，但可能无法充分捕获频率表示[^src-lscd]。

### 阶段 2：频谱精调

在得分匹配收敛后，加入 $L_{\text{SCons}}$ 精调：

$$L_{\text{reg}}(\theta) = \lambda_1 L(\theta) + \lambda_2 L_{\text{SCons}}(\theta)$$

这相当于在得分匹配最优解附近引入正则化，创建**分布精度**和**频谱保真度**之间的权衡[^src-lscd]。$L_{\text{SCons}}$ 会偏离参数在得分匹配目标上的最优解，但这种偏差被认为对高缺失率场景特别有益[^src-lscd]。

## 消融效果

在 PhysioNet 和 PM2.5 数据集上的消融实验显示[^src-lscd]：

| 配置 | PhysioNet 10% MAE | PM2.5 MAE |
|------|-------------------|-----------|
| 完整 LSCD | 0.211 | 9.069 |
| 删除 $L_{\text{SCons}}$ | 0.213 (+0.9%) | 9.085 (+0.2%) |
| 删除 $L_{\text{SCons}}$ + $E_{\text{spec}}$ | 0.218 (+3.3%) | 9.334 (+2.9%) |
| 删除全部 | 0.219 (+3.8%) | 9.669 (+6.6%) |

$L_{\text{SCons}}$ 单独贡献相对较小但有意义——它主要在高缺失率下起作用，确保即便观测值稀疏，频率分量也能被忠实重建[^src-lscd]。

## 设计动机

得分匹配目标在时域上优化分布拟合，但没有显式约束频谱结构[^src-lscd]。对于高缺失率的时间序列，时域损失可能不足以恢复正确的频率模式。$L_{\text{SCons}}$ 通过在频域中施加直接监督来填补这一空白——它告诉模型："你生成的时间序列在频率上必须看起来像真实观测信号的频率"[^src-lscd]。

## 计算成本

频谱精调阶段需要运行完整的推理流程来计算 $\hat{x}_0$，因此计算成本较高[^src-lscd]：
- PhysioNet: 288.7 秒/epoch
- PM2.5: 430.3 秒/epoch

考虑到阶段 1 的标准训练，$L_{\text{SCons}}$ 精调使总训练时间增加约 43-45%[^src-lscd]。

## 关联页面

- [[lscd]] — LSCD，使用 $L_{\text{SCons}}$ 的模型
- [[source-lscd]] — LSCD 源文件
- [[lomb-scargle-periodogram]] — Lomb–Scargle 周期图（$L_{\text{SCons}}$ 的基础算子）
- [[csdi]] — CSDI，不包含频谱损失的原始条件扩散框架
- [[diffusion-model]] — 扩散模型总览

[^src-lscd]: [[source-lscd]]
