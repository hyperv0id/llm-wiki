---
title: "Adaptive Frequency Modulation (AFM)"
type: technique
tags:
  - frequency-domain
  - extreme-weather
  - beta-distribution
  - spectral-filtering
  - weather-forecasting
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Adaptive Frequency Modulation (AFM)

自适应频率调制（AFM）是 [[uniextreme|UniExtreme]] 的核心模块之一，通过可学习 Beta 分布滤波器和多粒度频段聚合，自适应地区分正常与极端天气的频谱特征[^src-uniextreme]。

## 动机

极端天气区域在频域上呈现更强的"高频集中"特性（HFA 右移现象），但现有频域方法存在三类局限[^src-uniextreme]：

1. **仅通过特定频段**（如 OneForecast 的高通 GNN），忽略全局频谱和细粒度大气频谱细节
2. **刚性滤波参数**，缺乏区域自适应能力
3. **全可学习滤波器**，无法平滑地调控正常与极端天气的连续频谱分布

## 技术细节

### Beta 滤波

对区域天气状态做 2D FFT 后，定义 N 个区域自适应 Beta 滤波器[^src-uniextreme]：

$$B_{r;n}(x) = \frac{x^{\alpha_{r;n}-1}(1-x)^{\beta_{r;n}-1}}{\tilde{\lambda}_{r;n}^{\alpha_{r;n}-1}(1-\tilde{\lambda}_{r;n})^{\beta_{r;n}-1}}$$

两个关键参数由 $\tilde{\lambda}$（模式/mode）和 $\kappa$（展布/spread）确定：

- **模式 $\tilde{\lambda}_{r;n}$**：控制滤波器峰值对应的径向频率——通过**对数频段划分**手动指定（增长率 γ=1.3），确保低频段更细粒度
- **展布 $\kappa_{r;n}$**：控制分布围绕模式的集中度——通过对 FFT 实虚部的线性变换 + sigmoid **可学习**生成（范围 2 ≤ κ ≤ MAX_κ+2，MAX_κ=70）

这实现了**平滑性**（Beta 核函数保证）与**区域自适应性**（可学习展布参数）的平衡[^src-uniextreme]。

### 频段聚合

N 个局部滤波结果通过可学习权重加权聚合[^src-uniextreme]：

1. **空间权重**：CNN + MeanPool2D 从区域天气状态提取
2. **时间权重**：月/日/小时嵌入 + 线性变换
3. **融合**：时空权重拼接 → 线性层 → Softmax → 加权求和

时间嵌入的引入使 AFM 能捕获时间演化的频率差异。

### 后处理

聚合频谱经 IDFT 回到空间域，再通过 FFN + 残差连接 + LayerNorm 输出[^src-uniextreme]。

## 与相关技术的对比

| 技术 | 滤波方式 | 自适应性 | 频段覆盖 | 平滑性 |
|------|----------|----------|----------|--------|
| 高通 GNN (OneForecast) | 硬高通 | 无 | 仅高频 | N/A |
| 全可学习滤波 (FilterNet) | 纯学习 | 完全 | 全频段 | 无保证 |
| 固定 Beta (Rethink GNN) | 固定参数 | 无 | 全频段 | 有 |
| **AFM (UniExtreme)** | 可学习 Beta | 展布可学习 | 全频段（对数划分） | 有 |

## 消融实验验证

- 移除 AFM：极端 MAE 和 RMSE 显著上升[^src-uniextreme]
- 替换为全可学习滤波器（w/o BF）：性能下降，证明 Beta 核的平滑性约束至关重要
- 简单求和代替加权聚合（w/o BA）：性能下降，证明时空感知权重有效
- Band weight 的 HFA 分析显示与原始数据的"右移"模式一致，验证 AFM 确实捕获了正常-极端频谱差异[^src-uniextreme]

## 相关页面

- [[uniextreme]] — UniExtreme 模型
- [[event-prior-augmentation]] — EPA 模块（AFM 的互补模块）
- [[extreme-weather-forecasting]] — 极端天气预测概念
- [[fedformer]] — 频域分解 Transformer（不同领域的频域方法）
- [[frequency-aware-residual-representation]] — 傅里叶信号分解（相关频域技术）

[^src-uniextreme]: [[source-uniextreme]]
