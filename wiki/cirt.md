---
title: "CirT (Circular Transformer)"
type: entity
tags:
  - weather-forecasting
  - s2s
  - transformer
  - spherical-geometry
  - fourier-transform
  - iclr-2025
created: 2026-07-14
last_updated: 2026-07-15
source_count: 1
confidence: medium
status: active
---

# CirT (Circular Transformer)

**CirT（Circular Transformer）** 是 HKUST Guangzhou + Alibaba DAMO Academy 在 ICLR 2025 提出的 S2S 气候预测模型，核心贡献是将球面几何归纳偏置显式引入 Transformer 设计[^src-cirt]。

## 架构

CirT 由三个关键组件构成[^src-cirt]：

1. **[[circular-patching|圆形分块]]**：按纬度将输入 $X^{t_1} \in \mathbb{R}^{H \times W \times K}$ 分解为 $H$ 个 circular patches $\{X^{(h)}\}_{h=1}^H$，每个 patch $X^{(h)} \in \mathbb{R}^{W \times K}$ 对应一条纬线上的所有气象变量。Patch 间几何距离固定为 $R\Delta\phi$，消除了平面投影导致的面积失真。

2. **傅里叶域 Transformer Encoder**：$L$ 层编码器，每层内：
   - **DFT**：对每个 patch 的 embedding $E_h^{(l)} \in \mathbb{R}^D$ 做离散傅里叶变换，得到复频率表示 $S_h^{(l)} = \mathcal{F}(E_h^{(l)})$
   - **频域多头注意力**：将实部 $A^{(l)}$ 和虚部 $B^{(l)}$ 拼接为 $C^{(l)} \in \mathbb{R}^{H \times 2D}$，执行标准 scaled dot-product attention
   - **IDFT**：注意力输出拆分回实部/虚部后做逆傅里叶变换还原到空间域
   - **FFN + LayerNorm**

3. **直接预测头**：展平 MLP 同时输出 Weeks 3-4 和 Weeks 5-6 的平均值，避免迭代累积误差。

## 关键指标

| 指标 | CirT | GraphCast | PanguWeather |
|:-----|:-----|:----------|:-------------|
| 参数量 | 16M | 37M | 256M |
| FLOPs | 2.2G | 110T | 168T |
| z500 Weeks 3-4 RMSE | 477 | 618 | 649 |

## 与其他模型的关系

- vs **GraphCast**：GraphCast 用 mesh 建模球面但仅做局部消息传递，不显式编码空间周期性；CirT 通过傅里叶变换捕获全局周期性[^src-cirt]。
- vs **PanguWeather / FourCastNetV2**：两者用标准 grid/cube patching 隐式学习几何偏置；CirT 用圆形分块显式编码[^src-cirt]。
- vs **ClimaX**：ClimaX 同样将球面数据展平为平面图像，CirT 的几何偏置对此类方法有普遍改进意义[^src-cirt]。
- vs **FEDformer**：两者均利用傅里叶变换处理周期性信号，但 FEDformer 在频域使用随机频率子集配合可学习核做 element-wise 处理，CirT 保留完整频率表示并执行标准多头注意力，以充分利用 circular patch 的 DFT 完备基函数表示[^src-cirt]。

## 相关页面

- [[source-cirt]] — 论文源文件摘要
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测
- [[spherical-geometry-inductive-bias]] — 球面几何归纳偏置
- [[circular-patching]] — 圆形分块
- [[fourier-self-attention]] — 傅里叶域自注意力
- [[weather-foundation-model]] — 天气基础模型

[^src-cirt]: [[source-cirt]]