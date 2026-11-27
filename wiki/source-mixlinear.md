---
title: "MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - linear-model
  - low-rank
  - frequency-domain
  - lightweight
  - iclr-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters

长时序预测的轻量化路线各有取舍：纯时域下采样线性模型擅长局部形态、全局周期表达弱，纯频域压缩模型表达尖锐局部瞬态则需不成比例的参数量。MixLinear 利用序列的结构性对偶——局部形态在时域按段抽取最自然，全局趋势在频域天然稀疏——将两条极简支路并联相加，以约百个参数在 8 个基准上达到与既有轻量模型同量级的精度并实现最快推理[^src-mixlinear]。

## 核心做法

模型未引入新算子，组合的是下采样、分段线性分解与低秩频域投影三种标准组件[^src-mixlinear]。

- **时域支路**：固定步长下采样作隐式低通，切段后做段内、段间两次线性映射再上采样，两级因子分解把平方级参数压成线性增长（论文称 Segment-based Trend Extraction）[^src-mixlinear]。
- **频域支路**：对同一下采样序列做 FFT，谱不经全尺寸复数滤波器，先投影到仅 2 维低秩复数潜空间再重建，只保留主导频率成分，逆变换取实部上采样（FFT + 低秩复数投影 + iFFT，论文称 Adaptive Low-Rank Spectral Filtering）[^src-mixlinear]。
- 两支路输出直接相加，端到端训练[^src-mixlinear]。

## 证据

1. **定位与主结论**：约 0.1K 参数下 8 集精度与 SparseTSF、FITS 同量级，推理较 SparseTSF 快至多约 3.2 倍[^src-mixlinear]。
2. **真实涨点归因**：双支路贡献随维度分化——低维集去时域支掉点最重，高维集去频域支掉点最重（Traffic MSE 0.452→0.528），双域互补主要由高维支撑[^src-mixlinear]。
3. **证据缺口**：主表只报 MSE 无 MAE；缺 TimeBase 同台对比与融合方式消融[^src-mixlinear]。

## 相关页面

- [[sparsetsf]] · [[source-sparsetsf]] — 瓶颈接力 $\Delta$ 与复现对齐：保留下采样共享线性骨架，拓宽为段内/段间两级因子分解并新增频域支路，同台对比中 Exchange 上误差与延迟均优[^src-mixlinear]。
- [[source-timebase]] — 同槽位互斥：同处百参数级极轻量槽位，正交约束时间基与时频双低秩支路相加两种取向，缺同台对比
- [[cyclenet]] — 同槽位互斥：可学习周期残差剥离与时域双线性加低秩频域支路两种取向
- [[source-frets]] — 跨槽位正交：全谱复数 MLP 路线对下采样分段双线性加低秩谱支路

---

**出处**：ICLR 2026 · Aitian Ma, Dongsheng Luo, Mo Sha（Florida International University）[^src-mixlinear]

[^src-mixlinear]: [[source-mixlinear]]
