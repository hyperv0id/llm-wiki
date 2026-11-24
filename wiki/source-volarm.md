---
title: "VolaRM: Volatility-Whitened Probabilistic Residual Modeling for Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - probabilistic-forecasting
  - diffusion-model
  - ddim
  - residual-modeling
  - non-stationary
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# VolaRM: Volatility-Whitened Probabilistic Residual Modeling for Long-Term Time Series Forecasting

**来源**：Fan Zhang、Shiming Fan、Zexuan Ma、Shijun Chen、Meijia Wang（山东工商学院），Hua Wang（鲁东大学，通讯）；NeurIPS 2026[^src-volarm]。

在原始时间序列空间直接做加噪与去噪时，高波动通道的大幅值会压垮扩散损失，使低波动通道的精细结构欠监督；而两阶段残差精修又容易在历史观测与未来预测的交界处改出数值跳变。为此，VolaRM 先用局部多项式外推与历史周期插值融合生成带均值和标准差的概率基准分支，再把未来目标相对基准的残差做标准化白化压缩到统一尺度，送入八步截断的确定性扩散采样分支去噪，最后经门控加回基准并以接缝连续性正则约束历史—未来过渡[^src-volarm]。

## 去黑话算子配方与路线差异

模型未引入改变复杂度阶数的新基础算子，而是由局部多项式/季节锚点概率基准、白化残差 8 步截断 [[ddim|DDIM]] 与接缝条件正则组合而成[^src-volarm]：

- **边界特征与概率基准分支**：从输入窗口末尾提取值与一、二阶差分及局部波动统计 $s_c=[x_L,\Delta x_L,\Delta^2 x_L,\nu_L]$，池化投影得到接缝条件 token $u$；将二阶多项式外推与历史周期插值融合得锚点 $A=(1-\omega)A^{\text{poly}}+\omega A^{\text{sea}}$，并由线性映射结合上下文特征 $c$ 与时间编码 $e_h$ 输出均值 $\mu=A+W_\mu(c+e_h)$ 及截断正标准差 $\sigma=\operatorname{clip}(\operatorname{softplus}(\cdot))$[^src-volarm]。
- **波动率白化与 8 步截断 DDIM**：将目标相对基准映射为标准化白化残差 $r_0=\operatorname{clip}((Y-\mu)/(\sigma+\varepsilon))$ 后执行前向高斯加噪；去噪网络以 $[z_t,\tanh\mu,\log\sigma]$ 为输入、$c+u+e_t$ 为条件，通过 $T=8$ 步截断 [[ddim|DDIM]] 采样估计 $\hat{r}_0$，再经门控融合还原预测 $\hat{Y}=\mu+g\odot\sigma\odot\hat{r}_0$。训练目标联合 SmoothL1、近端加权 $\varepsilon$-MSE、高斯 NLL 与接缝一/二阶连续正则 $\mathcal{L}_{\text{seam}}$，以兼顾分布覆盖与历史—未来平滑衔接；其代价在于需同时维护三套输出头、引入手工多项式与季节锚点偏置，且 $\sigma$ 估计误差会直接放大或扭曲白化坐标[^src-volarm]。
- **有向关系对照**：相对 [[nsdiff|NsDiff]] 属**同槽位互斥**（将 [[location-scale-noise-model|Location-Scale Noise Model (LSNM)]] 的时变位置—尺度建模从噪声调度迁至残差白化坐标）；对 [[location-scale-noise-model|LSNM]] 属**同构对接**（基准 $(\mu,\sigma)$ 即对应 $(f_\phi,\sqrt{g_\psi})$ 的位置—尺度参数化）；相对 [[k2vae|K²VAE]] 属**同槽位机制对照**（同为「基准预测 + 残差概率精修」两阶段分工，以 8 步截断 DDIM 替代 Koopman + KalmanNet 与 VAE 一步解码）；相对 [[ddim|DDIM]] 属**采样特例**（8 步截断条件采样）；相对 [[instance-normalization|RevIN]] 属**参照对比**（从入口整窗统计归一化转为预测端逐点 $(\mu,\sigma)$ 残差白化）[^src-volarm]。

## 证据

1. **定位与主结论**：8 集 LTSF 7 集均值最优（较 [[itransformer|iTransformer]]/Pathformer 降 4%–12.7%，较 TimeDART 降 3.4%），Traffic（0.404）输 TimeDART（0.388）[^src-volarm]。
2. **真实涨点归因**：去白化掉点最重（ETTm1 0.275→0.397、Weather 0.144→0.172、ECL 0.130→0.161），去扩散仅留基准亦近不白化档，主收益在「$(\mu,\sigma)$ 基准 + 白化坐标」；固定 $\sigma$、去边界条件或 $\mathcal{L}_{\text{seam}}$ 仅约 0.01（接缝正则证据偏弱）[^src-volarm]。
3. **证据缺口与负结果**：缺 [[nsdiff|NsDiff]]/[[k2vae|K²VAE]]/[[csdi|CSDI]]/[[timegrad|TimeGrad]] 同台；$L$ 选择置附录 B，Table 1/2 口径不一（0.400 vs 0.359）；ETTm1 某消融 MSE 恶化而 MAE 更优[^src-volarm]。

## 相关页面

- [[nsdiff]] — 噪声调度异方差 vs 白化残差坐标（同槽位互斥）
- [[location-scale-noise-model]] — $(\mu,\sigma)$ 与 $(f_\phi,\sqrt{g_\psi})$ 参数化（同构对接）
- [[k2vae]] — 基准预测 + 残差概率精修两阶段分工（机制对照）
- [[ddim]] — 8 步截断条件采样（采样特例）
- [[instance-normalization]] — 入口整窗统计归一化 vs 预测端局部白化（参照对比）
- [[diffusion-model]] — 扩散模型前沿谱系矩阵条目（非平稳残差白化分支）

[^src-volarm]: [[source-volarm]]
