---
title: "Time-Gated Multi-Scale Velocity Heads"
type: technique
tags:
  - flow-matching
  - multi-scale
  - gating
  - spectral-bias
  - time-series-imputation
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# Time-Gated Multi-Scale Velocity Heads

**时间门控多尺度速度头**是 TG-MSFM（ICLR 2026）对速度场 $v_\theta$ 的参数化：把单一速度网络拆为固定 1D 金字塔上的尺度专用头，再用只依赖 $t$ 的门控做凸组合，让谱强调沿 ODE 轨迹确定性演化[^src-tgmsfm]。

## 机制

- **1D 金字塔**：stride 集合 $S=\{1,2,4\}$；共享表示 $h$ 经平均池化下采样 $h^{(s)}=\mathrm{Down}_s(h)$，每尺度一个轻量头（Conv-GELU-Conv）提出候选速度 $u^{(s)}$，线性上采样回原分辨率 $\tilde{u}^{(s)}$（结构引 FPN / U-Net 多分辨率设计，Sec 3.3）[^src-tgmsfm]
- **时间门**：$\alpha(t)=\mathrm{softmax}(\mathrm{MLP}(t))\in\Delta^{|S|-1}$，$v_\theta(z_t,t;\tilde{x})=\sum_{s\in S}\alpha_s(t)\,\tilde{u}^{(s)}$。门只看流相位 $t$、不看内容——早期粗尺度主导以稳定全局趋势，接近 $t=1$ 时权重转向最细分支以恢复局部细节[^src-tgmsfm]
- **细尺度抗混叠**：最细分支上叠加 3–5 tap 固定低通（unit DC gain）抑制伪振荡（Sec 3.3）[^src-tgmsfm]
- **有界性**（附录 A.2）：逐元素 tanh 压缩 + 门控凸组合 $\Rightarrow$ $\|v_\theta\|_\infty\le 1$，且不改变 ODE 不动点；unit DC 增益低通保持桥均值并降低高频能量（对 DC 正交成分 $\|Lx\|_2\le\|x\|_2$）[^src-tgmsfm]

论文附录 B 给出组件独立的诊断图（合成数据、数据集无关）：Fig B2 展示三尺度门调度示意（粗→细的过渡点 $\tau_1\approx0.29$、$\tau_2\approx0.61$）；Fig B3 用 early $[0.85,0.13,0.02]$ 与 late $[0.10,0.25,0.65]$ 两组门权重演示同一趋势上"早保低频、晚补细节"的合成方式[^src-tgmsfm]。

## 证据

消融（Table 2，Electricity / ETTh1 MSE）[^src-tgmsfm]：

| 变体 | Electricity | ETTh1 |
|---|---|---|
| 单尺度（$s{=}1$，去多尺度头） | 0.116 | 0.158 |
| 静态混合（去时间门） | 0.212 | 0.147 |
| Euler（去 Heun） | 0.115 | 0.143 |
| **full** | **0.101** | **0.126** |

三个组件中时间门退化最大。论文的解释分工：门控制"何时强调什么"（谱强调随流相位演化），Heun+DC 控制"更新如何穿过缺口"——两轴正交、贡献相加（Sec 4.4）[^src-tgmsfm]。长缺口扫描（Fig 3a）中，时间门延迟细尺度精化、避免长缺失段上过早对局部波动过拟合，TG-MSFM 在 12–72h 缺口上 MSE 增长最慢[^src-tgmsfm]。

## 谱系对照

- [[fgti|FGTI]]（NeurIPS 2024）：频率信息放**条件**——高频/主频双频域表示经 cross-attention 注入去噪网络；TG-MSFM 把尺度分解放**速度场参数化**并沿 $t$ 调度。论文自述区别在"多尺度建模的注入位置"：不只在编码器融合特征，而是参数化速度场本身（Sec 2）[^src-tgmsfm]
- [[timemixer|TimeMixer]]（ICLR 2024）：多尺度 mixing 作用于预测的表示层；TG-MSFM 在生成轨迹的速度场层面做尺度分解
- FPN（Lin et al., 2017）/ U-Net（Ronneberger et al., 2015）：金字塔下采样-上采样结构的引用来源[^src-tgmsfm]

## 相关页面

- [[tg-msfm]] — TG-MSFM 主页
- [[data-consistency-projection]] — 推理期配套机制
- [[heun-sampler]] — 积分器
- [[fgti]] / [[timemixer]] — 频率 / 多尺度对照
- [[flow-matching]] — FM 基础

[^src-tgmsfm]: [[source-time-gated-multi-scale-flow-matching]]
