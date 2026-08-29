---
title: "Improved Mean Flows — On the Challenges of Fastforward Generative Models"
type: source-summary
tags:
  - flow-matching
  - one-step-generation
  - meanflow
  - classifier-free-guidance
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 0
confidence: low
status: active
---

# Improved Mean Flows — On the Challenges of Fastforward Generative Models

**作者：** Zhengyang Geng（CMU/MIT/Adobe）, Yiyang Lu（THU/MIT）, Zongze Wu, Eli Shechtman（Adobe）, J. Zico Kolter（CMU）, Kaiming He（MIT）；前两人同等贡献
**发表：** arXiv:2512.02012 **v2**（cs.CV，2026-05-09）；首页水印 `arXiv:2512.02012v2 [cs.CV] 9 May 2026`，题名页印作 *Improved Mean Flows: On the Challenges of Fastforward Generative Models*；PDF 内无 venue 信息
**raw：** `raw/geng-improved-meanflows-arxiv-2025.pdf`

## 核心论点

[[meanflow|MeanFlow]]（MF）同团队后续工作。论文将显式纳入 ODE/SDE 加速的方法概括为 fastforward generative models，指出原 MF 两个挑战并给出修正（Sec. 1）：

1. **目标网络依赖**：原 MF 目标含 JVP($u_\theta$)，不是标准回归。论文将 MeanFlow identity 反解（Eq. 8），证明原目标完全等价于 v-loss（瞬时速度目标）经 $u_\theta$ 再参数化的复合函数 $V_\theta$（Eq. 9–10）；由此暴露 $V_\theta$ 以条件速度 $e-x$ 为额外输入的非合法形式，根源是 JVP 切向量本应取边缘速度 $v$ 却取了条件速度、其方差被放大。修正为 $V_\theta(z_t)=u_\theta(z_t)+(t-r)\,\mathrm{JVP}_{sg}(u_\theta;v_\theta)$（Eq. 12），$v_\theta$ 取边界条件 $u_\theta(z_t,t,t)$ 或辅助头；stop-gradient 移入预测函数内部[^src-improved-meanflows]。
2. **CFG 尺度固定**：把引导尺度 $\omega$ 与 CFG interval 端点 $\Omega=\{\omega,t_{min},t_{max}\}$ 改写为条件变量，训练时随机采样（$\omega\propto\omega^{-\beta}$，范围 [1.0, 8.0]），推理时可任选；作者报告最优 CFG 尺度随模型/训练/步数移动（Fig. 4）[^src-improved-meanflows]。
3. **架构**：多 token in-context conditioning（类 8、其余各 4 token）替代 adaLN-zero，模型缩小约 1/3（133M→89M），另加 SwiGLU/RMSnorm/RoPE（Sec. 4.3；通用 Transformer 改进见 Sec. 5.1，沿用 LightningDiT）[^src-improved-meanflows]。

## 实验结果（作者报告）

ImageNet 256×256 类条件、1-NFE、从头训练、FID-50K。消融（MeanFlow-B/2，Tab. 1）：目标修正 32.69→29.42（w/o CFG）、6.17→5.97（边界条件 w/ CFG）；ω 条件 5.52、Ω 条件 4.57；in-context 4.09；+高级块 3.82；+640ep 3.39。系统级（Tab. 2）：iMF-XL/2 610M 1-NFE FID 1.72（IS 282.0）vs MF-XL/2 3.43；2-NFE 1.54（Tab. 3）。对照含 iCT 34.24、Shortcut 10.60、α-Flow-XL/2+ 2.58、蒸馏 FACM 1.76 等。

## 范围与局限

实验仅 ImageNet 256×256 类条件；报告 FID 使用搜索到的最优引导尺度与区间（附录 A）；论文指出 tokenizer 开销在 1-NFE 时代不可忽略（Sec. 5.3）；称自身问题与同期改进（AlphaFlow/DMF/CMT）正交（Sec. 2）。

## 相关页面

[[improved-meanflows]] · [[meanflow]] · [[alphaflow]] · [[one-step-flow-generation]] · [[classifier-free-guidance]]

[^src-improved-meanflows]: [[source-improved-meanflows]]
