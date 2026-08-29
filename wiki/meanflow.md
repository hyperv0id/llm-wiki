---
title: "MeanFlow"
type: technique
tags:
  - flow-matching
  - consistency-models
  - one-step-generation
  - few-step-generation
  - average-velocity
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 4
confidence: medium
status: active
---

# MeanFlow

**MeanFlow**（Geng, Deng, Bai, Kolter, He；CMU/MIT；arXiv:2505.13447 v1，2025-05-19。PDF 内无接收信息，[[loft|LOFT]] 著录其为 NeurIPS Vol.38，未在 PDF 内核实[^src-loft]）是少步流生成框架：训练网络 $u_\theta(z_t,r,t)$ 直接预测区间 $[r,t]$ 上的平均速度，推理按 $z_r = z_t-(t-r)\,u(z_t,r,t)$ 跳到任意 $r<t$，1 步采样即 $z_0 = z_1 - u(z_1,0,1)$，$z_1=\epsilon\sim p_{prior}$（Eq. 12，Alg. 2）[^src-meanflow]。方法自足：从头训练，无需预训练、蒸馏或课程学习[^src-meanflow]。

论文对少步生成困难的归因：即使条件路径被设计为直的（rectified），边缘速度场通常仍诱导弯曲轨迹，且这种非直性来自 ground-truth 场本身而非网络近似误差；对弯曲轨迹做粗离散化时 ODE 求解器结果不准（Sec. 3）[^src-meanflow]。

## 平均速度场与 MeanFlow Identity

- **平均速度定义**（Eq. 3）：
$$u(z_t,r,t)\triangleq\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau$$
即位移除以时间区间。$u$ 是瞬时速度场 $v$ 诱导的泛函 $u=F[v]$，不依赖神经网络；论文将其定位为与 Flow Matching 的瞬时速度 $v$ 对应的新 ground-truth field[^src-meanflow]。
- **边界与一致性性质**：$r\to t$ 时 $u\to v$；取一大步 $[r,t]$ 与连续两小步 $[r,s]$、$[s,t]$ 满足可加性 $(t-r)u(z_t,r,t)=(s-r)u(z_s,r,s)+(t-s)u(z_t,s,t)$。论文称准确逼近真实 $u$ 的网络天然满足这种一致性关系（术语推广自 Consistency Models），无需显式约束[^src-meanflow]。
- **MeanFlow Identity**（Eq. 6）：对定义式两边关于 $t$ 求导（$r$ 视为与 $t$ 独立）：
$$u(z_t,r,t)=v(z_t,t)-(t-r)\,\tfrac{d}{dt}u(z_t,r,t)$$
沿轨迹的全导数展开（$\tfrac{dz_t}{dt}=v$，$\tfrac{dr}{dt}=0$，$\tfrac{dt}{dt}=1$）给出 $u$ 与 $v$ 的另一关系（Eq. 8）：
$$\tfrac{d}{dt}u(z_t,r,t)=v(z_t,t)\,\partial_z u+\partial_t u$$
即 Jacobian $[\partial_z u,\partial_r u,\partial_t u]$ 与切向量 $[v,0,1]$ 的 JVP（Jacobian-vector product），可用 `torch.func.jvp` / `jax.jvp` 高效计算[^src-meanflow]。

## 训练目标

$$L(\theta)=\mathbb{E}\bigl[\bigl\|u_\theta(z_t,r,t)-\mathrm{sg}(u_{tgt})\bigr\|^2\bigr],\qquad u_{tgt}=v_t-(t-r)\bigl(v_t\,\partial_z u_\theta+\partial_t u_\theta\bigr)$$

- 目标仅以条件速度 $v_t=\epsilon-x$（默认调度）为 ground-truth 信号，训练时无需计算积分（Eq. 9–11）[^src-meanflow]
- stop-gradient 施加在整个目标 $u_{tgt}$ 上，避免对 JVP 的二阶反向传播；JVP 本身只需一次额外 backward。作者报告 JAX 实现开销低于总训练时间 20%（Sec. 4.1），B/4 基准为 0.045→0.052 s/iter，约 16% wall-clock（附录 B.4）[^src-meanflow]
- 若 $u_\theta$ 达到零损失，则满足 MeanFlow Identity，从而满足原定义（Eq. 3）；论文认为 ground-truth 目标场的存在使最优解原则上不依赖具体网络，实践中带来更稳定训练（论文表述）[^src-meanflow]
- 消融证实 JVP 切向量必须为 $(v,0,1)$：错误切向 $(v,0,0)$、$(v,1,0)$、$(v,1,1)$ 的 1-NFE FID 分别为 268.06、329.22、137.96，正确配置 61.06（Tab. 1b）[^src-meanflow]

> [!note] α-Flow 的解读（非原文主张）
> [[alphaflow|α-Flow]] 将该目标恒等改写为轨迹流匹配 $L_{TFM}$ 与轨迹一致性 $L_{TCc}$ 之和（其附录 D.1），把目标解读为步长 $\Delta t\to0$ 的一致性目标，并把"75% 样本 $r=t$"解释为 $L_{TFM}$ 的代理损失——这些均为 α-Flow 的分析视角[^src-alphaflow]。原文的表述是：恒等式由平均速度的定义导出，"no extra consistency heuristic is needed"（Sec. 1；Sec. 4.1 另表述为 "naturally derived from this definition, with no extra assumption"）[^src-meanflow]。

## 设计选择：r=t 切片、损失度量与 (r,t) 采样

- **$r=t$ 切片**：原文设计节说明对一定比例的随机样本取 $r=t$（Sec. 4.3："We set a certain portion of random samples with $r=t$"）。消融（Tab. 1a）显示 $r\ne t$ 占比 0%（退化为标准 Flow Matching）时 1-NFE 生成无法产生合理结果（论文表述 "fails to produce reasonable results"，FID 328.91），25% 最优（61.06），100% 亦可行（67.32）；论文自己的解释是模型在"学习瞬时速度（$r=t$）与经修改目标传播到 $r\ne t$"之间取得平衡[^src-meanflow]。ImageNet 各模型配置取 25% $r\ne t$，即 75% 样本 $r=t$（Tab. 4）；CIFAR-10 配置则取 75% $r\ne t$（附录 A）——该比例是按设置调节的。$r=t$ 切片上修正项消失、目标退化为普通流匹配监督，这一点原文亦明确指出（Sec. 4.1）[^src-meanflow]。
- **损失度量**：自适应加权 $w=1/(\|\Delta\|^2+c)^p$（Eq. 22，$p=1-\gamma$，$c>0$ 如 $10^{-3}$；$p=0.5$ 近似 Pseudo-Huber）。消融：$p=1$ 最优（61.06），$p=0$（平方 L2）79.75（Tab. 1e）[^src-meanflow]。
- **(r,t) 采样与条件化**：logit-normal 采样最优（lognorm(−0.4, 1.0) 61.06 vs uniform 65.90，Tab. 1d）；条件化用位置嵌入，$(t, t-r)$ 最优（61.06），仅嵌区间 $t-r$ 也可行（63.13）（Tab. 1c）[^src-meanflow]。
- 后续 [[improved-meanflows|iMF]] 的超参表列出 $r\ne t$ 占比 50%（iMF Tab. 4），与本页记录的 ImageNet 配置 25% 不同；iMF 论文未讨论该配置差异[^src-improved-meanflows]。

## CFG 内建于目标场

论文不在采样时施加 CFG（那会使 NFE 翻倍），而是把 CFG 写进 ground-truth 场：$v^{cfg}=\omega\,v(\cdot|c)+(1-\omega)\,v(\cdot)$（Eq. 13），网络 $u^{cfg}_\theta$ 直接建模其平均速度 $u^{cfg}$，采样仍为 1-NFE（Sec. 4.2）[^src-meanflow]。附录 B.1 引入混合系数 $\kappa$（Eq. 20–21，有效尺度 $\omega'=\omega/(1-\kappa)$）：固定 $\omega'=2.0$ 时 $\kappa=0.9$ 最优（FID 18.63 vs $\kappa=0$ 的 20.15，Tab. 5）[^src-meanflow]。

## 实验结果（作者报告）

- **ImageNet-256 类条件生成**（DiT 骨干、SD-VAE 潜空间、从头训练 240 epochs，Tab. 2）：MeanFlow-XL/2 1-NFE FID **3.43**（B/2 6.17、M/2 5.01、L/2 3.84），相对 Shortcut-XL/2（10.60，1-NFE）提升近 70%、相对 IMM-XL/2（7.77，1×2 NFE）提升超 50%；2-NFE XL/2 2.93、XL/2+（1000 epochs）2.20，论文称与多步 DiT-XL/2（2.27，250×2 NFE）相当（正文引 SiT-XL/2 为 2.15，Tab. 2 列 2.06，两处不一致，此处分别记录）[^src-meanflow]。
- **CIFAR-10 无条件生成**（U-net 约 55M，无 EDM 前置条件化，Tab. 3）：1-NFE FID 2.92，与 iCT 2.83、sCT 2.97 相当[^src-meanflow]。
- 可扩展性：作者报告 1-NFE FID 随模型规模与训练时长持续改善（Fig. 4）[^src-meanflow]。
- [[alphaflow|α-Flow]] 论文在相同设置复现 MeanFlow-XL/2 为 3.47（1-NFE）/ 2.46（2-NFE），与原文自报值（3.43 / 2.93）存在差异；两组数字分别由各自论文报告[^src-meanflow][^src-alphaflow]。

## 与相关工作的关系（原文口径）

- **Consistency Models**：论文指出 CM 系将一致性约束施加在网络行为上、路径锚定数据侧（相当于固定 $r\equiv 0$），网络只条件化单一时间变量，训练需要离散化课程（Sec. 1）；MeanFlow 由平均速度定义导出恒等式，条件化 $(r,t)$ 两个时间变量，无需额外一致性假设（Sec. 4.1）[^src-meanflow]
- **Shortcut Models / IMM**：同样条件化两个时间变量，但依赖额外的两时自洽约束；MeanFlow 仅由平均速度定义驱动（Sec. 4.1）[^src-meanflow]
- **Flow Map Matching**（Boffi et al.）：Flow Map 对应位移本身（流的积分），MeanFlow 的平均速度是位移除以时间区间（Sec. 2）[^src-meanflow]

## 后续工作：iMF 对原目标的修正

同一团队的后续工作 [[improved-meanflows|iMF]]（arXiv:2512.02012 v2，2026-05-09）针对本框架指出两个问题并修正（改进而非取代，原框架的论断与数字保留在相应章节）[^src-improved-meanflows]：

- **目标网络依赖**：iMF 将 MeanFlow identity 反解，论文说明原 u-loss 目标完全等价于 v-loss（瞬时速度目标）经 $u_\theta$ 再参数化的复合函数（原文 "It is easy to show"），并由此指出该复合函数以条件速度 $e-x$ 为额外输入——按本页上述全导数展开（Eq. 8），JVP 切向量本应取边缘速度 $v$，原目标却代入条件速度，其方差被放大——不是标准回归的合法形式；修正为 JVP 切向量取网络预测的边缘 $v_\theta$（边界条件 $u_\theta(z_t,t,t)$ 或辅助头），stop-gradient 由目标整体移入预测函数内部（iMF Sec. 4.1）[^src-improved-meanflows]。
- **CFG 尺度固定**：iMF 把引导尺度 $\omega$ 与 CFG interval 端点 $\Omega$ 改写为条件变量，训练时随机采样、推理时可任选，单模型在 1-NFE 下支持可变引导（iMF Sec. 4.2）[^src-improved-meanflows]。

训练动态方面，iMF 作者报告（Fig. 3，MeanFlow-B/2、基本 $\ell_2$、无自适应加权、无 CFG 设置，仅统计 $t\ne r$ 样本）：按本页「训练目标」节的原始目标训练时损失非降且方差大，iMF 修正后损失正常下降。该观察与本页所引原论文"实践中带来更稳定训练"的表述并存：前者是 iMF 论文在特定设置下的训练动态报告，后者是原论文的表述，两组论断分别归因[^src-improved-meanflows]。数字上，iMF 论文报告 iMF-XL/2 从头训练 1-NFE FID 1.72，对照本框架 MF-XL/2 的 3.43（两值均出自 iMF 论文 Tab. 2）[^src-improved-meanflows]。

## 相关页面

- [[improved-meanflows]] — 同团队后续改进：v-loss 再参数化 + 修正 JVP 输入 + 灵活 CFG 条件化
- [[alphaflow]] — 对 MeanFlow 的分解分析与改进目标族
- [[consistency-models]] — 一致性模型源头工作（MeanFlow 原文对其路线有专门讨论）
- [[shortcut-models]] — 同期少步生成方法（两时自洽约束路线）
- [[average-velocity-modeling]] — 平均速度建模在时序预测侧的同类目标（CoGenCast）
- [[one-step-flow-generation]] — 一步流生成技术全景
- [[loft]] — MeanFlow 路线在时空插补中的对照
- [[flow-matching]] — 理论基础
- [[source-meanflow]] — 源文件摘要

[^src-meanflow]: [[source-meanflow]]
[^src-alphaflow]: [[source-alphaflow]]
[^src-loft]: [[source-loft]]
[^src-improved-meanflows]: [[source-improved-meanflows]]
