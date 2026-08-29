---
title: "Improved Mean Flows (iMF)"
type: technique
tags:
  - flow-matching
  - one-step-generation
  - meanflow
  - classifier-free-guidance
  - fastforward-generative-models
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: low
status: active
---

# Improved Mean Flows (iMF)

**Improved Mean Flows（iMF）**（Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He；CMU/MIT/Adobe/THU；arXiv:2512.02012 v2，2026-05-09）是 [[meanflow|MeanFlow]] 同一团队的后续工作。PDF 题名页印作 *Improved Mean Flows: On the Challenges of Fastforward Generative Models*，首页水印为 `arXiv:2512.02012v2 [cs.CV] 9 May 2026`。论文将"显式把 ODE/SDE 求解加速纳入训练目标"的一类方法概括为 **fastforward generative models**（借自物理模拟的用语：跨时间步做大跳跃的模型），并指出原 [[meanflow|MeanFlow]]（MF）遗留两个挑战（Sec. 1）：(i) 训练目标由网络自身的 JVP 构成、依赖网络，不是标准回归问题；(ii) CFG 引导尺度在训练前固定，牺牲推理时灵活性。iMF 针对这两点修正，全程从头训练、不蒸馏，作者报告 ImageNet 256×256 上 1-NFE FID 1.72（Tab. 2），并自称刷新该类（从头训练 1-NFE）方法的最好结果[^src-improved-meanflows]。

## 挑战一：训练目标依赖网络自身

### 等价改写：MeanFlow 即 v-loss

原 MF 训练的是 u-loss：目标 $u_{tgt}=(e-x)-(t-r)\,\mathrm{JVP}(u_\theta;e-x)$（iMF 论文 Eq. 6，对应原 MF 论文 Eq. 11），其中做了两处近似——边缘速度 $v$ 换成条件速度 $v_c=e-x$、JVP 中的真实 $u$ 换成网络预测 $u_\theta$[^src-improved-meanflows]。iMF 论文的第一步是把 MeanFlow identity $u=v-(t-r)\frac{du}{dt}$（原 MF 论文 Eq. 6，iMF 论文 Eq. 4）反解为（Sec. 4.1，Eq. 8）：

$$v(z_t)=u(z_t)+(t-r)\tfrac{d}{dt}u(z_t)$$

左边 $v$ 作回归目标（与标准 Flow Matching 同类），右边的复合函数由 $u_\theta$ 参数化：

$$V_\theta \triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP}_{sg}(u_\theta;e-x) \tag{Eq. 9}$$

损失为 $\mathbb{E}\,\|V_\theta-(e-x)\|^2$（Eq. 10）。论文称该改写与原 MF 目标完全等价（原文 "It is easy to show that the reformulation in (9)(10) is fully equivalent to the original MF objective in (6)(7)"，正文未附证明步骤），即 **MeanFlow 可以视为 v-loss 经 $u_\theta$ 再参数化的结果**；再参数化关系由 MeanFlow identity 驱动[^src-improved-meanflows]。

### 隐藏问题：复合函数多了一个输入

这一改写揭示出一个原表述中不可见的问题（Sec. 4.1）：$V_\theta$ 的输入不仅有 $z_t$，还有 $e-x$（Eq. 11）。从标准回归的角度（如 Flow Matching 的 v-loss，回归函数只吃 $z_t$），带 $e-x$ 输入的预测函数不是完全合法的预测函数（原文 "not a fully legitimate prediction function"）[^src-improved-meanflows]。

论文的机制分析是：v-loss 唯一的真目标是边缘速度 $v(z_t)=E[v_c\mid z_t]$，所以带 $e-x$ 输入并未直接泄漏真目标；但按全导数展开（原 MF 论文 Eq. 8，与本 wiki [[meanflow]] 页记法一致），JVP 的切向量输入本应是边缘 $v(z_t)$，原 MF 却代入条件速度 $e-x$，其方差经 JVP 被显著放大，而网络预测的 $v_\theta(z_t)$ 方差更低——论文认为 Fig. 3 的损失行为由该大方差主导（论文的分析表述）[^src-improved-meanflows]。

### 修正：把边缘速度参数化进 JVP

iMF 将复合函数重定义为（Sec. 4.1，Eq. 12）：

$$V_\theta(z_t)\triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP}_{sg}(u_\theta;v_\theta)$$

JVP 内的 $u_\theta$、$v_\theta$ 都只吃 $z_t$，$V_\theta$ 回到合法预测函数。$v_\theta$ 有两种实现：

- **边界条件**：利用 $v(z_t,t)\equiv u(z_t,t,t)$，直接取 $v_\theta(z_t,t)=u_\theta(z_t,t,t)$，零额外参数（论文报告这已足够解决本问题）[^src-improved-meanflows]
- **辅助 v-head**：在 $u_\theta$ 网络上加辅助头（末 $L=8$ 层不共享），以 Flow Matching 损失 $\|v_\theta-(e-x)\|^2$ 附加监督；训练期引入额外参数、推理不用（附录 A）[^src-improved-meanflows]

### stop-gradient 的位置

iMF 保留 stop-gradient，但位置改变：它在预测函数 $V_\theta$ 内部（$\mathrm{JVP}_{sg}$），而原 MF 的 stop-gradient 在回归目标整体上。论文说明此处的 stop-gradient 原则上非公式必需，实践中去掉会引入对 $\theta$ 的高阶梯度、加大优化难度，故仍保留（Sec. 4.1 "About Stop-gradient"）[^src-improved-meanflows]。

### 训练动态证据（Fig. 3）

作者报告的对照（MeanFlow-B/2、基本 $\ell_2$ 损失、无自适应加权、无 CFG，只统计 $t\ne r$ 样本；脚注说明若计入 $t=r$ 样本则原 MF 总损失仍可降，取决于占比）：两种目标虽只差在 $V_\theta(z_t,e-x)$ 与 $V_\theta(z_t)$，原 MF 的损失非降且方差大，iMF 损失正常下降（Fig. 3）[^src-improved-meanflows]。该观察与 [[meanflow|MeanFlow]] 原文"实践中带来更稳定训练"的表述存在张力，两组论断分别归因于各自论文（见下文与 [[meanflow]] 页的后续工作节）。

## 挑战二：CFG 尺度固定 → 引导即条件

原 MF 将引导场 $v^{cfg}=\omega\,v(\cdot|c)+(1-\omega)\,v(\cdot)$（原 MF 论文 Eq. 13）内建进训练目标，代价是引导尺度 $\omega$ 训练前固定。iMF 论文给出两点论据：固定尺度牺牲推理灵活性；且最优尺度随设置移动——作者以 iMF-B/2 在 ImageNet 256×256 展示，训练更久或推理步数更多时最优 CFG 尺度变小，即"强模型偏好小尺度"，事先无法确定最优值（Fig. 4）[^src-improved-meanflows]。

iMF 的做法是把引导相关量改写成条件变量，类比 $(r,t)$ 的时间条件化（Sec. 4.2，Eq. 15）：

- **尺度条件 $\omega$**：训练时从幂律分布 $p(\omega)\propto\omega^{-\beta}$ 采样（范围 $[1.0, 8.0]$，$\beta=1$ 或 2，附录 A），网络学习 $\omega$ 的嵌入
- **区间条件 $\Omega=\{\omega, t_{min}, t_{max}\}$**：把 CFG interval（原定义只作用在推理时的时间区间 $[t_{min},t_{max}]$）搬进训练——$t_{min}\sim U[0,0.5]$、$t_{max}\sim U[0.5,1.0]$，$t$ 落在区间外时置 $\omega=1$ 关闭引导；单模型即可在 1-NFE 下支持 CFG interval[^src-improved-meanflows]

论文注明：把引导作为条件在多步方法中已有研究（其引文 [34, 6, 49]），iMF 将其扩展到一步设定（Sec. 4.2）[^src-improved-meanflows]。

一个附带收益：推理时置 $\omega=1$ 即可模拟 w/o CFG。作者报告跨尺度训练显著改善 $\omega=1$ 推理的 FID（30.76 → 20.95，Tab. 1b），并认为这表明跨引导尺度训练改善模型泛化[^src-improved-meanflows]。

## 架构：多 token in-context conditioning

条件集合完整为 $u_\theta(z_t\mid r,t,c,\Omega)$（Eq. 16）。原 MF 沿用 DiT 的 adaLN-zero（各条件嵌入求和），论文认为异质条件增多后单一求和操作可能过载（论文表述）[^src-improved-meanflows]。

iMF 改用**多 token in-context conditioning**（Sec. 4.3）：DiT 曾探索过 in-context conditioning 但发现不如 adaLN-zero，iMF 的改动是每类条件复制成多个 token——类 8 个、其余条件各 4 个（Tab. 4），连续量经位置嵌入 + 2 层 MLP 处理，加可学习类型嵌入后与图像 latent token 沿序列维拼接（Fig. 5）[^src-improved-meanflows]。

副产物是可完全移除参数量大的 adaLN-zero：同深度宽度下模型缩小约 1/3（iMF-Base 133M → 89M）；初始化保留零残差块初始化，其余线性层用 $N(0,\sigma^2)$、$\sigma^2=0.1/\mathrm{fan\_in}$ 的高斯初始化（论文报告该初始化在去 adaLN-zero 后收敛更快，附录 A）。另引入 SwiGLU、RMSnorm、RoPE 等通用 Transformer 改进（沿用 LightningDiT，Sec. 4.3/5.1）[^src-improved-meanflows]。

## 实验结果（作者报告）

设置（Sec. 5）：ImageNet 类条件生成 256×256，预训练 VAE tokenizer 潜空间（32×32×4 latents，论文引文为 LDM 的 VAE），1-NFE，全部从头训练，FID-50K（5 万样本、每类 50 图）。消融基线为 MeanFlow-B/2（240 epochs，1-NFE FID 6.17 w/ CFG），训练设置与原 MF 完全一致。

**消融（Tab. 1，MF-B/2 骨干）**：

| 配置 | FID w/o CFG | FID w/ CFG |
|------|------------|-----------|
| 原 MF 目标（Tab. 1a） | 32.69 | 6.17 |
| iMF 目标，$v_\theta$=边界条件（Tab. 1a） | 29.42 | 5.97 |
| iMF 目标，$v_\theta$=辅助头（Tab. 1a） | 30.76 | 5.68 |
| + $\omega$ 条件（Tab. 1b） | 25.15（$\omega=1$ 推理） | 5.52 |
| + $\Omega$ 条件（Tab. 1b） | 20.95（$\omega=1$ 推理） | 4.57 |
| adaLN-zero → in-context（Tab. 1c） | — | 4.09 |
| + SwiGLU/RMSnorm/RoPE（Tab. 1c） | — | 3.82 |
| + 训练 640 epochs（Tab. 1c） | — | 3.39 |

- 目标修正的贡献随模型规模增大：同一边界条件变体在 MF-XL/2 上把 1-NFE FID 从 3.43 提到 2.99（Sec. 5.1）；作者推测容量更大的模型更能借 $u_\theta(z_t,t,t)$ 学好 $v_\theta$（论文的推测表述）[^src-improved-meanflows]
- $\omega$ 条件在 B/2 小模型上增益有限（5.68→5.52）——原 MF 对该小模型已调出近最优的固定 $\omega$；作者报告该增益随模型增大而更显著（Tab. 1b 及正文）[^src-improved-meanflows]
- $\Omega$ 条件的 1.11 增益（5.68→4.57，基线为 Tab. 1(a) 最佳行）主要来自 CFG interval 在推理时的可用性（Tab. 1b）[^src-improved-meanflows]

**系统级对照（Tab. 2）**：iMF-B/2 89M/3.39、M/2 174M/2.27、L/2 409M/1.86、XL/2 610M/1.72（IS 282.0）；对照原 MF-XL/2 676M/3.43。1.72 相对 3.43 为约 50% 的相对降幅（作者报告）。B/M/L/XL 记号仅便于称呼：移除 adaLN-zero 后参数量与计算量无法同时与原 MF 对齐，L/XL 实际比 MF 对应型号小约 10%（Sec. 5.2 说明）[^src-improved-meanflows]。

**与其他方法对照（Tab. 3，均为 iMF 论文报告的数字）**：1-NFE 从头训练组中 iCT-XL/2 34.24、Shortcut-XL/2 10.60、MeanFlow-XL/2 3.43、TiM-XL/2 3.26、α-Flow-XL/2+ 2.58、iMF-XL/2 1.72；2-NFE 组中 iMF-XL/2 1.54（对照 MeanFlow-XL/2+ 2.20、α-Flow-XL/2+ 1.95）；蒸馏 1-NFE 组 π-Flow 2.85、DMF 2.16、FACM 1.76。作者据此指出：iMF 高于所有从头训练同类方法，也高于蒸馏方法，且缩小了与多步方法的差距（多步参照含 DiT-XL/2 2.27、SiT-XL/2 2.06、LightningDiT-XL/2 1.35、DDT-XL/2 1.26 等）[^src-improved-meanflows]。

训练配置（Tab. 4）：240（消融）/640 epochs；batch 256/1024；lr $10^{-4}$ 恒定、warmup 10 epochs；EMA 0.9999；$r\ne t$ 样本占比 50%；$(t,r)$ 采样 logit-normal($-0.4, 1.0$)；实现基于原 MF 公开 JAX/TPU 代码库（附录 A）[^src-improved-meanflows]。

## 与 MeanFlow 原文及同期改进的关系（论文口径）

- 论文对原 MF 的修正是**改进而非取代**：等价改写（Eq. 9-10）保留原目标的解，修正的是参数化形式（JVP 输入）与 CFG 机制；原 MF 的 1-NFE FID 3.43 与 iMF 的 1.72 并列报告于 Tab. 2[^src-improved-meanflows]
- 论文将同期 MeanFlow 改进概括为（Sec. 2）：AlphaFlow 分解 MF 目标并用调度从 Flow Matching 插值到 MF；Decoupled MeanFlow 微调预训练 FM 模型（对网络末端 blocks 以第二时间步条件化）；CMT 用预训练 FM 提供固定显式回归目标做 mid-training。论文称 iMF 关注的是 MF 目标的根本限制与 CFG 的实际问题，"与这些同期改进正交"（原文 "orthogonal to other concurrent improvements"）[^src-improved-meanflows]
- 论文对 fastforward 路线的概括（Sec. 2）：Consistency Models 从中间时刻跳到轨迹终点；Consistency Trajectory Models 靠显式积分学任意两时刻间轨迹；Flow Map Matching 回归流场的零阶/一阶导；Shortcut Models 建立在两时刻与中点的关系上；IMM 用不同时刻的矩匹配；MeanFlow 参数化任意两时刻间的平均速度[^src-improved-meanflows]

## 与 α-Flow 的对照（本课程层面）

以下对照为本课程（wiki）归纳，非任一论文的原文表述：

- **对"原 MF 目标哪里不理想"给出两种不同机制解释**：[[alphaflow|α-Flow]] 将 MF 损失分解为轨迹流匹配 $L_{TFM}$ 与轨迹一致性 $L_{TCc}$，归因于两者梯度强负相关，并以课程退火分离（α-Flow 附录 D.1 及实证）[^src-alphaflow]；iMF 不采用分解视角，将问题归因于目标网络依赖与 JVP 切向量误用条件速度（方差放大），修正是参数化层面的（Sec. 4.1）[^src-improved-meanflows]。iMF 论文提及 AlphaFlow 时只概括为"分解 MF 目标并以调度插值"，未讨论其梯度冲突分析（Sec. 2）[^src-improved-meanflows]
- **修正手段不同**：α-Flow 改目标（课程调度 + stop-gradient 目标网络 $\theta^-$，消融后不使用 EMA），iMF 改参数化（$v_\theta$ 进 JVP）+ 改条件机制（CFG 条件化）+ 改架构（in-context conditioning）[^src-alphaflow][^src-improved-meanflows]
- **数字转引的口径差异**：iMF Tab. 3 将 α-Flow-XL/2+ 列为 2-NFE FID 1.95；α-Flow 原文的常规 2-NFE 自报值为 2.15，1.95 是其"均衡类采样"设定下的结果——转引时口径有出入，本页两组数字分别归因[^src-alphaflow][^src-improved-meanflows]

## 范围与局限

- 论文无独立局限性章节；实验仅覆盖 ImageNet 256×256 类条件生成（无 CIFAR-10、无文本条件实验）[^src-improved-meanflows]
- 启用 CFG 条件化后，每个模型的报告 FID 使用搜索得到的最优引导尺度与区间（附录 A Evaluation）——最优尺度的选取依赖事后评估[^src-improved-meanflows]
- 辅助头变体在 w/o CFG 下的增益小于边界条件变体（30.76 vs 29.42，Tab. 1a），作者报告其增益在 w/ CFG 下相对更大（Sec. 5.1）[^src-improved-meanflows]
- 作者在结论前指出：1-NFE 时代 tokenizer 的推理开销开始不可忽略，高效 tokenizer 或像素空间生成是正交于本工作的后续方向（Sec. 5.3）[^src-improved-meanflows]

## 相关页面

- [[meanflow]] — 被改进的原框架（本页问题的来源）
- [[alphaflow]] — 对 MeanFlow 的分解分析（另一种机制解释，见对照节）
- [[consistency-models]] — fastforward 路线源头工作（论文 Sec. 2 概括）
- [[shortcut-models]] — 步长条件化路线（论文 Sec. 2 概括）
- [[one-step-flow-generation]] — 一步流生成技术全景
- [[average-velocity-modeling]] — 平均速度建模（时序侧同类目标）
- [[classifier-free-guidance]] — CFG 机制（iMF 的 $\omega/\Omega$ 条件化为其在 1-NFE 下的变体）
- [[flow-matching]] — 理论基础（v-loss 的来源）
- [[source-improved-meanflows]] — 源文件摘要

[^src-improved-meanflows]: [[source-improved-meanflows]]
[^src-alphaflow]: [[source-alphaflow]]
