---
title: "YaRN"
type: entity
tags:
  - transformer
  - rope
  - context-extension
  - interpolation
created: 2026-04-28
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# YaRN (Yet another RoPE extensioN method)

YaRN 是 Nous Research 团队（Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole）提出的 RoPE 上下文窗口扩展方法。微调数据（PG19）不足原始预训练数据的 0.1%：Llama 2 7B/13B 从 4k 起，$s=16$ 用 400 步扩至 64k，再由该 checkpoint 续训 200 步（$s=32$）外推到 128k；与 Dynamic Scaling 结合后，不微调也可扩展 2 倍以上[^src-yarn]。

## 组合结构

YaRN 的定义包含两项调整：NTK-by-parts 插值修改 RoPE 各维频率，注意力温度缩放调整 softmax 前的 logits（Definition 2）[^src-yarn]。Dynamic Scaling 不在定义内，它是推理时决定缩放因子 $s$ 取值的调度方式：与 NTK-aware 组合称为 Dynamic NTK，与 YaRN 组合称为 Dynamic-YaRN[^src-yarn]。

四块组件的定义、单个方法的动机与各自的证据在：

[[ntk-aware-interpolation]] 修改 base $b'$ 将插值压力分摊到各维，交代 $b'=b\cdot s^{|D|/(|D|-2)}$ 的来源；[[ntk-by-parts-interpolation]] 说明波长分段规则、$\alpha,\beta$ 与 ramp 权重；[[attention-temperature-scaling]] 涵盖 Eq. 14 的温度 $t$、拟合式 Eq. 15 与 A.3 的扫描实验；[[dynamic-scaling]] 说明每次 forward 用 $s=\max(1,l'/L)$ 更新及 KV-cache 的处理。

论文 Figure 1 给出的谱系即这四块的嵌套：PI → NTK-aware → NTK-by-parts → YaRN，Dynamic Scaling 横切在推理阶段[^src-yarn]。

## 数据流与维度变换

**维度符号**：$|D|$ 为单个注意力头参与旋转的通道数（偶数，复维数 $|D|/2$）；$d=0,\dots,|D|/2-1$ 为复维下标，$d$ 越小则 $\lambda_d$ 越短、频率越高；$b$ 为 RoPE base（10000）；$L$ 为预训练上下文长度，$L'=sL$ 为目标长度；$N$ 为当前序列长度；$m,n$ 为 token 绝对位置。形状记为 $[\text{行},\text{列}]$。各步插值只在 $[\,|D|/2\,]$ 逐维标量上做逐元素运算，不跨维混合；复通道求和只出现在 Stage 3 内积行。

### Stage 1 · 算各维波长与预训练窗口内的圈数

$$\begin{aligned}
\theta_d &= b^{-2d/|D|},\quad d=0,\dots,|D|/2-1 && \text{// } \text{标量}\to[\,|D|/2\,]\ \text{逐维向量——第 } d \text{ 个复维的角频率（rad/token）}\\
\lambda_d &= \frac{2\pi}{\theta_d}=2\pi b^{\,2d/|D|} && \text{// } [\,|D|/2\,]\ \text{逐元素取倒数再乘 }2\pi\text{——该维转满一圈需要的 token 数}\\
r(d) &= \frac{L}{\lambda_d}=\frac{L\,\theta_d}{2\pi} && \text{// } [\,|D|/2\,]\ \text{——预训练窗口 }L\text{ 内该维转过的圈数，}L'\text{ 不进入这一步}
\end{aligned}$$

$\theta_d$ 随 $d$ 指数衰减，$\lambda_d$ 随 $d$ 指数增长：$d=0$ 是最高频维，$d=|D|/2-1$ 是最低频维[^src-yarn]（Eq. 8）。

### Stage 2 · 按圈数把维度切成三段

$$\begin{aligned}
\gamma_d &= \operatorname{clip}\!\left(\frac{r(d)-\alpha}{\beta-\alpha},\,0,\,1\right) && \text{// } [\,|D|/2\,]\to[\,|D|/2\,]\ \text{——Eq. 11 分段函数的等价 clip 写法；}\gamma_d=0\text{ 全插值，}\gamma_d=1\text{ 不插值}\\
\mathcal{D}_{\mathrm{interp}} &= \{\,d:\gamma_d=0\,\},\ \ \mathcal{D}_{\mathrm{ramp}}=\{\,d:0<\gamma_d<1\,\},\ \ \mathcal{D}_{\mathrm{keep}}=\{\,d:\gamma_d=1\,\} && \text{// } \text{三个互斥下标集，并集是全部 }|D|/2\text{ 个复维}
\end{aligned}$$

代入 $|D|=128$（LLaMA 系 head dim，[INFERENCE]）、$b=10000$、$L=4096$、$\alpha=1$、$\beta=32$：$r(20)=36.66$、$r(21)=31.75$，故 $\mathcal{D}_{\mathrm{keep}}=\{d\le 20\}$（21 个复维，$\lambda_d\le 112$）；$r(45)=1.004$、$r(46)=0.87$，故 $\mathcal{D}_{\mathrm{interp}}=\{d\ge 46\}$（18 个，$\lambda_{63}\approx 5.4\times 10^{4}$）；中间 25 个复维落在 ramp。$L$ 换成 2048 时两个阈值各向低维移动约 5 个复维，三段仍非空。

分组只由 $(|D|,b,L,\alpha,\beta)$ 决定，与目标长度 $L'$ 无关——$L'$ 只进入 Stage 3 的压缩量 $1/s$，因而该分段策略能用一套 $\alpha,\beta$ 覆盖 $s=16$ 与 $s=32$。

### Stage 3 · 分段缩放频率并构造旋转

$$\begin{aligned}
\theta'_d &= \left(\frac{1-\gamma_d}{s}+\gamma_d\right)\theta_d && \text{// } [\,|D|/2\,]\ \text{逐维线性混合的是频率而非波长；}\gamma_d{=}0\to\theta_d/s\text{，}\gamma_d{=}1\to\theta_d\\
f_W(x_m,m,\theta') &= e^{\,im\theta'}\odot\big(Wx_m\big) && \text{// } [\,|D|/2\,]\ \text{相位向量}\odot[\,|D|/2\,]\ \text{复通道}\to[\,|D|/2\,]\ \text{——逐元素乘单位复数，不跨维混合}\\
\operatorname{Re}\big(q_m^{*}k_n\big) &= \sum_{d=0}^{|D|/2-1}\operatorname{Re}\!\Big(e^{\,i(n-m)\theta'_d}\,\overline{(W_qx_m)_d}\,(W_kx_n)_d\Big) && \text{// } \text{标量——旋转只留下相位因子 }e^{\,i(n-m)\theta'_d}\text{，}m\text{ 与 }n\text{ 各自消失}
\end{aligned}$$

中间段平滑过渡：$d=30$ 处 $r=8.69$、$\gamma=0.248$，$s=16$ 时该维频率为 $0.295\,\theta_{30}$；$d=40$ 处 $r=2.06$、$\gamma=0.034$，频率几乎按 PI 压到 $0.094\,\theta_{40}$。

实现上 $\theta'$ 是一维向量，$\cos,\sin$ 表按位置展开一次算好，再按序列绝对位置取行：

$$\begin{aligned}
C_{m,d} &= \cos(m\theta'_d),\quad S_{m,d}=\sin(m\theta'_d) && \text{// } [L',\,|D|/2]\ \text{——位置}\times\text{复维两张表，与 batch、头无关}\\
\mathcal{I} &= (m_1,\dots,m_N) && \text{// } [N]\ \text{——本序列的绝对位置下标}\\
C_{\mathcal{I}} &= \operatorname{Gather}(C,\mathcal{I}),\quad S_{\mathcal{I}}=\operatorname{Gather}(S,\mathcal{I}) && \text{// } [L',|D|/2]\xrightarrow{\text{按 }\mathcal{I}\in[N]\text{ 取行}}[N,|D|/2]\ \text{——再对每个头复用同一对表}
\end{aligned}$$

### Stage 4 · 把温度并进 logits 的缩放

$$\begin{aligned}
\ell_{mn} &= \frac{q_m^{\top}k_n}{t\sqrt{|D|}} = \frac{c\,q_m^{\top}\,c\,k_n}{\sqrt{|D|}},\quad c=\sqrt{1/t} && \text{// } [N,\bcancel{|D|}]\times[\bcancel{|D|},N]\to[N,N]\ \text{——消去通道维；}c^{2}=1/t\text{ 把温度与 }1/\sqrt{|D|}\text{ 合成一个分母}\\
A &= \operatorname{softmax}\!\left(\ell\right) && \text{// } [N,N]\xrightarrow{\text{沿 key 维逐行归一化}}[N,N]\ \text{——}c^{2}=1/t>1\text{ 抬高 logits，行分布更尖、熵更低}\\
O &= A\,V && \text{// } [N,\bcancel{N}]\times[\bcancel{N},|D|]\to[N,|D|]
\end{aligned}$$

$c$ 不必进入 attention 代码：把 $c$ 乘到 Stage 3 复旋转嵌入（等价于乘 $C,S$）上，$q$ 与 $k$ 各自放大 $c$ 倍，内积自然放大 $c^{2}=1/t$；$\cos,\sin$ 表预先算好，训练与推理均无额外算子[^src-yarn]（§3.3）。

### Stage 5 · 决定缩放因子来源与重算时机

$$\begin{aligned}
s_{\text{fixed}} &= L'/L && \text{// } \text{标量，微调阶段固定；}C,S\text{ 随之固定，跨 forward 与所有 batch 复用}\\
s_{\text{dyn}} &= \max(1,\,l'/L) && \text{// } \text{标量，每次 forward 由当前序列长度 }l'\text{ 重算；}C,S\text{ 需重建，kv 须在旋转前缓存}
\end{aligned}$$

固定 $s$ 会导致短序列过度插值、超过 $L'$ 后崩溃；Dynamic Scaling 将两端改为缓降[^src-yarn]（§3.4，曲线见 Figure 8）。

## NTK-by-parts 为什么成立

前提有两条。其一，RoPE 各复维频率独立，内积按 Stage 3 第三行逐维相加，各维对位置的表征即「周期为 $\lambda_d=2\pi/\theta_d$ 的相位」，$d$ 越大波长越长，不同维类似一组不同周期的 Fourier 特征[^src-yarn]（§3.1 引 Tancik et al. 2020 的类比）。其二，$r(d)=L/\lambda_d$ 是恒等式，读作「预训练窗口内这一维转了几圈」[^src-yarn]（Eq. 10）。

三段判据分别由相位在 $[0,L]$ 上的覆盖方式给出：

$r>\beta$（短波长、高频）不插值：该维在窗口内已转过几十圈，相位在整个圆周密铺，位置超过 $L$ 只是继续在同一分布取值，不会落入未见相位；按 $1/s$ 压缩频率则会缩减相邻 token 的相位差，毁掉这组维度对近距离次序的分辨力，PI 在 $s\approx 8$ 以后失效正源于此[^src-yarn]（§3.1）。$r<\alpha$（长波长、低频）必须插值：该维在窗口内未转满一圈，相位只落在圆周一小段弧上，且因从不重复，每个位置在该弧上的取值唯一，网络可直接作为绝对位置读取[^src-yarn]（§3.2 正文与脚注）；位置超过 $L$ 时相位走出该弧并成为分布外取值，把 $\theta_d$ 除以 $s$ 恰好把 $[0,L']$ 的相位压回 $[0,L]$ 的弧内。$\alpha\le r\le\beta$ 则按 $\gamma_d$ 混合：该段维度转动不足一圈至几十圈，相位覆盖部分圆周；全插值损失分辨率，完全不插值又轻微越界，于是让 $\theta_d/s$ 与 $\theta_d$ 按 $(r-\alpha)/(\beta-\alpha)$ 线性加权[^src-yarn]（Eq. 13）。

将 ramp 施加于频率 $\theta_d$ 而非波长 $\lambda_d$ 属于实现选择：论文脚注列出的等价替代包括「对 $\theta_d/s$ 与 $\theta_d$ 取调和平均」与「先对波长线性插值再转回 $\theta$」，三者均可采用[^src-yarn]（§3.2 脚注 ‡）。

**这是工程规则，不是定理。** 三段相位覆盖只说明对应处理在该段不越界、不丢分辨率，并不推导边界落在 $r=1$ 与 $r=32$；$\alpha,\beta$ 是自由参数，论文按模型族实测标定，Llama 系取 $\alpha=1$、$\beta=32$[^src-yarn]（§3.2）。NTK 理论表明低维输入下网络难学高频成分，因而不能抹掉高频[^src-yarn]（§3.1），并不导出分段。

## 温度缩放的动机与实现

论文观察到 softmax 前引入温度 $t$ 对 perplexity 的影响跨样本、跨 token 位置一致，并将其归结于熵上升：logits 除以 $t<1$（$q_m,k_n$ 各乘 $\sqrt{1/t}>1$ 即得）后行分布更尖、熵更低[^src-yarn]（Eq. 14、§3.3）。文中未给出熵与 $t$ 之间的定量关系。

$\sqrt{1/t}$ 的取值来自拟合而非推导：在 LLaMA 7b/13b/33b/65b 上不微调，使用 NTK-by-parts 扫描不同 $s$ 并取 perplexity 最低的 $\sqrt{1/t}$，得到

$$\sqrt{\frac{1}{t}}=0.1\ln(s)+1,$$

Llama 2（7b/13b/70b）沿用同一组取值[^src-yarn]（Eq. 15）。代入得 $\sqrt{1/t}$：$s=8$ 为 1.208（原文 Eq. 23 示例）、$s=16$ 为 1.277、$s=32$ 为 1.347。该式跨模型复用表现出经验「普遍性」[^src-yarn]（§3.3）。

A.3 扫描证实「同一 $t$ 对所有样本与位置有效」：固定 $s=8$ 时，896 篇 16k-token RedPajama 文档中最低 perplexity 对应的 $\sqrt{1/t}$ 跨样本、跨 2048-token 分段位置基本一致（Figure 4–6）[^src-yarn]。

谱系里的 PI 一项见 [[position-interpolation]]：它把全部 $d/2$ 对频率统一除以 $s$、波长整体放大 $s$ 倍[^src-pi]，本页「等比例拉伸所有维度」指的就是这一步。

## 关键参数

| 参数 | 含义 | 论文取值 |
|------|------|----------|
| $s$ | 目标扩展倍数 $L'/L$ | 16、32（4k → 64k / 128k） |
| $\alpha$ | 完全插值的圈数上界 | 1 |
| $\beta$ | 不插值的圈数下界 | 32 |
| $t$ | 注意力温度 | $\sqrt{1/t}=0.1\ln(s)+1$ |
| 训练步数 | 微调步数 | $s=16$ 用 400 步；$s=32$ 在 $s=16$ checkpoint 上再 200 步 |

$\alpha,\beta$ 需按模型族实测；$s$ 与 $t$ 的对应关系沿用 Eq. 15[^src-yarn]（§3.2、§3.3；训练步数见 §4.1、Table 1；训练成本见 Table 4）。

## 证据

协议：滑动窗口 perplexity 采用 $S=256$（Press et al. 2022 的做法），微调数据将 PG19 切成 64k 段（LLaMA 7B 消融切 32k 段），学习率 $2\times10^{-5}$、warmup 20 步、AdamW $(\beta_1,\beta_2)=(0.9,0.95)$、全局 batch 64，全程用 Flash Attention 2[^src-yarn]（§4.1）。

表内 PI 行来自 YaRN 自建的微调对照，其步数与超参不同于 PI 原文的 Pile 1000 步全参数微调；原文 7B 扩到 32768（$s=16$）的模型在 8k/16k 评估窗口为 2.39/2.24（Table 2）[^src-pi]。两套数值口径不同，不互相校准。

### 标准基准（Llama 2 7B, s=16）

**同预算消融。** LLaMA 7B、$2k\times16$、微调 400 步，十篇 128k Proof-pile 文档的滑动窗口 perplexity（Table 5）：

| 方法 | 2k | 4k | 8k | 16k | 32k |
|------|-----|-----|------|------|------|
| PI（微调） | 5.70 | 4.95 | 4.64 | 3.97 | 3.57 |
| NTK-aware（微调） | 4.39 | 3.92 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts（微调） | 4.14 | 3.75 | 3.62 | 3.12 | 2.81 |
| YaRN（微调） | 4.19 | 3.77 | 3.30 | 3.09 | 2.77 |

NTK-aware 在 32k 列越界至 8.49[^src-yarn]。YaRN 在 2k、4k 列落后 NTK-by-parts 0.02–0.05（4k 列 3.77 对 3.75），从 8k 起各列均更低；YaRN 与 NTK-by-parts 在五列上均单调下降，PI 虽单调但各列均高于两者，NTK-aware 在 16k 之后仅于 32k 列翘起。不微调的 $2k\times16$ 行差距更大：YaRN 为 4.61/4.24/4.18/3.66/3.45（2k/4k/8k/16k/32k），PI 全列 $>10^{2}$，NTK-aware 在 32k 列 $>10^{1}$[^src-yarn]（Table 5；Figure 2 为同批模型训练损失曲线，Figure 3 为其 perplexity 与 passkey 曲线）。

**扩展到 128k。** Llama 2 7B 仅用 64k 数据训练，$s=16$ 在 64k 处为 2.42、131k 处越界（$>10$）；$s=32$ 从 $s=16$ checkpoint 再训 200 步，在 131072 处为 2.37，13B 同设置为 2.24[^src-yarn]（Table 1）。对照开源长上下文模型，Code Llama 13B（NTK）在 98304 处为 2.37、131k 处升至 2.54，YaRN 13B $s=32$ 为 2.23 与 2.24[^src-yarn]（Table 7、Figure 7）。GovReport 固定 32k 窗口 perplexity：YaRN 7B $s=16$ 为 3.59，Together 的 PI 32k 模型为 3.67，Code Llama 7B 为 4.44[^src-yarn]（Table 8）。

**passkey 检索。** 各上下文长度重复 10 次，passkey 在窗口内均匀随机放置：7B/13B 的 128k 模型（$s=32$）准确率均为 99.4%，7B/13B 的 64k 模型（$s=16$）为 96.3%/97.5%[^src-yarn]（Table 9）。passkey 与 perplexity 排序不同：Code Llama 13B 在 100k 以上 perplexity 上升，却仍能在 128k 检索到 passkey；Code Llama 7B 的 passkey 准确率为 94.3%[^src-yarn]（Table 9 与 §B.5）。

**标准基准。** Llama 2 7B 在 ARC-c/HellaSwag/MMLU/TruthfulQA 上从 53.1/77.8/43.8/39.0 变为 YaRN $s=16$ 的 52.3/78.8/42.5/38.2 与 $s=32$ 的 52.1/78.4/41.7/37.3；13B 从 59.4/82.1/55.8/37.4 变为 58.1/82.3/52.8/37.8[^src-yarn]（Table 3、Table 10）。$s=16$ 与 $s=32$ 之间平均下降 0.49%，64k→128k 的迭代扩展损失可忽略[^src-yarn]（§4.4）。

**训练成本。** LLaMA 7B $2k\times16$ 扩展至 32k 耗时 128 A100 小时，Llama 2 7B $4k\times16$ 耗时 256 小时，$4k\times32$ 为 $256+128$ 小时；同表 PI 基线（Chen et al.）为 640 小时[^src-yarn]（Table 4）。与 PI 的直接对照为 400 步对 1000 步：$4k\times2$ 设置下 2048/4096/6144/8192 四列，PI 1000 步为 3.92/3.51/3.51/3.34，YaRN 400 步为 3.91/3.50/3.51/3.35[^src-yarn]（Table 6、Appendix B.2）。

**边界。** 评测最长至 131072，未测更长序列；$s=32$ 模型只在 64k 数据上训练，128k 属于外推；$\alpha,\beta$ 按模型族标定（Llama 系取 1 与 32）；Eq. 15 拟合自 LLaMA 未微调模型，Llama 2 沿用而未重新拟合[^src-yarn]。

## 与 ALiBi 的关系

两者均针对位置编码无法外推到训练长度之外的问题，但路径不同：[[alibi|ALiBi]] 为相对位置编码，YaRN 不动位置编码接口，只改 RoPE 各维频率与 logits 整体缩放，作为 PI 的 drop-in 替代，适用于 LLaMA、GPT-NeoX、PaLM 等 RoPE 模型[^src-yarn]（§1、§5）。论文指出 ALiBi 类方法仅能有限外推，无法外推到远长于预训练长度的序列[^src-yarn]（§1）；实验未设与 ALiBi 的同协议对照，扩展幅度见前述各表。

## 引用

[^src-yarn]: [[source-yarn]]
[^src-pi]: [[source-position-interpolation-2306-15595]]
