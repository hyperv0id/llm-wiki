---
title: "UniST"
type: technique
tags:
  - spatio-temporal
  - foundation-model
  - prompt-learning
  - masked-autoencoder
  - one-for-all
  - zero-shot
  - traffic-forecasting
created: 2026-05-31
last_updated: 2026-11-07
source_count: 2
confidence: high
status: active
---

# UniST

**UniST**（Universal Spatio-Temporal model）是 Yuan Yuan、Jingtao Ding、Jie Feng、Depeng Jin、Yong Li（清华电子工程系 + BNRist，KDD 2024, DOI 10.1145/3637528.3671662）提出的 one-for-all 城市时空预测模型：单一预训练模型覆盖 21 个数据集（14 城、6 类域），跨城市跨域完成短期/长期预测、few-shot 与 zero-shot 适配[^src-unist]。论文将其定位为 universal 时空预测的 first attempt；代码见 github.com/tsinghua-fib-lab/UniST[^src-unist]。

## Core Insight

直接联合训练会失败：PatchTST(one-for-all) 在所有短期数据集上劣于各数据集专用版（Table 2：Crowd RMSE $10.25\to 13.51$、Cellular $43.40\to 56.80$、BikeNYC $5.27\to 9.97$）——朴素共享参数无法适配异构分布[^src-unist]。UniST 的解法建立在三条假设上（§3.5.1）：不同城市/域的数据集整体分布不同，但存在细粒度模式对（$x_a\approx x_b,\ y_a\approx y_b$）；不同模式应对应不同 prompt；存在可学的 $f_\theta$ 把输入模式映射到 prompt。因此先用多域数据掩码重建预训练学共享模式，再在冻结骨干上用 memory pool 检索生成逐样本自适应 prompt。

## 数据流与维度

形状记作 $[\text{行}, \text{列}]$。**维度符号**：$T$ 时间步数；$C$ 通道数；$H,W$ 网格高宽；patch 尺寸 $(l,h,w)=(2,2,2)$；$L'=T/l,\ H'=H/h,\ W'=W/w$；掩码率 $r=0.5$；隐藏维 $d=128$；memory pool 条目数 $N=512$；$M$ 时间近邻回看步数；$P$ 周期回看天数。超参为官方代码默认配置（`size='middle'`，模型 6.71M 参数）。

### Stage 1 · channel-independent 拆分与 ST patch 化

$$\begin{aligned}
X^{(c)} &= \mathrm{Split}_C(X) && \text{// } [T, C, H, W] \xrightarrow{\text{沿 } C \text{ 维拆成独立实例}} C\times[T, H, W]\ \text{——各通道独立处理，容纳跨数据集可变的通道数}\\
E_x &= \mathrm{Conv3d}_{(l,h,w)}\!\left(X^{(c)}\right) && \text{// } [T, H, W] \xrightarrow{\text{kernel=stride=}(2,2,2)\text{ 的非重叠 3D 卷积}} [L', H', W']\ \text{——每个 token 是一个 } 2\times2\times2 \text{ 时空体，保留空间拓扑}\\
\tilde{E}_x &= \mathrm{Flatten}\!\left(E_x\right)+\mathrm{PE}_{\sin/\cos}+\mathrm{Emb}_{t} && \text{// } [L', H', W'] \to [L'H'W', d]\ \text{——展平为 token 序列；正弦位置编码分别施加于空间维与时间维，另加时间戳嵌入}
\end{aligned}$$

### Stage 2 · 四路掩码与重建

对 token 序列施加掩码率 $r$，四种策略各针对一种能力（§3.4）。预训练时每个 mini-batch 随机选一种（Algorithm 1）：

$$\begin{aligned}
\text{random:}\quad &E_x = E_x\!\left[M<1-r\right], && M\sim U[0,1]\in\mathbb{R}^{L'\times H'\times W'} && \text{// 逐 token 随机丢弃——细粒度时空关系}\\
\text{tube:}\quad &E_x = E_x\!\left[:,\,M<1-r\right], && M\sim U[0,1]\in\mathbb{R}^{H'\times W'} && \text{// 同一空间位置沿全时间轴一起遮蔽——模拟传感器失效，练空间外推}\\
\text{block:}\quad &E_x = E_x\!\left[:,\ \Omega\right], && \Omega\subset[H']\times[W'],\ |\Omega|=rH'W' && \text{// 连续矩形空间块沿全时间轴遮蔽——练空间迁移}\\
\text{temporal:}\quad &E_x = E_x\!\left[M{=}1\right], && M=\left[\mathbf{1}^{(1-r)L'H'W'};\ \mathbf{0}^{rL'H'W'}\right] && \text{// 展平序列尾部 } r \text{ 比例（未来）被遮——练因果时间建模}
\end{aligned}$$

$$\begin{aligned}
E_{enc} &= \mathrm{Encoder}\!\left(E_x\right) && \text{// 6 层 8 头 Transformer，只见可见 token；被遮位置不进编码器}\\
Y_{dec} &= \mathrm{Decoder}\!\left(E_{enc},\ E_{mask}\right) && \text{// 4 层 4 头 full-sized decoder：mask token（128 维可学习向量）填回被遮位置后解码重建}
\end{aligned}$$

预训练损失为被掩 patch 上的 MSE。**预测任务即 temporal 掩码重建**：微调与推理时掩码策略固定为 temporal、掩码率取 $\text{pred\_len}/(\text{his\_len}+\text{pred\_len})$（6→6 时即 0.5），未来 token 全遮、由 decoder 重建——掩码重建框架天然涵盖预测。

### Stage 3 · 提取四类知识表示（prompt 阶段输入）

空间侧先用 attention 把时间维合并为 $E_s$，再分两路；时间侧取两个窗口（§3.5.2）：

$$\begin{aligned}
E_{sc} &= \mathrm{Conv2D}_{3}\!\left(E_s\right) && \text{// 空间近邻：}3\times3\text{ 卷积——近邻单元互相影响}\\
E_{sh} &= \left\{\mathrm{Conv2D}_{2i+1}\!\left(E_s\right)\right\}_{i\in\{2,3,4\}} && \text{// 空间层级：}5/7/9\text{ 三尺度卷积——城市结构的多层感知}\\
E_{tc} &= \mathrm{Attention}\!\left(X_c\right) && \text{// 时间近邻：对最近 } M \text{ 步做注意力聚合——近因依赖}\\
E_{tp} &= \mathrm{Attention}\!\left(X_p\right) && \text{// 时间周期：对之前 } P \text{ 天对应时刻做注意力聚合——日/周周期}
\end{aligned}$$

官方实现默认启用空间（近邻+层级合并为多尺度卷积）+ 周期 + 近邻三路 prompt。

### Stage 4 · memory pool 检索生成 prompt 并注入

两个 key-value 池：$K_s, V_s\in\mathbb{R}^{N\times d}$（空间，512×128）、$K_t, V_t\in\mathbb{R}^{N\times d}$（时间）。知识表示经线性投影 + tanh 作查询，softmax 检索（官方实现）：

$$\begin{aligned}
\alpha &= \mathrm{softmax}\!\left(K\,q^{\top}\right) && \text{// } [N, d]\times[d, 1]\to[N, 1]\ \text{——查询对 } 512 \text{ 个 key 打分归一化}\\
P &= \alpha^{\top}V && \text{// } [1, N]\times[N, d]\to[1, d]\ \text{——加权聚合 value，得到该样本的 prompt}
\end{aligned}$$

$E_{sc},E_{sh}$ 查空间池，$E_{tc},E_{tp}$ 查时间池，各得一路 prompt。注入点在 **decoder 输入端的未来 token**：对末尾 $\text{pred\_len}/l\cdot H'W'$ 个（待预测）token 逐元素相加（$X_p=X+P$，Algorithm 2）——prompt 只引导重建目标，encoder 侧不注入。三个空间卷积尺度共享一个空间池，近邻/周期共享一个时间池。

## 两阶段训练

- **预训练**：21 个数据集轮流训练（各数据集独立 DataLoader，数据集顺序打乱一次后固定），每 mini-batch 随机选一种掩码策略；Adam，lr 3e-4，损失为被掩 patch 的 MSE（Algorithm 1；官方代码 AdamW + weight decay 1e-6 + 5 步 warmup + cosine 退火）。
- **Prompt 微调**：加载预训练权重（`strict=False`），按 Algorithm 2 冻结骨干 attention/FFN 层、只训 prompt 网络，损失 MSE + L2 正则，lr 5e-5。官方代码未实现显式冻结——prompt 模块随机初始化后与骨干共同训练，复现时需注意。

## 实验结果

### 短期预测（Table 2，6→6，RMSE/MAE；Crowd 与 Cellular 数值 ×10³）

| 数据集 | UniST | 最强基线 | 改进(RMSE) |
|---|---|---|---|
| TaxiBJ | **26.84 / 13.95** | STID 27.36 / 14.01 | 1.9% |
| Crowd | **3.00 / 1.38** | PromptST 3.52 / 1.54 | 14.8% |
| Cellular | **14.29 / 6.50** | PromptST 15.74 / 7.20 | 9.2% |
| BikeNYC | **3.50 / 1.27** | ACFM 3.93 / 1.67 | 10.9% |
| TrafficJN | **0.843 / 0.430** | PredRNN 0.852 / 0.463 | 1.1% |
| TDrive | **44.97 / 19.67** | SimVP 46.8 / 22.9 | 3.9% |
| TrafficSH | **0.665 / 0.405** | STID 0.742 / 0.469 | 10.4% |

### 长期预测（Table 3，64→64，RMSE/MAE；Crowd ×10³）

| 数据集 | UniST | 最强基线 | 改进(RMSE) |
|---|---|---|---|
| TaxiNYC | **19.83 / 6.71** | SimVP 20.18（RMSE）/ PredRNN 7.31（MAE） | 1.7% |
| Crowd | **4.25 / 2.26** | STID 4.91 / 2.63 | 13.4% |
| BikeNYC | **3.56 / 1.31** | TAU 3.89 / 1.54 | 8.5% |

相对各数据集最强基线平均改进 10.1%（三数据集 RMSE 与 MAE 六项提升的均值）。

### 完整基准中的反例（arXiv App. Table 11/13）

「consistently outperforms」只对 Table 2/3 所选数据集成立。完整基准中 UniST 并非全部最优：短期 TrafficBJ RMSE 0.689 输 PredRNN 0.651、TrafficWH RMSE 0.762 输 TAU 0.747（MAE 反超）；长期 BikeNYC-2 11.91 输 PredRNN 9.72、TDrive 90.60 输 STGSP/STID 83.70。

### Few-shot 与 zero-shot（arXiv App. Table 14–16；Crowd ×10³）

目标数据集不参与预训练，仅用其训练集 1%/5%/10% 微调：

| 设置 | Crowd | BikeNYC | TaxiBJ |
|---|---|---|---|
| 1% | **13.952**（MetaST 14.242、PatchTST 14.491） | **7.811**（TAU 8.50） | **35.09**（MetaST 39.08） |
| 5% | 13.859 | 6.113 | 31.19 |
| 10% | 13.411 | 5.318 | 27.59 |
| zero-shot | 14.665 | 9.06 | 51.4 |

zero-shot 的强势集中在 Crowd：14.665 超过 ACFM 1%-shot 21.166 与 PredRNN 1%-shot 24.901，与自身 10% 微调 13.411 仅差 9.4%。但 BikeNYC zero-shot 9.06 输给 TAU 8.50、MetaST 8.58、SimVP 8.661、MAML 8.98 的 1% 结果；TaxiBJ zero-shot 51.4 差于多数 1% 基线——zero-shot 能力跨数据集并不同质。

### 效率（arXiv App. Table 9）

6.71M 参数、2875MB 显存、训练 1.4 min/epoch（21 数据集一轮）、推理 0.034 min。训练时长约为 STID（0.054 min/epoch）的 26 倍。

## 消融

- **掩码策略**（arXiv App. Table 10，TrafficCD）：四策略全用最优（prediction/imputation/spatial extrapolation = 0.781/0.761/0.729）；去 temporal 对 prediction 伤害最大（1.44），去 random 对 imputation 伤害最大（1.72），去 block 对空间外推伤害最大（1.02）——四种掩码与三类任务一一对应。
- **prompt 成分**（Figure 5a，四个 Traffic 数据集）：去掉空间、周期、时间近邻任一均下降，各成分贡献随数据集变化。
- **memory pool 规模**（Figure 5b）：条目数 128→512 提升，512→1024 持平，取 512。
- **位置编码**（arXiv App. Table 7/8）：正弦编码在 0–10% 噪声下几乎不掉点；可学习编码 10% 噪声下崩溃（TaxiBJ 43.80 vs 28.76、Crowd 11.436 vs 3.29），零噪声时也已全面落后。
- **可解释性**（Figure 6/7 + arXiv App. E.1）：memory embedding 训练后从均匀初始化发散；同一数据集 train/test 的 prompt 相似，同城跨域（Crowd vs Cellular，均南京）相似，跨城跨域明显不同。

## 边界

- 仅网格数据：空间划分是经纬度等分网格；grid/sequence/graph 的统一列为 future work。
- 参数从 2M 扩至 30M（五档，arXiv App. Table 6）边际收益递减，论文归因于时空数据多样性低于语言/视觉。
- 四个知识提取网络是朴素实现（CNN+attention），论文明示可替换（如 Fourier 方法捕获周期）。

## 谱系与对比

UniST 与 [[urbandit|UrbanDiT]] 同出清华 FIB Lab（Yuan Yuan 为 UniST 一作、UrbanDiT 共同作者），从 MAE+prompt（UniST, KDD 2024）走到 diffusion+prompt（UrbanDiT, NeurIPS 2025）[^src-urbandit]。[[gpt-st|GPT-ST]]（NeurIPS 2023）先验证了 ST 图数据上逐数据集 MAE 预训练有效，UniST 把它推到跨数据集 one-for-all 并加 memory prompt[^src-unist]。

| 维度 | UniST (KDD 2024) | [[urbangpt|UrbanGPT]] (KDD 2024) | [[opencity|OpenCity]] (2024) | [[urbandit|UrbanDiT]] (NeurIPS 2025) |
|---|---|---|---|---|
| 范式 | MAE 预训练 + prompt | LLM 指令微调 | instance norm + TimeShift Transformer | Diffusion Transformer + prompt |
| 数据格式 | 仅网格 | 逐传感器 | 图 | 网格 + 图 |
| zero-shot | 跨城/跨域（Crowd 最强） | 跨区域/跨城 | 跨城 | 跨数据类型（5 任务） |
| 参数量 | 6.71M | ~7B (Vicuna) | 2M/5M/26M | 未披露 |
| 推理 | 0.034 min | 174 s/传感器 | <3 s | 快（rectified flow） |

## Related Pages

- [[source-unist]] — source summary 页
- [[mae]] — MAE，掩码自编码预训练范式来源
- [[videomae]] — VideoMAE，tube masking 同名策略的视频域先例
- [[gpt-st]] — GPT-ST，逐数据集 MAE 预训练前身
- [[std-mae]] / [[st-mae]] — 时空解耦/双掩码 MAE 预训练
- [[step]] — STEP，ST 掩码自编码预训练（图数据）
- [[patchtst]] — PatchTST，one-for-all 对照基线
- [[unified-prompt-learning]] — UrbanDiT 的三域 memory pool prompt，UniST 双池设计的后继
- [[urbandit]] / [[urbangpt]] / [[opencity]] / [[uniflow]] / [[urbanfm]] / [[factost]] — 同谱系或对照 ST 基础模型
- [[spatio-temporal-foundation-model]] — 时空基础模型范式
- [[spatio-temporal-foundation-model-landscape]] — 全景分析

[^src-unist]: [[source-unist]]
[^src-urbandit]: [[source-urbandit]]
