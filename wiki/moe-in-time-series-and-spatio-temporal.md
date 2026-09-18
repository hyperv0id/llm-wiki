---
title: "MoE 在时序与时空领域的机制谱系"
type: analysis
tags:
  - mixture-of-experts
  - routing
  - load-balancing
  - time-series
  - spatio-temporal
  - survey
created: 2026-09-17
last_updated: 2026-09-17
source_count: 31
confidence: medium
status: active
---

# MoE 在时序与时空领域的机制谱系

本页把 [[mixture-of-experts|MoE]] 在时序（time series）与时空（spatio-temporal）领域的用法按**插入位置、门控信号、稀疏与均衡、专家形态**四个维度整理成谱系，覆盖本地 wiki 已摄入的源文件与 Zotero 库（`/run/media/jcheng/WD-Data/yjs/Zotero/storage/`）中尚未建页的论文。

## 范围与证据基础

- **本地**：`wiki/` 下已有 MoE 机制描述的 35 个页面，以及 `raw/` 中的一手源文件（FaST、Time-MoE、Moirai-MoE、TiMi、MoST、MAGE、TESTAM、STAMImputer、FEDformer、DynaMix、HEPHAESTUS、AdaFre、xCPD、MMCKM、ExoST、StormInsight 等）。
- **Zotero storage**：对全库 155 个命中 `mixture of experts` 的 `.zotero-ft-cache`（PDF 全文纯文本）逐篇定性，筛出时序/时空相关工作。

检索得到一个需要先说明的结果：**关键词命中量远大于真实使用量**。155 篇里大多数只是把 Time-MoE / Moirai-MoE 当零样本基线（TimeMosaic、Timer-XL、Zeus、AdaPTS、TimeTIC、MSHLLM），或只在 related work 里引一句（STUnet 引 TESTAM 的 MoE 路由）。名字像 MoE 但机制不是的也要排除：MSHLLM 的 mixture of prompts (MoP) 是提示融合而非专家路由。本页只收录**把 MoE 当作真实组件**的工作，并在第七节列出边界案例。这一段的判定依据是本次检索对各论文 `.zotero-ft-cache` 的逐篇核对，未建 source 页的论文不挂脚注，检索记录见 [[ingest-reports/2026-09-17-moe-ts-st|WHY 报告]]。

> [!note] 证据等级
> 本页中 FaST、Time-MoE、Moirai-MoE、MoST、MAGE、TESTAM、FEDformer、AdaFre、xCPD、MMCKM、AMS-MoE 的数字来自已摄入的一手论文；SoftShape、InterpGN、TFPS、DutyTTE、TransferTraj、TS-RAG、FlowNet、SMARTraj2、AdapTraj、Graph Mixture Density Networks 的细节来自 Zotero 本地全文缓存；ExoST 截至 2026-07-28 仍无公开接收证据[^src-exost]。UniSTD 只有综述描述（「rank-adaptive mixture-of-experts」），无一手正文可核，本页只作提及[^src-large-models-ts-st-survey]。

## 一、动机：时序/时空为什么要用 MoE

四类动机在文献里可以清楚分开，它们决定了 MoE 被放在哪一层。

### 1.1 异质性：不同节点、区域、时段、频率的行为分布不同

- FaST 的出发点是**时空图的节点异质性**：不同传感器在不同时段的模式不同，用同一组参数拟合会互相拉扯；其 HA-Router 直接用原始序列 $X_t$ 加空间/日内/周内三组自适应 bias 计算专家分数 $G_\ell=\mathrm{softmax}(g_\ell(X_t)\oplus R_{S,\ell}\oplus R_{t}^{T,\ell}\oplus R_{t}^{W,\ell})$，让路由不被单层特征锁死[^src-fast-long-horizon-forecasting]。
- MoST 的出发点是**空间模式高度局部化**：邻近学校、商业区的区域空间模式相近，因此用多模态背景挑专家，每个专家只做「传感器与其 top-k 邻居」的交叉注意力，避开全图 $O(N^2)$[^src-most]。
- AdaFre 的出发点是**频率重要性随上下文变化**：低频通勤模式在高峰主导、高频事故在平峰更重要，而既有模型对所有频率固定权重融合[^src-adafre]。
- TFPS 的出发点是**patch 级分布偏移**：同一序列内部不同 patch 的模式不同，作者用子空间聚类得到每个 patch 的 pattern affinity，再据此路由专家[^src-tfps]。
- MAGE 的出发点是**线性自适应图学习的低秩瓶颈**：单个核的秩被 $d_G$ 限制，用 $K$ 个专家各自生成一张差分图可把秩上限提到 $\min\{d,K\cdot d_G\}$[^src-mage]。

### 1.2 容量—算力：用稀疏激活换参数规模

- Time-MoE 把每层 FFN 换成「N 个独立专家 + 1 个共享专家」的 Top-K 稀疏 MoE，论文报告 Time-MoE$_\text{ultra}$（1.1B 激活 / 2.4B 总参）零样本平均 MSE 相对 Moirai/TimesFM/Chronos 降低 20% 以上，训练成本比等激活量 dense 模型低 78%、推理低 39%[^src-time-moe]。
- Moirai-MoE 用 M=32 专家、每 token 激活 K=2，论文报告 Moirai-MoE-S（11M 激活 / 117M 总参）在 Monash 聚合 MAE 上比 dense Moirai-S 提升 17%，激活参数比 Chronos-L 少 65 倍[^src-moirai-moe]。
- FaST 用 dense MoE + GLU 专家，把复杂度写成 $O(N\cdot e\cdot d)$（线性于节点数），并报告推理 1.4× 加速[^src-fast-long-horizon-forecasting]。MAGE 则用稀疏激活把每专家约 6.25% 的激活率固定下来[^src-mage]。

### 1.3 多源与多模态：让路由信号来自数值之外

- TiMi 用冻结 LLM 从外生文本推出结构化因果知识，**用它当门控输入**：TMoE 的分数 $s_{i,t}=\mathrm{Softmax}_i(W_t\bar{H})$，SMoE 的分数 $s_{i,s}=\mathrm{Softmax}_i(W_s[h_1,\cdots,h_N])$，两路共享 FFN 池但门控独立[^src-timi]。
- MoST 用模态融合嵌入做路由，同时保留「每种激活模态一个共享专家」[^src-most]。
- TransferTraj 的门控输入是空间/时间/POI/路网四模态融合后的 token 表征，泛化对象是**区域**（空间上下文分布漂移）[^src-transfertraj]。
- AdapTraj 把「每个源域一个 extractor」当专家集合，聚合器用域标签（训练时以一定概率 mask 成未知域）来做域泛化；论文在 related work 里把 MoE 分成 sparse selection 与 full aggregation 两类，并自称属于后者[^src-adaptraj]。
- TS-RAG 把**检索到的历史序列本身当专家**：「Each embedding is treated as an expert」，k+1 个专家在检索增强模块里做 softmax 加权[^src-tsrag]。

### 1.4 机制切换与不确定性：专家=一种动力学/一组区间

- MMCKM 用 MoE 在 5 个参数受限的 Koopman 算子之间选，每个算子对应一种驾驶模式（自由流/跟驰/换道/汇入/紧急）[^src-mmckm]。
- DynaMix 用 J=10 个 AL-RNN 专家做动力系统重建，专家自然分化到不同动力学区域[^src-dynamix]。
- DutyTTE 把 noisy top-k MoE 插在路段编码器与不确定性头之间，用 C=8、k=4 的专家表示路段行程时间不确定性，再由两个预测头输出区间上下界、用 MIS 损失监督[^src-dutytte]。
- TESTAM 的动机更具体：回归任务里门控在初始化后几乎不再变化，专家无法分化，于是把路由改造成**带伪标签的分类问题**[^src-testam]。

## 二、插入位置谱系

同样是「用 MoE」，插入点差别决定了它能改什么。已收录工作可归为七类。

| 插入位置 | 代表工作 | 被替换/新增的对象 |
|---|---|---|
| 层内 FFN 替换 | [[time-moe\|Time-MoE]]、[[moirai-moe\|Moirai-MoE]]、[[timi\|TiMi]]、[[source-fast-long-horizon-forecasting\|FaST]]、[[source-transfertraj\|TransferTraj]] | Transformer 每层的前馈网络 |
| 输入/尺度/频率选择 | [[hephestus\|AMS-MoE]]、[[adafre\|AdaFre]]、[[xcpd\|xCPD]]、FaST（时间压缩） | 走哪条尺度/频带通路 |
| 空间与图结构 | [[mage\|MAGE]]、[[most\|MoST]] | 自适应图的构造、空间依赖的建模方式 |
| 分解滤波器 | [[fedformer\|FEDformer]] MOEDecomp | 趋势提取的滑动平均核 |
| 框架外层/模块间路由 | [[stamimputer\|STAMImputer]]、[[testam\|TESTAM]]、[[source-exost\|ExoST]]、[[mmckm\|MMCKM]]、[[dynamix\|DynaMix]]、TS-RAG | 用哪个子模块处理当前输入 |
| 输出/形状层 | SoftShape、InterpGN、FlowNet | 输出分布/形状嵌入的正向计算 |
| 事后装配 | FusionBench（WE-MoE / SMILE） | 已微调检查点之间的组合 |

### 2.1 层内 FFN 替换

这是最常见的一类，机制上最接近 Switch Transformer / Mixtral 的稀疏专家层。Time-MoE 每层放 N 个独立专家 + 1 个共享专家、Top-K 激活并配辅助均衡损失[^src-time-moe]；Moirai-MoE 每层放 M=32 专家、每 token 激活 K=2[^src-moirai-moe]；TiMi 的 MMoE 是即插即用模块，换掉 backbone 的 FFN[^src-timi]；TransferTraj 在每层 TRIE 注意力块之后接一个 SC-MoE 块，对每个轨迹点 token 做 noisy top-k[^src-transfertraj]；FaST 同时把 HA-MoE 用在时间压缩和 backbone 的 FFN 上[^src-fast-long-horizon-forecasting]。

### 2.2 输入、尺度与频率的选择

AMS-MoE 让每个专家对应一个固定 patch 尺寸（= 一个时间分辨率），用 noisy top-k 选尺度，论文报告 METR-LA/PEMS08 上 M=4、K=2 最优，且消融中移除 AMS-MoE 的退化幅度大于移除周期嵌入或空间注意力[^src-hephestus]。AdaFre 把选择维度固定在频率：$P=4$ 个频带、节点级温度 softmax 打分后取 top-$K{=}2$，各带过独立 backbone 再加权融合[^src-adafre]。xCPD 的选择对象是 channel-patch 节点在三频段（低/中/高）滤波器上的归属，其 DyMoE 按**累计概率阈值**选 1–3 个专家，而不是固定 $K$[^src-xcpd]。

### 2.3 空间与图结构

MAGE 的 MoE 作用于**图学习本身**：每个专家对应一种空间拓扑假设，用差分图 $A^{(k)}=\mathrm{Softmax}(E_1^{(k)})\mathrm{Softmax}(E_2^{(k)\top})-\lambda\,\mathrm{Softmax}(E_3^{(k)})\mathrm{Softmax}(E_4^{(k)\top})$ 增强多样性，从 KG=16 个候选中每节点激活 Top-K=4[^src-mage]。MoST 走另一条：专家是「与 top-k 邻居的交叉注意力块」，分模态共享专家与路由专家两类，路由专家默认 4 个、取概率最高者[^src-most]。

### 2.4 分解滤波器

FEDformer 的 MOEDecomp 把趋势提取写成 $X_{\text{trend}}=\mathrm{Softmax}(L(x))\cdot\mathcal{F}(x)$，即 K 个不同核宽的平均滤波器的数据自适应混合，替代固定窗口的平均池化[^src-fedformer]。论文报告在 ETT 与 Weather 上相对单一固定窗口方案平均提升 2.96%[^src-fedformer]。这是「专家 = 一个滤波核」的形态，粒度比 FFN 级 MoE 细。

### 2.5 框架外层路由

STAMImputer 把 MoE 上移到框架外层，由观测专家（O-Expert）依据稀疏度特征在时间/空间注意力专家之间路由，论文自称首次把 MoE 用于交通数据填补[^src-stamimputer]。TESTAM 用记忆库查询决定异构专家（恒等映射/静态图/空间注意力）的权重[^src-testam]。ExoST 的门控专家选择器建在潜在空间：$W(X_\tau)=\sum_k g_k^\tau(X_\tau)W_k^\tau$，是**输入依赖的线性算子混合**，作者把它类比推荐系统中的 MoE 用法[^src-exost]。StormInsight 的纵向交互编码器用 MoE 驱动 4 条定向跨层路径（中→高、高→中、中→低、低→中），每条路径一个 Transformer 专家[^src-storminsight]。

### 2.6 输出与形状层

SoftShape 在 shape embedding 之上用 router 激活 k=1 个 class-specific MLP 专家学 intra-shape 模式，专家总数等于数据集类别数，另有一个 Inception **共享专家**沿 shape 序列方向学 inter-shape 模式[^src-soft-shape]。InterpGN 只用 2 个专家做输出混合（可解释的 SBM 与 DNN），门控值由 SBM 自身输出算出的修改版 Gini Index 决定，越不自信越倚重 DNN，不做 top-k 稀疏激活[^src-interpgn]。FlowNet 在 M-MLP 里用 Mixture of Linears (MoL)：把**每条线性投影**当独立专家，16 个专家，L 层堆叠得到 L×E 种线性变换组合空间，论文以此对照「把整个 block 当专家」的 MoE[^src-flownet]。

### 2.7 事后装配

FusionBench 把 WE-MoE、SMILE 归为 model mixing：输入是多个已微调的同构检查点，输出常为扩参或稀疏专家结构，与训练期学路由的 MoE 分属两类[^src-jmlr-25-1243]。论文报告 CLIP-ViT-B/32 八任务 AVG 上 WEMoE 89.2、SMILE 89.3，对照多任务学习 88.6 与单任务上界 90.3[^src-jmlr-25-1243]。

## 三、门控与路由机制谱系

路由信号是这类方法最有区分度的一维。

| 门控信号 | 形式 | 代表工作 |
|---|---|---|
| 输入/token 线性投影 | $\mathrm{Softmax}(\mathrm{TopK}(xW))$ | Time-MoE[^src-time-moe]、TFPS（输入换成 subspace affinity）[^src-tfps] |
| 加噪 top-k | $\mathrm{Softmax}(xW_r+\epsilon\cdot\mathrm{Softplus}(xW_{noise}))$ | AMS-MoE[^src-hephestus]、TransferTraj[^src-transfertraj]、DutyTTE[^src-dutytte] |
| 原始序列 + 自适应 bias | $\mathrm{softmax}(g(X_t)\oplus R_S\oplus R_T\oplus R_W)$ | FaST[^src-fast-long-horizon-forecasting] |
| 预训练表示的簇中心距离 | $\mathrm{Softmax}(\mathrm{TopK}(\mathrm{Euclidean}(\tilde{x}^l,C^l)))$ | Moirai-MoE[^src-moirai-moe] |
| 多模态/外生文本嵌入 | $\mathrm{Softmax}(W\bar{H})$、$\mathrm{Softmax}(W[h_1..h_N])$ | TiMi[^src-timi]、MoST[^src-most] |
| 记忆库检索 | 元节点库相似度 + 路由分类损失 | TESTAM[^src-testam] |
| 观测稀疏度特征 | O-Expert 路由 | STAMImputer[^src-stamimputer] |
| 抽象状态/上下文注意力 | CNN 编码 + 状态注意力 + MLP | DynaMix[^src-dynamix] |
| 频率能量/相关性打分 | 节点级温度 softmax + top-K | AdaFre[^src-adafre]、xCPD（累计概率阈值）[^src-xcpd] |
| 域标签 | extractor 集合 + 域标签 mask | AdapTraj[^src-adaptraj] |
| 可解释模型自身输出 | modified Gini index 当门控，无额外参数 | InterpGN[^src-interpgn] |
| 当前状态 + 宏观观测 | MLP 硬选择 | MMCKM[^src-mmckm] |

几点值得单独指出：

- **路由粒度是设计变量**。AdaFre 的打分矩阵形状是 $N\times P$，即**节点级**而非样本级：每个节点独立选自己的频带组合[^src-adafre]；MAGE 同样是每节点 Top-K[^src-mage]；而 TS-RAG 是**检索样本级**（每个检索序列一个专家）[^src-tsrag]；TFPS 是 **patch 级**[^src-tfps]。
- **偏置项用来修正路由分布**。MAGE 的可学标量 $\gamma_k$ 趋于 $+\infty$ 时强制激活、趋于 $-\infty$ 时强制抑制[^src-mage]；TS-RAG 的门控 $\alpha=\mathrm{Softmax}(W_g E_{concat}+b_g)$ 带偏置项，且 softmax 覆盖全部 k+1 个专家、不做稀疏 top-k[^src-tsrag]。
- **噪声与阈值是稀疏化的两种替代**。TransferTraj 与 DutyTTE 用 Shazeer 式加噪 top-k[^src-transfertraj][^src-dutytte]；xCPD 用累计概率阈值决定选 1–3 个专家[^src-xcpd]；InterpGN 训练时软混合、推理时按阈值硬切换回纯可解释模型[^src-interpgn]。

## 四、稀疏性与负载均衡谱系

### 4.1 稀疏程度

- **dense（全专家加权）**：FaST[^src-fast-long-horizon-forecasting]、DynaMix（J=10 全部激活）[^src-dynamix]、ExoST（软路由）[^src-exost]、TS-RAG（softmax 覆盖全部专家）[^src-tsrag]、FEDformer MOEDecomp（K 个滤波核全部参与）[^src-fedformer]。
- **sparse top-k**：Time-MoE（Top-K，另加 1 个共享专家）[^src-time-moe]、Moirai-MoE（K=2/M=32）[^src-moirai-moe]、AMS-MoE（K=2/M=4）[^src-hephestus]、MAGE（K=4/KG=16）[^src-mage]、AdaFre（K=2/P=4）[^src-adafre]、SoftShape（k=1）[^src-soft-shape]、TransferTraj（k=4/C=8）[^src-transfertraj]、DutyTTE（k=4/C=8）[^src-dutytte]、MoST（路由专家取概率最高者 + 模态共享专家）[^src-most]。
- **阈值式**：xCPD 的 DyMoE 按累计概率阈值选 1–3 个[^src-xcpd]。
- **硬选择**：MMCKM 的意图判别器从 5 个算子中选一个[^src-mmckm]。

### 4.2 负载均衡：这是文献里差异最大的一维

| 均衡机制 | 形式 | 使用方 |
|---|---|---|
| Switch 式辅助损失 $M\sum f_i r_i$ | 按 batch 内专家选择频率×平均路由概率 | Time-MoE[^src-time-moe]、AMS-MoE[^src-hephestus]、MoST[^src-most] |
| 变差系数（CV）形式 | $\mathcal{L}_{imp}$ + $\mathcal{L}_{load}$，权重 $\lambda=0.001$ | SoftShape[^src-soft-shape] |
| 频率均衡项 | 惩罚各频带平均选择概率偏离 $1/P$ | AdaFre[^src-adafre]、xCPD（另有 entropy loss）[^src-xcpd] |
| 符号 SGD 优先级调制器 | $\beta_k\leftarrow\beta_k-\mu\,\mathrm{sgn}(N_k-NK/K_G)$ | MAGE[^src-mage] |
| 路由分类损失代替均衡 | worst-route avoidance + best-route selection（$q=0.7$） | TESTAM[^src-testam] |
| 用偏置/噪声代替均衡 | HA-Router 的异质性 bias；噪声促探索 | FaST[^src-fast-long-horizon-forecasting]、TransferTraj[^src-transfertraj] |
| 无 | 论文未提及任何均衡策略 | TFPS[^src-tfps]、DutyTTE[^src-dutytte]、TS-RAG[^src-tsrag]、ExoST[^src-exost] |

需要明确写下来的一条观察：**Zotero 中新增的四篇时序/时空 MoE 工作里，三篇（TFPS、DutyTTE、TS-RAG）完全没有负载均衡损失**[^src-tfps][^src-dutytte][^src-tsrag]，TransferTraj 用加噪代替均衡[^src-transfertraj]。均衡不是这类方法的默认配置，而是可选项。反过来，少数方法把均衡当作核心主张：MAGE 用 βk 保证每个专家约 6.25% 激活率以免模型坍缩到少数专家[^src-mage]；Time-MoE 的消融显示移除辅助均衡损失后专家坍缩为更小的 FFN，指标从 0.262 退化到 0.275[^src-time-moe]。

## 五、专家形态谱系

| 专家形态 | 代表工作 |
|---|---|
| FFN / GLU | Time-MoE（FFN + 共享专家）[^src-time-moe]、FaST（GLU，并行性更好）[^src-fast-long-horizon-forecasting]、TiMi（共享 FFN 池）[^src-timi] |
| 线性映射 | FlowNet 的 MoL（16 个）[^src-flownet] |
| 平均滤波核 | FEDformer MOEDecomp[^src-fedformer] |
| 注意力块 | TESTAM（空间注意力）[^src-testam]、STAMImputer（时间/空间注意力）[^src-stamimputer]、MoST（邻居交叉注意力）[^src-most] |
| 图生成器 | MAGE（每专家一张差分图）[^src-mage] |
| 形状嵌入 MLP | SoftShape（class-specific MLP + Inception 共享专家）[^src-soft-shape]、InterpGN（2 个可解释专家）[^src-interpgn] |
| 动力学算子 | MMCKM（5 个 Koopman 算子）[^src-mmckm]、DynaMix（AL-RNN）[^src-dynamix] |
| 检索序列 | TS-RAG（检索到的历史序列）[^src-tsrag] |
| 可学习滤波器组 | xCPD（低/中/高三个频率专属滤波器）[^src-xcpd]、AdaFre（频带 + 空间谱嵌入配对）[^src-adafre] |
| 域特定网络 | AdapTraj（每源域一个 extractor）[^src-adaptraj] |

## 六、共识、缺口与共同弱点

### 6.1 已报告的量级

Zotero 一侧新增工作给出的数字（作者报告，口径与表号注明）：

| 工作 | 任务/设定 | 报告结果 | 对照 |
|---|---|---|---|
| SoftShape[^src-soft-shape] | 128 个 UCR 数据集分类（Table 1） | avg acc 0.9334、avg rank 2.72 | TSLANet 0.9205、InceptionTime 0.9181（Wilcoxon p=1.06E-03） |
| InterpGN[^src-interpgn] | 30 个 UEA 数据集分类（Table 1） | avg acc 0.760、avg rank 3.500 | SBM 单独 0.726、FCN 0.746 |
| TFPS[^src-tfps] | 9 数据集 × 4 horizon × 2 指标（Table 1） | 论文自称取得 top-1 的设定占 57/72 | PatchTST、DLinear、TimesNet、iTransformer、FEDformer 等；亦记录 ETTh1-96 上 0.398 低于 TSLANet 0.387 |
| DutyTTE[^src-dutytte] | 成都/西安 OD 通行时间区间，90% 置信水平（Table 1） | RMSE 278.22/270.04、PICP 91.02%/91.67%；论文报告相对最强基线提升 9.34%/11.65%/15.65%/7.63%（RMSE/MAE/MAPE/PICP） | T-WDR-MIS RMSE 306.89、PICP 84.57% |
| TransferTraj[^src-transfertraj] | 成都 OD 通行时间估计消融（Table 3 vs Table 6） | 去掉 SC-MoE 后 RMSE 3.240 / MAE 2.585 / MAPE 11.584%，论文自述该任务性能下降 14.52% | 全模型 2.861 / 2.060 / 9.360% |
| TS-RAG[^src-tsrag] | 7 数据集 zero-shot（Table 1） | MSE 与 MAE 均最低，平均较 backbone Chronos-Bolt 降 3.54% MSE | Chronos-Bolt、MOMENT、TTM、Moirai、TimesFM、Chronos |

### 6.2 共识

**共识**：把「用哪个子模块」变成输入相关的决策，在时序与时空任务上普遍带来增益，且这一增益在多层设置下与参数规模解耦（稀疏激活的 MoE 基础模型是当前最清晰的一档证据）[^src-time-moe][^src-moirai-moe]。

### 6.3 缺口与弱点

按可得证据逐条列出，不做外推。

1. **专家数与稀疏度的选择缺少统一证据**。已报告的最优配置各不相同：AMS-MoE 的 M=4/K=2 来自 METR-LA 与 PEMS08 的消融[^src-hephestus]；MAGE 取 KG=16/K=4[^src-mage]；Moirai-MoE 取 M=32/K=2[^src-moirai-moe]；DynaMix 报告 J≥5 即足够、J=10 已在性能平台期[^src-dynamix]；而 AdaFre 把 $K$ 列入「systematic study」对象但正文与图题都没给出 $K$ 的结论[^src-adafre]；TFPS 的专家数 $K_t/K_f$ 随数据集漂移程度变化（Weather 用 $K_t{=}4,K_f{=}8$，ETTh1 用 $K_t{=}4,K_f{=}2$），但 **top-k 的 $k$ 取值正文未给出**，只在附录的网格 $\{1,2,4,8\}$ 里出现[^src-tfps]。TS-RAG 是少数给出扫描曲线的：$k\in\{1,4,8,16,20\}$ 上 MSE 先降后平，论文给出「最优区间 $k$ 在 6 到 10 之间」的取舍结论[^src-tsrag]。
2. **路由可解释性的直接证据少**。多数论文不报告路由分布随节点/时间的变化；例外是 TiMi 报告 SMoE 路由与 Mann–Kendall 趋势方向相关[^src-timi]、TransferTraj 给出专家激活分布图[^src-transfertraj]、AdaFre 未报告该可视化[^src-adafre]。
3. **MoE 常常没有单独的消融**。FlowNet 的 Table 2 只消融 retained flow / allocation flow / conservation law，**没有任何针对 MoL 或专家数量的消融**[^src-flownet]。
4. **组件描述不完整的情况确实存在**。DutyTTE 全文未给出专家网络的内部结构[^src-dutytte]；FGTI 的附录图 7 画了 "MoE Module" 但正文未说明该模块的作用与配置，只能如实记录（不以图推正文）[^src-fgti]；UniSTD 只有综述转述「rank-adaptive mixture-of-experts」，无一手正文可核[^src-large-models-ts-st-survey]。
5. **论文内部数字不一致要按原样保留**。TFPS 的数据集数 9 与 8 不一致、Table 12 与正文运行时间 6.114 vs 6.457 ms 不一致[^src-tfps]。
6. **二手转述的路线要标清**。TFMoE 在 EAC 的对照表中被归为「Mixture of experts」的持续时空预测路线，且与 TrafficStream、STKEC、PECPM 一样每个周期调整全部 STGNN 参数[^src-eac]；SGL 把 MoE 路线（引 TESTAM）与预训练路线一起归为「提精度但算力开销大」的一类[^src-lets-group]。这两处都是二手记录，本页不据此下结论。

## 七、边界：容易与 MoE 混淆的机制

- **MoE vs Mixture Density Network**。Graph Mixture Density Networks 明确选择 MDN 而非 MoE，论文自述的理由是 MoE 需要为每个输出分布配一个独立的 DGN encoder，参数多且在大图上不可扩展，MDN 共享 $h_g^V$ 让子网络 cooperate[^src-graph-mixture-density-networks]。两者都是「混合」，区别在混合的是**专家函数**还是**输出分布参数**。
- **MoE vs 借用专家思想但无门控**。AdapTraj 自述 inspired by mixture-of-experts，但实现上是每源域一个 extractor + 域标签聚合，**没有门控网络、没有 top-k、没有 softmax 路由**[^src-adaptraj]。
- **MoE vs 固定特征门控**。SMARTraj2 全文的 expert/MoE 命中只出现在参考文献与 3 处引用，正文机制是城市级与轨迹级的 sigmoid 特征门控（含 stop-gradient），没有专家集合[^src-smartraj2]。
- **MoE vs Mixture of Prompts**。MSHLLM 的 MoP 是提示融合，不是专家路由；其 MoE 命中仅来自 Time-MoE 的文献条目（见范围一节）。
- **MoE vs 固定多尺度混合**。TimeMixer 用固定下采样金字塔与线性混合，PathFormer 用固定 patch 尺寸 + 可学习通路，AMS-MoE 的对照表把二者列为「无路由」与「软路由」的基线[^src-hephestus]。
- **MoE vs 事后模型合并**。FusionBench 的 WE-MoE/SMILE 属于 mixing，与训练期学 gating 的 MoE 是两类；资源受限时应先看无参合并与 RegMean，而不是默认上 MoE[^src-jmlr-25-1243]。

## 相关页面

- [[mixture-of-experts|MoE]] — 通用框架页
- [[heterogeneous-moe-routing|异质性感知 MoE 路由]]、[[cluster-based-gating|簇基门控]]、[[memory-augmented-gating|记忆增强门控]]、[[sparse-balanced-mixture-of-experts-st|稀疏平衡 MoE]]、[[multi-modality-guided-spatial-expert|多模态引导空间专家]]、[[frequency-pathway-routing|频率通路路由]]、[[ams-moe|AMS-MoE]]、[[token-level-specialization|Token 级专业化]]
- [[time-moe|Time-MoE]]、[[moirai-moe|Moirai-MoE]]、[[timi|TiMi]]、[[mmoe|MMoE]]、[[source-fast-long-horizon-forecasting|FaST]]、[[most|MoST]]、[[mage|MAGE]]、[[testam|TESTAM]]、[[stamimputer|STAMImputer]]、[[hephestus|HEPHAESTUS]]、[[adafre|AdaFre]]、[[xcpd|xCPD]]、[[dynamix|DynaMix]]、[[mmckm|MMCKM]]、[[source-exost|ExoST]]、[[fedformer|FEDformer]]
- [[traffic-forecasting-architecture-trends|交通预测架构趋势]]、[[spatio-temporal-foundation-model-landscape|时空基础模型全景]]

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-time-moe]: [[source-time-moe]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-timi]: [[source-timi]]
[^src-most]: [[source-most]]
[^src-mage]: [[source-mage]]
[^src-testam]: [[source-testam]]
[^src-stamimputer]: [[source-stamimputer]]
[^src-fedformer]: [[source-fedformer]]
[^src-dynamix]: [[source-dynamix]]
[^src-hephestus]: [[source-hephestus]]
[^src-adafre]: [[source-adafre]]
[^src-xcpd]: [[source-xcpd]]
[^src-mmckm]: [[source-mmckm]]
[^src-exost]: [[source-exost]]
[^src-storminsight]: [[source-storminsight]]
[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
[^src-fgti]: [[source-fgti]]
[^src-eac]: [[source-eac]]
[^src-lets-group]: [[source-lets-group]]
[^src-large-models-ts-st-survey]: [[source-large-models-ts-st-survey]]
[^src-soft-shape]: [[source-soft-shape]]
[^src-interpgn]: [[source-interpgn]]
[^src-tfps]: [[source-tfps]]
[^src-dutytte]: [[source-dutytte]]
[^src-transfertraj]: [[source-transfertraj]]
[^src-adaptraj]: [[source-adaptraj]]
[^src-tsrag]: [[source-tsrag]]
[^src-flownet]: [[source-flownet]]
[^src-smartraj2]: [[source-smartraj2]]
[^src-graph-mixture-density-networks]: [[source-graph-mixture-density-networks]]
