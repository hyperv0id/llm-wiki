---
title: "How Good Can Linear Models Be for Time-Series Forecasting? (SearchCast)"
type: source-summary
tags:
  - time-series-forecasting
  - linear-model
  - hyperparameter-search
  - preprocessing
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# How Good Can Linear Models Be for Time-Series Forecasting?

> [!note] 出处口径
> arXiv:2606.27282v2（2026-06-29），PDF 首页仅印 "Preprint."，无 venue 信息。ICLR 2027 投稿（OpenReview xJaVGhG9Bc）著录来自用户，未在 PDF 内核实。源文件 `raw/2606.27282-ltsf-linear-models.pdf`，转换文本 `raw/2606.27282-ltsf-linear-models.txt`。

## 核心论点

论文反对"更大架构解锁精度"的领域趋势（从专用 Transformer 到 foundation model），主张大部分精度差距可通过调预处理而非扩模型在远低成本下弥合（Abstract）[^src-2606-27282-ltsf-linear-models]。因果解释：固定一套预处理超参的评测惯例系统性不利于低容量模型——百万参数 Transformer 能部分吸收糟糕的 lookback 或归一化选择，线性模型不能，导致线性方法"看起来比实际弱"（§1）[^src-2606-27282-ltsf-linear-models]。

## 方法

以 Toner & Darlow 统一结论（DLinear/NLinear 等变体等价于增强特征上的无约束线性回归）为起点，模型固定为闭式解 Ridge 回归、逐 variate 独立（channel-independent），预算全部投向四条预处理轴：context length $L \in [32, 2048]$（log）、local normalization（尾部分数 $r \in [0.001, 1]$）、augmentation（时/频域加噪 $\sigma \in [0.001, 0.5]$）、正则 $\alpha$ 内层 21 点网格；通过 $(g_h, g_s)$ 分组搜索在"全局共享"与"逐 horizon 逐序列"两极端间连续移动（§3）[^src-2606-27282-ltsf-linear-models]。详见 [[searchcast]]。

## 关键证据

8 个 benchmark 上（§4.2，Table 1）：线性类内 7/8 数据集平均 MSE 最优（ETTh2 与 FITS 打平 0.334 vs 0.333），对 OLS 降幅 ETTm2 4.8%、ETTh1 4.3%、Exchange 14.8%；对非线性基线（PatchTST/iTransformer/TimeMixer/TimesNet/Autoformer）6/8 平均最优，两处失利均输给 PatchTST 且集中于最大两数据集（Electricity 0.159 平、Traffic 0.404 vs 0.391）[^src-2606-27282-ltsf-linear-models]。三项诊断发现：最优 lookback 与 horizon 的幂律指数从 +0.46（ETTm2）到 −0.19（Exchange/Traffic）（§5.1）；局部归一化在 62–100% 单元被选中且最优尾部分数 0.3%–30%，全窗口归一化几乎从未被选中（§5.4）；跨序列超参共享的最优程度因数据集而异（§5.2）[^src-2606-27282-ltsf-linear-models]。

## 局限（论文自述，Appendix D）

仅限标准数值长程 benchmark；点估计 MSE 下小差距应视作平手；非线性基线取原论文数字而非联合重调；per-series 搜索随通道数增长；结论仅针对 Ridge + 搜索式预处理[^src-2606-27282-ltsf-linear-models]。计算成本仅有定性口径（单 trial "a few milliseconds on a GPU"、训练 "orders of magnitude cheaper"），无具体墙钟/FLOPs 数字；SearchCast 正文发布链接为占位符 "(URL)"[^src-2606-27282-ltsf-linear-models]。

## 落点

- 方法与主结果 → [[searchcast]]
- lookback–horizon 幂律与长程自相关证据 → [[lookback-horizon-scaling]]
- 尾部比例局部归一化 → [[trailing-fraction-local-normalization]]
- 分组超参搜索 → [[grouped-hyperparameter-search]]

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
