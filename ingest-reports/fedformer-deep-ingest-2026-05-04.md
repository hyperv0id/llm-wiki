# Ingest 报告：FEDformer (ICML 2022) — 深度增强

## 创建
- wiki/frequency-enhanced-block.md — WHY：FEB-f/FEB-w 是 FEDformer 的核心自注意力替代组件，包含完整 DFT 公式、递归 Wavelet 分解流程和与标准 attention 的全面对比，具有独立参考价值
- wiki/frequency-enhanced-attention.md — WHY：FEA-f/FEA-w 是 FEDformer 的交叉注意力替代组件，包含频域 attention 公式、消融证据（V2 改进 12/16）和与标准 cross-attention 的对比表
- wiki/moe-decomposition.md — WHY：MOEDecomp 是 FEDformer 提出的输入自适应季节性-趋势分解方法（多尺寸平均滤波器 + 数据依赖 softmax 混合权重），与固定窗口分解有本质区别（+2.96% 提升），是独立的可复用技术

## 修改
- wiki/source-fedformer.md — WHY：原页面为批量 ingest 时创建的简要摘要。深度增强为包含 Theorem 1 理论分析、完整架构公式（eq.1-10）、消融研究详细结果、KS 检验分析、复杂度对比、与 Autoformer 的 5 个关键差异、加强的批判性分析
- wiki/fedformer.md — WHY：原实体页面为轻量级概述。重写为包含完整架构流程、复杂度对比表、性能汇总、11 个关联页面的 Connection 链
- wiki/autoformer.md — WHY：更新 FEDformer Connection 引用新创建的 FEB/FEA/MOEDecomp 技术页面
- wiki/dualsformer.md — WHY：更新 FEDformer Connection 引用 FEB/FEA 技术页面，准确描述 Dualformer 对固定模式选择的改进
- wiki/informer.md — WHY：更新 FEDformer 条目添加上下文和量化结果（14.8% multivariate, 22.6% univariate）
- wiki/hyperd.md — WHY：更新两处 FEDformer 引用，链接到 FEB/FEA/MOEDecomp 技术页面
- wiki/frequency-aware-residual-representation.md — WHY：更新 FEDformer 引用链接到 FEB-f/FEB-w 技术页面
- wiki/tslib.md — WHY：更新 FEDformer Connection 引用 FEB/FEA/MOEDecomp 技术页面
- wiki/traffic-forecasting.md — WHY：增强频域方法小节，链接到 FEB/FEA/MOEDecomp 技术页面
- wiki/periodicity-modeling-in-time-series.md — WHY：扩展 FEDformer 小节，补充 Theorem 1 的数学表述、KS 检验结果、架构组成（FEB/FEA/MOEDecomp）及具体量化指标
- wiki/index.md — WHY：添加 3 个新建技术页面条目
- wiki/log.md — WHY：记录本次详细 ingest 活动

## 新建交叉链接
- [[frequency-enhanced-block]] ↔ [[fedformer]], [[frequency-enhanced-attention]], [[moe-decomposition]], [[informer]], [[autoformer]], [[dualsformer]]
- [[frequency-enhanced-attention]] ↔ [[fedformer]], [[frequency-enhanced-block]], [[moe-decomposition]], [[autoformer]], [[informer]], [[dualsformer]]
- [[moe-decomposition]] ↔ [[fedformer]], [[frequency-enhanced-block]], [[frequency-enhanced-attention]], [[autoformer]], [[cyclenet]], [[mixture-of-experts]]