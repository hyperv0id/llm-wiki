# Ingest 报告：UniCA

## 基本信息
- **日期**：2026-04-29
- **源文件**：`Han 等 - 2025 - UniCA Unified Covariate Adaptation for Time Series Foundation Model.pdf`
- **来源**：Zotero storage (GN69X52A)
- **会议**：ICLR 2026 Poster

---

## 创建的页面

### Source Summary
- **wiki/source-unca.md** — WHY：UniCA 论文摘要页面，需要记录核心论点、贡献和实验结果
- **wiki/source-timesfm.md** — WHY：TimesFM 基础模型占位符，因 UniCA 论文引用
- **wiki/source-chronos.md** — WHY：Chronos 基础模型占位符，因 UniCA 论文引用

### Entities
- **wiki/unified-covariate-adaptation.md** — WHY：UniCA 核心技术，作为新方法需要独立页面
- **wiki/timesfm.md** — WHY：TimesFM 实体页面，被 UniCA 验证的基础模型之一
- **wiki/chronos.md** — WHY：Chronos 实体页面，被 UniCA 验证的基础模型之一

### Concepts
- **wiki/covariate-homogenization.md** — WHY：UniCA 核心创新概念，将异构协变量转换为统一表示
- **wiki/heterogeneous-covariates.md** — WHY：核心问题定义，UniCA 解决的主要挑战
- **wiki/multimodal-time-series-forecasting.md** — WHY：多模态预测概念，UniCA 处理的场景之一

### Techniques
- **wiki/conditional-attention-pooling.md** — WHY：UniCA 的 CAP 融合机制，核心技术创新点

---

## 修改的页面

- **wiki/index.md** — 添加所有新页面的索引
- **wiki/log.md** — 记录本次 ingest

---

## 新建交叉链接

| 页面 | 链接到 |
|------|--------|
| source-unca | unified-covariate-adaptation |
| unified-covariate-adaptation | timesnet, conditional-attention-pooling, heterogeneous-covariates, covariate-homogenization, multimodal-time-series-forecasting |
| covariate-homogenization | unified-covariate-adaptation, heterogeneous-covariates, conditional-attention-pooling |
| heterogeneous-covariates | covariate-homogenization, unified-covariate-adaptation, multimodal-time-series-forecasting, timesnet |
| conditional-attention-pooling | unified-covariate-adaptation, gated-linear-units, timesnet |
| multimodal-time-series-forecasting | heterogeneous-covariates, covariate-homogenization, unified-covariate-adaptation, timesnet |
| timesfm | unified-covariate-adaptation, timesnet, chronos |
| chronos | unified-covariate-adaptation, timesfm, timesnet |

---

## 技术亮点

1. **协变量同质化**：将分类、图像、文本协变量转换为统一的时间序列表示
2. **双注意力融合**：Pre-Fusion（历史协变量）+ Post-Fusion（未来协变量）
3. **冻结骨干设计**：保持预训练 TSFMs 不变，仅训练适配器模块
4. **广泛兼容性**：适配 Chronos、TimesFM、Time-MoE、Moirai 等主流 TSFMs

---

## 实验结果

- 单模态：Chronos-Bolt + UniCA 达到 0.506 MAPE（最优）
- 多模态图像：TimesFM + UniCA 降低 6.5% MAE
- 多模态文本：Chronos-Bolt + UniCA 降低 13.0% MAPE