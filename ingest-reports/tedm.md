# Ingest 报告：TEDM (ICLR 2026)

## 创建
- [[source-tedm]] — WHY：TEDM 论文的 source-summary 页面，记录论文核心方法、实验和局限性
- [[tedm]] — WHY：TEDM 实体/技术页面，详细讲解扩散时间=物理时间的创新、经验 schedule、结构化噪声、EDM 预处理推广

## 修改
- [[edm]] — 添加指向 tedm 的交叉链接
- [[simdiff]] — 添加指向 tedm 的对比链接
- (无已有 wiki 页面内容被实质性修改，TEDM 为全新 ingest)

## 新建交叉链接
- [[tedm]] ↔ [[edm]] — TEDM 直接扩展 EDM 框架
- [[tedm]] ↔ [[score-based-sde]] — TEDM 基于 Score-Based SDE 框架
- [[tedm]] ↔ [[ddpm]] — DDPM 是 VP SDE 的离散化，属同一理论脉络
- [[tedm]] ↔ [[dpm-solver]] — DPM-Solver 提供快速 ODE 求解
- [[tedm]] ↔ [[simdiff]] — SimDiff 是另一个扩散时间序列预测方法
- [[tedm]] ↔ [[informer]] — Informer 是非扩散 Transformer 基线
- [[tedm]] ↔ [[autoformer]] — Autoformer 是分解式 Transformer 基线
- [[tedm]] ↔ [[ltsf-linear]] — LTSF-Linear 是简单线性基线

## 未创建
- [[structured-noise-for-ts]] — 时间序列结构化噪声注入技术（太细粒度，在 tedm.md 内解释即可）
- [[diffusion-physical-time-alignment]] — 扩散-物理时间轴对齐概念（当前仅 TEDM 使用，暂不独立成页）
