# Ingest 报告：数学直觉系列（一）：缩放因子 1/√dₖ

## 创建
- wiki/source-bluuuuue-scaling-factor-intuition.md — WHY：源文件摘要，记录核心论点（方差膨胀→Softmax饱和→梯度消失→缩放归一化）
- wiki/scaling-factor-sqrt-dk.md — WHY：核心概念页面，系统整理缩放因子的数学原理、三个特性、与现有概念的关系

## 修改
- wiki/attention-entropy-collapse.md — WHY：添加"根本原因：点积方差膨胀"章节，从数学根源解释熵崩溃，链接到缩放因子概念页，source_count 1→2
- wiki/attention-logit-explosion.md — WHY：在现象描述中链接缩放因子为预防 logit 爆炸的第一道防线，说明缩放因子不足以完全控制训练中范数增长，source_count 1→2
- wiki/attention-temperature-scaling.md — WHY：说明温度缩放与缩放因子在数学上同构，温度缩放是缩放因子在 RoPE 上下文扩展场景下的推广，source_count 1→2
- wiki/key-normalization.md — WHY：说明缩放因子（控制初始化阶段方差）与键归一化（约束训练中范数增长）互补，source_count 1→2

## 新建交叉链接
- [[scaling-factor-sqrt-dk]] ↔ [[attention-entropy-collapse]] — 缩放因子预防熵崩溃
- [[scaling-factor-sqrt-dk]] ↔ [[attention-logit-explosion]] — 缩放因子控制 logit 尺度
- [[scaling-factor-sqrt-dk]] ↔ [[attention-temperature-scaling]] — 数学同构：两者都是 Softmax 前线性缩放
- [[scaling-factor-sqrt-dk]] ↔ [[key-normalization]] — 互补：缩放因子控方差 vs 键归一化控范数
