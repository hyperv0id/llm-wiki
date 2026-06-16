# Ingest 报告：DiffusionBlocks (2506.14202)

**日期**: 2026-06-16  
**来源**: DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation (Makoto Shing, Masanori Koyama, Takuya Akiba; Sakana AI, University of Tokyo; ICLR 2026)

## 创建的页面

### 1. wiki/source-diffusionblocks.md
**WHY**: 论文源文件摘要页面，记录核心贡献、实验结果和理论洞察

### 2. wiki/diffusionblocks.md
**WHY**: 实体页面，DiffusionBlocks 作为一个具体方法框架需要独立实体页

### 3. wiki/block-wise-training.md
**WHY**: 重要概念页面，论文提出的块级训练范式是深度学习训练方法的新方向，需要概念页面对比历史方法（Forward-Forward、Greedy Layer-wise）与新方法

### 4. wiki/residual-connections-as-diffusion.md
**WHY**: 核心理论概念页面，残差连接与扩散过程的对应关系是 DiffusionBlocks 的理论基础，这一洞察具有独立的理论价值

### 5. wiki/equi-probability-noise-partitioning.md
**WHY**: 关键技术页面，均概率噪声划分是 DiffusionBlocks 成功的技术关键，实验证明其显著优于均匀划分

### 6. wiki/memory-efficient-training.md
**WHY**: 概念综述页面，需要将 DiffusionBlocks、激活检查点、混合精度等内存优化技术进行系统对比

### 7. wiki/activation-checkpointing.md
**WHY**: 技术页面，论文详细对比了激活检查点与 DiffusionBlocks，需要独立页面说明其机制、优缺点和组合方式

## 修改的页面

### 1. wiki/edm.md
**WHY**: DiffusionBlocks 基于 EDM 框架（对数正态噪声分布、损失加权），需要在 EDM 页面添加交叉引用

### 2. wiki/dit.md
**WHY**: DiT 是 DiffusionBlocks 实验中的主要测试架构之一（ImageNet 实验），需要在 DiT 影响列表中添加 DiffusionBlocks

### 3. wiki/index.md
**WHY**: 添加新创建的所有页面到索引的相应分类中（Sources、Entities、Concepts、Techniques）

### 4. wiki/log.md
**WHY**: 记录本次 ingest 操作的时间、创建/更新的页面列表

## 新建交叉链接

本次 ingest 建立的主要链接关系：

- [[diffusionblocks]] ↔ [[block-wise-training]]
- [[diffusionblocks]] ↔ [[residual-connections-as-diffusion]]
- [[diffusionblocks]] ↔ [[equi-probability-noise-partitioning]]
- [[diffusionblocks]] ↔ [[memory-efficient-training]]
- [[diffusionblocks]] ↔ [[activation-checkpointing]]
- [[diffusionblocks]] ↔ [[edm]]
- [[diffusionblocks]] ↔ [[dit]]
- [[residual-connections-as-diffusion]] ↔ [[neural-ordinary-differential-equation]]
- [[residual-connections-as-diffusion]] ↔ [[probability-flow-ode]]
- [[equi-probability-noise-partitioning]] ↔ [[edm]]
- [[memory-efficient-training]] ↔ [[activation-checkpointing]]
- [[block-wise-training]] ↔ [[memory-efficient-training]]

## 质量评估

**论文质量**: ⭐⭐⭐⭐⭐ (非常高)

理由：
1. **理论扎实**: 基于扩散模型理论提供了块级训练的第一个原则性框架
2. **实验全面**: 覆盖 5 种架构（ViT、DiT、AR、Masked Diffusion、Recurrent-depth）和多个任务
3. **结果显著**: 在保持性能的同时实现 B× 内存减少，部分情况甚至优于端到端训练
4. **影响深远**: 为大规模模型训练的民主化提供实用工具，理论洞察（残差=扩散步）具有普遍价值

**覆盖广度**: 论文涉及扩散模型、Transformer、内存优化、训练方法等多个领域，与 wiki 现有内容（EDM、DiT、Neural ODE）有良好连接。
