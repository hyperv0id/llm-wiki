# Ingest 报告：CBSA — Towards Interpretable and Efficient Attention

**日期**：2026-04-29
**源文件**：`Wen 等 - 2025 - Towards Interpretable and Efficient Attention Compressing All by Contracting a Few.pdf`
**会议**：NeurIPS 2025 (Spotlight)

## 创建

### Source Summary
- **[[source-cbsa]]** — WHY：NeurIPS 2025 Spotlight 论文，首次通过算法展开统一可解释性与高效率注意力机制

### Entities
- **[[cbsa]]** — WHY：核心贡献——Contract-and-Broadcast Self-Attention，通过压缩少数代表实现线性复杂度
- **[[cbt]]** — WHY：基于 CBSA 构建的完整 Transformer 架构，验证可解释性与效率
- **[[crate-white-box-transformer]]** — WHY：CRATE 是 CBSA/MSSA 的前身，white-box transformer 先驱工作

### Concepts
- **[[algorithm-unrolling]]** — WHY： CBSA 论文的核心方法论，将优化目标梯度步骤展开为网络层
- **[[mcr2]]** — WHY：MCR² (Maximal Coding Rate Reduction) 是 CBSA/MSSA 的优化目标基础
- **[[coding-rate]]** — WHY：编码率是 MCR² 目标的核心度量，��于量化 token 紧凑程度
- **[[union-of-subspaces-model]]** — WHY：MCR² 和 CBSA 的数据假设前提

### Techniques
- **[[contract-and-broadcast-mechanism]]** — WHY： CBSA 的核心两阶段操作机制
- **[[representative-token-extraction]]** — WHY： CBSA 实现高效压缩的关键——通过 cross-attention 提取代表性 token

## 修改

- **[[index]]** — 添加 12 个新页面到 Sources、Entities、Concepts、Techniques 类别
- **[[log]]** — 添加 ingest 记录

## 新建交叉链接

- **[[cbsa]]** ↔ **[[mcr2]]**：CBSA 从 MCR² 目标通过算法展开导出
- **[[cbsa]]** ↔ **[[algorithm-unrolling]]**：CBSA 是算法展开的典型应用
- **[[cbsa]]** ↔ **[[coding-rate]]**：压缩目标基于编码率
- **[[cbsa]]** ↔ **[[representative-token-extraction]]**：代表提取是 CBSA 的第一步
- **[[cbsa]]** ↔ **[[contract-and-broadcast-mechanism]]**：CBSA 的核心机制
- **[[cbt]]** ↔ **[[cbsa]]**：CBT 使用 CBSA 作为 token mixer
- **[[cbt]]** ↔ **[[crate-white-box-transformer]]**：CRATE+CBT 混合模型
- **[[crate-white-box-transformer]]** ↔ **[[mcr2]]**：CRATE 基于 MCR² 目标
- **[[crate-white-box-transformer]]** ↔ **[[algorithm-unrolling]]**：CRATE 通过算法展开导出
- **[[mcr2]]** ↔ **[[union-of-subspaces-model]]**：MCR² 假设数据来自子空间并集

## 与现有页面的链接

### 注意力机制类页面
- **[[cbsa]]** → **[[attention-logit-explosion]]**：对比 CBSA 与 QUEST 在训练稳定性上的不同方法
- **[[cbsa]]** → **[[attention-entropy-collapse]]**：对比 CBSA 与 QUEST 在注意力分布上的不同方法
- **[[cbsa]]** → **[[key-normalization]]**：对比 CBSA（子空间压缩）与 QUEST（键归一化）的稳定性机制
- **[[cbsa]]** → **[[spurious-patterns.md]]**：对比 CBSA 与 QUEST 在虚假模式学习上的不同方法
- **[[cbsa]]** → **[[attention-dilution]]**：对比 CBSA（固定代表数线性复杂度）与标准注意力的 O(N²) 复杂度

## 技术亮点

1. **统一框架**：首次揭示 softmax/linear/channel/agent attention 可以统一为 CBSA 的不同代表结构实例
2. **白盒设计**：每层操作对应优化目标的梯度步骤，完全可解释
3. **线性复杂度**：固定代表数 m 时复杂度为 O(N)，突破标准注意力 O(N²) 瓶颈
4. **实证验证**：ImageNet 分类、ADE20K 语义分割均取得 SOTA 或可比结果

## 局限性记录

- 子空间并集假设可能不适用于所有模态（论文排除 NLP 任务）
- 从预训练 ViT 适配时性能略低于 linear attention
- 早期层出现"解压缩"现象的原因不明