# Ingest 报告：QUEST - A Robust Attention Formulation

## 创建

- **wiki/source-quest.md** — WHY：论文 source-summary，核心贡献是 QUEST 注意力机制
- **wiki/quest-attention.md** — WHY：QUEST 实体页，详细定义公式和实验结果
- **wiki/key-normalization.md** — WHY：键归一化技术页，QUEST 的核心机制
- **wiki/attention-logit-explosion.md** — WHY：训练不稳定问题页
- **wiki/attention-entropy-collapse.md** — WHY：注意力熵崩溃问题页
- **wiki/spurious-patterns-in-attention.md** — WHY：虚假模式问题页

## 修改

- **wiki/index.md** — 添加 6 个新页面到 Sources/Entities/Techniques
- **wiki/log.md** — 记录本次 ingest

## 新建交叉链接

- [[quest-attention]] ↔ [[key-normalization]]
- [[quest-attention]] ↔ [[attention-logit-explosion]]
- [[quest-attention]] ↔ [[attention-entropy-collapse]]
- [[quest-attention]] ↔ [[spurious-patterns-in-attention]]

## 核心创新点

1. **仅归一化键**：消除键范数的"窃取"效应，保持查询独立控制锐度
2. **Q-K 交叉依赖分析**：首次形式化解释训练不稳定的根源
3. **虚假模式实验**：设计玩具实验证明标准注意力易陷入次优解
4. **广泛适用性**：视觉、语言、图神经网络、点云、时序均验证