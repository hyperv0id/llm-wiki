# Ingest 报告：Vetcha 2026 - Towards Infinite Length Extrapolation

## 创建

- **wiki/source-vetcha-2026-towards-infinite-length-extrapolation.md** — WHY：论文 source-summary，核心贡献是统一位置编码框架和 APE 方法
- **wiki/adaptive-positional-encoding.md** — WHY：APE 实体页，详细定义公式和理论性质
- **wiki/generalized-positional-encoding-framework.md** — WHY：GPE 概念页，统一 RoPE/ALiBi 的理论框架
- **wiki/convergent-normalization.md** — WHY：无限外推的关键技术条件之一
- **wiki/entropy-boundedness.md** — WHY：无限外推的关键技术条件之一
- **wiki/long-distance-correlation-preservation.md** — WHY：无限外推的关键技术条件之一
- **wiki/gradient-positional-sensitivity.md** — WHY：无限外推的关键技术条件之一
- **wiki/long-tiny-stories-dataset.md** — WHY：新数据集实体页，用于长上下文评估

## 修改

- **wiki/index.md** — 添加 8 个新页面到 Sources/Entities/Concepts/Techniques
- **wiki/log.md** — 记录本次 ingest
- **wiki/position-extrapolation.md** — 添加 APE 到外推能力对比表，更新 source_count
- **wiki/context-window-extension.md** — 添加 APE 到方法分类，更新 source_count

## 新建交叉链接

- [[position-extrapolation]] ↔ [[adaptive-positional-encoding]]
- [[position-extrapolation]] ↔ [[convergent-normalization]]
- [[context-window-extension]] ↔ [[adaptive-positional-encoding]]
- [[context-window-extension]] ↔ [[generalized-positional-encoding-framework]]
- [[alibi]] ↔ [[adaptive-positional-encoding]] (通过对比表)
- [[yarn]] ↔ [[generalized-positional-encoding-framework]] (通过 GPE 框架)

## 理论创新点

1. **LDCP vs 收敛归一化矛盾**：首次形式化证明两者不可兼得
2. **四个关键性质**：收敛归一化、熵有界性、LDCP、GPS
3. **APE 次线性衰减**：log + √|n| 项保留更多长程依赖