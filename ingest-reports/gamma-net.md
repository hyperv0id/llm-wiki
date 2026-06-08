# Ingest Report: GAMMA-Net (arXiv 2604.16859)

## 创建
- **wiki/source-gamma-net.md** — WHY：摘要 arXiv 2026 GAMMA-Net 论文，交错式 GAT-Mamba 时空预测，SOTA 于 6 个基准，达到 16.25% MAE 降幅
- **wiki/gamma-net.md** — WHY：首个以交替闭环将 GAT 和双轴 Mamba 结合的时空交通预测实体页，突破三难困境（长时记忆+自适应图推理+轻量部署）
- **wiki/interleaved-gat-mamba.md** — WHY：核心技术贡献——(GAT → Mamba_Temporal) → (GAT → Mamba_Spatial) 的交错设计是区分 GAMMA-Net 于所有先前混合模型的关键

## 修改
- **wiki/s-mamba.md** — WHY：添加 GAMMA-Net 交叉引用和 Mamba 家族演进表，GAMMA-Net 代表了 Mamba 在交通预测中的最新里程碑（交错闭环 vs. 纯 Mamba/分解 Mamba）
- **wiki/stgcn.md** — WHY：添加 GAMMA-Net 交叉引用和演进链（STGCN→GWNet→GAMMA-Net），GAMMA-Net 的 GAT 组件直接回应 STGCN"图结构时不变"的局限
- **wiki/index.md** — WHY：注册新 source-summary、entity、technique 页面
- **wiki/log.md** — WHY：记录 2026-06-08 ingest 操作

## 新建交叉链接
- [[gamma-net]] ↔ [[interleaved-gat-mamba]]
- [[gamma-net]] ↔ [[s-mamba]]
- [[gamma-net]] ↔ [[stgcn]]
- [[gamma-net]] ↔ [[mamba]]
- [[gamma-net]] ↔ [[dst-mamba]]
- [[gamma-net]] ↔ [[gwnet]]
- [[gamma-net]] ↔ [[dcrnn]]
- [[gamma-net]] ↔ [[traffic-forecasting]]
- [[interleaved-gat-mamba]] ↔ [[s-mamba]]
- [[interleaved-gat-mamba]] ↔ [[stgcn]]
- [[stgcn]] ↔ [[interleaved-gat-mamba]]

## 关键决策
1. **技术页而非概念页**：交错式 GAT-Mamba 是一个具体的架构设计模式（明确的操作序列、消融验证的顺序必要性），适合 `type: technique` 而非 `type: concept`。
2. **不可创建独立概念页**：GAMMA-Net 使用的 GAT 和 Mamba 均为已有技术；其核心贡献在于交错排列方式而非新基础算子，因此仅需一个技术页即可捕捉其独特性。
3. **confidence: medium**：单一 arXiv 预印本来源，代码未公开，论文状态未经同行评审最终确认。
4. **2026-06-08 时间戳**：所有新页面的 `created` 和 `last_updated` 使用当前日期。
