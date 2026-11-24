# Ingest 报告：mixlinear

- **分级**：Tier 2（将下采样后的二维分段双线性分解与频域秩-2 复投影相加压至 45–176 参数，但主表仅报 MSE 不报 MAE、缺 iTransformer/TimeMixer/TimeBase 等同生态位基线对照，且仅 2 组简单消融；按轻量化标准不建 `wiki/mixlinear.md`）

## 创建
- `wiki/source-mixlinear.md` — WHY：按 Tier 2 轻量化入库，补齐人话大义、去黑话双域算子配方、按低维/高维分化的消融真相与只报 MSE/缺基线/参数削减口径出入等缺口。

## 修改
- `wiki/sparsetsf.md` — WHY：补充与 MixLinear 的瓶颈接力 $\Delta$（下采样骨架加宽为 2D 分段双线性 + 频域秩-2 支）与 Exchange 复现提速对照，刷新 `source_count`（4→5）与 `last_updated`。
- `wiki/source-sparsetsf.md` — WHY：补充 MixLinear 沿用下采样骨架的瓶颈接力 $\Delta$ 与主表缺 MAE 口径对照，刷新 `source_count`（1→2）与 `last_updated`。
- `wiki/source-timebase.md` — WHY：补充同在 0.1K 参数极轻量槽位的互斥对照（正交约束 $R=6$ 时间基 vs 时频双低秩分支），刷新 `source_count`（1→2）与 `last_updated`。
- `wiki/cyclenet.md` — WHY：补充与 MixLinear 的同槽位互斥对照（全局可学习周期表查表残差 vs 下采样时频双低秩分支），刷新 `source_count`（3→4）与 `last_updated`。
- `wiki/ltsf-linear.md` — WHY：补充极低参数因子化演进对照（$O(LH)$ 稠密线性压缩至 0.1K 因子化表示），刷新 `source_count`（5→6）与 `last_updated`。
- `wiki/source-frets.md` — WHY：补充与 MixLinear 的跨槽位正交对照（全谱复数 MLP 与通道学习器 vs 下采样时域双线性 + 低秩频谱），刷新 `source_count`（1→2）与 `last_updated`。
- `wiki/index.md` — WHY：在 Sources 段按字母序插入 `[[source-mixlinear]]`。
- `wiki/log.md` — WHY：在文首追加 `[2026-11-24] ingest | MixLinear` 条目（按全库最新条目置顶约定）。

## 新建交叉链接（标明关系类型）
- [[source-mixlinear]] ↔ [[sparsetsf]] · [[source-sparsetsf]]（瓶颈接力 Δ 与复现对齐）
- [[source-mixlinear]] ↔ [[source-timebase]]（同槽位互斥）
- [[source-mixlinear]] ↔ [[cyclenet]]（同槽位互斥）
- [[source-mixlinear]] ↔ [[ltsf-linear]]（演进对照）
- [[source-mixlinear]] ↔ [[source-frets]]（跨槽位正交）
