# Ingest 报告：Stochastic Interpolants / InterFlow

## 创建
- **wiki/source-stochasticinterpolants.md** — WHY: ICLR 2023 论文 source-summary；slug=`src-stochasticinterpolants`；覆盖随机插值、二次目标 $G$、OT max-min、$W_2$ 界、score 对偶与实验
- **wiki/stochastic-interpolant.md** — WHY: 核心概念页——任意 $\rho_0,\rho_1$ 的有限时间插值过程与概率流速度变分刻画
- **wiki/interflow.md** — WHY: 方法/技术实体页——仿真无关二次训练 + ODE 采样/似然的生成模型管线与基准结果

## 修改
- **wiki/index.md** — WHY: 登记 source-stochasticinterpolants、stochastic-interpolant、interflow
- **wiki/log.md** — WHY: 记录 ingest 操作
- **wiki/flow-matching.md** — WHY: 同期 SI/InterFlow 对照、SB 段补充 max-min OT 桥接、相关链接与脚注；source_count 5→6
- **wiki/rectified-flow.md** — WHY: 与 SI 的 reflow vs max-min 对比表与脚注；source_count 2→3
- **wiki/continuous-normalizing-flow.md** — WHY: 应用段补充仿真无关速度学习；链接 InterFlow/SI；source_count 2→3
- **wiki/optimal-transport.md** — WHY: 动态 OT 与 SI max-min 桥接；相关链接；source_count 3→4
- **wiki/benamou-brenier-algorithm.md** — WHY: 生成模型意义段连接 Proposition 2；链接；source_count 1→2
- **wiki/building-schrodinger-bridges.md** — WHY: §6 Stochastic Interpolants 回链原论文与概念页；source_count 1→2
- **wiki/source-flow-matching.md** — WHY: 意义段补充同期 SI 工作交叉链接
- **wiki/source-rectified-flow.md** — WHY: 与 SI 关系条目交叉链接

## 新建交叉链接
- [[source-stochasticinterpolants]] ↔ [[stochastic-interpolant]] ↔ [[interflow]]
- [[source-stochasticinterpolants]] ↔ [[source-flow-matching]] / [[source-rectified-flow]]
- [[stochastic-interpolant]] ↔ [[flow-matching]] / [[rectified-flow]] / [[continuous-normalizing-flow]] / [[optimal-transport]] / [[benamou-brenier-algorithm]] / [[building-schrodinger-bridges]]
- [[interflow]] ↔ [[flow-matching]] / [[rectified-flow]] / [[score-based-sde]]

## 未创建（已有足够覆盖）
- 未新建独立 `probability-current` 页：电流 $j_t$ 已在 stochastic-interpolant 正文定义
- 未新建独立 `trigonometric-interpolant` 页：作为 SI 默认实例在概念页说明
- 未修改 raw/：按不可变策略只读外部 PDF 完成 ingest

## 源文件
- 外部路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/StochasticInterpolants_Albergo_2023_ICLR.pdf`
- arXiv:2209.15571v3 (9 Mar 2023)；ICLR 2023
- 作者：Michael S. Albergo, Eric Vanden-Eijnden
