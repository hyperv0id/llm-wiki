# Ingest 报告：DynaMix

## 创建
- wiki/source-dynamix.md — WHY：论文源文件摘要，NeurIPS 2025 首个零样本 DSR 基础模型
- wiki/dynamix.md — WHY：DynaMix 核心实体页面，涵盖架构（MoE+AL-RNN+STF）、训练、评估和消融
- wiki/dynamical-systems-reconstruction.md — WHY：DSR 概念页面，wiki 此前缺少对动力系统重建这一领域的系统定义
- wiki/almost-linear-rnn.md — WHY：AL-RNN 技术页面，DynaMix 的专家基础单元，此前无独立页面
- wiki/sparse-teacher-forcing.md — WHY：STF 训练方法页面，DynaMix 消融中最为关键的训练组件

## 修改
- wiki/chronos.md — WHY：新增 DynaMix 对比章节，记录 Chronos 在 DSR 任务上的系统性失败（长期动力学坍缩、context parroting、Lyapunov 指数偏差）
- wiki/timesfm.md — WHY：新增 DynaMix 对比章节，记录 TimesFM 的 DSR 失败和多变量耦合缺失
- wiki/mixture-of-experts.md — WHY：新增 DynaMix 在动力系统重建中的应用章节，扩展 MoE 概念到 DSR 领域
- wiki/index.md — WHY：注册所有新页面到索引
- wiki/log.md — WHY：记录 Ingest 操作

## 新建交叉链接
- [[dynamix]] ↔ [[almost-linear-rnn]]
- [[dynamix]] ↔ [[sparse-teacher-forcing]]
- [[dynamix]] ↔ [[dynamical-systems-reconstruction]]
- [[dynamix]] ↔ [[mixture-of-experts]]
- [[dynamix]] ↔ [[chronos]]
- [[dynamix]] ↔ [[timesfm]]
- [[chronos]] ↔ [[source-dynamix]]
- [[timesfm]] ↔ [[source-dynamix]]
- [[mixture-of-experts]] ↔ [[source-dynamix]]
- [[dynamical-systems-reconstruction]] ↔ [[source-dynamix]]
