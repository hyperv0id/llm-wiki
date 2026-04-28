# Ingest 报告：mfrl ch1/ch2/ch7 deep ingest

## 创建
- wiki/source-chapter-1-basic-concepts.md — WHY：为第 1 章建立可引用的章节级来源，沉淀 MDP 与任务设定事实。
- wiki/source-chapter-2-state-values-and-bellman-equation.md — WHY：为值函数与贝尔曼方程建立章节级来源。
- wiki/source-chapter-7-temporal-difference-methods.md — WHY：为 TD/Sarsa/Q-learning 建立章节级来源。
- wiki/mdp-formal-definition.md — WHY：补齐强化学习形式化建模页面。
- wiki/policy-evaluation.md — WHY：将“给定策略求值”从贝尔曼页面中独立成方法页。
- wiki/action-value-function.md — WHY：补齐动作值函数定义与算法关联。
- wiki/sarsa-algorithm.md — WHY：补齐 on-policy 代表算法专题页。
- wiki/expected-sarsa.md — WHY：补齐 Sarsa 低方差变体专题页。
- wiki/n-step-sarsa.md — WHY：补齐 TD/MC 统一桥梁页面。
- wiki/q-learning-algorithm.md — WHY：补齐 off-policy 代表算法专题页。
- wiki/on-policy-vs-off-policy.md — WHY：增加策略范式对照分析。

## 修改
- wiki/bellman-equation.md — WHY：加入第 2 章矩阵形式、闭式解与策略评估上下文。
- wiki/temporal-difference-learning.md — WHY：加入第 7 章统一更新视角与算法分化。
- wiki/grid-world-environment.md — WHY：补充第 1 章中的奖励设定语义。
- wiki/rl-learning-path-mfrl.md — WHY：把学习路径细化到 Ch1/2/7 的页面级节点。
- wiki/math-foundation-of-reinforcement-learning.md — WHY：补充本书主线的章节级深度链接。
- wiki/index.md — WHY：登记新增 Source/Concept/Technique/Analysis 页面。
- wiki/log.md — WHY：追加深度 ingest 活动记录。

## 新建交叉链接
- [[source-chapter-1-basic-concepts]] ↔ [[mdp-formal-definition]]
- [[source-chapter-2-state-values-and-bellman-equation]] ↔ [[bellman-equation]] ↔ [[policy-evaluation]]
- [[source-chapter-7-temporal-difference-methods]] ↔ [[temporal-difference-learning]]
- [[action-value-function]] ↔ [[sarsa-algorithm]] / [[expected-sarsa]] / [[n-step-sarsa]] / [[q-learning-algorithm]]
- [[on-policy-vs-off-policy]] ↔ [[sarsa-algorithm]] ↔ [[q-learning-algorithm]]
- [[rl-learning-path-mfrl]] ↔ 上述所有关键节点
