# Ingest 报告：math-foundation-rl

## 创建
- wiki/source-math-foundation-rl-readme.md — WHY：摘要《Mathematical Foundations of Reinforcement Learning》Readme 源文件，记录教材核心主张、内容结构、目标读者和配套资源
- wiki/source-grid-world-code-readme.md — WHY：摘要网格世界环境代码说明源文件，记录 Python 和 MATLAB 版本的用法和可视化功能
- wiki/math-foundation-of-reinforcement-learning.md — WHY：创建教材实体页面，记录其内容概述、涵盖主题和配套资源
- wiki/shiyu-zhao.md — WHY：创建作者实体页面，记录其学术背景和贡献
- wiki/grid-world-environment.md — WHY：创建网格世界环境实体页面，记录环境描述、代码实现和教学用途
- wiki/bellman-equation.md — WHY：创建贝尔曼方程技术页面，记录其作为 RL 核心理论基础的定义和求解方法
- wiki/temporal-difference-learning.md — WHY：创建时序差分学习技术页面，记录 Sarsa、Q-learning 等算法及其统一视角
- wiki/rl-learning-path-mfrl.md — WHY：创建分析页面，基于教材内容设计系统性 RL 学习路径

## 修改
- wiki/index.md — WHY：在 Sources、Entities、Techniques、Analyses 小节追加新页面条目
- wiki/log.md — WHY：追加 ingest 活动记录

## 新建交叉链接
- [[source-math-foundation-rl-readme]] ↔ [[source-grid-world-code-readme]]（同一教材的配套资源）
- [[shiyu-zhao]] ↔ [[math-foundation-of-reinforcement-learning]]（作者 ↔ 著作）
- [[shiyu-zhao]] ↔ [[grid-world-environment]]（作者 ↔ 代码）
- [[math-foundation-of-reinforcement-learning]] ↔ [[grid-world-environment]]（教材 ↔ 示例环境）
- [[math-foundation-of-reinforcement-learning]] ↔ [[bellman-equation]]（教材 ↔ 核心技术）
- [[math-foundation-of-reinforcement-learning]] ↔ [[temporal-difference-learning]]（教材 ↔ 核心技术）
- [[bellman-equation]] ↔ [[temporal-difference-learning]]（理论基础 ↔ 算法实现）
- [[rl-learning-path-mfrl]] ↔ 所有上述页面（学习路径 ↔ 各知识节点）