# Ingest 报告：CSDI + PriSTI (2026-05-31)

## 创建
- [[source-csdi]] — WHY：CSDI (NeurIPS 2021) 论文的 source-summary 页面，记录首个条件扩散+时间序列插补方法的核心论点、自监督训练策略、双轴 Transformer 架构和关键结果
- [[csdi]] — WHY：CSDI 技术页面，详细讲解条件扩散形式化、四种自监督策略、双轴注意力架构规格、与 DDPM/TimeGrad/DiffWave 的继承关系、CRPS/MAE 性能和局限性
- [[source-pristi]] — WHY：PriSTI (ICDE 2023) 论文的 source-summary 页面，记录先验引导条件扩散时空插补框架的核心论点、条件信息增强、先验引导注意力、虚拟节点降采样和实验
- [[pristi]] — WHY：PriSTI 技术页面，详细讲解 CSDI 的两个关键缺陷（忽略空间信息、条件信息使用粗糙）、线性插值增强+先验引导注意力的分离式设计、消融实验、与 CSDI/DiffSTG/SpecSTG 的对比

## 修改
- [[index]] — 在 Sources 添加 [[source-csdi]] 和 [[source-pristi]]，在 Techniques 添加 [[csdi]] 和 [[pristi]]
- [[log]] — 追加 CSDI + PriSTI integration ingest 条目
- [[cofill]] — 将 CSDI 和 PriSTI 纯文本提及替换为 [[csdi|CSDI]] 和 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[specstg]] — 将 PriSTI 纯文本提及替换为 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[traffic-forecasting]] — 将 PriSTI 纯文本提及替换为 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[imputeformer]] — 将 CSDI 和 PriSTI 纯文本提及替换为 [[csdi|CSDI]] 和 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[freqflow-ts]] — 将 PriSTI 纯文本提及替换为 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[gsli]] — 将 PriSTI 纯文本提及替换为 [[pristi|PriSTI]] wikilink；更新 last_updated
- [[generative-time-series-forecasting]] — 添加 CSDI 和 PriSTI 作为扩散+时序插补路线的代表工作；更新 last_updated

## 新建交叉链接
- [[csdi]] ↔ [[ddpm]] — CSDI 将 DDPM 的 ε-prediction 直接扩展到条件形式 ε_θ(x_t^ta, t | x_0^co)
- [[csdi]] ↔ [[timegrad]] — 同期扩散+时序工作：TimeGrad 做预测（RNN 隐状态为条件），CSDI 做插补（观测值为条件）
- [[csdi]] ↔ [[cofill]] — CoFILL 将 CSDI 的单流架构扩展为时域+频域双流 Cross-Attention
- [[csdi]] ↔ [[diffstg]] — CSDI 设为 diffstg 关联页面
- [[pristi]] ↔ [[csdi]] — PriSTI 直接继承和改进 CSDI 的条件扩散框架，解决空间信息缺失和条件信息噪声问题
- [[pristi]] ↔ [[cofill]] — CoFILL 在 PriSTI 基础上添加频域处理和双流架构
- [[pristi]] ↔ [[specstg]] — SpecSTG 在局限性对比中提及 PriSTI
- [[pristi]] ↔ [[diffstg]] — 同为条件扩散+空间建模，任务不同（插补 vs 预测）
- [[pristi]] ↔ [[traffic-forecasting]] — PriSTI 在交通数据和空气质量数据上验证
- [[pristi]] ↔ [[imputeformer]] — ImputeFormer 相关工作分类中提及 PriSTI 为扩散模型路线代表
- [[pristi]] ↔ [[gsli]] — GSLI 实验结果中对比 PriSTI
- [[pristi]] ↔ [[freqflow-ts]] — FrèqFlow 实验结果中对比 PriSTI

## 未创建
- (无 — 所有必要页面已在 prior subagents 中创建)
