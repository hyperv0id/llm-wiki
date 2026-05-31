# Ingest 报告：TimeGrad + DiffSTG (2026-05-31)

## 创建
- [[source-timegrad]] — WHY：TimeGrad (ICML 2021) 论文的 source-summary 页面，记录首个扩散+时序预测方法的核心论点、DDPM 条件化架构、关键结果和局限性
- [[timegrad]] — WHY：TimeGrad 实体页面，详细讲解 RNN+DDPM 二段式架构、训练/推理流程、与 DDPM/DeepAR/Vec-LSTM 的关系、6 数据集性能和消融分析
- [[source-diffstg]] — WHY：DiffSTG (AAAI 2023) 论文的 source-summary 页面，记录首个扩散+时空图预测方法的广义条件扩散框架、UGnet 架构、非自回归加速
- [[diffstg]] — WHY：DiffSTG 实体页面，详细讲解问题动机（填补 TimeGrad 缺空间、CSDI 缺预测适配的裂缝）、UGnet 设计规则、噪声调度敏感性、推理加速策略

## 修改
- [[index]] — 在 Sources 添加 [[source-timegrad]] 和 [[source-diffstg]]，在 Entities 添加 [[timegrad]] 和 [[diffstg]]
- [[log]] — 追加两篇论文的 ingest 条目和一条 integration 条目
- [[source-diffstg]] — 将 TimeGrad 纯文本提及替换为 [[timegrad|TimeGrad]] wikilink（2 处）
- [[diffstg]] — 将 TimeGrad 纯文本提及替换为 [[timegrad|TimeGrad]] wikilink（4 处）
- [[specstg]] — 将已有 TimeGrad 和 DiffSTG 纯文本提及转为 wikilink；更新 last_updated
- [[traffic-forecasting]] — 将已有 TimeGrad 和 DiffSTG 纯文本提及转为 wikilink；更新 last_updated
- [[generative-time-series-forecasting]] — 添加 TimeGrad 作为扩散预测路线奠基工作的条目；更新 source_count (4→5)，last_updated

## 新建交叉链接
- [[timegrad]] ↔ [[ddpm]] — TimeGrad 直接继承 DDPM 的 ε-prediction、L_simple、β 调度、Markov 链
- [[timegrad]] ↔ [[diffstg]] — DiffSTG 在问题动机中对比 TimeGrad 的自回归速度瓶颈
- [[timegrad]] ↔ [[generative-time-series-forecasting]] — TimeGrad 是生成式时间序列预测扩散路线的奠基工作
- [[diffstg]] ↔ [[specstg]] — SpecSTG 在局限性中对比 DiffSTG 的空间信息利用不足
- [[diffstg]] ↔ [[traffic-forecasting]] — DiffSTG 是时空图交通预测概率方法的代表性工作
- [[timegrad]] ↔ [[traffic-forecasting]] — TimeGrad 在 traffic-forecasting 中作为原始域扩散方法的代表被提及
- [[diffstg]] ↔ [[generative-time-series-forecasting]] — DiffSTG 通过 specstg 页面间接关联

## 未创建
- (无 — 所有必要页面已在 prior subagents 中创建)
