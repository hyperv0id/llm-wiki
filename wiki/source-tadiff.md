---
title: "TADiff: Counterfactual Time Series Forecasting with Textual Conditions"
type: source-summary
tags:
  - time-series-forecasting
  - counterfactual
  - diffusion-model
  - ddim
  - multimodal-time-series
  - contrastive-learning
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# TADiff: What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions

**来源**：Shuqi Gu、Yongxiang Zhao、Baoyu Jing、Kan Ren（致谢显示为 ShanghaiTech；核心切片未见具体会议年份水印，本仓库 OpenReview CCF-A 调研记录其录用于 ICML 2026）[^src-tadiff]。

同一段历史在「明天世界杯决赛」等不同文本假设下应演化出不同未来，但现实中的反事实条件只能以自然语言表达，无法写成固定类别的干预变量，也缺少真实反事实标签用于监督训练。为此，TADiff 先通过条件感知的前向扩散逆过程，从观测历史中剥离历史文本条件，得到仅保留序列固有演化的隐状态作为初始噪声，再由目标或反事实文本引导确定性去噪生成；对于无真值的反事实样本，则以解耦对比模型 DTTC 分别度量「守住历史固有特征」与「兑现目标外生语义」，直接用该分数微调模型[^src-tadiff]。

## 去黑话算子配方与路线差异

模型基于 [[diffusion-model|DDIM]] 条件去噪框架（核心切片正文记为 DDIM，未展开具体去噪网络结构）[^src-tadiff]：

- **条件剥离前向归因（TA）**：对前向扩散过程逆用概率流转移 $\psi^{-1}$，从观测历史 $x_{h,0}$ 与历史文本 $c_h$ 迭代推回终端隐态 $x_{h,T} = (\psi_{T-1}^{-1}\circ\cdots\circ\psi_0^{-1})(x_{h,0}, c_h)$，剥离历史外生语义以保留固有演化特征[^src-tadiff]。
- **历史回填条件去噪**：令初始隐态 $\hat{x}_{f,T}=x_{h,T}$，拼接历史前缀并在每步将前 $L_h$ 步回填为真实历史，在目标文本 $c_f$ 引导下经 $\psi_1\circ\cdots\circ\psi_T$ 去噪输出未来轨迹[^src-tadiff]。
- **事实监督与 DTTC 反事实微调（CF）**：有真值事实样本用预测损失与归因重建损失加权监督（仅监督未来段掩码）；无真值反事实样本直接最大化解耦对比评分 DTTC 的目标 $-\mathbb{E}[\lambda_I\langle I_h, \hat{I}_f\rangle + \lambda_E\langle \hat{E}_f, \tilde{E}_f\rangle]$[^src-tadiff]。
- **有向关系对照**：相对常规 [[diffusion-model|DDIM]] 条件扩散，TADiff 将初始噪声由 $\mathcal{N}(0, I)$ 换为归因隐态 $x_{h,T}$，推理额外多一条 $T$ 步前向链使采样计算约翻倍；相对同槽位因果生成模型 [[doflow|DoFlow]]，TADiff 不建显式 SCM/DAG，以放弃 [[causal-counterfactual-recovery|外生噪声溯因可识别性]]与单一确定性轨迹为代价换取自由文本条件支持；相对 [[time-mmd|Time-MMD]] 的真实对齐文本与 [[source-whatif-tsf|What If TSF（WIT）]] 的专家手工场景方向基准，TADiff 在无反事实真值的真实序列上采用启发式条件构造并输出连续轨迹[^src-tadiff]。

## 证据

1. **定位与主结论**：Synth 及四真实集（ETTm1/Traffic/Exchange/Weather）上 DTTC 领先 VerbalTS、CT、IATSF 等；事实预测在 Synth、ETTm1、Traffic 多数领先，ETTm1 MAE 近 VerbalTS[^src-tadiff]。
2. **真实涨点归因**：去 TA 掉点最重（Traffic MSE $0.27\to 0.78$）；去 CF 后 Traffic 事实数值反更好（MSE $0.24$ vs $0.27$、MAE $0.31$ vs $0.35$），表明精度源自前向归因初始化，CF 仅提语义一致性[^src-tadiff]。
3. **证据缺口与负结果**：Weather 正文称 MSE 最优，Table 1 实测 IATSF（$0.09$）低于 TADiff（$0.11$，MAE $0.12$ vs $0.15$）；真实集反事实靠启发式构造，DTTC 评测与微调目标同源；采样链翻倍、变长不可靠[^src-tadiff]。

## 相关页面

- [[multimodal-time-series-forecasting]] — 多模态与文本反事实预测对照矩阵（含评价同源提醒）
- [[doflow]] — 同槽位对比（显式 SCM + CNF 溯因可识别 vs 无 SCM 文本扩散）
- [[causal-counterfactual-recovery]] — 外生噪声可识别性口径对照
- [[source-whatif-tsf]] — 同生态位场景基准对照（专家手工场景方向基准 vs 启发式构造与 DTTC）
- [[diffusion-model]] — 机制继承（DDIM 条件去噪改初始归因隐态）
- [[time-mmd]] — 数值–文本对齐设定背景参考

[^src-tadiff]: [[source-tadiff]]
