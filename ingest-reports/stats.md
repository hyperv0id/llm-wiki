# Ingest 报告：StaTS (arXiv 2603.00037)

## 创建
- [[stats]] — WHY：StaTS 实体页面，描述联合频谱轨迹调度学习+频率引导去噪的扩散预测框架
- [[source-stats]] — WHY：源文件摘要，记录论文核心论点、方法、结果和贡献
- [[spectral-trajectory-scheduler]] — WHY：STS 技术页面，描述通过频域正则化 PGD 学习自适应噪声调度的机制，包含优化目标、收敛性定理和学习到的非单调调度模式
- [[frequency-guided-denoiser]] — WHY：FGD 技术页面，描述通过多频带频谱失真估计调制去噪强度的机制及与 STS 的协同

## 修改
- [[diffusion-models]] — WHY：添加 StaTS 作为时序扩散模型新方法，以频谱轨迹调度学习推进固定调度到自适应调度
- [[nsdiff]] — WHY：添加 StaTS 作为 NsDiff 的后续超越者，记录 CRPS 和 MAE 全胜结果
- [[timegrad]] — WHY：添加 StaTS 作为从 TimeGrad 固定调度到自适应调度演进的证据

## 新建交叉链接
- [[stats]] ↔ [[nsdiff]]：StaTS 在所有基准上超越 NsDiff
- [[stats]] ↔ [[timegrad]]：从固定到自适应调度的演进路径
- [[stats]] ↔ [[spectral-trajectory-scheduler]]：STS 是 StaTS 核心组件
- [[stats]] ↔ [[frequency-guided-denoiser]]：FGD 是 StaTS 核心组件
- [[diffusion-models]] ↔ [[stats]]：扩散模型在时序预测中的应用进展
