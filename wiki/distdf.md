---
title: "DistDF"
type: entity
tags:
  - time-series-forecasting
  - direct-forecast
  - learning-objective
  - optimal-transport
  - wasserstein
  - plug-and-play
  - iclr-2026
created: 2026-11-09
last_updated: 2026-11-09
source_count: 3
confidence: high
status: active
---

# DistDF

**DistDF（Distribution-aware Direct Forecast）** 是一套训练目标框架：保留 [[direct-forecast|Direct Forecast (DF)]] 一次前向输出全部 $T$ 步的结构，把逐点 MSE 换成「预测序列与标签序列在给定历史 $x$ 下的条件分布对齐」，并可与 MSE 混合成 plug-and-play 正则项接到任意骨干上。作者 Hao Wang、Licheng Pan、Yuan Lu、Zhixuan Chu、Xiaoxi Li、Shuting He、Zhichao Chen、Qingsong Wen、Haoxuan Li、Zhouchen Lin（小红书、浙江大学、上海财经大学、北京大学、Squirrel AI），ICLR 2026，arXiv:2510.24574v3，代码 `github.com/Master-PLC/DistDF`[^src-distdf]。

## 问题定位

DF 的学习目标是标签序列的条件负对数似然，实践中用 MSE 估计：$\mathcal{L}_{\mathrm{MSE}}=\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_2^2$[^src-distdf]。论文指出该估计在标签序列存在自相关时有偏：MSE 把 $T$ 个未来步当作独立，等价于假设条件协方差 $\Sigma_{\mid x}=I$，而真实标签满足 $y_t$ 依赖 $y_{<t}$，于是 MSE 与条件 NLL 之间存在偏差，即 [[autocorrelation-bias|自相关偏差]]（Theorem 3.1，Eq. 2）[^src-distdf]。

前作的处理路线是先把标签变换到「条件去相关」的分量再逐点算误差：[[fredf|FreDF]] 用 Fourier 变换取频域分量，Time-o1 用 PCA 取主成分。论文承认这类变换能消除偏差的前提是分量真的条件去相关，但 Fourier 与 PCA 只保证边缘去相关（$\Sigma$ 对角），不保证给定 $x$ 后条件去相关（$\Sigma_{\mid x}$ 对角），偏差因此残留（§3.1）[^src-distdf]。FreDF 自身对频域分量的论证也止于边缘层面：正交基下用 DML 测得的频率分量依赖约为 $3.6\%$ 非对角元超过 $0.1$（Weather，$T=192$），对应的是分量间边缘结构，而非条件结构[^src-fredf]。

论文给出一组交通数据上的直接测量：Traffic（$T=192$）原始标签的偏相关矩阵中逾 $50.3\%$ 的非对角元 $|\rho|>0.1$；换成 FreDF 的频率分量、Time-o1 的主成分后，非对角元量级下降但仍留有不可忽略的残余相关（Fig. 1(a)–(c)、Appendix C）[^src-distdf]。偏相关的计算方式是给定历史序列 $x$ 后求未来步两两相关（等价于 MATLAB `partialcorr` 的做法），目的是剔除「$y_t\leftarrow x\rightarrow y_{t'}$」这条共因路径带来的伪相关[^src-distdf]。

## 方法与核心定理

框架目标换成直接最小化条件分布差异：让 $\mathbb{P}_{\hat{y}\mid x}$ 逼近 $\mathbb{P}_{y\mid x}$，绕开似然估计的因式分解[^src-distdf]。直接做法 $\min W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x})$ 无法估计：给定 $x$ 通常只有一条 $y$，模型也只输出一条 $\hat{y}$，两侧经验集各只有一个样本，任何分布差异度量都会退化成无信息量（Footnote 3）[^src-distdf]。

论文的替代量是 [[joint-distribution-wasserstein-alignment|联合分布 Wasserstein 差异]] $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})$，三件性质构成整条论证链：

1. **上界性**（Lemma 3.3，引自 Kim et al. 2022）：对任意 $p\geq 1$，

$$
\int W_p\!\left(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x}\right)d\mathbb{P}(x)\ \leq\ W_p\!\left(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}}\right),
$$

$p=1$ 或条件项对 $x$ 为常数时取等。因此压低右侧就压低了左侧的期望条件差异[^src-distdf]。

2. **可估计性**：联合样本 $(x_b,y_b)$ 与 $(x_b,\hat{y}_b)$ 可从整个数据集构造，样本量不受「单个 $x$ 只有一条 $y$」的限制[^src-distdf]。

3. **零差异即条件分布相等**（Theorem 3.4）：若 $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})=0$，则由 Lemma 3.3 得 $\int W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x})d\mathbb{P}(x)=0$，几乎处处 $W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x})=0$，进而几乎处处 $\mathbb{P}_{y\mid x}=\mathbb{P}_{\hat{y}\mid x}$（Appendix A.3 用离散传输计划证明 $W_p=0\Rightarrow$ 两分布质量函数逐点相等）[^src-distdf]。

落到可微实现时加高斯假设：$\mathbb{P}_{x,y}$ 与 $\mathbb{P}_{x,\hat{y}}$ 分别为 $\mathcal{N}(\mu_{x,y},\Sigma_{x,y})$、$\mathcal{N}(\mu_{x,\hat{y}},\Sigma_{x,\hat{y}})$，平方 $W_2$ 有闭式 [[bures-wasserstein|Bures–Wasserstein]] 解（Lemma 3.5，引自 Peyré & Cuturi 2019 的 Remark 2.31）[^src-distdf]：

$$
\mathrm{BW}=\lVert\mu_{x,y}-\mu_{x,\hat{y}}\rVert_2^2+\mathcal{B}(\Sigma_{x,y},\Sigma_{x,\hat{y}}),\qquad
\mathcal{B}(\Sigma_1,\Sigma_2)=\operatorname{Tr}\!\left(\Sigma_1+\Sigma_2-2\left(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\right)^{1/2}\right).
$$

BW 只对齐一阶与二阶矩，丢弃了 batch 内样本与标签的逐元素对应；因此最终目标是两项加权（Eq. 6、Algorithm 1）：

$$
\mathcal{L}_{\mathrm{DistDF}}=\gamma\,\mathcal{L}_{\mathrm{Dist}}+(1-\gamma)\,\mathcal{L}_{\mathrm{MSE}},\qquad 0\leq\gamma\leq1 .
$$

论文把这一路线与领域自适应（Domain Adaptation）里的联合分布最优传输区分开：领域自适应对齐的是输入侧的边缘分布，服务于跨域泛化；DistDF 对齐的是模型输出与标签的条件分布，属于多任务监督学习下的训练目标设计（§3.2）[^src-distdf]。

## 算法实现与计算开销

给定 batch $X\in\mathbb{R}^{B\times H}$、$Y\in\mathbb{R}^{B\times T}$（论文按单变量书写，多变量把通道维并入 batch/样本轴或逐变量计算），Algorithm 1 的步骤为[^src-distdf]：

1. 骨干前向：$\hat{Y}=g(X)$，一次输出全部 $T$ 步；
2. 沿时间轴拼接：$Z=[X,Y]\in\mathbb{R}^{B\times(H+T)}$、$\hat{Z}=[X,\hat{Y}]\in\mathbb{R}^{B\times(H+T)}$；
3. 在 batch 维估计一阶与二阶矩：均值 $\mu_Z,\mu_{\hat{Z}}\in\mathbb{R}^{H+T}$、样本协方差 $\Sigma_Z,\Sigma_{\hat{Z}}\in\mathbb{R}^{(H+T)\times(H+T)}$；
4. 分布项与逐点项：$\mathcal{L}_{\mathrm{Dist}}=\mathrm{BW}(\mu_Z,\mu_{\hat{Z}},\Sigma_Z,\Sigma_{\hat{Z}})$、$\mathcal{L}_{\mathrm{MSE}}=\lVert Y-\hat{Y}\rVert_2^2$；
5. 混合：$\mathcal{L}_{\mathrm{DistDF}}=\gamma\,\mathcal{L}_{\mathrm{Dist}}+(1-\gamma)\,\mathcal{L}_{\mathrm{MSE}}$；
6. 反向传播：$Z=[X,Y]$ 全为观测常量，$\mu_Z$ 与 $\Sigma_Z^{1/2}$ 不含可学习参数、也不接收梯度；梯度只经依赖模型输出 $\hat{Y}=g(X)$ 的 $\mu_{\hat{Z}}$ 与 $\Sigma_{\hat{Z}}$（即通过 $(\Sigma_Z^{1/2}\Sigma_{\hat{Z}}\Sigma_Z^{1/2})^{1/2}$ 与 $\mathcal{L}_{\mathrm{MSE}}$）回传，按 Adam 更新模型参数[^src-distdf]。

推理阶段只运行 $\hat{Y}=g(X)$，分布项不参与[^src-distdf]。额外计算集中在 $(H+T)\times(H+T)$ 的协方差与矩阵平方根上：论文在 $B=128$、$D=21$、$T=1024$ 下测得训练单步前向 $0.74\text{--}0.80\text{ ms}$、反向 $0.58\text{--}0.64\text{ ms}$，均低于 $1\text{ ms}$（Appendix D.5、Fig. 7）[^src-distdf]。

## 实验证据

设置：ETT、ECL、Weather 为主，$H=96$，$T\in\{96,192,336,720\}$，关闭 drop-last；Adam + 早停（验证损失连续 3 轮不降），骨干超参保留原设定，只调 $\gamma\in(0,1]$ 与学习率 $\eta\in[5\times10^{-5},10^{-3}]$；硬件为 Intel Xeon Platinum 8383C + RTX H100（§4.1）[^src-distdf]。

### 学习目标对比（Table 1）

Table 1 是 4 个 horizon 的均值，单元格为 MSE / MAE：

| 骨干 | 数据集 | DistDF | Time-o1 | FreDF | Koopman | DILATE | Soft-DTW | DF |
|---|---|---|---|---|---|---|---|---|
| TimeBridge | ETTm1 | 0.383 / 0.397 | 0.383 / 0.397 | 0.386 / 0.398 | 0.460 / 0.438 | 0.387 / 0.400 | 0.395 / 0.402 | 0.387 / 0.400 |
| TimeBridge | ETTh1 | 0.434 / 0.436 | 0.439 / 0.438 | 0.439 / 0.436 | 0.459 / 0.449 | 0.464 / 0.452 | 0.452 / 0.445 | 0.442 / 0.440 |
| TimeBridge | ECL | 0.172 / 0.267 | 0.175 / 0.268 | 0.175 / 0.267 | 0.182 / 0.277 | 0.176 / 0.271 | 0.173 / 0.268 | 0.176 / 0.271 |
| TimeBridge | Weather | 0.248 / 0.275 | 0.250 / 0.275 | 0.254 / 0.276 | 0.269 / 0.293 | 0.252 / 0.277 | 0.260 / 0.280 | 0.252 / 0.277 |
| Fredformer | ETTm1 | 0.378 / 0.394 | 0.379 / 0.393 | 0.384 / 0.394 | 0.389 / 0.400 | 0.389 / 0.400 | 0.397 / 0.402 | 0.387 / 0.398 |
| Fredformer | ETTh1 | 0.430 / 0.429 | 0.431 / 0.429 | 0.438 / 0.434 | 0.452 / 0.443 | 0.453 / 0.442 | 0.460 / 0.449 | 0.447 / 0.434 |
| Fredformer | ECL | 0.173 / 0.266 | 0.178 / 0.270 | 0.179 / 0.272 | 0.190 / 0.282 | 0.187 / 0.280 | 0.206 / 0.298 | 0.191 / 0.284 |
| Fredformer | Weather | 0.255 / 0.277 | 0.255 / 0.276 | 0.256 / 0.277 | 0.257 / 0.279 | 0.258 / 0.280 | 0.261 / 0.280 | 0.261 / 0.282 |

在 2 个骨干 × 4 个数据集的 8 组 4-horizon 均值对比中，DistDF 的 MSE 全部最低（完整单步结果见 Appendix Table 7），FreDF 与 Time-o1 两类似然变换目标次之，DILATE 与 Soft-DTW 等形状对齐目标相对于纯 MSE 提升有限[^src-distdf]。

### 均值与协方差消融（Table 2）

| 数据集 | DF | $\text{DistDF}^\dagger$（仅均值） | $\text{DistDF}^\ddagger$（仅协方差） | DistDF |
|---|---|---|---|---|
| ETTm1 | 0.387 / 0.398 | 0.381 / 0.396 | 0.386 / 0.399 | 0.379 / 0.395 |
| ETTh1 | 0.447 / 0.434 | 0.435 / 0.430 | 0.439 / 0.437 | 0.430 / 0.429 |
| ECL | 0.176 / 0.271 | 0.175 / 0.269 | 0.174 / 0.269 | 0.172 / 0.267 |
| Weather | 0.252 / 0.277 | 0.251 / 0.277 | 0.251 / 0.277 | 0.248 / 0.275 |

仅对齐均值（$\mathrm{DistDF}^\dagger$）与仅对齐协方差（$\mathrm{DistDF}^\ddagger$）单独接入时均降低 MSE，两项组合的完整 DistDF 在四个数据集上 MSE 与 MAE 均最低[^src-distdf]。

### 分布差异度量对比（Table 3）

| 骨干 | 数据集 | Ours（BW） | EMD | MMD@Linear | MMD@RBF | KL | DF |
|---|---|---|---|---|---|---|---|
| TimeBridge | ETTm1 | 0.383 / 0.398 | 0.388 / 0.400 | 0.385 / 0.400 | 0.387 / 0.399 | 0.387 / 0.400 | 0.387 / 0.400 |
| TimeBridge | ETTh1 | 0.433 / 0.437 | 0.441 / 0.439 | 0.438 / 0.437 | 0.441 / 0.440 | 0.437 / 0.438 | 0.442 / 0.440 |
| TimeBridge | ECL | 0.172 / 0.267 | 0.177 / 0.272 | 0.174 / 0.269 | 0.172 / 0.266 | 0.176 / 0.271 | 0.176 / 0.271 |
| TimeBridge | Weather | 0.248 / 0.275 | 0.251 / 0.276 | 0.253 / 0.278 | 0.250 / 0.276 | 0.253 / 0.277 | 0.252 / 0.277 |
| Fredformer | ETTm1 | 0.379 / 0.395 | 0.386 / 0.397 | 0.380 / 0.395 | 0.385 / 0.397 | 0.385 / 0.397 | 0.387 / 0.398 |
| Fredformer | ETTh1 | 0.429 / 0.431 | 0.445 / 0.435 | 0.437 / 0.432 | 0.444 / 0.435 | 0.444 / 0.435 | 0.447 / 0.434 |
| Fredformer | ECL | 0.183 / 0.275 | 0.187 / 0.280 | 0.188 / 0.280 | 0.187 / 0.280 | 0.187 / 0.279 | 0.191 / 0.284 |
| Fredformer | Weather | 0.257 / 0.279 | 0.261 / 0.282 | 0.262 / 0.282 | 0.262 / 0.282 | 0.261 / 0.282 | 0.261 / 0.282 |

各分布差异度量均不劣于纯 MSE（DF），其中 Bures–Wasserstein（Ours）在 16 组对比中取得 14 组最优[^src-distdf]。

### 超参 $\gamma$ 敏感性（Table 4、Table 5）

$0\to1$ 扫描显示最优点远小于 $1$：TimeBridge 在 ECL 上 $\gamma=0.001\text{--}0.01$ 落在 $0.172\text{--}0.173/0.267$，$\gamma=1$ 退化到 $0.197/0.282$；Weather 从 $\gamma=0.01$ 的 $0.249/0.276$ 退到 $\gamma=1$ 的 $0.265/0.286$；ETTh2 的最优点后移到 $\gamma=0.05\text{--}0.1$（$0.375/0.399$）。Fredformer 在 ECL 上 $\gamma=0$ 为 $0.191/0.284$，$\gamma=0.002$ 为 $0.181/0.274$，$\gamma=1$ 为 $0.239/0.326$[^src-distdf]。按数据集不同，最优 $\gamma$ 落在 $0.001\text{--}0.1$；$\gamma=1$ 即丢弃样本级逐点对应、只用矩统计，性能一致退化[^src-distdf]。

### 跨模型、跨范式与扩展实验（Appendix Table 6、8–13）

| 实验 | 设置 | 报告结果 |
|---|---|---|
| Table 6 经典模型横比 | 9 数据集（ETT 四套、ECL、Traffic、Weather、PEMS03、PEMS08）× 10 模型；DistDF 接各数据集平均最优骨干 | 40 项 MSE 第一、41 项 MAE 第一；Traffic 平均 $0.417/0.279$ vs TimeBridge $0.426/0.282$、PEMS03 $0.104/0.215$ vs $0.112/0.223$、PEMS08 $0.123/0.223$ vs $0.139/0.239$ |
| Table 8 骨干通用性 | [[tqn\|TQNet]]、TimeBridge、Fredformer、[[itransformer\|iTransformer]]、[[source-frets\|FreTS]]，6 数据集 | 平均 MSE 均下降或持平：TQNet ETTh1 $0.449\to0.438$、TimeBridge ETTm1 $0.387\to0.383$、Fredformer ECL $0.191\to0.173$、iTransformer Weather $0.269\to0.258$、FreTS ETTh2 $0.524\to0.466$；FreTS+ECL 为 $0.199\to0.199$（MAE $0.288\to0.287$） |
| Table 9 自回归多步滚动 | TimeBridge、Fredformer，$H=96$ | 抑制长程误差累积：TimeBridge Weather $T=720$ 由 $9.363/1.374$ 降至 $0.461/0.421$，平均 MSE $3.970\to0.344$；Fredformer ETTm1 $T=720$ 由 $1.821/0.837$ 降至 $0.833/0.563$ |
| Table 10 概率预测 | D3U，指标含 [[crps\|CRPS]]、$\text{CRPS}_{\text{sum}}$ | Weather 平均 MSE $0.259\to0.255$、MAE $0.295\to0.290$、CRPS $0.221\to0.218$、$\text{CRPS}_{\text{sum}}$ $0.275\to0.255$；ETTm1 平均 MSE $0.383\to0.380$、CRPS 同为 $0.293$、$\text{CRPS}_{\text{sum}}$ $0.776\to0.762$ |
| Table 11 多尺度架构 | [[timemixer\|TimeMixer]]、SCINet，4 数据集 | 均获改善：TimeMixer ETTh1 平均 $0.501/0.476\to0.456/0.446$；SCINet ETTh1 $0.467/0.450\to0.459/0.443$；TimeMixer Weather $0.262/0.285\to0.252/0.280$ |
| Table 12 变历史长度 | Weather，$H\in\{96,192,336,720\}$ | [[patchtst\|PatchTST]] 在 $H=192,T=720$ 的退化（$0.511/0.451$）经 DistDF 降至 $0.335/0.338$；$H=336,T=192$ 由 $0.235/0.291$ 降至 $0.200/0.249$；TimeBridge 同设置 $0.338/0.344\to0.335/0.342$ |
| Table 13 随机种子 | ECL、Weather，种子 2021–2025 | ECL 平均 $0.172\pm0.000/0.266\pm0.001$ vs DF $0.177\pm0.002/0.272\pm0.001$；Weather $0.250\pm0.001/0.276\pm0.001$ vs $0.252\pm0.001/0.277\pm0.000$ |

补充证据：ECL 上 iTransformer 与 Fredformer 接 DistDF 后 MSE 降幅最高分别为 $2.7\%$ 与 $4.3\%$（§4.4、Fig. 3）；论文报告 BW 目标单调下降并在若干 epoch 后进入平台，验证集 MAE/MSE 与 BW 曲线同向变化（Appendix D.7、Fig. 8）[^src-distdf]。主实验统一取 $H=96$，变历史长度只在 Weather 上测（Table 12）；全部结果来自单卡 H100，不涉及分布式设定。

## 局限性与同组演进关系

高斯假设下 Bures–Wasserstein 只匹配一阶与二阶矩，非高斯分布需要高阶统计量才能完整刻画条件分布；作者把「引入高阶统计量且保持可计算」列为后续方向（Conclusion）。接入新数据集时需联合调节学习率 $\eta$ 与权重 $\gamma$：分布项的尺度与梯度动态随数据集变化，$\gamma$ 的有效区间跨数据集相差两个数量级（$\gamma\in[0.001,0.1]$，§4.1、Table 4–5）[^src-distdf]。

同一作者组沿「MSE 的似然偏差」这条线留下三种解法，路线互不相同：

| 方法 | 问题切入点 | 数学路线 | 去相关性质 | 训练代价 |
|---|---|---|---|---|
| [[fredf\|FreDF]] | DF 的 MSE 隐含未来步条件独立 | 正交频域分量上的模长似然 + 时域 MSE 混合 | 边缘去相关：频域分量依赖降到约 $3.6\%$ 非对角元 $>0.1$（Weather，$T=192$） | 一次 FFT，无新增参数[^src-fredf] |
| **DistDF** | 条件 NLL 的因式分解本身不可靠 | 联合分布 Wasserstein（高斯下 BW）差异 + MSE 混合 | 不做因式分解，用联合差异上界直接逼条件分布；$W_p=0\Rightarrow\mathbb{P}_{y\mid x}=\mathbb{P}_{\hat{y}\mid x}$ | 每步 $(H+T)\times(H+T)$ 协方差与矩阵平方根，$<1$ ms，推理零开销[^src-distdf] |
| [[qdf\|QDF]] | DF 的二次型似然等价于取 $\Sigma=I$，条件依赖与步间权重都无从表达 | 双层学习 PSD 矩阵 $\Sigma$ 的二次型似然 $L_\Sigma=\lVert Y-\hat{Y}\rVert_{\Sigma^{-1}}^2$ | $\Sigma^{-1}$ 非对角元显式建模条件依赖，对角元建模 [[heterogeneous-task-weights\|异构任务权重]] | 双层元学习 + $T\times T$ Cholesky 参数化[^src-qdf] |

三者的关系是同一诊断下的三种取舍：FreDF 换坐标系但不换似然形式，DistDF 换掉似然估计改为分布对齐，QDF 保留二次型似然但把 $\Sigma$ 变成可学习量。技术细节见 [[joint-distribution-wasserstein-alignment]]、[[bures-wasserstein]]、[[frequency-enhanced-direct-forecast]]、[[quadratic-form-weighted-objective]]。

---

[^src-distdf]: [[source-distdf]]
[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
