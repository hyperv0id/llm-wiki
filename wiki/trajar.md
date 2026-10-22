---
title: "TrajAR"
type: entity
tags:
  - trajectory-prediction
  - intersection
  - multi-scale
  - autoregressive
  - mixture-of-experts
  - graph-attention
created: 2026-10-20
last_updated: 2026-10-20
source_count: 1
confidence: medium
status: active
---

# TrajAR

**TrajAR**（long-term **Traj**ectory prediction multi-scale **A**uto**R**egressive framework）是北京交通大学、南洋理工大学、Aalborg University 的作者在 IJCAI-26 提出的路口多道路使用者轨迹预测模型，采用 encoder-decoder 结构：编码器从历史轨迹中提取运动趋势、类型、信号与隐式交互，解码器把自回归预测的单位从「一步」换成「一个尺度」，由粗到细生成未来轨迹[^src-trajar]。

## 问题设定

论文的形式化如下[^src-trajar]：

- 道路使用者 $\nu$ 在时刻 $t$ 的特征 $z^\nu_t$ 含位置 $\tau^\nu_t=(x^\nu_t,y^\nu_t)$ 与信号状态 $s_t \in \mathbb{R}^{N_s}$；使用者类型是 one-hot $a_\nu \in \mathbb{R}^{N_a}$（机动车、自行车、行人等）。
- 交互使用者集合 $V_I$ 取距目标最近 $N_R$ 个使用者，实验中 $N_R=15$。
- 给定历史 $H=\langle z^\nu_{t_0-t_h+1},\dots,z^\nu_{t_0}\rangle$（$\nu \in V_I \cup \{\nu_T\}$），预测未来位置 $\langle \tau^{\nu_T}_{t_0+1},\dots,\tau^{\nu_T}_{t_0+t_f}\rangle$。
- 实验取 $t_h=t_f=32$ 步（论文称「prediction step 设为 32，与观察窗口等长」）。

## 编码器

编码器由三层组成：运动趋势层、Motion MoE 层、交互感知层[^src-trajar]。

### 运动趋势层

用单层 Transformer encoder 处理每个使用者的轨迹特征序列。因使用者进出画面导致序列长度不一，先做时间对齐、把序列 padding 到统一长度 $t_h$；输出 $z^{f\nu}_t$ 表示该使用者在 $t$ 步的运动趋势[^src-trajar]。

### Motion MoE

Motion MoE 把离散属性与连续趋势分给两类专家，输出为残差叠加[^src-trajar]：

$$
z^\nu_t = z^{f\nu}_t + \sum_{n=1}^{N_a} I^a_n \mathrm{FFN}^{(a)}_n(z^{f\nu}_t) + \sum_{n=1}^{N_l} I^s_n \mathrm{FFN}^{(l)}_n(z^{f\nu}_t) + \sum_{n=1}^{8} g_n \mathrm{FFN}^{(r)}_n(z^{f\nu}_t)
$$

- **indicator experts**：$I^a_n$、$I^s_n$ 是指示当前使用者类型与信号状态所在位置的 one-hot，按属性直接选中对应专家（$N_a$ 个类型专家、$N_l$ 个信号专家）。
- **route experts**：8 个，走 top-$K_r$ 路由。门控值 $g_n=g'_n/\sum_j g'_j$，其中 $g'_n=p_{n,t}$ 当 $p_{n,t}\in\mathrm{TopK}(\{p_{j,t}\}_{j=1}^8,K_r)$，否则为 0；亲和度 $p_{n,t}=\mathrm{Sigmoid}((z^{f\nu}_t)^\top c_n)$，$c_n$ 是第 $n$ 个 route expert 的质心向量。

论文的表述是：indicator experts 处理类型与信号这类离散特征，route experts 建模运动趋势这类连续特征[^src-trajar]。机制细节与证据见 [[motion-moe]]。

### 交互感知层

用 GAT 在目标使用者与每个交互使用者之间算注意力[^src-trajar]：

$$
\tilde\alpha^{\nu,\nu_T}_t=\frac{\exp\!\big(a_{\tilde\alpha}^\top\gamma(W_{\tilde\alpha}[z^\nu_t\,\|\,z^{\nu_T}_t\,\|\,\beta^{\nu,\nu_T}_t])\big)}{\sum_{\nu\in V^I_t}\exp\!\big(\cdot\big)},\qquad
\beta^{\nu,\nu_T}_t=\exp\!\big(-(d_{\nu,\nu_T}/\sigma_e)^2\big)
$$

$d_{\nu,\nu_T}$ 是两者欧氏距离，$\sigma_e$ 是学习参数，用来压低远距离邻居的边权；边权本身作为特征参与注意力计算。隐式交互表征为 $h^{\nu_T}_t=W_1z^{\nu_T}_t+\sum_{\nu\in V^I_t}\tilde\alpha^{\nu,\nu_T}_t W_2z^\nu_t+b$。此后用单层 Transformer encoder 处理序列 $\langle h^{\nu_T}_t\rangle$，把各步输出 $o^{\nu_T}_t$ 沿历史窗口拼接成 $O_{his}$。论文把这一步定位为把隐式交互与显式路权规则（转弯让直行、右转让左转、机动车让行人）一起纳入情境感知[^src-trajar]。

## 解码器

解码器把「预测下一步」换成「预测下一个更细的尺度」，机制与插值细节见 [[scale-wise-autoregressive-decoding]]。要点[^src-trajar]：

- 未来 32 个位置被分成 $K$ 个尺度 $S_1,\dots,S_K$，$S_K$ 对齐原始分辨率；第 $k$ 步的源序列 $T_k=[S_0\|\hat S_{k-1}]$，$T_1=[O_{his}\|S_0]$，同一步内的 $\hat S_k$ 并行生成。
- $k>1$ 时用三次插值在 $T_k$ 相邻位置间插点、再丢掉 $\tau^{\nu_T}_{t_0}$ 得到 $\bar T_k$，用全连接层 $\phi_k$ 映射为嵌入 $E_k$。
- 解码器是 $l_d$ 层 Transformer，用 block-wise causal mask 保证 $\hat S_k$ 只能看分辨率不细于 $k$ 的源序列；输出经一层 MLP（BatchNorm + Sigmoid）得到 $\hat S_k$。
- 损失为所有尺度上的 RMSE 之和：$\mathcal{L}=\sum_{k=1}^{K}\sum_{\tau\in S_k}\sqrt{(\tau-\hat\tau)^2}$。

## 训练设置

- 数据：SinD-Tianjin、SinD-Xi'an（天津、西安信号交叉口，10 Hz）、InD-Bendplatz、InD-Frankenburg（德国路口，25 Hz，四类使用者）。按录制时间 6:2:2 划分训练/验证/测试[^src-trajar]。
- 超参数：学习率 1e-4，训练 100 epoch，early stopping patience 5。按验证集选优的参数组合 $\{K_r, l_d, N_a, N_l\}$ 为 SinD-Tianjin {2,4,7,6}、SinD-Xian {2,6,7,4}、InD-Bendplatz {4,6,4,−}、InD-Frankenburg {2,8,4,−}，其中 $N_a$、$N_l$ 取该数据集唯一使用者类型数与信号状态数[^src-trajar]。InD 无信号信息，实验中去掉信号专家；生成阶段不引入额外上下文[^src-trajar]。尺度数 $K$ 与各尺度的位置划分未在实现设置中列出。
- 硬件：Intel Rapids-SP CPU + NVIDIA A40 GPU[^src-trajar]。
- 复现口径：论文称每组实验跑 5 次报均值；统计检验一节称 5 个随机种子报 mean±std，并用 paired t-test 与 KI-GAN 比较[^src-trajar]。

## 实验结果

主结果（表 1、表 2，ADE/FDE，表头口径 e-2）[^src-trajar]：

| 数据集 | TrajAR | KI-GAN（正文选作最强基线） | 表内最低值 |
|---|---|---|---|
| SinD-Tianjin | 0.409 / 0.798 | 0.495 / 0.945 | TrajAR |
| SinD-Xi'an | 0.509 / 0.942 | 0.666 / 1.245 | TrajAR |
| InD-Bendplatz | 0.437 / 1.029 | 0.523 / 1.312 | TrajAR |
| InD-Frankenburg | 0.493 / 1.284 | 0.593 / 1.555 | TrajAR |

表中共 12 个基线，论文分三类：物理模型 CA、CV、Cubic；车辆轨迹预测 C2F-TP、WSiP、HLTP、PiP、STDAN、BAT；多使用者轨迹预测 MTP-GO、FJMP、KI-GAN[^src-trajar]。评价指标沿用 MTP-GO 的四类：ADE、FDE、MR（Miss Rate）、APDE（Average Path Displacement Error），另加 step@8/16/24 的分步误差[^src-trajar]。按表值核算，TrajAR 相对 KI-GAN 的 ADE 降幅为 17.4%/23.6%/16.4%/16.9%，FDE 降幅为 15.6%/24.3%/21.6%/17.4%（分母取基线值，即常规口径）。各列的最优基线不总是 KI-GAN：InD-Frankenburg 的 ADE 最优基线是 STDAN 0.572（FJMP 0.583、KI-GAN 0.593），FDE 最优基线是 BAT 1.421（KI-GAN 1.555）[^src-trajar]。论文表 1/2 的 Imp.(Δ%) 行以逐列最优基线为参照、分母取本方法自身，见下文的待查项。

消融（表 3，SinD-Tianjin / SinD-Xi'an 的 ADE/FDE）[^src-trajar]：

| Motion MoE | GAT & ESA | 编码器整体换成 FC | Decoder | SinD-Tianjin | SinD-Xi'an |
|---|---|---|---|---|---|
| × | × | ✓ | ✓ | 1.231 / 2.542 | 1.423 / 2.782 |
| × | ✓ | × | ✓ | 0.892 / 1.642 | 1.023 / 2.122 |
| ✓ | × | × | ✓ | 0.882 / 1.621 | 0.929 / 1.837 |
| ✓ | ✓ | × | × | 1.623 / 3.241 | 1.892 / 3.912 |
| ✓ | ✓ | × | ✓ | 0.409 / 0.798 | 0.509 / 0.942 |

- 去掉 decoder 的行（第 4 行）误差上升最大（SinD-Tianjin ADE/FDE 0.409/0.798 → 1.623/3.241），论文据此支持 next-scale 路线[^src-trajar]。
- 第 2 行只把 Motion MoE 置 ×，其余列与完整模型一致：0.892/1.642，相对完整模型 +118.1%/+105.8%，与正文「Removing the Motion MoE」的归因一致[^src-trajar]。
- 第 3 行把 GAT & ESA（正文称作 Interaction-Aware 层）置 ×：0.882/1.621，相对 +115.6%/+103.1%，与正文「Excluding the Interaction-Aware layer」一致[^src-trajar]。
- 第 1 行把编码器整体换成全连接层：1.231/2.542，与正文「Replacing encoders with fully connected layers leads to substantial performance drops」一致[^src-trajar]。
- 表 3 的表头在 PDF 里是两行排版（Motion/MoE、GAT &/ESA、Fully/Connection），四列应读作 Motion MoE｜GAT & ESA｜Fully Connection｜Decoder；本页初版把前两列并成「Motion GAT & MoE｜ESA」，因此曾误判正文归因与表格不符，已按上表纠正。

其它分析[^src-trajar]：

- **按类型与峰谷**（表 6，SinD-Tianjin，仅 ADE）：机动车 0.295 → 峰值 0.320（+8.5%），自行车 0.610 → 0.680（+11.5%），行人 0.405 → 0.435（+7.4%）；KI-GAN 对应 +27.4%/+30.8%/+28.6%。
- **插值方式**（表 7，SinD-Tianjin，ADE/FDE）：三次插值在机动车 0.303/0.564、自行车 0.613/1.171、行人 0.408/0.701 上最低；线性插值机动车 1.321/2.212、Lagrange 行人 1.723/3.345、「None」行人 0.512/1.134。论文据此称三次插值的稳定性—精度平衡最好，线性插值对快速使用者退化、Lagrange 有 Runge 效应。
- **概率与安全指标**（表 4，对 KI-GAN）：minADE/NLL/Collision 在 SinD-Tianjin 为 0.392/1.0987/0.01982 vs 0.451/1.2345/0.04567，SinD-Xi'an 0.421/1.2876/0.01651 vs 0.487/1.5432/0.05123，InD-Bendplatz 0.378/1.1894/0.00912 vs 0.432/1.3210/0.03891，InD-Frankenburg 0.407/1.2765/0.01765 vs 0.469/1.4123/0.04218。论文自述为「尽管用确定性 RMSE 目标」，四个数据集上三项指标均优于 KI-GAN。
- **MoE 分析**（图 3(b)(c)、图 4）：不同转向（左转、右转、直行）下 route expert 的负载分布不同；去掉用户类型 0.613/1.197、去掉信号 0.531/1.037、打乱 route 路由 0.736/1.436（对比完整模型 0.409/0.798，SinD-Tianjin ADE/FDE）；8 个专家 + top-$K_r$ 时 ADE 最低。
- **效率**（图 3(a)）：论文称在对比方法中精度最高且推理延迟低于 100 ms。

## 论文自述的边界与本页记录的待查项

- 论文结论把**概率多模态预测**与**更广城市泛化**列为未来工作，且被定位为「轻量」框架；主实验在四个路口数据集、32 步预测、确定性 RMSE 目标下完成[^src-trajar]。
- 摘要的「平均提升 **23.8%**」对应哪种聚合，正文未写。可复算的口径有三个。其一，表 1/表 2 的 Imp.(Δ%) 行以本方法为分母、逐列最优基线为参照：28 个格子逐列复算与 Imp 行的差均在 0.1pp 内，该行算术平均 19.2%。其二，同口径但只取 KI-GAN 为参照、ADE/FDE 八个格子：均值 23.9%（与摘要差 0.1pp，是唯一落在 0.1pp 内的口径）。其三，换成常规的基线分母口径：同八个格子 19.1%。即摘要数字是以本方法为分母的降幅，比常规降幅高约 4–5pp；论文未声明这一约定[^src-trajar]。
- 表 4 的 minADE/NLL/Collision 由确定性 RMSE 模型给出，实验设置未描述概率输出来源；表中 NLL 保留 4 位、Collision 保留 5 位小数，位数口径不统一[^src-trajar]。
- 「lightweight」没有参数量或 FLOPs 支撑，只有图 3(a) 的延迟—精度曲线[^src-trajar]。
- 下采样口径：正文称输入与目标下采样 3 倍后有效采样间隔 0.12 s，该值对应 InD 的 25 Hz；SinD 为 10 Hz，同倍下采样应为约 0.3 s，论文未说明两组是否取同一间隔，因此 32 步预测对应的物理时长（约 3.84 s 或约 9.6 s）在文中不确定[^src-trajar]。

## Related Pages

- [[source-trajar]] — 论文源摘要
- [[scale-wise-autoregressive-decoding]] — next-scale 自回归解码（本文核心贡献 1）
- [[motion-moe]] — Motion MoE（本文核心贡献 2）
- [[intersection-trajectory-prediction]] — 路口轨迹预测任务与输出框架对比
- [[mixture-of-experts]] — MoE 通用机制
- [[moe-in-time-series-and-spatio-temporal]] — 时序/时空 MoE 机制谱系

[^src-trajar]: [[source-trajar]]
