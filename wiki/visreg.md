---
title: "VISReg: Variance-Invariance-Sketching Regularization"
type: technique
tags:
  - self-supervised-learning
  - jepa
  - regularization
  - optimal-transport
  - sliced-wasserstein
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# VISReg: Variance-Invariance-Sketching Regularization

Wu、Balestriero、Levine（Altos Labs；Brown University，arXiv:2606.02572v1，2026-06-01 preprint）提出的联合嵌入自监督正则化方法（论文标题自称 for JEPA training，训练对象为 ViT 骨干）。定位在 VICReg 与 [[lejepa|LeJEPA]] 的 SIGReg 之间：保留 VICReg 的方差项控制 scale，用基于最优传输的 Sliced Wasserstein sketching（见 [[sliced-wasserstein-distance]]）替换协方差项控制 shape[^src-visreg]。

## 问题：两类正则各缺一角

无负样本的联合嵌入训练依赖对嵌入分布的正则防止塌缩（见 [[non-contrastive-ssl]]）。两条既有路线各有缺口：

- **VICReg 的协方差项只约束二阶统计量**。均值与协方差都对上仍可能远离各向同性高斯——去相关不等于分布形状正确；且协方差矩阵计算随嵌入维度 D 二次增长（$O(ND^2)$）[^src-visreg]。
- **SIGReg 在塌缩时梯度消失**。SIGReg 经 Epps-Pulley 检验把嵌入各 1D 投影拉向标准正态（复杂度 $O(D)$），但 VISReg 论文的测量显示：嵌入越塌缩，其梯度越小（Figure 2，Barlow Twins 与 VISReg 在塌缩区梯度保持强信号）——恰在需要纠正时纠正信号最弱；此外 SIGReg 的单一检验统计量把 scale 与 shape 捆在一起，无法分权重调节；17 节点采样也拖慢计算[^src-visreg]。

VISReg 的设计即补这两个缺口：把正则拆成 scale 与 shape 两个独立目标，shape 用塌缩时梯度有界的 OT 度量[^src-visreg]。

## 机制：scale / shape / center 三项解耦

对中心化嵌入 $\hat{Z}\in\mathbb{R}^{N\times D}$（N 为 batch 大小、D 为嵌入维度）：

- **scale（方差项，沿用 VICReg 直觉）**：$L_{scale}=\frac{1}{D}\sum_{j=1}^{D}(1-\sigma_j(\hat{Z}))^2$，$\sigma_j$ 为第 j 维标准差。塌缩时其梯度趋于常数，保证纠正信号不消失[^src-visreg]。
- **shape（sketching 项）**：先做 stop-gradient 归一化 $\tilde{Z}=\hat{Z}/(\mathrm{sg}(\sigma)+\epsilon)$，使 shape 梯度不干扰方差调节；再将 $\tilde{Z}$ 投影到 K 个归一化随机方向 $w_k$，逐方向排序后与标准正态分位数 $q_N$ 做 L2 回归：$L_{shape}=\frac{1}{K}\sum_{k=1}^{K}\lVert\mathrm{sort}(\tilde{Z}w_k)-q_N\rVert^2$。依据 Cramér-Wold 定理，分布的全部信息包含在其所有 1D 投影中，因此对齐每个投影方向的完整边缘分布严格强于仅去相关[^src-visreg]。
- **center**：$L_{center}=\lVert\mu\rVert_2^2$（batch 均值）。最终精度贡献小（ImageNette 上差 0.41%）但加快收敛[^src-visreg]。

不变项沿用 LeJEPA：全局视图均值 $\mu_g$ 与全部 V 个视图嵌入 $z_i$ 的 MSE（$L_{pred}=\frac{1}{V}\sum_i\lVert\mu_g-z_i\rVert_2^2$）。总目标 $L=(1-\lambda)L_{pred}+\lambda L_{Reg}$，$L_{Reg}=\lambda_{scale}L_{scale}+\lambda_{shape}L_{shape}+\lambda_{center}L_{center}$。相比依赖训练稳定性启发式的方法，全部超参只有四个：λ、学习率、batch size、投影维度[^src-visreg]。

$L_{Reg}$ 的成本为 $O(NDK)+O(KN\log N)$，排序项中 $\log N\ll D$，有效复杂度 $O(NDK)$——对所有缩放参数线性，对比 VICReg 的 $O(ND^2)$。模拟缩放实验（单张 H100 80GB、batch 50K、投影维 10K、切片 2.5K、8 视图）中，VISReg 速度略优于 SIGReg，显存需求为 SIGReg 的 13.7%[^src-visreg]。

切片数 K 与嵌入维度 D 存在耦合：ImageNette 上固定 K=4096 增大 D 时，最优精度要求 K 超过 D 一个因子 C>1（Cramér-Wold 类方法共同的规律，VISReg 所需 C 最小）；固定 D=256 时，OT 类方法在 K=⅛D 仍保持鲁棒，VISReg 全程精度最高。论文的处理是把切片按 GPU 数分摊：每 GPU 生成 K/M 个切片即可合成总 K=CD。1 GPU 上 K=128 与 K=1024 的精度差为 SIGReg 13.88%、SWD 2.21%、VISReg 2.44%；8 GPU 且 per-GPU K 不变时，差距分别缩至 0.27%、0.24%、0.22%——K 因此可在缩放时保持常数，复杂度维持 $O(NDK)$[^src-visreg]。

## 证据

以下数字均出自该论文。除注明外：骨干 ViT-B/16 与 ViT-L/14 在 ImageNet-1K 上从零训练 400 epoch（4 全局 + 6 局部视图，增广与训练设置沿用 LeJEPA；超参消融另用 2 全局 + 6 局部、100 epoch），下游评估沿用 DINOv2 协议——冻结编码器、末 4 层 CLS 拼接特征、线性头 10 epoch、13 个学习率网格取最优（ImageNet-1K 探针为 100 epoch）[^src-visreg]。

**域内线性探针（Table 5，8 数据集平均 + ImageNet-1K）**：无启发式组内，VISReg ViT-B/16 79.1 / 75.7，ViT-L/14 78.5 / 77.0，MAE ViT-L/16（1600 ep）77.8 / 75.1；100 epoch 探针下 VISReg ViT-L/14 80.7 / 75.6，同协议 LeJEPA 79.5 / 75.6。有启发式组仍更高：iBOT ViT-L/16 84.5 / 81.0，DINO ViT-B/16 83.1 / 78.2[^src-visreg]。

**OOD（Table 6，DTD/Galaxy10/AID/ChestXRay/RetinaMNIST/OrganAMNIST 六数据集平均）**：VISReg ViT-L/14 70.63、ViT-B/16 70.19，高于 ImageNet-1K 预训练组的全部对比方法与骨干规模（次高 iBOT ViT-B/16 69.64）；ImageNet-22K 预训练（100 ep，16×H100）的 VISReg ViT-L/14 达 72.94，与用 10 倍数据（LVD-142M）训练的 DINOv2 ViT-L/14 的 72.93 相当[^src-visreg]。

**迁移微调（Table 7，ViT-B/16，DINO 微调协议）**：CIFAR-10 99.2、CIFAR-100 91.8、Flowers 99.0、ImageNet-1K 83.0、Galaxy10 87.0，均高于 DINO（99.1 / 91.7 / 98.8 / 82.8 / 86.6）与有监督基线（99.0 / 89.5 / 98.5 / 81.5）[^src-visreg]。

**密集预测（Table 8，ADE20K 线性分割 mIoU，复现值）**：VISReg 30.16，介于 DINO 29.40 与 MoCoV3 31.69 之间（data2vec 21.99、MAE 23.60）[^src-visreg]。

**生成引导（Table 9，SiT-B/2 训练 100K 步、iREPA 特征引导、50K 样本无 CFG 评估）**：VISReg gFID 40.36 / IS 33.48 / Precision 51.38 / Recall 61.26，DINO 为 41.15 / 33.47 / 50.51 / 60.70[^src-visreg]。

**损失–精度相关性（Figure 7）**：ViT-L/14 于 ImageNet-1K 训练 100 epoch，训练损失与在线探针精度的 Pearson 相关为 −0.996，损失曲线可直接反映学习进度[^src-visreg]。

## 低质量数据集

长尾（ImageNet-LT，115K 图像 1000 类）与低秩（Galaxy10，17736 张星系图像 10 类；类别少且图像含大量黑像素，按低秩处理）设定下，ViT-S/8 从零训练 400 epoch（lr 1e-4、K=4096、D=256、128px 输入）：ImageNet-LT 总体精度 VISReg*（shape 权重 4:1 重加权）35.14 > VICReg 33.08 > VISReg 32.11 > SIGReg 32.00 > SWD 31.85，DINO 5.13（未学到有效表征）；Galaxy10 上 VISReg* 80.76 > SWD 80.60 > SIGReg 80.50 > VISReg 80.51 > VICReg 79.93 > DINO 73.49。权重比消融（Table 12，三项权重和恒为 3）：shape 4:1 在 ImageNet-LT +3.2%、Galaxy10 +1.3%，scale 4:1 分别 −4.6%、−2.7%；高质量数据（ImageNette）则默认 1:1:1 最好[^src-visreg]。

## 消融

- 三个正则分量均必要（Table 10，ImageNette，ViT-S/8、800 ep）：去 scale 后 20.80（降 71.02%），去 shape 后 33.42（降 58.4%），去 center 91.41，完整 91.82[^src-visreg]。
- λ（总正则权重，Table 3，ViT-B/16、100 ep 在线探针）：0.9 最优 69.19（0.7→67.98、0.8→68.64、0.95→69.04）。与 SIGReg 需随 batch 缩放正则不同，VISReg 天然 batch 不变；小数据集（ImageNette、Galaxy10）λ=0.6 起步，ImageNet-1K λ=0.9 起步[^src-visreg]。
- 学习率与 batch size（Table 3）：lr 9e-4 最优 69.66（5e-4–1e-3 为 ImageNet-1K 可用区间），batch 512 最优 69.66（2048 时 68.79）[^src-visreg]。
- 投影维度（Table 3/4）：域内精度随维度升高（300 ep 下 512 最优 73.53）；下游任务上方向分裂——域内分类偏好大维度、OOD 分类与密集预测偏好小维度，折中取 256[^src-visreg]。

## 局限

- 域内精度与有启发式方法（DINO、iBOT）仍有差距；ADE20K 分割与 MoCoV3 差距不可忽略（作者列为后续工作方向）[^src-visreg]。
- λ 需按数据集规模调节（小数据 0.6、ImageNet-1K 0.9）；投影维度对域内与 OOD 的影响方向相反，需按下游任务选择[^src-visreg]。
- 单 GPU 低 K 时 VISReg 自身也有精度损失（K=128 vs 1024 差 2.44%），需多 GPU 分摊切片才能在缩放时保持 K 常数[^src-visreg]。

## Related Pages

- [[jepa]] — JEPA 架构与训练判据，本方法是正则化路线的实现之一
- [[vicreg]] — 被替换的协方差项来源；方差项被保留
- [[lejepa]] — 不变项与训练设置的沿用对象；SIGReg 为对比基线
- [[non-contrastive-ssl]] — 防塌缩路线谱系
- [[sliced-wasserstein-distance]] — shape 项的数学工具
- [[hierarchical-jepa]] — JEPA 的层级扩展（架构谱系）

[^src-visreg]: [[source-visreg]]
