---
title: "VISReg: Variance-Invariance-Sketching Regularization for JEPA training"
type: source-summary
tags:
  - self-supervised-learning
  - jepa
  - regularization
  - optimal-transport
  - sliced-wasserstein
  - preprint
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# VISReg: Variance-Invariance-Sketching Regularization for JEPA training

**Haiyu Wu, Randall Balestriero, Morgan Levine（Altos Labs；Brown University），arXiv:2606.02572v1 [cs.CV]，2026-06-01 提交，preprint（PDF 未标注 venue）**

完整论文（17 页）：`raw/2606.02572.pdf`。

## 核心论题

自监督联合嵌入架构依赖正则项防止嵌入塌缩。VICReg 的协方差项仅约束二阶统计量且协方差矩阵计算达 $O(ND^2)$；[[lejepa|LeJEPA]] 的 SIGReg 复杂度为 $O(D)$，但在嵌入塌缩时梯度消失，且单一检验统计量无法分开调节 scale 与 shape。VISReg 把正则拆为 scale（方差项）与 shape（Sliced Wasserstein sketching）两个独立目标：stop-gradient 归一化隔离两者；shape 项在 K 个随机投影上以 1D W2 闭式解对齐完整边缘分布（Cramér-Wold 定理），塌缩时梯度有界，复杂度 $O(NDK)$。

## 证据

ImageNet-1K 从零预训练 ViT-B/16 与 ViT-L/14（400 ep，32×H100，约 1120/2060 GPU 时）：域内线性探针落后 DINO、iBOT 等有启发式方法；OOD 六数据集平均 70.63，高于 ImageNet-1K 预训练组的全部对比方法；ImageNet-22K 预训练（100 ep，16×H100）后 OOD 平均 72.94，与 10 倍数据训练的 DINOv2（72.93）相当；迁移微调五个数据集均高于 DINO 与有监督基线。低质量数据（ImageNet-LT 长尾 35.14、Galaxy10 低秩 80.76）上 shape 权重 4:1 重加权后为对比方法最好，DINO 仅 5.13/73.49。消融确认 scale 与 shape 项缺一不可（移除后分别下降 71.02% 与 58.4%），detach 与 center 项各有小幅增益。

## 局限

域内精度与 DINO、iBOT 仍有差距；ADE20K 线性分割落后 MoCoV3（30.16 vs 31.69 mIoU）；切片数 K 与嵌入维度 D 耦合，需按 GPU 数分摊切片；正则权重 λ 需按数据规模调节（小数据集 0.6、ImageNet-1K 0.9 起步）。

[^src-visreg]: [[source-visreg]]
