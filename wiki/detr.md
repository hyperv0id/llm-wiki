---
title: "DETR: End-to-End Object Detection with Transformers"
type: technique
tags:
  - transformer
  - object-detection
  - set-prediction
  - learned-queries
  - non-autoregressive-decoding
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# DETR（End-to-End Object Detection with Transformers）

DETR（DEtection TRansformer）是 Carion et al.（Facebook AI）提出的目标检测框架：论文将检测建模为直接的集合预测（direct set prediction）问题，用基于二部匹配（bipartite matching）的集合损失与 Transformer encoder-decoder 取代 proposals/anchors 等代理任务及 NMS、anchor 启发式分配等手工组件[^src-detr]。论文自述其推理代码可在 PyTorch 中以少于 50 行实现（Sec 3.2）[^src-detr]。

## 论文针对的问题

论文认为现代检测器以间接方式处理集合预测：在 proposals（两阶段）、anchors（单阶段）或窗口中心网格上定义代理回归与分类问题，性能受近重复预测的后处理（NMS）、anchor 集合设计与目标-锚点启发式分配规则影响（Sec 1）[^src-detr]。此前的端到端集合预测尝试（如基于 RNN 的 recurrent detectors）或引入额外先验，或未在强基线与大规模基准上证明竞争力（Sec 1）[^src-detr]。

## 机制

论文自述直接集合预测需要两个成分：(1) 强制预测与真值唯一匹配的集合预测损失；(2) 一次前向预测一组对象并建模其相互关系的架构（Sec 3）[^src-detr]。

### 集合预测损失与二部匹配

DETR 在一次 decoder 前向中推断固定大小为 N 的预测集合，N 显著大于图像中典型目标数；真值集合用 ∅（no object）填充到 N（Sec 3.1）[^src-detr]。

- **匹配**：搜索 N 元置换 σ 最小化 $\sum_i L_{match}(y_i, \hat{y}_{\sigma(i)})$（式 1），用匈牙利算法（Hungarian algorithm）高效求解；匹配代价同时考虑类别预测与框相似度（Sec 3.1）[^src-detr]。论文将这一步的角色类比为 Faster R-CNN 中 proposal/anchor 与真值的启发式分配，区别在于这里要求一对一匹配、无重复（Sec 3.1）[^src-detr]。
- **损失**：对匹配好的对计算 Hungarian loss（式 2）= 类别负对数似然 + 框损失；∅ 类的 log 概率项降权 10 倍以应对类别不平衡（Sec 3.1）[^src-detr]。匹配代价中使用概率而非 log 概率，论文报告这样做使类别项与框损失量纲可比、经验效果更好（Sec 3.1）[^src-detr]。
- **框损失**：框直接以绝对坐标预测（相对图像尺寸归一化），不做相对初始猜测的 Δ 回归；由于 L1 对不同大小框尺度不一致，采用 $\lambda_{iou} L_{IoU} + \lambda_{L1}\|\cdot\|_1$ 的线性组合，其中 $L_{IoU}$ 为尺度不变的 generalized IoU（GIoU）损失（Sec 3.1）[^src-detr]。

### 架构：CNN + Transformer encoder-decoder

三个组件（Sec 3.2, Fig 2）[^src-detr]：

1. **CNN backbone**：常规 CNN（实验用 ImageNet 预训练 ResNet）产生低分辨率激活图（C=2048，H/32×W/32）。
2. **Transformer encoder-decoder**：1×1 卷积降维到 d 并将空间维展平为 d×HW 序列；encoder 各层为多头自注意力 + FFN，由于 Transformer 置换不变，在每层注意力输入加入固定位置编码。decoder 采用标准结构但**逐层并行解码 N 个对象**——与 Vaswani et al. 原始 Transformer 的自回归逐元素生成不同；decoder 的 N 个输入嵌入是可学习的位置编码，论文称为 **object queries**（详见 [[object-queries]]），同样加到每层注意力输入。
3. **预测 FFN**：每个 decoder 输出嵌入独立过共享的 3 层 perceptron（ReLU，隐藏维 d）预测归一化框中心/宽高，线性投影 + softmax 预测类别；∅ 类扮演标准检测中背景类的角色。

训练时使用 auxiliary decoding losses：在每个 decoder 层后接共享参数的预测 FFN 与 Hungarian loss，帮助模型输出各类别正确的目标数（Sec 3.2）[^src-detr]。

论文自述与以往 direct set prediction 工作的主要区别是 bipartite matching loss 与 Transformer 并行（非自回归）解码的结合：以往 recurrent detectors 基于 RNN 自回归生成（Sec 1, Sec 2.3）[^src-detr]。

## 证据

实验在 COCO 2017 detection 与 panoptic 分割数据集（118k 训练 / 5k 验证图像，平均每图 7 个实例）上进行（Sec 4）[^src-detr]。训练用 AdamW（transformer lr 1e-4、backbone 1e-5），消融用 300 epoch schedule、与 Faster R-CNN 对比用 500 epoch（后者比前者 +1.5 AP）；基线模型在 16 块 V100 上训练 3 天，总 batch size 64（Sec 4）[^src-detr]。

**与 Faster R-CNN 对比（Table 1，COCO val）**。作者报告：DETR（ResNet-50）42.0 AP / AP50 62.4 / APS 20.5 / APM 45.8 / APL 61.1（86 GFLOPS、28 FPS、41M 参数，其中 transformer 17.8M）；强化后的 Faster R-CNN-FPN+（9x schedule + GIoU + 随机裁剪增强）为 42.0 AP / APS 26.6 / APL 53.4。论文的结论是 DETR 在同参数量下与强化的 Faster R-CNN 相当，且在大目标 APL 上显著更好、在小目标 APS 上落后；DETR-R101 与 DETR-DC5-R101 分别达 43.5 与 44.9 AP（Table 1, Sec 4.1）[^src-detr]。

**消融（Sec 4.2）**：

- **Encoder 层数**（Table 2）：0 层 encoder 时 AP 降 3.9、大目标降 6.0；作者假设 encoder 的全局场景推理用于分离实例，并报告 Fig 3 中 encoder 自注意力图似乎已分离实例（Sec 4.2）[^src-detr]。
- **Decoder 层数**（Fig 4）：各层后接的辅助预测头显示 AP/AP50 逐层提升，第一层到最后一层 +8.2/9.5 AP；NMS 在第一层输出上提升性能（单层无跨元素通信、易对同一目标重复预测），第二层起自注意力抑制重复，在最后层 NMS 反而误删真阳性、降低 AP——论文以此验证 DETR 设计上不需要 NMS（Sec 4.2）[^src-detr]。
- **FFN**：移除 FFN 后参数从 41.3M 降到 28.7M，AP 降 2.3，作者结论 FFN 对性能重要（Sec 4.2）[^src-detr]。
- **位置编码**（Table 3）：spatial positional encodings 在 encoder/decoder 均不传入时仍达 32.8 AP（−7.8）；sine spatial enc 与输出位置编码均只在输入传入一次（原 Transformer 方式）而非每层注意力输入时 −1.4；encoder 不传 spatial enc（decoder 仍逐层传入）仅 −1.3；输出位置编码（object queries）必需、不能移除（Sec 4.2）[^src-detr]。
- **损失项**（Table 4）：class + GIoU 组合（无 L1）仅比完整损失低 0.7 AP；class + L1（无 GIoU）低 4.8 AP——作者报告 GIoU 贡献了大部分框性能，L1 单独效果差、与 GIoU 组合后改善 APM/APL（Sec 4.2）[^src-detr]。

**Query slot 行为（Sec 4.3）**。Fig 7 将 20 个（共 N=100）query slot 在 COCO val 全集上的预测框中心可视化：各 slot 学到对特定区域与框尺寸的特化、且有多个操作模式，几乎所有 slot 都有预测全图大框的模式（作者假设与 COCO 目标分布相关）[^src-detr]。Fig 5 的分布外实验：训练集中没有超过 13 只长颈鹿的图像，DETR 能在合成图中检出全部 24 只——论文以此说明 object query 没有强类别特化[^src-detr]。

**Panoptic segmentation 扩展（Sec 4.4, Table 5）**。在冻结的 DETR 上加 mask head：decoder 输出对 encoder 输出做 M 头注意力得到每目标 M 张低分辨率热图，FPN 式 CNN 升采样到 stride 4，用 DICE/F-1 loss + Focal loss 监督；预测时逐像素 argmax 保证掩码无重叠，无需 Panoptic FPN 类启发式对齐（Sec 4.4）[^src-detr]。作者报告 COCO val 上 DETR-R101 PQ 45.1（对比重训的 PanopticFPN++ R101 44.1、UPSNet 42.5），stuff 类 PQst 37.0 明显高于基线（32.3–34.1），things mask AP 落后至多 8 mAP 但 PQth 仍有竞争力；COCO test set 46 PQ（Table 5）[^src-detr]。

## 论文自述局限

- 小目标性能低于 Faster R-CNN；作者期望未来工作改进，类比 FPN 之于 Faster R-CNN 的发展（Sec 1）[^src-detr]。
- 新设计带来训练、优化与小目标性能方面的新挑战；作者指出既有检测器花费多年改进才解决类似问题（Sec 5）[^src-detr]。
- 训练需要额外长的 schedule 并受益于 decoder 辅助损失（Sec 1）[^src-detr]。

## 在 wiki 中的位置

在 wiki 的设计谱系整理中，DETR 的 learned object queries + Transformer decoder 逐 query 并行解码是后续把可学习 query 引入其他领域解码器（包括时空预测）设计的对照起点；此谱系定位是 wiki 层面的组织，DETR 论文本身未做此声明。时序方向的 query 变体见 [[tqn]] 与 [[temporal-query-technique]]（可学习向量作注意力 Query），[[query-aggregate-attention]]（时间 token 对空间结构基的查询-聚合）。在本 wiki 中，DETR 也被用作非自回归并行解码在集合预测领域的对照例子，见 [[ar-vs-nar-decoding]] 与 [[generative-style-decoder]]。

## 相关页面

[[object-queries]] · [[source-detr]] · [[ar-vs-nar-decoding]] · [[generative-style-decoder]] · [[tqn]] · [[temporal-query-technique]] · [[query-aggregate-attention]]

[^src-detr]: [[source-detr]]
