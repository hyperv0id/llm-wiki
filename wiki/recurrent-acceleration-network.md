---
title: "Recurrent Acceleration Network (RAN)"
type: technique
tags:
  - recurrent
  - parallelization
  - time-series
  - efficiency
  - rnn
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Recurrent Acceleration Network (RAN)

RAN 是 WITRAN（NeurIPS 2023）提出的递推加速框架：对二维网格上的双向递推，把 $r+c=\text{const}$ 的反对角线（一条"波前"）上互不依赖的格点打包成 slice 批量计算，串行递推步数从序列长度 $L=R\times C$ 降到 $R+C-1$；取 $R,C=O(\sqrt L)$ 时论文记作时间复杂度 $O(\sqrt L)$、内存 $O(L)$（§3.3，Table 2）[^src-witran-jia-2023]。

## 可并行性的来源

传统 RNN 相邻时间步串行等待；WIT 重排成 $R\times C$ 网格后，某格算完，其右邻（horizontal）与下邻（vertical）互不依赖即可开始（§3.3）——同一条反对角线上的格点恰无先后关系，即"water-wave"类比的计算含义。每 slice 一次批量矩阵乘，slice 数=串行迭代次数（Figure 4(c) 绿框个数）。

## 实现要点（Algorithm 3，PyTorch）

- **错位填充**：第 $r$ 行输入整体右移 $r$ 位填入 `input_transfer[slice_num, batch, slice_len, input_size]`，使每条反对角线在时间轴上对齐为同一列；`for slice in range(...)` 循环内对 slice 内所有点一次性 `linear(cat([h_row, h_col, x]))`。
- **门切分**：每层输出 $6d_{model}$，其中 $4d_{model}$ 过 sigmoid（横/纵各一对 update/output gate）、$2d_{model}$ 过 tanh（横/纵候选），与 GSC 式 (4) 一一对应。
- **纵向传递**：每步末 `torch.roll` 把纵向状态整体下移一行；横向传递由 slice 内位置偏移天然完成。
- **RAN-max**：slice 尺寸取 $\max(R,C)$ 而非 $\min(R,C)$——后者在行数 < 列数时中间变量被占满、需反复 shift+reset；以少量空间换掉 reset（App. D.1，Figure 9）。
- 可选层残差（`layer_res`）：各层输出与第 0 层输出相加。

## 复杂度口径辨析

论文称"reduced the time complexity to $O(\sqrt L)$"（§3.3），但每格的 GSC 仍各算一次，**总算术量仍是 $O(L)$**；降为 $O(\sqrt L)$ 的是串行步数（关键路径长度 / 循环迭代次数）[INFERENCE，由 Algorithm 3 代码结构确证]。内存 $O(L)$ 的论证：须保存每点输出（$O(L)$），并行缓冲仅 $O(\sqrt L)$。与 Table 2 中 RNN $O(L)$ 对照时须区分这两个口径，否则自相矛盾。

## 通用性

论文两处自述 RAN 是 **generic / universal 加速框架**，可整合其他模型做信息融合与传递（§1、§3）：加速对象是任何 2D 网格双向递推，不绑定 GSC；实现以 `num_layers / water_rows / water_cols / res_mode` 参数化为通用 encoder 类（Algorithm 3）[^src-witran-jia-2023]。

## 边界

- slice 间仍有信息等待，作者自述 Python 实现效率非最优，计划做成 C++ 接口（类似 `nn.GRU/nn.LSTM` 的融合算子，§5）。
- 并行格点增加内存，靠 $O(\sqrt L)\ll O(L)$ 才可接受；RAN-max 相对 RAN-min 明确引入额外空间。
- WITRAN 的消融显示 RAN 对精度"无显著影响"（App. G），效率证据只有 Figure 5 定性曲线，全文无加速倍数或显存绝对值。

## 引用

[^src-witran-jia-2023]: [[source-witran-jia-2023]]
