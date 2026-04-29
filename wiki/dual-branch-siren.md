---
title: "Dual-Branch SIREN Architecture"
type: technique
tags:
  - positional-encoding
  - siren
  - neural-representation
  - temporal-modeling
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Dual-Branch SIREN Architecture

SIREN-RoPE 中用于将时间戳映射为旋转角的神经网络架构，结合周期性 SIREN 分支和非周期性 DNN 分支[^src-siren-rope]。

## 架构公式

$$f_\phi(T) = f_{sin}(T) + f_{DNN}(T)$$

- $f_{sin}$: 周期性分支，使用 SIREN 架构
- $f_{DNN}$: 非周期性分支，使用标准 MLP[^src-siren-rope]

## 分支 1：周期性 SIREN

使用周期激活函数 $\sin(\omega_0 Wx + b)$ 的全连接网络：

- **优势**：能够自主发现数据中隐藏的时间周期（如日/周/月周期）
- **原理**：SIREN 的谱偏差克服能力使其擅长学习多尺度周期函数
- **应用**：捕获用户行为的周期性模式（日间活跃度、周末vs工作日）[^src-siren-rope]

## 分支 2：非周期性 DNN

使用标准 ReLU 激活的全连接网络：

- **优势**：能够捕获单调趋势
- **应用**：近因衰减（recency decay）——近期交互应获得更高注意力权重
- **注意**：即使输入是周期性的 (cos, sin) 对，DNN 仍保持建模周期分量的灵活性[^src-siren-rope]

## 实现细节

- SIREN 分支：2 层，每层 64 隐藏单元
- DNN 分支：2 层，每层 64 隐藏单元
- 输入：5 维时间特征分解 $t(T)$
- 输出：$d_k/2$ 维旋转角向量[^src-siren-rope]

## 消融实验结论

- **仅 DNN + 全时间特征**：达到最佳 NE，说明 (cos, sin) 对已提供周期结构
- **仅 SIREN**：性能下降明显
- **保留 SIREN 分支的理由**：自主发现未在手动特征工程中指定的隐藏周期[^src-siren-rope]

## 相关页面

- [[siren-rope]] — 主 entity 页面
- [[temporal-rotation]] — 时间旋转概念
- [[ordinal-temporal-fusion]] — 序数-时间融合技术

[^src-siren-rope]: [[source-siren-rope]]