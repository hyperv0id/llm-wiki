# Ingest 报告：FSTLLM — 补全（ICML 2025）

> **状态**：先前于 2026-06-09 批量 ingest 已创建 source-summary 和 entity 页。本次为补全 re-examination。

## 创建

- [[llm-enhanced-graph-construction]] — WHY：FSTLLM 的核心技术创新之一（LLM 编码节点文本 → 图注意力 → α-Entmax 稀疏邻接矩阵），可作为可复用的图构建模式独立引用
- [[domain-knowledge-injection]] — WHY：FSTLLM 的核心技术创新之二（六组件 prompt + QLoRA SFT 校准数值预测），是多模态外生信息与 LLM 推理融合的范式性技法

## 修改

- [[fstllm]] — WHY：在 Connections 部分添加两条新技术页的交叉引用；更新 `last_updated`
- [[source-fstllm]] — WHY：在 Links 部分添加新技术页链接；更新 `last_updated`
- [[index]] — WHY：在 Techniques 部分以字母序插入两个新条目；更新 `last_updated`
- [[log]] — WHY：记录本次 re-ingest 操作

## 新建交叉链接

- [[llm-enhanced-graph-construction]] ↔ [[fstllm]]
- [[domain-knowledge-injection]] ↔ [[fstllm]]
- [[llm-enhanced-graph-construction]] ↔ [[alpha-entmax]]
- [[domain-knowledge-injection]] ↔ [[few-shot-traffic-forecasting]]

## 未创建

- **QLoRA** — 通用技术（Xu et al., ICLR 2024），非 FSTLLM 原创，仅有 FSTLLM 一个引用不足以支撑独立技术页；待未来多源积累后创建
- **Graph Diffusion Convolution** — 复用 GTS 已有技术，非 FSTLLM 原创

## 源文件

`raw/fstllm-icml2025.pdf`（不可变，已存在）
