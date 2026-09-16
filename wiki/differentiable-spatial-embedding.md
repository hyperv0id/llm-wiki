---
title: "可微空间嵌入（SE-L / SE-H）"
type: technique
tags:
  - spatial-embedding
  - llm
  - geohash
  - physics-informed
created: 2026-09-16
last_updated: 2026-09-16
source_count: 2
confidence: medium
status: active
---

# 可微空间嵌入（SE-L / SE-H）

把经纬度编成**对位置可求偏导**的连续向量。两个约束同时要求这样做：物理残差需要对空间坐标求导；无观测区域预测需要把地点语义与具体传感器实例解耦[^src-gencast]。GenCast 给两套方案，差别在输入可得性与是否参与训练[^src-gencast]。

## SE-L：LLM 文本描述

先由提示词把坐标、周边几何属性、POI、最近路段属性组织成描述文本 $L_t\in\mathbb{R}^{N_o\times S}$，送入**冻结的 LLaMA3-8B Instruct**，取末层隐藏状态最后一 token 的嵌入 $L_{llm}$[^src-gencast]。嵌入不更新，故不可导，GenCast 靠正弦时间嵌入 $TE_{enc}$ 提供梯度路径。

## SE-H：GeoHash + 字符 BERT

坐标先做 GeoHash 编码成定长字符串（长度决定精度），用预训练 character-BERT 编码得 $L'_{hash}\in\mathbb{R}^{N_o\times S\times d_{bert}}$，再经多层 Transformer encoder，沿字符维度均值池化得 $L_{hash}$[^src-gencast]。只依赖坐标，训练中持续更新。

时间嵌入按日内索引 $TE[i]=i \bmod T_d$ 做正余弦编码；STE 层把 $TE_{enc}$ 与 $L_{enc}$ 相加后与观测序列拼接为初始特征 $H^0$[^src-gencast]。

## 实证差异

| 数据集 | 更优变体 | 论文的解释 |
|---|---|---|
| PEMS07、PEMS08 | SE-L | 采集时间近，与当期 OpenStreetMap 信息匹配 |
| PEMS-Bay、METR-LA、Melbourne | SE-H | 前两者数据年代久远，环境变化导致文本嵌入与采集期错配；Melbourne 只覆盖小而同质的 CBD，SE-L 提不出可区分特征 |

消融显示耦合关系：GenCast-L 去掉物理约束后冻结嵌入明显变差（w/o-TE 误差最高），GenCast-H 的可训练嵌入在无物理约束下仍能工作[^src-gencast]。

## 与 LLMGeovec 的差别

[[source-geolocation-llm-st|LLMGeovec]]（AAAI 2025）同样用 LLM + OpenStreetMap，但定位是训练无关的增强器：生成后拼接即用，不参与训练，覆盖全球[^src-geolocation-llm-st]。GenCast 的 SE-L 沿用生成方式，改变了使用位置——进入物理残差的微分路径，并增设可训练分支 SE-H[^src-gencast]。

## 相关页面

- [[source-gencast|GenCast]] — 提出者
- [[lwr-traffic-pde|LWR 交通流偏微分方程]] — 要求可微的物理模块
- [[unobserved-region-forecasting|无观测区域交通预测]]

[^src-gencast]: [[source-gencast]]
[^src-geolocation-llm-st]: [[source-geolocation-llm-st]]
