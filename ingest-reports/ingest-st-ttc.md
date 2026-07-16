# Ingest 报告：ST-TTC (NeurIPS 2025 Spotlight)

## 创建
- `wiki/source-st-ttc.md` — WHY：300-500 字 source-summary，覆盖核心思想（test-time computing via learning with calibration）、方法（SD-Calibrator + Flash Gradient Update）、实验（6 backbone × 6 数据集）和局限性
- `wiki/st-ttc.md` — WHY：实体页，详细记录 ST-TTC 的 motivation、architecture、理论保证、结果表和局限性
- `wiki/test-time-computing-st.md` — WHY：概念页，形式化 test-time computing 范式及其与 OOD/continual/TTT/online 的辨析，阐述 label autocorrelation 和 timeliness 两个使能属性
- `wiki/spectral-domain-calibration.md` — WHY：技术页，SD-Calibrator 的完整设计（空间感知分解→分组调制→逆变换）、复杂度分析和理论界
- `wiki/flash-gradient-update.md` — WHY：技术页，FIFO streaming memory queue + 单样本单步梯度更新机制、信息泄漏防护和受控下降理论

## 修改
- `wiki/traffic-forecasting.md` — WHY：在 Test-Time Adaptation & Computing 节添加 ST-TTC 条目及性能数据
- `wiki/index.md` — WHY：在 Sources、Entities、Concepts、Techniques 四个类别添加对应条目
- `wiki/log.md` — WHY：记录 2026-06-09 批量摄入（13 篇论文之一）

## 补全交叉链接（2026-07-15）
- `[[label-autocorrelation]]` ↔ `[[st-ttc]]` — WHY：ST-TTC 的核心使能属性是 spatio-temporal data 的 label autocorrelation，二者互为关键交叉引用
- `[[label-autocorrelation]]` ↔ `[[test-time-computing-st]]` — WHY：test-time computing 范式依赖 label autocorrelation 才得以做显式监督校准
- `[[label-autocorrelation]]` ↔ `[[source-st-ttc]]` — WHY：source-summary 页面中多次提及 label autocorrelation，应建立 wikilink

## 备注
- 原始 PDF：`raw/st-ttc-neurips2025.pdf`（2026-06-09 摄入时拷贝）
- 用户提供副本 hash：`e7e9f2530c3507656bd08a1459ac2699`，与 raw/ 中版本（`1e2974d0a928585b253cc1dc27837a56`）不同但内容一致（NeurIPS 2025 同一论文的不同版本/元数据差异），按 raw/ 不可变策略不重复拷贝
- 论文原始 slug：`st-ttc`，source slug：`src-st-ttc`
