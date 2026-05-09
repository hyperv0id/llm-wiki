# Ingest 报告：source-snr-t-bias (2604.16044)

## Source
- **标题**: Elucidating the SNR-t Bias of Diffusion Probabilistic Models
- **作者**: Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, Kun Zhan (AMAP Alibaba Group + Lanzhou University)
- **会议**: CVPR 2026

## 创建
- wiki/source-snr-t-bias.md — source-summary 页面，>100 行，含完整数学公式推导（Eq. 1-21, Theorem 5.1 证明）
- wiki/snr-t-bias.md — 概念页面，定义 SNR-t Bias、与 exposure bias 的关系、理论证明
- wiki/dcw.md — 技术页面，DCW 方法详述（像素域→小波域→动态权重调度）

## 修改
- wiki/diffusion-model.md — 更新 source_count (12→13)，添加 [[snr-t-bias]] 和 [[dcw]] 链接到相关概念部分
- wiki/tweedies-formula.md — 更新 source_count (0→1)，添加 SNR-t Bias 证明中使用 Tweedie 公式的描述及引用
- wiki/index.md — 添加 [[source-snr-t-bias]] 到 Sources，[[snr-t-bias]] 到 Concepts，[[dcw]] 到 Techniques
- wiki/log.md — 追加 ingest 记录

## 新建交叉链接
- [[snr-t-bias]] ↔ [[diffusion-model]] — SNR-t bias 是扩散模型推理阶段的核心偏置
- [[snr-t-bias]] ↔ [[tweedies-formula]] — Tweedie 公式是 SNR-t bias 理论证明的基础工具
- [[snr-t-bias]] ↔ [[dcw]] — DCW 是缓解 SNR-t bias 的具体方法
- [[dcw]] ↔ [[diffusion-model]] — DCW 作为扩散模型的即插即用后处理

## 质量统计
- source-snr-t-bias.md: 约 180 行，含 15+ 条数学公式
- snr-t-bias.md: 约 60 行
- dcw.md: 约 100 行
