# Ingest 报告：Multi-scale Attention Flow (MANF) / arXiv:2205.07493

## 创建
- wiki/source-maf.md — WHY：source-summary，slug `src-maf`；记录 MANF 非自回归多尺度注意力 + 条件 RealNVP 的核心论点、方法、实验与局限
- wiki/manf.md — WHY：实体页面，模型架构、复杂度、六数据集结果与消融
- wiki/multi-scale-attention.md — WHY：技术页面，MANF 多尺度窗口注意力 + 动态相对位置形式化

## 修改
- wiki/index.md — WHY：Sources / Entities / Techniques / Concepts 登记新页面与关联更新
- wiki/log.md — WHY：追加 ingest 日志
- wiki/generative-time-series-forecasting.md — WHY：补充归一化流/NAR 方法分支与 MANF 条目
- wiki/normalizing-flow.md — WHY：补充时序条件流应用与 MANF / RealNVP 链接
- wiki/ar-vs-nar-decoding.md — WHY：加入 MANF 作为 NAR + 精确似然流的早期证据
- wiki/timegrad.md — WHY：交叉链接 MANF（NAR 流 vs AR 扩散）

## 新建交叉链接
- [[source-maf]] ↔ [[manf]] ↔ [[multi-scale-attention]]
- [[manf]] ↔ [[normalizing-flow]] / [[generative-time-series-forecasting]] / [[ar-vs-nar-decoding]]
- [[manf]] ↔ [[timegrad]]
- [[multi-scale-attention]] ↔ [[multi-scale-linear-prediction]]（相关多粒度）

## 源文件
- 仓库内：`raw/2205.07493.pdf`（不可变，已存在）
- 外部任务路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/2205.07493.pdf`
