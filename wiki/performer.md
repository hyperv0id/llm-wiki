---
title: "Performer：FAVOR+ 线性注意力 Transformer"
type: technique
tags:
  - transformer
  - linear-attention
  - kernel-methods
  - random-features
  - efficiency
  - iclr-2021
created: 2026-08-30
last_updated: 2026-08-30
source_count: 3
confidence: medium
status: active
---

# Performer：FAVOR+ 线性注意力 Transformer

Performer 是 Choromanski 等人（Google / Cambridge / DeepMind / Alan Turing Institute）提出的 Transformer 架构，论文将其定位为：能以可证明精度估计常规 softmax 全秩注意力、但只用线性空间与时间复杂度、不依赖稀疏或低秩先验的架构；论文自述这是首个具备这些性质的 Transformer（摘要、Sec 1，结论处措辞为 "to our knowledge the first"）[^src-performer]。机制核心是 FAVOR+（Fast Attention Via positive Orthogonal Random features），数学细节见 [[positive-random-features|正随机特征（PRF）]][^src-performer]。

## 问题

常规 dot-product attention 需显式构造并存储 L×L 注意力矩阵，时间 O(L²d)、空间 O(L²+Ld)，长序列下不可行（Sec 2.1）[^src-performer]。论文对已有高效方案的归类：多数方法给注意力加结构先验——局部窗口、稀疏模式、池化压缩、聚类/分桶、LSH 分组等；另一类用低秩核替换 softmax 做稠密注意力（Sec 1）[^src-performer]。论文认为这些方案不是去近似常规注意力，而是提出更简单但受限的替代机制，且缺少表示能力方面的严格保证（Sec 1，论文自述口径）[^src-performer]。与之相对，Performer 的路线是直接无偏估计 softmax 核本身（Sec 1）[^src-performer]。

## 机制：FAVOR+ 的两个组件

### FA：广义核化注意力 + 结合律重排

FAVOR+ 作用于形如 A(i,j)=K(qᵢ,kⱼ) 的注意力矩阵，其中核 K(x,y)=E[φ(x)ᵀφ(y)] 由随机特征映射 φ: R^d → R^r 定义（Sec 2.2）[^src-performer]。利用矩阵乘法结合律，注意力输出按 Q′((K′)ᵀV) 顺序计算（Q′、K′ 的行分别为 φ(qᵢ)、φ(kⱼ)），无需构造 L×L 矩阵：时间 O(Lrd)、空间 O(Lr+Ld+rd)（Sec 2.2）[^src-performer]。这构成 FA 部分；剩下的 OR+ 部分回答两个问题——该核化形式能否覆盖 softmax 注意力，以及能否取 r ≪ L（Sec 2.2）[^src-performer]。注意力矩阵本身可在 O(Ld²log d) 时间内近似到任意精度，作为对比，LSH 类方法（Reformer）为 O(Ld²log L)（Sec 2.1）[^src-performer]。

### OR+：softmax 核的随机特征估计

- **三角特征为什么不够**：sin/cos 特征（SMm^trig）给出 softmax 核的无偏估计，但特征值可负；核值趋 0（大量低相关 token 对）时其 MSE 发散（Lemma 2），负值使归一化对角阵出现负数，导致训练不稳定或次优（Sec 2.3，论文归因这也是此前无人提出鲁棒随机特征 softmax 近似的原因之一）[^src-performer]。
- **正随机特征（PRF）**：Lemma 1 给出恒等式 SM(x,y)=E[exp(ωᵀx−‖x‖²/2)·exp(ωᵀy−‖y‖²/2)]，ω~N(0,I)；对应的无偏估计器只用正值特征，且 MSE 随核值趋 0 而趋 0（Sec 2.3, Lemma 1/2）[^src-performer]。详见 [[positive-random-features|PRF 子页]]。
- **正交随机特征（ORF）**：对各向同性分布用 Gram-Schmidt 把 m 个样本严格正交化，保持无偏，要求 m ≤ d（Sec 2.4）[^src-performer]。论文称首次证明正交化能降低 softmax/高斯核估计的方差且对任意维度 d 成立（此前结果仅对大 d 渐近成立），并给出首个严格小于非正交情形的指数级小偏差概率界（Sec 2.4, Theorems 2/3 与附录 Theorems 5/6）[^src-performer]。
- **一致收敛**：存在只依赖嵌入维度 d、精度 ε 与 query/key 范数上界 R、不依赖序列长度 L 的投影数 m，使注意力矩阵以任意常数概率满足 ‖Â−A‖∞ ≤ ε（Theorem 4；附录 Theorem 7 为更一般版本）[^src-performer]。附录讨论指出：固定 m 时 FAVOR 无法近似无限长序列上的 hard attention——序列变长时常规注意力本身也需要更大的向量范数来让 softmax 集中；作者报告在该文实验长度内该限制未显现（Appendix F.6）[^src-performer]。
- **因果（单向）注意力**：通过 prefix-sum（并行 O(log L) 深度、共 O(L) 步）沿序列累加 φ(k)ᵀv 与 φ(k)，同样不构造 L×L 矩阵；算法 1 给出双向/单向统一伪代码，时间 O(Lmd)（Sec 2.1、Appendix B/B.1/B.3）[^src-performer]。

## 命名细节：Performer 与 Performer-SOFTMAX

附录 A 说明：除非特别标注（如 "Performer-SOFTMAX"），"Performer" 默认指 generalized attention 设置——即用 ReLU 核（kernel=ReLU，256 特征）替代 softmax 核的最优设置；softmax 近似设置单独标注（命名约定见附录 A 开头说明，两组默认超参见 A.3/A.4）[^src-performer]。阅读引用 Performer 的实验文献时需要区分这两种设置：[[wire|WIRE]] 与 [[graphgps|GraphGPS]] 使用的 "ReLU Performer" 即 generalized 设置[^src-2509-22259][^src-graphgps]。

## 实验（作者报告口径）

- **计算成本**：默认规模 (nheads, nlayers, dff, d)=(8, 6, 2048, 512)、V100 16GB 上，Performer 反向传播时间随 L 近线性、内存次二次（不存储显式注意力矩阵），接近注意力直接返回 V 矩阵的 "X"(OPT) 理论线（Sec 4.1, Fig 3）[^src-performer]。
- **近似误差**：L=4096、d=16 下，正交特征误差低于 IID 特征、正特征低于三角特征，两项合起来验证 PORF 机制（Sec 4.2, Fig 4）[^src-performer]。
- **文本建模**：LM1B 上把预训练 Transformer 权重直接迁入 Performer，初始准确率 0.07，经小量微调在远少于原训练步数内恢复（Fig 5）；PG-19 上三角特征训练高度不稳定，正特征加重采样才能匹配常规 Transformer 的困惑度，SMREG 正则化加快收敛（Sec 4.3, Fig 5）[^src-performer]。论文把误差传播归因于非注意力组件的 Lipschitz 放大（Sec 4.3, Fig 14）[^src-performer]。
- **蛋白质序列建模**：36 层模型（8 头、dff=1024、d=512）在 TrEMBL 上训练；论文报告 Reformer 与 Linformer 在该数据集上准确率显著下降，generalized ReLU 核的 Performer 在 (U) 与 (B) 两种设置中取得最高准确率；Table 2 的 test 准确率：UNI Transformer 30.80 vs Performer (generalized) 31.58、BID Transformer 33.32 vs Performer (generalized) 36.09、Performer (softmax) 33.00；OOD 集上 Performer (generalized) 为 18.44（UNI）/24.10（BID），低于 Transformer 的 19.70/25.07（Sec 4.4, Fig 6, Table 2）[^src-performer]。注意正文 "softmax 近似与精确 softmax Transformer 同精度" 的说法对应 BID test 33.00 vs 33.32 的近似持平，正文与表格数字应分别看待（Sec 4.4 vs Table 2）[^src-performer]。
- **长序列**：ImageNet64（L=12288，常规 Transformer 不可行）上 Performer/6 层匹配 Reformer/12 层、Performer/12 层匹配 Reformer/24 层；硬件相关（TPU/GPU）条件下 (U) 设置中 Performer 经 Jax 优化可比 Reformer 快约 2 倍（Sec 4.5, Fig 7）[^src-performer]。拼接 TrEMBL 长序列（L=8192）上常规 Transformer 每芯片 batch 1 仍内存溢出，缩小版 3 层 Transformer 准确率被限制在约 19%，Performer 以标准架构、batch 8 持续训练到约 24%（Sec 4.5, Fig 7）[^src-performer]。
- **设置消融**：ImageNet64 (U) 100K 步时 Performer-ReLU / Performer-Softmax / Performer-Softmax (SMREG) 分别为 3.67 / 3.69 / 3.67 BPD（Appendix D.2）[^src-performer]。
- **Long Range Arena**：附录转引 LRA 基准结果（Tay et al., 2021, Fig 19），作者报告 Performers 在速度大于 100 examples/sec 的可扩展 Transformer 方法中取得最高 LRA 分数（Appendix D.5，作者自述口径）[^src-performer]。
- **对照 Linear Transformer**：ProGen 设置下以 elu(x)+1 为特征映射的 Linear Transformer（Katharopoulos et al., 2020）3 个种子均在训练早期梯度爆炸终止，双向设置在约 125K 步处同样爆炸（Appendix D.4, Fig 18）[^src-performer]。
- **训练即插即用**：Performer 使用与常规 Transformer 相同的训练超参数取得竞争性结果，作者称 FAVOR 可作为不需大量调参的 drop-in 组件（Appendix A.2，作者自述）[^src-performer]。

## 论文自述局限

- 固定 m 下无法近似无限长序列的 hard attention；m 依赖 d、ε 与 query/key 范数上界 R（Theorem 4、Appendix F.6）[^src-performer]。
- 小的注意力近似误差可被多层网络的 Lipschitz 常数放大，有时需要很紧的近似；预训练权重的向后兼容因此需要小量微调（Sec 4.3, Fig 5/Fig 14）[^src-performer]。
- 三角特征在注意力场景训练不稳定，随机特征需周期性重采样以避免个别"unlucky"特征集造成的退化（Sec 4.2/4.3, Fig 5, Appendix D.2）[^src-performer]。
- ORF 机制要求 m ≤ d；m > d 时可在 W 的每个 d×d 块内局部使用正交化（Sec 2.4, Appendix B.2）[^src-performer]。

## 在本 wiki 中的位置

Performer 是 [[linear-attention-unified-framework|线性注意力统一框架]]中「核随机特征」路线的代表性机制（区别于 Mamba 的 SSM 循环形式）。下游应用实例：[[long-sequence-feature-extractor|BigST 的 LSFE]] 用 PRF 线性化时间自注意力、[[linearized-spatial-convolution|BigST 的 LSC]] 用同一 PRF 核分解自适应邻接、[[urbanpg|UrbanPG]] 的 STCA 模块用随机特征映射做线性时空注意力；[[graphgps|GraphGPS]] 把 Performer 列为可替换的全局注意力模块之一；[[spectral-kernel-linear-attention]] 把 [[wire|WIRE]] 的旋转读作核的随机特征，与 Performer 同属随机特征线性注意力谱系。

## 相关页面

- [[positive-random-features]] — softmax 核估计的数学机制（trig vs PRF、ORF、SMREG）
- [[linear-attention-unified-framework]] — 线性注意力与 Mamba 的统一框架
- [[long-sequence-feature-extractor]] — BigST 时间维 PRF 线性化
- [[linearized-spatial-convolution]] — BigST 空间维 PRF 核分解
- [[graphgps]] — 以 Performer 为可替换注意力组件的图 Transformer 框架
- [[wire]] — 图旋转位置编码，实验骨干为 GraphGPS + ReLU Performer
- [[spectral-kernel-linear-attention]] — 旋转 = 随机特征的线性注意力读法
- [[urbanpg]] — STCA 使用随机特征映射线性注意力
- [[source-performer]] — 论文源摘要

[^src-performer]: [[source-performer]]
[^src-2509-22259]: [[source-2509-22259]]
[^src-graphgps]: [[source-graphgps]]
