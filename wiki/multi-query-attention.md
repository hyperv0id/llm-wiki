---
title: "Multi-Query Attention (MQA)"
type: technique
tags:
  - attention
  - kv-cache
  - efficient-inference
  - transformer
  - memory-bandwidth
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Multi-Query Attention (MQA)

保留 $H$ 组 query head，K/V 只留一份，全部 query head 共享同一组 $K$、$V$（§2.2、§2.4.1）[^src-shazeer-2019-mqa]。动机在于增量解码：整段并行时算术量 $\Theta(bnd^2)$ 对访存量 $\Theta(bnd+bhn^2+d^2)$ 的比值足够高，足以占满算力（§2.3.1）；解码按 token 串行，同一 $\Theta(bnd^2)$ 的算术量分摊到 $\Theta(bn^2d+nd^2)$ 的访存上，比值降为 $\Theta(n/d+1/b)$，在 $n\approx d$ 或 $b\approx1$ 时接近 1，受限于带宽（§2.4.1）[^src-shazeer-2019-mqa]。多头把同一份记忆存成 $H$ 份，$\Theta(n/d)$ 随之放大 $H$ 倍——这正是论文的着力点，也是「一个写头」命名的由来。

## 数据流与维度变换

记 $N$ 为并行 query token 数（增量解码取 $N=1$），$M$ 为被注意的 memory 位置数（自注意力中 $M=N$），$H$ 为 query head 数，$d$ 为模型维，$d_k=d_v$ 为单头 key/value 维且满足 $Hd_k=d$（实验配置 $d=1024$、$H=8$、$d_k=d_v=128$）。形状按单样本记作 $[\text{行},\text{列}]$；$b$ 在算术强度一节出现。

论文单头公式省略 $1/\sqrt{d_k}$ 缩放并折算进 $P_q$、$P_k$，不改变任何维度（§2.2）[^src-shazeer-2019-mqa]。

### Stage 1 · $H$ 组 query 投影、一份 K/V 投影

$$\begin{aligned}
Q &= X\,P_q && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, Hd_k]\to[N, Hd_k]\ \text{——每个 token 投出 } H \text{ 组 } d_k \text{ 维 query，消去 } d\\
K &= M\,P_k && \text{// } [M, \bcancel{d}]\times[\bcancel{d}, d_k]\to[M, d_k]\ \text{——只做一次投影：全部 query head 共享这份 } K\\
V &= M\,P_v && \text{// } [M, \bcancel{d}]\times[\bcancel{d}, d_v]\to[M, d_v]\ \text{——同上，} V \text{ 也只有一份，形状里没有 } H
\end{aligned}$$

多头版 $P_k,P_v$ 形状为 $[H,d,d_k]$，投影结果带出 $[M,H,d_k]$ 头维（einsum `bmd,hdk->bhmk`）；MQA 去掉参数张量中的字母 $h$（`bmd,dk->bmk`），$K$、$V$ 元素数由 $HMd_k$ 降至 $Md_k$，投影乘加数同比降至 $1/H$（§2.3、§3 代码清单）[^src-shazeer-2019-mqa]。$P_q$ 与 $P_o$ 不动，省下的参数靠加宽 $d_{ff}$ 补齐以对齐总参数量（§4.1：4096→5440）[^src-shazeer-2019-mqa]。

### Stage 2 · 各头独立打分，读同一份 K

$$\begin{aligned}
S^{(j)} &= Q^{(j)}{K}^{\top}/\sqrt{d_k} && \text{// } [N, \bcancel{d_k}]\times[\bcancel{d_k}, M]\to[N, M]\ \text{——第 } j \text{ 个头各得一个同形打分矩阵，} K \text{ 不按头复制}\\
A^{(j)} &= \operatorname{softmax}\!\left(S^{(j)}\right) && \text{// } [N, M] \xrightarrow{\text{沿 } M \text{ 维逐行归一化}} [N, M]\ \text{——每行是一个 query 对 } M \text{ 个位置的分布}
\end{aligned}$$

$H$ 个头叠成 $[H,N,M]$，与多头版同形；差别只在 $K$ 由 $H$ 份副本换成一份被读 $H$ 次。

### Stage 3 · 各头对同一份 V 加权

$$\begin{aligned}
O^{(j)} &= A^{(j)}V && \text{// } [N, \bcancel{M}]\times[\bcancel{M}, d_v]\to[N, d_v]\ \text{——消去 } M\text{，同一份 } V \text{ 被 } H \text{ 个头依次取用}
\end{aligned}$$

### Stage 4 · 拼头投影回残差流

$$\begin{aligned}
Y &= \operatorname{Concat}_{j}\!\left(O^{(j)}\right)P_o && \text{// } [N, \bcancel{Hd_v}]\times[\bcancel{Hd_v}, d]\to[N, d]\ \text{——与多头版完全一致，差异止于 K/V}
\end{aligned}$$

### Stage 5 · 单步解码的增量形态

自注意力单步（新 token $x_t$，已有 $M$ 个位置）：

$$\begin{aligned}
q_t &= x_tP_q && \text{// } [1, \bcancel{d}]\times[\bcancel{d}, Hd_k]\to[1, Hd_k]\ \text{——只有一个新 query，故 } N=1\\
K_t &= [\,K_{t-1};\, m_tP_k\,] && \text{// } [M, d_k]\oplus[1, d_k]\to[M{+}1, d_k]\ \text{——只追加一行，不复制 } H \text{ 份}\\
S_t &= q_t{K_t}^{\top}/\sqrt{d_k} && \text{// } [1, H\bcancel{d_k}]\times[\bcancel{d_k}, M{+}1]\to[1, H, M{+}1]\ \text{——新 token 的 } H \text{ 个 query 同时乘整段 } K\\
O_t &= \operatorname{softmax}(S_t)V_t && \text{// } [1, H, \bcancel{M{+}1}]\times[\bcancel{M{+}1}, d_v]\to[1, H, d_v]\ \text{——} H \text{ 个头各自对同一份 } V_t \text{ 加权}
\end{aligned}$$

每步成本由重读整段 $K_t,V_t$ 决定而非乘加数：$M$ 随生成长度增长，重读量线性增长，$H$ 决定重读几份。

## 关键推导：解码每步的算术强度

一步解码（batch $b$，memory 长 $M$，fp32 每元素 4 字节）：

$$\begin{aligned}
B_{KV}^{\mathrm{MH}} &= 4\cdot 2\,bHMd_k && \text{// } K,V\ \text{两个张量}\times b\times H\times M\times d_k\ \text{个元素——各 } H \text{ 份，每份都要读}\\
B_{KV}^{\mathrm{MQ}} &= 4\cdot 2\,bMd_k && \text{// } 同上少一个 } H\ \text{——只有一份}\\
F_{\mathrm{attn}} &= 2\left(bHMd_k+bHMd_v\right) && \text{// } 打分 } bHMd_k \text{ 次乘加 + 加权 } bHMd_v \text{ 次乘加}\ \text{——共享不删这两项，} N=1 \text{ 时两结构相等}\\
I^{\mathrm{MH}} &= \frac{F_{\mathrm{attn}}}{B_{KV}^{\mathrm{MH}}}=\frac{bHM(d_k+d_v)}{4bHMd_k} && \text{// } =\frac{1+d_v/d_k}{4}\ \text{FLOP/byte}\ \text{——}d_k=d_v\ \text{时 }0.5\\
I^{\mathrm{MQ}} &= \frac{F_{\mathrm{attn}}}{B_{KV}^{\mathrm{MQ}}}=\frac{bHM(d_k+d_v)}{4bMd_k} && \text{// } =H\cdot\frac{1+d_v/d_k}{4}\ \text{FLOP/byte}\ \text{——}H=8\ \text{时 }4
\end{aligned}$$

算术强度只随共享份数 $H$ 放大，乘加数不变：MQA 省的不是算力，而是同一批算力所需的字节数。论文估计机器平衡点的算力比带宽高约两个数量级（§2.3.1）；$0.5$ 与 $4$ FLOP/byte 都落在这条线之下，解码由带宽定速[INFERENCE]。

代入测速设定（$b=1024$ 条序列、$M=128$、$H=8$、$d_k=d_v=128$，§4.3）：

$$B_{KV}^{\mathrm{MH}}=4\times2\times1024\times8\times128\times128=1.07\ \text{GB/步},\qquad B_{KV}^{\mathrm{MQ}}=134\ \text{MB/步}$$

$$F_{\mathrm{attn}}=2\times1024\times8\times128\times(128+128)=537\ \text{MFLOP/步}$$

字节比 $=H=8$，乘加数逐个相同。

### 「$H$ 份 K/V 重写」的具体开销

K/V 不只被读，每步还要重写：实现把新一行拼到旧张量上（`new_K = concat([prev_K, expand_dims(...)])`），固定形状下每步新建整段 $[b,H,M,d_k]$，即 $1024\times8\times128\times128=1.34\times10^{8}$ 个元素（$K$、$V$ 各 0.537 GB）各读一遍写一遍；MQA 在同样长的时间步里只需新建 $[b,M,d_k]$（$K$、$V$ 各 67 MB）（§2.4 代码清单）[^src-shazeer-2019-mqa]。query 是读端，可以有 $H$ 种不同的读法；K/V 是写入记忆的那一端，留一个写头就够——标题的 write-head 即指此事，正文表述为「去掉 K/V 的 head 维、只保留 query 的 head 维」（§2.4.1）[^src-shazeer-2019-mqa][INFERENCE]。

把每步 K/V 的读、写（各一遍，共 $4bHMd_k$ 对 $4bMd_k$ 次元素访问，每核 $b=128$）与投影参数（$4d^2$ 对 $2d^2+2d^2/H$ 个元素）一并计入，单步总访存比约 $6.6\times$；实测解码加速 $12.1\times$（46→3.8 µs）明显高于该比值，差额来自逐 head 读-拼-写的 kernel 行为与每步固定开销，论文未拆解[INFERENCE]。

## 证据

**翻译质量**（WMT14 EN-DE，6 层、$d=1024$、$d_{ff}=4096$、$H=8$、$d_k=d_v=128$、211M 参数、100K 步、32 核 TPUv3 约 2 小时，§4.1）：Table 1 中 dev $\ln(\text{PPL})$ 1.424→1.439、dev BLEU 26.7→26.5、test greedy 27.7→27.5、test beam-4 28.4→28.5（多 query 最高）[^src-shazeer-2019-mqa]。同一参数预算下削 $H$ 或削 $d_k$ 的方案更差：$h=1$ 为 1.518/25.8，$(h,d_k)=(2,64)$ 为 1.480/26.2，$(4,32)$ 为 1.488/26.1，$(8,16)$ 为 1.513/25.8（Table 1）[^src-shazeer-2019-mqa]。

**语言模型质量**（Billion-Word，6 层、$d_{ff}=8192$、192M 参数、136K 步、batch 64K token，§4.1）：dev per-word PPL 29.9→30.2，削头方案为 30.9–31.2（Table 3）[^src-shazeer-2019-mqa]。

**速度**（Table 2，单张 TPUv2 8 核，序列长 128，逐输出 token 微秒）：训练 13.2→13.0；encoder 1.7→1.5；decoder 1.7+46→1.5+3.8；beam-4 2.0+203→1.6+32[^src-shazeer-2019-mqa]。训练与 encoder 整段并行、算力受限，耗时几乎不变；收益全在串行解码。

**与局部注意力的正交性**：decoder 自注意力窗口限 32 时，多头 local 为 23 µs、多 query local 为 3.3 µs，与 MQA 叠加后 46→3.3 µs（Table 2、§4.1）[^src-shazeer-2019-mqa]。

## 边界

实测质量代价轻微：翻译 dev 两指标均退，test beam-4 高 0.1 BLEU；LM 的 dev PPL 退 0.3。削 $H$/$d_k$ 的四行对照均保持 $Hd_k=128$，未设「8 个 query head 配 2 或 4 组 K/V」等中间共享档位，除局部窗口外亦无质量-速度折中曲线（Table 1、Table 3）[^src-shazeer-2019-mqa]。该设计属训练期层定义（§3），非事后改造已训练多头模型；将现有模型 $H$ 份 K/V 合并为一份（取平均或选一个）的代价与质量，论文未测。评测仅设两组翻译与 LM 任务，$M$ 固定为 128（local 变体为 32），更长上下文与更高 K/V 访存占比下的收益未做实验。测速采用固定形状 padding，每步耗时相同并摊平序列早期的实际成本，阶梯增长实现在早期更快（脚注 1）[^src-shazeer-2019-mqa]。

## 相关页面

[[source-shazeer-2019-mqa]] 为源摘要；[[transformer-architecture]] 为被替换的多头基线与拼头投影；[[scaling-factor-sqrt-dk]] 为本文公式省掉的缩放因子；[[kv-cache-compression]] 为压缩 KV 条目本身而不减少副本数的另一路径；[[native-sparse-attention]] 说明块稀疏访存节省以组内共用同一批 KV 为前提；[[deepseek-sparse-attention]] 对应在 MLA 的 MQA 模式上实例化逐 query 选择。

[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
