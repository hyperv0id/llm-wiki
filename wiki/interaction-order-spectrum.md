---
title: "Interaction 阶谱（Interaction-Order Spectrum）与残差缩放的阶压制"
type: technique
tags:
  - residual-networks
  - interaction-order
  - scaling-analysis
  - interpretability
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Interaction 阶谱与残差缩放的阶压制

Möbius 系数给出 $2^L-1$ 个可加项后，下一个问题是哪些项、哪一阶在承载功能贡献。论文用平方系数质量在阶上的分布回答，并用推理期残差缩放做因果干预[^src-resnets-route-without-routers]。系数见 [[residual-interaction-decomposition]]。

## 定义

对读数 $r\in\{v,h\}$ 与幅值 $a_S^r$（标量取 $|\Delta_S v_x|$，向量取 $\lVert\Delta_S h_x\rVert_2$，见 [[residual-interaction-decomposition|Stage 3]]）[^src-resnets-route-without-routers]：

$$E_k^r=\sum_{|S|=k}\left(a_S^r\right)^2,\qquad \widetilde E_k^r=\frac{E_k^r}{Z^r},\qquad C^r_{\le q}=\frac{\sum_{j=1}^{q}E_j^r}{Z^r},\qquad \kappa^r=\frac{\sum_{j=1}^{L}j\,E_j^r}{Z^r},\qquad M_k^r=\frac{E_k^r}{\binom{L}{k}}$$

$Z^r=\sum_{j=1}^{L}E_j^r$ 为非零阶总质量（不含 shortcut 基线）[^src-resnets-route-without-routers]。$M_k=\cdot/\binom{L}{k}$ 校正「中间阶天生项数多」的组合效应：若峰只由组合数造成，$M_k$ 应随 $k$ 递减。

## 自然工作点：峰在中高阶

| 模型 | 读数 | 峰值阶 $k$（$\widetilde E_k$） | $C_{\le3}$ | $C_{\le L/2}$ | $\kappa$ | $M_L/M_1$ |
|---|---|---|---|---|---|---|
| ResNet-18 | 标量 | 5（0.2721） | 0.1417 | 0.3545 | 5.001 | 32.21 |
| ResNet-18 | 向量 | 5（0.2951） | 0.1356 | 0.3675 | 4.929 | 22.31 |
| ResNet-34 | 标量 | 10（0.1951） | 0.001153 | 0.2421 | 9.887 | 341.4 |
| ResNet-34 | 向量 | 10（0.1956） | 0.001116 | 0.2419 | 9.884 | 296.4 |

（Table 4，$\lambda=1$；ResNet-18 $N=10{,}000$、$L=8$，ResNet-34 $N=1{,}000$、$L=16$。）完整阶谱见 Table 5[^src-resnets-route-without-routers]：

| $k$ | 18-标量 | 18-向量 | 34-标量 | 34-向量 |
|---|---|---|---|---|
| 1 | 0.008619 | 0.007611 | 0.000038 | 0.000036 |
| 2 | 0.032546 | 0.027096 | 0.000212 | 0.000196 |
| 3 | 0.100549 | 0.100901 | 0.000903 | 0.000884 |
| 4 | 0.212804 | 0.231893 | 0.003625 | 0.003595 |
| 5 | 0.272092 | 0.295098 | 0.012134 | 0.012054 |
| 6 | 0.231087 | 0.220151 | 0.032453 | 0.032361 |
| 7 | 0.111121 | 0.096990 | 0.070109 | 0.070110 |
| 8 | 0.031182 | 0.020260 | 0.122593 | 0.122682 |
| 9 | — | — | 0.172787 | 0.172977 |
| 10 | — | — | 0.195111 | 0.195567 |
| 11 | — | — | 0.174654 | 0.175310 |
| 12 | — | — | 0.121618 | 0.121884 |
| 13 | — | — | 0.063775 | 0.063326 |
| 14 | — | — | 0.023746 | 0.023134 |
| 15 | — | — | 0.005600 | 0.005306 |
| 16 | — | — | 0.000641 | 0.000577 |

两点结论（§3 Observation ❷、附录 F.3）[^src-resnets-route-without-routers]：

- 峰值阶高于中点（ResNet-18 5 对 $L/2=4$，ResNet-34 10 对 $L/2=8$）；4–6 阶占 ResNet-18 标量/向量 71.6%/74.7%，1–3 阶仅 14.17%/13.56%；8–12 阶占 ResNet-34 78.68%/78.84%，1–3 阶约 0.1%。
- $M_k$ 随阶上升、$k=L$ 最大（$M_L/M_1$ 为 32.21/22.31 与 341.4/296.4），峰不是组合数造成的。

直接推论是「短结构路径 ≠ 低阶功能 interaction」：低阶 interaction 里的支路可以相隔很远，短路径也不必产生低阶项（附录 F.1 区分二者）[^src-resnets-route-without-routers]。

## 命题 2.3：残差缩放压制高阶项

对缩放后的残差堆 $x^{(\lambda)}_{l+1}(m)=x^{(\lambda)}_l(m)+\lambda m_l F_l(x^{(\lambda)}_l(m))$，$0<\lambda\le\lambda_0$。若 $F_l$ 与标量读出在 shortcut 轨迹附近 $C^L$ 且 $L$ 阶导数一致有界，则对每个 $|S|=k$ 存在与 $\lambda$ 无关的 $B_{S,x}>0$ 使[^src-resnets-route-without-routers]

$$\left|\Delta_S v_x^{(\lambda)}\right|\le B_{S,x}\,\lambda^{k}.$$

证明分两步（附录 D）[^src-resnets-route-without-routers]：Lemma D.1 把 Möbius 系数写成混合偏导积分，

$$\Delta_S v=\int_{[0,1]^{|S|}}\partial_S v\left(u_S,0_{\bar S}\right)\,du_S,\qquad \partial_S=\prod_{s\in S}\frac{\partial}{\partial m_s};$$

Lemma D.2 对块序号归纳证 $\left|\partial_A x^{(\lambda)}_l(m)\right|\le C_{l,A,x}\lambda^{|A|}$（多取一次门导数多乘一个 $\lambda$）；两步合成 $\lambda^k$ 界。命题是逐系数渐近结论，不要求未缩放网络本身低阶主导[^src-resnets-route-without-routers]。

## 推理期缩放实验

不重训，只把 $\lambda$ 在 $\{1.0,0.95,0.9,0.85,0.8,0.75,0.5,0.25,0.1\}$ 上扫[^src-resnets-route-without-routers]：

| 模型 | $\lambda$ | $\kappa^v$ | $\kappa^h$ | $C^v_{\le3}$ | $C^h_{\le3}$ | top-1 一致率 |
|---|---|---|---|---|---|---|
| 18 | 1.00 | 5.0009 | 4.9295 | 0.1417 | 0.1356 | 1.0000 |
| 18 | 0.95 | 4.9756 | 4.9002 | 0.1491 | 0.1447 | 0.9375 |
| 18 | 0.90 | 4.9448 | 4.8706 | 0.1573 | 0.1541 | 0.8759 |
| 18 | 0.85 | 4.9086 | 4.8389 | 0.1671 | 0.1640 | 0.8093 |
| 18 | 0.80 | 4.8635 | 4.8022 | 0.1786 | 0.1751 | 0.7314 |
| 18 | 0.75 | 4.8042 | 4.7572 | 0.1933 | 0.1882 | 0.6365 |
| 18 | 0.50 | 4.3103 | 4.2689 | 0.3211 | 0.3254 | 0.0518 |
| 18 | 0.25 | 2.0431 | 2.7209 | 0.8314 | 0.7268 | 0.0019 |
| 18 | 0.10 | 1.1444 | 1.4517 | 0.9968 | 0.9777 | 0.0014 |
| 34 | 1.00 | 9.8872 | 9.8835 | 0.0012 | 0.0011 | 1.0000 |
| 34 | 0.95 | 9.8462 | 9.8390 | 0.0014 | 0.0013 | 0.9200 |
| 34 | 0.90 | 9.8027 | 9.7938 | 0.0017 | 0.0016 | 0.8780 |
| 34 | 0.85 | 9.7616 | 9.7474 | 0.0020 | 0.0020 | 0.8140 |
| 34 | 0.80 | 9.7013 | 9.6957 | 0.0025 | 0.0024 | 0.7430 |
| 34 | 0.75 | 9.6391 | 9.6373 | 0.0032 | 0.0030 | 0.6630 |
| 34 | 0.50 | 9.2639 | 9.2544 | 0.0188 | 0.0118 | 0.0450 |
| 34 | 0.25 | 7.6038 | 8.0708 | 0.1986 | 0.1267 | 0.0050 |
| 34 | 0.10 | 2.7084 | 3.4369 | 0.8082 | 0.7219 | 0.0030 |

（Table 6、7；一致率对比同模型 $\lambda=1$ 的全掩码预测，测试集无标签，故是 prediction agreement 而非 accuracy。）

三条读法[^src-resnets-route-without-routers]：

- 方向与命题一致：$\kappa$ 单调降、$C_{\le3}$ 单调升，两个读数、两个模型都成立。
- 深层移得更慢：$\lambda=0.25$ 时 ResNet-18 低阶质量已到 0.8314/0.7268，ResNet-34 只有 0.1986/0.1267，需压到 $\lambda=0.1$ 才把大部分质量移进 1–3 阶。
- 阶谱移动与预测保持不可兼得：$\lambda=0.95$ 一致率已降到 0.9375/0.9200，$\lambda=0.8$ 时约四分之一预测改变而阶谱几乎不动；低阶集中只出现在 $\lambda\in\{0.25,0.1\}$，那时一致率接近 0。

论文的限定：命题是逐系数界，$\kappa$ 与 $C_{\le3}$ 是归一化聚合统计，故实验只能算方向性证据；标量坐标随 $\lambda$ 可能切到另一预测类，向量读数不受该选择影响且趋势相同（附录 G）[^src-resnets-route-without-routers]。这组结果也不构成「少算几个 interaction 就能保住预测」的依据。

[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
