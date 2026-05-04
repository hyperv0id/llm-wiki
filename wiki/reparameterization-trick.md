---
title: "重参数化技巧"
type: technique
tags:
  - reparameterization
  - vae
  - gradient-estimation
  - variational-inference
  - diffusion
created: 2026-05-04
last_updated: 2026-05-04
source_count: 2
confidence: high
status: active
---

# 重参数化技巧

**重参数化技巧**（Reparameterization Trick）是一种将随机采样操作重排为"确定性仿射变换 + 固定噪声源"的数学变换，使得梯度可以穿过原本不可微的采样节点，同时显著降低蒙特卡洛梯度估计的方差。它是 VAE、扩散模型、SAC 等概率生成模型端到端可训练的根本保证。[^src-bluuuuue-reparameterization-trick]

## 问题：采样不可导

给定变分推断目标：

$$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$$

编码器输出参数化分布 $q_\phi(\mathbf{z}|\mathbf{x})$，从中采样 $\mathbf{z}$ 的操作在计算图上不可导——反向传播路径在此断裂。[^src-bluuuuue-reparameterization-trick]

### REINFORCE 的困境

得分函数估计（REINFORCE）通过恒等变换绕开采样不可导问题：

$$\nabla_\phi \mathbb{E}_{\mathbf{z} \sim q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\mathbf{z} \sim q_\phi}[f(\mathbf{z}) \cdot \nabla_\phi \log q_\phi(\mathbf{z})]$$

该估计无偏，但在高维空间方差极大：重建误差标量 $f(\mathbf{z})$ 与得分函数的乘积在训练早期波动严重，梯度估计噪声几乎不可用。[^src-bluuuuue-reparameterization-trick]

## 重参数化：两步分离

### 第一步：将随机节点移出计算图

将 $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma_\phi^2(\mathbf{x}))$ 改写为等价仿射变换：

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

$\epsilon$ 来自固定的标准正态分布，与模型参数 $\phi$ 完全无关。[^src-bluuuuue-reparameterization-trick]

### 第二步：固定噪声源替代参数依赖的期望

代入原始期望：

$$\mathbb{E}_{\epsilon \sim p(\epsilon)}[\log p_\theta(\mathbf{x} | \mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon)]$$

期望下标与 $\phi$ 无关，$\phi$ 仅出现在方括号内的确定性计算路径上。梯度沿 $\phi \to \mu_\phi, \sigma_\phi \to \mathbf{z} \to$ 损失函数 正常回传，唯一的随机性来自外部注入的 $\epsilon$。[^src-bluuuuue-reparameterization-trick]

## 双重功效

### 结构层面：打通反向传播

计算图分离为两条通路：噪声注入之前的路径是确定性的（$\phi$ 在此路径上）；噪声注入由不可训练的 $\epsilon$ 引起。梯度沿确定性路径顺畅回传，无需统计近似。[^src-bluuuuue-reparameterization-trick]

### 优化层面：降低梯度方差

重参数化的梯度每一项都直接携带"$\mu_\phi$ 和 $\sigma_\phi$ 该如何调整"的结构信息，而非依赖标量 $f(\mathbf{z})$ 乘上得分函数来估计方向。$\epsilon$ 从标准正态分布中抽取，波动范围受常数约束，优化轨迹比 REINFORCE 稳定得多。[^src-bluuuuue-reparameterization-trick]

> [!note] 理论保证
> Xu et al. (2019, AISTATS) 证明：在高斯平均场变分近似下，重参数化梯度估计的逐维度方差严格小于得分函数估计，训练初期差异可达数量级。[^src-bluuuuue-reparameterization-trick]

## 适用前提

| 条件 | 说明 | 不满足时的替代 |
|------|------|----------------|
| 隐变量分布连续 | 离散变量不存在等价可导变换 | Gumbel-Softmax 松弛、REINFORCE + 控制变量 |
| 分布族可重参数化 | 需存在"确定性变换 + 固定噪声"的分解 | 仅限位置-尺度族（高斯、拉普拉斯、Student-t 等）|

## 应用场景

### 扩散模型

前向加噪每一步天然使用重参数化：

$$\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

这是扩散框架可以端到端训练的根本保证——每一步都是可导的。等价地，$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$ 也是重参数化的直接推论。[^src-bluuuuue-reparameterization-trick][^src-understanding-diffusion-models]

### VLA 隐动作空间推理

策略输出隐动作的概率分布时，从该分布采样作为后续去噪/流匹配的起点，完全依赖重参数化打通梯度。[^src-bluuuuue-reparameterization-trick]

### SAC（Soft Actor-Critic）

策略网络输出均值和方差，动作通过重参数化采样得到低方差策略梯度估计。[^src-bluuuuue-reparameterization-trick]

## 与相关概念的关系

| 概念 | 关系 |
|------|------|
| [[variational-autoencoder\|VAE]] | VAE 的核心训练依赖——ELBO 中重建项 $\mathbb{E}_{q}[\log p(\mathbf{x}\|\mathbf{z})]$ 的梯度回传 |
| [[elbo\|ELBO]] | ELBO 的可优化性以重参数化为前提 |
| [[diffusion-model\|扩散模型]] | 前向过程每步使用重参数化，是端到端训练的基础 |
| [[score-function\|分数函数]] | REINFORCE 使用得分函数 $\nabla_\phi \log q_\phi$，重参数化绕过了它的高方差问题 |
| [[ddpm-simplified-training-objective\|$L_{\text{simple}}$]] | DDPM 训练目标中 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 即重参数化 |
| [[scaling-factor-sqrt-dk\|缩放因子 $1/\sqrt{d_k}$]] | 同系列文章：缩放因子解决 Softmax 饱和，重参数化解决采样不可导——两者都是数值/梯度稳定性的结构性方案 |

## 引用

[^src-bluuuuue-reparameterization-trick]: [[source-bluuuuue-reparameterization-trick]]
[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]