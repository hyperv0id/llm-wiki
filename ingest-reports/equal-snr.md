# Ingest 报告：EqualSNR / A Fourier Space Perspective on Diffusion Models

## 创建

- `wiki/source-equal-snr.md` — WHY：为 arXiv:2505.11278 建立 source-summary，保留 DDPM 频域 SNR 公式、反向高斯假设反例、EqualSNR 训练目标和实验结论。
- `wiki/equal-snr.md` — WHY：EqualSNR 是论文核心技术，需要独立技术页详细记录 $\Sigma_{ii}=cC_i$、SNR 校准、ELBO 损失和 DDIM 采样公式。
- `wiki/frequency-hierarchy-in-diffusion.md` — WHY：论文最重要的概念贡献是 DDPM 的低频到高频隐式生成层级，需要独立概念页连接 DDPM、频域噪声控制和 SNR 偏置。

## 修改

- `wiki/diffusion-model.md` — WHY：在扩散模型总览中加入 EqualSNR 作为前向过程归纳偏置塑造的新证据，并补充每频率 SNR 公式。
- `wiki/frequency-based-noise-control.md` — WHY：把 EqualSNR 接入频域噪声控制谱系，区分“噪声方差均匀”和“SNR 均匀”。
- `wiki/ddpm.md` — WHY：为 DDPM 页面补充傅里叶空间解释，说明白噪声如何诱导频率层级和高频后验问题。
- `wiki/inductive-bias-shaping.md` — WHY：增加“信号方差与噪声方差相对比例”作为前向过程塑造归纳偏置的新机制。
- `wiki/index.md` — WHY：将新 source、concept、technique 页面加入目录。
- `wiki/log.md` — WHY：按 ingest 规则记录本次摄取。

## 新建交叉链接

- [[equal-snr]] ↔ [[frequency-hierarchy-in-diffusion]]
- [[equal-snr]] ↔ [[frequency-based-noise-control]]
- [[equal-snr]] ↔ [[diffusion-model]]
- [[equal-snr]] ↔ [[ddpm]]
- [[frequency-hierarchy-in-diffusion]] ↔ [[inductive-bias-shaping]]
