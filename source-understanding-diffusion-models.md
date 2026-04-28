---
title: "Understanding Diffusion Models: A Unified Perspective"
type: source-summary
tags:
  - diffusion
  - vae
  - score-based
  - generative-model
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

*Understanding Diffusion Models: A Unified Perspective* by Calvin Luo (2022) provides a comprehensive mathematical derivation that unifies [[variational-autoencoders|Variational Autoencoders (VAEs)]], [[diffusion-models|diffusion models]], and [[score-based-generative-models|score-based generative models]] under a single framework[^src-understanding-diffusion-models].

The paper's core contribution is demonstrating that [[variational-diffusion-models|Variational Diffusion Models (VDMs)]] can be viewed as a special case of Markovian Hierarchical Variational Autoencoders (MHVAEs) with three key restrictions: latent dimension equals data dimension, encoder transitions are predefined linear Gaussian models, and the final latent distribution is standard Gaussian[^src-understanding-diffusion-models]. This perspective allows deriving the Evidence Lower Bound (ELBO) for VDMs, which decomposes into reconstruction, prior matching, and consistency terms[^src-understanding-diffusion-models].

A crucial insight is that optimizing VDMs reduces to learning a neural network to predict the original ground truth image from arbitrarily noisified versions[^src-understanding-diffusion-models]. The paper derives three equivalent training objectives: predicting the original image $x_0$, predicting the source noise $\epsilon_0$, or predicting the score $\nabla \log p(x_t)$[^src-understanding-diffusion-models]. These objectives are mathematically equivalent through applications of the reparameterization trick and Tweedie's formula[^src-understanding-diffusion-models].

The connection to [[score-based-generative-models|score-based generative models]] is established by showing that the score function $\nabla \log p(x_t)$ is proportional to the negative source noise: $\nabla \log p(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_0$[^src-understanding-diffusion-models]. This reveals that [[langevin-dynamics|Langevin dynamics]] sampling in score-based models corresponds to denoising in diffusion models[^src-understanding-diffusion-models].

For conditional generation, the paper discusses both [[classifier-guidance|classifier guidance]] and [[classifier-free-guidance|classifier-free guidance]] approaches[^src-understanding-diffusion-models]. Classifier guidance uses gradients from a separately trained classifier to steer generation, while classifier-free guidance jointly trains conditional and unconditional models and interpolates between their predictions[^src-understanding-diffusion-models].

The paper acknowledges limitations of diffusion models, including slow sampling due to sequential denoising steps and lack of interpretable latent representations compared to VAEs[^src-understanding-diffusion-models]. However, it highlights their strong performance in generative modeling tasks and provides a unified mathematical foundation connecting multiple generative modeling approaches[^src-understanding-diffusion-models].

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]