# Ingest Report: DiT (Scalable Diffusion Models with Transformers)

## Created
- [[source-dit]] — WHY: Source-summary page for the DiT paper (Peebles & Xie, ICCV 2023). Core contribution is replacing U-Net with ViT Transformer in diffusion models, demonstrating Gflops-based scaling law in generative modeling.
- [[dit]] — WHY: Technique page documenting the DiT architecture in detail: patchify design, adaLN-Zero conditioning mechanism (zero-initialized adaptive layer norm), four conditioning scheme comparison, model configurations (S/B/L/XL), Gflops scaling analysis, SOTA results, training recipe, and subsequent impact on Sora/SD3/Flux/UrbanDiT.

## Modified
- [[diffusion-model]] — WHY: Added DiT to Key Implementations section and Related Concepts; updated source_count (12->13).
- [[ddpm]] — WHY: Added DiT to Subsequent Development section and Related Pages; DiT directly inherits DDPM's diffusion framework.
- [[latent-diffusion-models]] — WHY: Added DiT cross-link; DiT reuses LDM's pretrained VAE (f8, 84M) as its latent space encoder.
- [[classifier-free-guidance]] — WHY: Added DiT as a notable CFG user (best cfg=1.50 on ImageNet, FID 2.27); updated source_count (2->3).
- [[urbandit]] — WHY: Converted plain-text "Diffusion Transformer (DiT)" to [[dit|Diffusion Transformer (DiT)]] wikilink; added DiT to related resources.
- [[mae]] — WHY: Added DiT cross-link; DiT inherits ViT/MAE patchify + Transformer design pattern.

## New Cross-Links
- [[dit]] <-> [[diffusion-model]]
- [[dit]] <-> [[ddpm]]
- [[dit]] <-> [[latent-diffusion-models]]
- [[dit]] <-> [[classifier-free-guidance]]
- [[dit]] <-> [[urbandit]]
- [[dit]] <-> [[mae]]
- [[source-dit]] <-> [[dit]]
- [[source-dit]] -> [[source-urbandit]] (via index)
