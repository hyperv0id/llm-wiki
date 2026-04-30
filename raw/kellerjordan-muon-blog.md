---
created: 2026-04-30T09:45:47 (UTC +08:00)
tags: []
source: https://kellerjordan.github.io/posts/muon/
author: Keller Jordan
type: url
---

# Muon: An optimizer for hidden layers in neural networks

Source URL: https://kellerjordan.github.io/posts/muon/

Author: Keller Jordan

Published: 2024

Summary: Blog post introducing Muon optimizer - an optimizer for 2D parameters in neural network hidden layers using Newton-Schulz orthogonalization. Used in current NanoGPT and CIFAR-10 speedrunning records.

Key results:
- CIFAR-10 speed record: 3.3 → 2.6 A100-seconds (94% accuracy)
- NanoGPT speed record: 1.35x improvement (3.28 val loss)
- 1.5B model to GPT-2 XL level in 10 8xH100-hours (vs 13.3h AdamW)

Related: [[source-muon-optimizer]]