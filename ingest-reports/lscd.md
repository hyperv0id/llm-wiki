# Ingest Report: LSCD

## Created
- **wiki/source-lscd.md** — WHY: source-summary for the LSCD paper (ICML 2025). First source introducing differentiable Lomb–Scargle into diffusion-based time series imputation.
- **wiki/lscd.md** — WHY: entity/technique page for LSCD. Describes the full architecture (differentiable LS layer + spectrum encoder + spectral consistency loss), two-stage training, theoretical foundation, and relationship to CSDI.
- **wiki/lomb-scargle-periodogram.md** — WHY: concept page. The Lomb–Scargle periodogram is a significant mathematical concept that LSCD introduces to the ML community. Explains its mathematical derivation (least-squares sinusoid fitting), FAP statistical filtering, and contrast with FFT.
- **wiki/spectral-consistency-loss.md** — WHY: technique page. $L_{\text{SCons}}$ is a novel loss function introduced by LSCD that penalizes discrepancies between observed and reconstructed Lomb–Scargle periodograms, enforcing frequency-domain fidelity.

## Modified
- **wiki/csdi.md** — WHY: added LSCD as a follow-up work in the "Subsequent Impact" section, describing how LSCD extends CSDI's conditional diffusion framework with spectral conditioning. Added cross-link to LSCD, Lomb–Scargle periodogram, and spectral consistency loss. Updated `source_count` (3→4) and `last_updated`.
- **wiki/index.md** — WHY: added [[source-lscd]] to Sources, [[lscd]] to Entities, [[lomb-scargle-periodogram]] to Concepts, and [[spectral-consistency-loss]] to Techniques. Updated `last_updated`.
- **wiki/log.md** — WHY: recorded ingest entry with full description and page lists. Updated `last_updated`.

## New Cross-links
- [[lscd]] ↔ [[csdi]] — LSCD directly builds on CSDI's conditional diffusion framework
- [[lscd]] ↔ [[lomb-scargle-periodogram]] — LS periodogram is the core conditioning signal
- [[lscd]] ↔ [[spectral-consistency-loss]] — LSCons is the fine-tuning loss in LSCD
- [[lomb-scargle-periodogram]] ↔ [[frequency-aware-conditioning]] — LS is a specific frequency-aware conditioning mechanism
- [[csdi]] ↔ [[lscd]] — CSDI page now references LSCD as follow-up
