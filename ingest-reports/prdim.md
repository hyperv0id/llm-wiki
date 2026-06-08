# Ingest Report: PRDIM

Source: PRDIM — Missing Pattern Recognized Diffusion Imputation Model for Missing Not At Random (Sim, Lee, Bae, Na, Kwon, Moon [KAIST]; Hwang, Lim [SNU]; arXiv:2605.25439v1, May 2026 preprint). Raw: `raw/2605.25439.pdf`.

## Created
- [[source-prdim]] — WHY: Source summary for arXiv 2605.25439 (required one-per-raw-file).
- [[prdim]] — WHY: Core entity for the PRDIM model — EM framework, two phases, pattern recognizer, multi-modal results.
- [[missing-not-at-random]] — WHY: MCAR/MAR/MNAR taxonomy + ignorability was absent from the wiki; created as a reusable hub for the imputation cluster, recording each model's missing-mechanism assumption (CSDI/NuwaTS/T1/PRDIM).
- [[pattern-recognizer-guidance]] — WHY: PRDIM's central novel mechanism (discriminator approximating p(M|X) + EM + diffusion reverse-process guidance); a classifier-guidance analogue worth a standalone technique page.

## Modified
- [[csdi]] — WHY: PRDIM extends CSDI's conditional-diffusion framework to MNAR and critiques its MCAR artificial-masking assumption; added to CSDI's successor list + related pages. source_count 6→7.
- [[classifier-guidance]] — WHY: PRDIM's pattern-recognizer guidance is structurally classifier guidance (condition = missing mask M, classifier = D_φ); added as an imputation application. source_count 1→2.
- [[nuwats]] — WHY: Added structural cross-link to the new MNAR concept (NuwaTS assumes random missing / ignores the missing process).
- [[t1]] — WHY: Added structural cross-link to the new MNAR concept (T1 handles point/block/natural missing but assumes randomness).
- [[index]] — WHY: Added 4 new pages to Sources, Entities, Concepts, Techniques.
- [[log]] — WHY: Recorded ingest activity per wiki protocol.

## New Cross-Links
- [[prdim]] ↔ [[missing-not-at-random]]
- [[prdim]] ↔ [[pattern-recognizer-guidance]]
- [[prdim]] ↔ [[classifier-guidance]] (guidance is a classifier-guidance analogue)
- [[prdim]] ↔ [[tweedies-formula]] (posterior mean for guidance)
- [[prdim]] ↔ [[csdi]] (MNAR extension of conditional diffusion)
- [[pattern-recognizer-guidance]] ↔ [[classifier-guidance]]
- [[missing-not-at-random]] ↔ [[csdi]] / [[nuwats]] / [[t1]] (records each model's missing assumption)
