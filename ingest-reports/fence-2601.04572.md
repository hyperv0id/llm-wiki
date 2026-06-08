# Ingest Report: FENCE (arXiv 2601.04572)

## Created
- `wiki/source-fence.md` — **WHY**: source-summary for the FENCE paper per ingest workflow, covering core innovations (feedback guidance, cluster-aware guidance, two-stage training) and experimental results
- `wiki/fence.md` — **WHY**: entity/technique page for FENCE as a distinct method in the diffusion imputation lineage (CSDI → PriSTI → FENCE)
- `wiki/feedback-diffusion-guidance.md` — **WHY**: technique page for the general dynamic guidance mechanism (posterior-driven λ adjustment), reusable beyond traffic imputation
- `wiki/cluster-aware-guidance.md` — **WHY**: technique page for the cluster-level posterior aggregation strategy, generalizable to other multi-node diffusion tasks

## Modified
- `wiki/csdi.md` — **WHY**: added FENCE to the evolution chain and associated pages; FENCE addresses CSDI's fixed guidance scale limitation
- `wiki/pristi.md` — **WHY**: added FENCE to the improvement lineage and updated subsequent impact analysis
- `wiki/classifier-free-guidance.md` — **WHY**: added "动态 CFG（反馈引导）" section documenting FENCE's extension of standard CFG
- `wiki/diffusion-models.md` — **WHY**: added dynamic guidance mechanism as a future direction in challenges section
- `wiki/index.md` — **WHY**: added source-fence, fence, feedback-diffusion-guidance, cluster-aware-guidance entries
- `wiki/log.md` — **WHY**: recorded ingest activity per workflow

## New Cross-links
- [[fence]] ↔ [[csdi]] (diffusion imputation lineage)
- [[fence]] ↔ [[pristi]] (diffusion imputation lineage)
- [[fence]] ↔ [[classifier-free-guidance]] (dynamic extension of CFG)
- [[feedback-diffusion-guidance]] ↔ [[classifier-free-guidance]] (dynamic vs fixed CFG)
- [[cluster-aware-guidance]] ↔ [[fence]] (component relationship)
- [[fence]] ↔ [[diffusion-models]] (domain reference)
- [[source-fence]] ↔ [[fence]] (source-entity pair)
