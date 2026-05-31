# Ingest Report: UrbanGPT (KDD 2024)

## Source
- **Paper**: UrbanGPT: Spatio-Temporal Large Language Models (Li et al., KDD 2024, arXiv:2403.00813)
- **File**: `/home/jcheng/Downloads/papers/UrbanGPT_2024_2403.00813.pdf`
- **Draft reference**: `/tmp/opencode/paper-ingest/urbangpt-2024.org`

## Created
- `wiki/source-urbangpt.md` — **WHY**: Source-summary for the first spatio-temporal LLM paper per CLAUDE.md ingest workflow. Covers architecture (ST encoder + alignment + instruction-tuning + regression layer), experiments (zero-shot superiority across 4 datasets), limitations (7B params, 174s inference), and 6 critical concerns.
- `wiki/urbangpt.md` — **WHY**: Technique page for UrbanGPT as a method. Includes full architecture breakdown (4 components), training details, quantitative results table, ablation analysis, comparison with other ST foundation models (UrbanDiT, GPT-ST, OpenCity), robustness analysis, and limitations.

## Modified
- `wiki/spatio-temporal-foundation-model.md` — **WHY**: Added wikilinks to `[[urbangpt|UrbanGPT]]` at plaintext mentions; updated source_count and last_updated for consistency.
- `wiki/traffic-forecasting.md` — **WHY**: Added Foundation Model section entry for UrbanGPT as the first LLM-based ST model; added source citation and wikilink.
- `wiki/gpt-st.md` — **WHY**: Added cross-reference to `[[urbangpt]]` in Related Pages section as a contrasting ST approach (LLM-based vs pure numerical pre-training).
- `wiki/opencity.md` — **WHY**: Added `[[urbangpt]]` cross-link in Related section as another ST foundation model approach.
- `wiki/index.md` — **WHY**: Added new pages under Sources (source-urbangpt) and Techniques (urbangpt).
- `wiki/log.md` — **WHY**: Appended ingest entry with created/updated pages.

## New Cross-Links
- `[[urbangpt]]` ↔ `[[spatio-temporal-foundation-model]]` (ST foundation model overview)
- `[[urbangpt]]` ↔ `[[gpt-st]]` (contrasting ST pre-training approaches)
- `[[urbangpt]]` ↔ `[[urbandit]]` (comparison: LLM-based vs diffusion-based ST FM)
- `[[urbangpt]]` ↔ `[[opencity]]` (cross-city ST foundation model comparison)
- `[[urbangpt]]` ↔ `[[traffic-forecasting]]` (traffic prediction task context)
- `[[urbangpt]]` ↔ `[[source-urbangpt]]` (source ↔ technique link)
- `[[urbangpt]]` ↔ `[[stgcn]]`, `[[gwnet]]`, `[[mtgnn]]` (zero-shot baselines)
- `[[urbangpt]]` ↔ `[[urbandit-paper-river]]` (already had detailed UrbanGPT section)
