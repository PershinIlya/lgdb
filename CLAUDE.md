# lgdb: trade-off axes in guidance-on-living literature

## Working with the researcher (Ilia)
- Reply in Russian. Never use em or en dashes. Numbers as numerals. Answer first, short clear sentences, explain jargon plainly.
- No markdown reports for him to read; put review content in chat.
- No API budget. All model calls run as Claude subagents inside his subscription, in batches (tools/agents.py: QueueLLM queue -> batch files -> agents write answers_*.jsonl -> ingest -> rerun stage). About 70k tokens overhead per agent; budget a few million tokens per day.

## Research goal
Place influential books on how to live (before 1500 to after 1945; Western, East Asian, South Asian; aphorism collections and single-thesis books, incl. modern bestsellers) on data-driven dimensions of disagreement between their recommendations. Annotator agents must stay blind to the hypothesis (books simplify dilemmas by taking one pole of trade-off axes; positions depend on era and culture). Every new prompt must pass the leakage audit (fresh opus, sonnet, haiku guess the hypothesis; see audit/leakage_audit.json).

## State (November 8, 2026)
- Corpus selection: corpus/candidates.csv, corpus/selection.csv (mechanical rule in lgdb/select.py), cleaned public-domain texts in corpus/raw/, sources in corpus/sources.csv (fetch.py; trim patterns in tools/trim/).
- Mini pilot on 10 full texts (work_pilot1/): 895 claims, 12 topics, 2292 pairwise relations. 80% "compatible", 3 "incompatible", 0 bipolar topics. The pairwise compatibility question (P5) does not detect trade-offs; this matches literature on LLM entailment bias. P5-based camps are abandoned.
- Model memory trial (work_memory/): 3 models x 23 books + 3 invented books; all invented books answered "unknown"; recall is book-specific but drops conditions.
- Calibration on 10 modern bestsellers (work_calib/): step 1 recall done; argued_against field is very informative.
- Method review (work_method/): literature summaries, 3 independent designs (A, B, C), skeptical critique (critique_1.md), merged design (critique_2_merged.md).

## Agreed next step: stance-matrix pilot (not yet started, awaiting researcher's go)
Merged design from work_method/critique_2_merged.md with critique_1.md changes:
- Statement pool from recommendations and argued_against items; concrete statements with a visible cost; cap 30 items per book; build at 2 clustering thresholds; drop near-unanimous statements; flag own-source cells.
- Stance codes: advises / advises against / not addressed / unsure, plus a separate "with conditions" flag (no middle code). Evidence note id required, else not addressed. Title hidden. 2 raters (sonnet, opus, paraphrase, reversed options), haiku tie-break; 40 reversed statement pairs for acquiescence.
- Dimensions: ordinal IRT or factor analysis on the book x statement matrix, silence = missing; check factors against coverage, polemic style, source type, religious vs secular; report unipolar factors too.
- Pilot: 10 classics (text + memory), 10 modern (memory), 3 invented books, 2 null dossiers, about 2.5M tokens. Go criteria in critique_2_merged.md section 3; reject share >= 10% of addressed cells.
- Researcher hand checks: 40 blind stance cells after the pilot (offer a simple form).

## Not in git
Ambedkar text (US copyright), downloaded sources (python run.py fetch restores them), cache/, pilot fragments.
