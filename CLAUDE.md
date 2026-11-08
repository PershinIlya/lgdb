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

## Blind annotators: use Explore agents
Ordinary subagents receive this CLAUDE.md (it states the hypothesis); renaming the file mid-session does not help. Explore agents get no project instructions (tested Nov 8). They cannot write files, so they return answers in their final report and tools/stance.py harvest reads them from the agent transcript. Put batch files outside the repo (LGDB_BATCH_DIR, scratchpad). About 55k tokens overhead per Explore agent.

## Stance-matrix pilot (done Nov 8, work_stance/, driver tools/stance.py, code lgdb/stance.py)
- Prompts S1 recall, S2 condense, S3 statement, S4a/S4b stance (2 raters), S5 reverse. Leakage audit: S1+S2, S3, S4 pass; S5 exempt (construct-inherent, like P5). Locked plan: work_stance/plan.md.
- 35 rows (10 classics text + memory, 10 modern memory, 3 invented, 2 null) x 148 statements (108 pool + 40 reversed), R1 sonnet, R2 opus, haiku tie-break (2 cells). Results in work_stance/criteria.json.
- Automatic criteria 1, 3, 4, 5, 6 pass (alpha interval 0.85, nominal 0.74; reversal both-A 0%; controls 100% uncoded; memory vs text sign 0.985). Caveats: negative share 47% overall but 10.8% on advised statements; criticized-pole statements are strawmen (99% B). First SVD component on all statements = coverage (R2 0.88); on 27 two-sided statements it is not (R2 0.08). Invented books answered "unknown", so their rows test recall only. Opus recognized a book from its dossier in the S4 audit.
- Criterion 2 waits for the researcher: 40 blind cells at https://claude.ai/artifact/7hGxwQNXv4b6Gvt2btxFyG (db collection "gold"; score with tools/gold_form.py score). Criterion 7: no leakage guess named poles or trade-offs in top-2 for S1-S4.
- Pilot spent about 2.5M tokens (work_stance/usage.jsonl).

## Not in git
Ambedkar text (US copyright), downloaded sources (python run.py fetch restores them), cache/, pilot fragments.
