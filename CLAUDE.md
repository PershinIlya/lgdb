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

## Full stance run (done Nov 22, work_full/, code lgdb/stance_full.py, lgdb/stance_analysis.py)
- 125 books (7 per period x region x format cell, rule in stance_full.books), 122 known to memory; 192 statements; R1 sonnet, R2 opus, 8 A-vs-B ties by haiku. About 5.55M tokens (work_full/usage.jsonl).
- `LGDB_FULL=1 python tools/stance.py stage analyze` writes work_full/analysis.json. Negatives are rare (7.4% of addressed cells); 47 statements two-sided enough, 58 books placed (34 South Asian).
- Parallel analysis k=4. By the locked keep rules 0 axes pass: dim 1 (faith and guidance vs own reason) congruence 0.93, rater r 0.93, text vs memory 0.94, fails on R2 with negative count 0.33 > 0.30; dim 2 (nonviolence and truth vs inner path) congruence 0.849; dims 3 (letting go vs disciplined cultivation) and 4 unstable. Blind pole names: work_full/poles.json.
- No extremity: tail shares below the permutation null, unimodal. Period eta2 <= 0.08 (all p > 0.09); region weak (dims 2, 4).
- Map page: human/stance_map.html, https://claude.ai/artifact/SmdbNEVET8ypmkW4JvU9wW

## Dilemma axes (done Nov 28, work_axes/, code lgdb/axes.py, map tools/axes_map.py + tools/axes_map_template.html)
- Locked plan with amendment: work_axes/plan.md. 12 axes = everyday questions with 2 answers, written by opus (A1) from the dossiers' 1,616 "advice || instead of" pairs, clustered (work_axes/tensions.jsonl, axes.jsonl). Each answer coded as a separate statement with the audited S4a/S4b prompts (R1 sonnet, R2 opus); position = s(x) - s(y), -2..+2.
- Leakage audit: A1 and A2 failed (2 of 3 models guessed "opposed advice"); A2 dropped; A1 used, its exemption is the researcher's call (raw in work_axes/audit_raw). Do not self-mark exemptions.
- 24 well-known extra titles (work_axes/extra_books.json) recalled with S1 and drawn on the map, excluded from tests. Russian titles and fame tiers: work_axes/display.json (tiers are the analyst's judgment, display only).
- Results (work_axes/analysis.json): 11 of 12 axes pass keep rules (rater r 0.88 to 0.98, no opposite signs, controls uncoded); K12 (arguing) fails. Era: K01 improve vs accept (eta2 0.29, p 0.001; after 1945 leans to accept) and K02 withdraw vs practice in daily life (0.26, p 0.001; after 1945 leans to daily life). Region: K11 moderation vs renunciation (0.27, p 0.001; South Asia renunciation). Middle positions rare (0 to 9%), but the coding makes "middle" require advising both answers.
- About 2.3M tokens (work_axes/usage.jsonl). Map: same artifact as before (version 3).

## Not in git
Ambedkar text (US copyright), downloaded sources (python run.py fetch restores them), cache/, pilot fragments.
