Design B: stance matrix with negated twins, then factor discovery, then pairwise confirmation

CORE IDEA
Stop asking models whether two recommendations conflict. That question triggers the "compatible" bias. Instead, build a roll-call matrix. Each book "votes" on a shared pool of short probe statements. Disagreement then comes from books voting differently on the same probe, not from a model's judgment of conflict. Dimensions come from factoring that matrix.

STAGE 1. EVIDENCE DOSSIER PER BOOK
- Full-text books (about 40): run the existing extraction. Keep recommendation, domain, conditions, argued_against, and passage ID.
- All books (about 120): run memory recall with the existing prompt on 3 models. Add 3 invented books as controls.
- Modern books: union of the 3 recalls, plus third-party summaries if available.
- Normalize every dossier item into plain modern English with the same rewrite prompt. This removes era and translation style before any rating.
Result: about 60 to 100 items per book, about 4k tokens per dossier.

STAGE 2. PROBE POOL
- Pool all recommendations AND all argued_against items. argued_against items matter because they often name the opposite pole directly.
- Embed, cluster, and ask a model: "Rewrite each group as one short instruction to a reader, in plain modern English. Merge items that say the same thing."
- Keep probes that come from at least 2 books, or from at least 1 argued_against item. Target 300 probes.
- For each probe, write a negated twin in a separate call: "Write the instruction that tells the reader to do the opposite." Example: "Spend time alone regularly" and "Avoid long periods alone."
- Twins and originals are never rated in the same batch.

STAGE 3. STANCE MATRIX (THE CORE MEASUREMENT)
Prompt, close to final wording:
"Below is a set of notes on what one book advises. Then comes a list of instructions. For each instruction, say what this book's author would tell a reader about it. Answer with one code:
U = urges it; A = agrees with it; C = advises it only in some circumstances; S = the book does not address this; D = advises against it; W = warns strongly against it; ? = cannot tell.
For any code except S and ?, give the ID of the note that supports it."
- Half the runs list the codes in reverse order (option order bias).
- Two conditions:
  (a) Dossier-blind: the model sees the normalized dossier with the book title and author removed. This blocks the canonical-book prior.
  (b) Memory: the model sees only the title and author.
  Run (a) for all books. Run (b) for the 40 full-text books and the 3 invented books.
- Raters: 4 per cell. Sonnet with paraphrase 1, sonnet with paraphrase 2, opus with paraphrase 1 in reversed order, opus with paraphrase 2. Temperature 0. JSON output.
- Batching: 5 books per call, 300 probes each, compact output ("p017:D:n42"). Book groupings are reshuffled for each rater, so cross-book contamination does not repeat.
- Scoring: U=+2, A=+1, C=+0.5, S=missing, D=-1, W=-2, ?=missing. Item score = (original minus twin) / 2. Acquiescence index = how often a book endorses both a probe and its twin. Report it per book and per model.

STAGE 4. DIMENSION DISCOVERY
- Matrix: about 120 books by 300 items, with many cells missing. Missing means silent. Silent is never zero.
- Fit ordinal factor analysis or multidimensional IRT that tolerates missing cells.
- Choose the number of dimensions k by held-out prediction. Mask 10% of observed cells and pick the k that predicts them best.
- Rotate with oblimin. Name each dimension from its top positive and top negative probes.
- Keep a dimension only if:
  - at least 4 books sit clearly on each side (|score| > 0.5 SD), from at least 2 era or culture cells;
  - it is stable under bootstrap over books and over raters (Tucker congruence >= 0.85);
  - it is not explained by dossier source (text, memory, or summary). Source type as a predictor must explain under 10% of the variance.
- Convergent check (cheap, optional): cluster pairs of recommendation and its argued_against into candidate tensions, then see which factors they match.

STAGE 5. PLACEMENT CONFIRMATION BY PAIRWISE JUDGMENTS
Discovery is finished at this point. Now dimension wording is allowed in prompts.
- For each kept dimension, ask: "Which of these two books more strongly advises the reader to [pole description]? Answer A, B, equal, or neither addresses this."
- Run both book orders and both pole wordings (pole X and pole Y). Use 2 models.
- Each book gets about 6 random opponents. That is about 360 pairs times 4 runs per dimension, at 200 pairs per call.
- Fit Bradley-Terry. Require r >= 0.7 between Bradley-Terry scores and factor scores. If r is lower, the dimension is flagged on the map.

SILENT VERSUS MIDDLE VERSUS EXTREME
For each book and dimension, coverage = share of high-loading items that are not silent.
- Coverage under 30%: the book is shown as "silent" on the map, not placed in the middle.
- Among covered books, "balanced" means the book endorses items at both poles, or uses C (conditional) on them. This is a different state from silence.
- The question "do books sit at the ends?" is tested only on covered books: compare the bimodality of their scores with a null from permuted matrices.

VALIDATION
- Human gold, written before seeing any model output: 80 cells (10 books by 8 probes), oversampled toward D and W ratings and toward cells where raters disagree. Plus a rank order of 10 books on each of the top 2 dimensions (about 20 items). Report accuracy, recall on D and W, and the S versus D confusion. Report accuracy for each model separately, never only the consensus.
- Rater stability: Krippendorff alpha across the 4 raters. Target 0.8 overall. Drop probes with alpha under 0.67.
- Leakage test: give each prompt to 3 fresh models and ask them to guess the hypothesis. Any guess naming trade-offs, poles, axes, or opposition sends the prompt back for rewording. The stance prompt rates one instruction at a time, so it should pass.
- Locked plan: write down the rule for choosing k, all thresholds, and the gold set before the main run. Tune prompts only on a 3-book pilot that is not in the gold set. This guards against "LLM hacking".

RISKS AND GUARDS
- Bias toward "compatible": no pair-conflict question during discovery. Negated twins and the acquiescence index measure "agree with everything". Gold recall on D and W tests whether models ever say "against".
- Hypothesis leakage: probes are single instructions. Dimension words appear only after discovery. Every prompt passes the leakage test.
- Memory versus text: for the 40 full-text books, compare memory rows with dossier-blind rows (item correlation and factor position). Memory-only books are admitted if the memory positions of full-text books fall within the bootstrap interval of their text positions. Otherwise, memory-only books get wider intervals and are marked. Invented books must be almost all S.
- Era and translation: dossiers and probes are rewritten in plain modern English, and rating hides the title. Test: 4 books in 2 translations each (for example, Tao Te Ching and Meditations). The distance between translations must be well under the median distance between books. Era and culture are then predictors in the analysis, not hidden in the measurement.
- Silent versus disagreeing: separate S code, silent treated as missing, coverage shown on the map, and the S versus D confusion measured on gold.
- Shared model bias: report results per model. Gold accuracy, not agreement between models, is the evidence of validity.

TOKEN COST (about 70k fixed overhead per call)
- Extraction for about 30 new full-text books, 15 passages per call: 20 calls, about 2.2M.
- Memory recall, 120 books plus 3 controls by 3 models, 10 books per call: 37 calls, about 3M.
- Normalization, probe pool, and twins: about 8 calls, about 0.8M.
- Stance matrix, condition (a): 24 calls by 4 raters, about 115k each, about 11M.
- Stance matrix, condition (b): 9 calls by 4 raters, about 4M.
- Pairwise confirmation, 6 dimensions: about 45 calls, about 3.6M.
- Leakage tests and pilot: about 1M.
Total: about 25M tokens, roughly 8 days at 3M per day. Cheaper option: 3 raters instead of 4, and 250 probes. That gives about 18M.
