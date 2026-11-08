Design C: stance matrix over data-derived statements

CORE IDEA
Drop pairwise relation labels. They failed because models call almost everything "compatible". Instead, turn each book into a dossier of normalized recommendations. Then derive about 100 single, one-sided statements from the pooled data. Then ask, for each book and each statement, one closed question: does this book endorse it, reject it, or not address it? This yields a book x statement matrix, like roll-call votes. Dimensions come from factoring that matrix. No model ever sees two poles side by side, and no model is asked about tensions.

STAGE 1: DOSSIERS (what the models are asked about each book)
1a. Text books (before 1931, about 60 books). Reuse the working extraction prompt. Passages of about 1,500 words, 20 passages per agent, sonnet. Fields: recommendation, domain, force, conditions, reason, argued_against.
1b. Memory, all books (including the 60 text books, for calibration). Prompt, close to verbatim: "From your knowledge of the book [title, author, year], list the main pieces of practical advice it gives to readers. For each, give the advice in one sentence, any conditions or exceptions the author attaches, and any view, habit, or advice the author criticizes or warns against. If you do not know this book well, answer UNKNOWN." Two models (opus, sonnet), 15 books per agent. Keep 3 invented books as a guard.
1c. Condensation. Sonnet merges each book's items into at most 40 dossier entries in plain modern English, each with a stable id. This step removes translation style and archaic wording before any scoring. Text and memory dossiers stay separate files.

STAGE 2: CANDIDATE STATEMENTS (how candidate dimensions are induced)
2a. Embed all dossier recommendations and, separately, all argued_against items. Cluster each set (HDBSCAN, as in the topic step).
2b. For each cluster, opus writes one statement that summarizes it: "Write one sentence of the form 'The book advises the reader to ...' that a person could check against a book. Do not mention any other cluster." Clusters of argued_against items are rewritten as the advice that is being criticized, so they become ordinary statements too.
2c. Keep a statement only if at least 3 distinct books contribute endorsing items. This gives about 80 to 150 statements. Statements from argued_against clusters are kept only if 3 books endorse them, which means some author genuinely holds the criticized view.
2d. Data-driven pairing, used only for later interpretation: a statement A is linked to statement B when items in A's cluster carry argued_against items that fall in B's cluster. No model proposes pairs.

STAGE 3: STANCE SCORING
Each agent gets 5 book dossiers and all statements, in shuffled order. Per cell the question is: "Based only on the advice listed for this book, which is true? (a) The book gives this advice or clearly equivalent advice. (b) The book advises against it or criticizes it. (c) The book does not address it. (d) Unsure. For (a) or (b), cite the dossier entry id." The option order is reversed in half the batches. A label (a) or (b) without a valid id is recoded to (c). This evidence rule targets the bias directly: the model must point to a specific entry, not judge general fit.
Raters: 2 prompt paraphrases x 2 models (sonnet, opus) on 30 books chosen at random, as a stability check. All other books get 2 raters (sonnet P1, opus P2). A third rater (haiku, P1) scores the cells where the two disagree.
Aggregation per cell: endorse = +1, reject = -1. The cell is missing if most raters say "not addressed". Otherwise the cell value is the mean of the non-missing labels. Dawid-Skene is not used, because the raters share errors. Human gold does the calibration instead.

STAGE 4: DIMENSIONS
Statement correlations are computed pairwise over books that address both statements, requiring at least 15 such books. Parallel analysis picks the number of factors, then oblique factor analysis runs on that matrix. Stability: 200 bootstraps over books; keep a factor only if Tucker congruence is at least 0.85. As a second route, check each linked A/B pair from 2d. If A and B load with opposite signs on the same factor, that factor is a confirmed axis with two real poles. A blind opus agent names each factor from its top loading statements. It is not told what the study is about. Book positions come from factor scores, using only addressed cells. Each position also gets a coverage count.

STAGE 5: VALIDATION
- Human gold, about 100 cells, labeled by the researcher before seeing model labels. The sample is stratified: 35 predicted endorse, 35 predicted reject, 30 predicted not addressed, with rater disagreements oversampled. Report per-class precision and recall. Reject recall is the key number. If it is below 0.6, downstream estimates use design-based correction (Egami et al. 2023) with this sample.
- Stability: Krippendorff alpha across the 4 raters on the 30 stability books, with a target of 0.8 or higher. Below that, the statement is rewritten once and rescored. If it fails again, it is dropped.
- Anchors: the researcher places 15 well-known books on the top 3 factors from their own reading, before seeing the scores. Report Spearman correlation.
- Optional convergent check: Bradley-Terry over about 400 pairwise prompts of the form "which book's advice is closer to statement S", both orders, for the 20 anchor books on the top 2 factors.

RISKS AND GUARDS
- Compatible bias. Pairwise judgments are gone. Each cell is a single-statement stance that needs an evidence id. "Not addressed" is an explicit option and is treated as missing, not as agreement. Reject recall is measured against human gold.
- Hypothesis leakage. Prompts ask only for advice, summaries, and stance. Poles come from data co-occurrence, never from a model asked for trade-offs. Before use, 3 fresh haiku agents read each prompt and answer "What hypothesis might the authors of this prompt be testing?". If any guess mentions one-sidedness, opposites, trade-offs, or extremes, the prompt is reworded. Annotators never see pairs, factors, or era labels.
- Memory versus text. The 60 text books get both dossiers. For each factor, compare text and memory positions (correlation and mean shift). Memory drops conditions, so it may make books look more extreme. For the extremity question, the main result uses text dossiers only. Memory results are reported with the measured shift. Source type is a covariate in every era or culture model.
- Era and translation. Condensation into plain modern English happens before scoring. Embeddings are used only to group items, never to place books. For 4 books with two translations (for example the Tao Te Ching and the Enchiridion), both versions are processed. The difference between translations must be well below the spread between books. Era is a research question, so it is not regressed out. Instead, check that each factor has loading statements from at least 2 eras and 2 cultures. A factor carried by one era is reported as era-specific, not universal.
- Silent versus disagreeing books. Silence stays missing. It is never coded as 0. A middle position counts as "balanced" only if the book endorses statements at both ends of the factor, often with conditions. Otherwise a middle position from low coverage is flagged as "thin". The test of whether books sit at the ends uses only books that address at least 4 statements on the factor. It applies a dip test and compares against a null where stances are permuted within each statement.
- Batching contamination. Books are shuffled across batches, and 10% of cells are rescored in different batch company. Report how often those labels flip.

TOKEN COST (about 70k overhead per agent)
- 1a text extraction: about 23 agents, about 3.2M.
- 1b memory: 150 books x 2 models, about 20 agents, about 2.3M.
- 1c condensation: about 30 agents, about 3M (text and memory dossiers).
- 2 statements and naming: about 0.6M.
- 3 scoring: 30 batches x 2 raters at about 95k each, plus stability raters, tie-breaks, and rescoring. About 7M.
- Leakage tests and Bradley-Terry: about 1M.
Total about 17M tokens, or 6 to 8 working days at a few million per day. Human time: about 100 gold cells plus 15 anchor placements.
