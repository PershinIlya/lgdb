Design A: books as voters on a shared statement pool

CORE IDEA
Stop asking whether two recommendations conflict. Ask instead: does this one book endorse, reject, or ignore this one statement? Disagreement then appears as opposite stances on the same statement, and nobody has to name it. This is the roll-call design from political science. Books are the senators, normalized statements are the bills, and a multidimensional ideal-point model finds the axes. Single-book stance coding never asks the compatibility question that failed.

STAGE 1. EVIDENCE DOSSIER PER BOOK
Old books: sample 15 passages of about 1,500 words, spread evenly, and run the existing extraction unchanged. Modern books: memory extraction by sonnet and opus. Prompt, close to verbatim:
"From your knowledge of [title] by [author], list up to 25 things the book tells the reader to do, value, or avoid. For each give: (1) the advice as one plain modern English sentence, not a quotation; (2) any view or practice the author argues against while giving it, or 'none'; (3) who or when it applies to, if the book says; (4) how central it is: core, recurring, or passing. If you do not know the book well, say 'unknown'."
Also run memory extraction on 30 old books, so memory and text can be compared on the same books. Each dossier is a list of numbered evidence lines.

STAGE 2. STATEMENT POOL
Pool all recommendations and all argued_against entries. Each argued_against entry becomes a statement of its own; this supplies opposite poles from the data. Embed and cluster at a tight threshold. For each cluster, sonnet writes one neutral canonical statement ("A person should ..."), seeing only that cluster. Keep a statement if at least 3 books support it or at least 2 argue against it. Target 250 to 350 statements. For 40 random statements, a separate agent that sees only that statement writes its reversal ("Write the sentence a person who disagrees with this would assert"). These pairs only measure acquiescence, the habit of agreeing with any pleasant advice.

STAGE 3. STANCE CODING (THE CORE MEASUREMENT)
Each agent gets 4 books with dossiers and the statement list in shuffled order. Prompt:
"For each statement, judge what the author of this book would say, using the evidence lines and your knowledge of the book. Code: 2 = clearly urges this; 1 = leans this way; 0 = does not address this; -1 = leans against; -2 = clearly rejects; U = unsure. For 2 or -2, give the evidence line number, or 'M' if from memory only. 'Does not address' is the expected answer for most statements."
The last sentence counters the pull toward finding support everywhere. Raters:
- R1: sonnet, wording P1, codes listed from 2 down to -2.
- R2: opus, paraphrase P2, codes listed from -2 up to 2, against option order bias.
- R3: haiku, only as a tie-breaker on disputed cells.
- Memory-only: the 30 dual-source books coded by R1 without their text dossier.
Codes 0 and U are missing values, not a middle position.

STAGE 4. AGGREGATION AND DIMENSIONS
1. Subtract each rater's mean endorsement on reversed pairs (acquiescence correction).
2. Cell value = mean of R1 and R2. On a sign conflict, R3 decides; if all 3 differ, the cell is missing and flagged.
3. Fit a Bayesian multidimensional ordinal IRT (ideal-point) model, ignoring missing cells. 120 books by 300 statements resembles a Senate session, where these models work well.
4. Choose the number of dimensions by cross-validation: hide 10% of coded cells, fit 1 to 6 dimensions, keep the count that predicts hidden cells best.
5. Opus names each pole from the top 12 statements at each end, with no book names shown.
6. Keep a dimension only if: at least 5 books sit clearly at each end (otherwise it marks coverage, not disagreement); scores fitted from R1 alone and R2 alone correlate at r >= 0.8; a split-half of statements reproduces it at r >= 0.8; and source type (memory or text) does not mainly explain it.
7. Analyze the pattern of 0 codes separately as a coverage map (who talks about what).

STAGE 5. VALIDATION
Machine checks:
- Stability: Krippendorff alpha of R1 versus R2 on coded cells, target >= 0.8.
- Convergence: per kept dimension, sonnet judges 300 book pairs ("which book pushes more toward [pole A]") in both orders; Bradley-Terry scores should correlate with ideal points at r >= 0.7.
- Controls: 2 real dossiers under disguised titles should land near the originals; 3 null dossiers of random lines should land central and never define a pole; invented books should get "unknown".
- Translation: for 8 old books, extract from 2 translations; scores should agree within standard error.
Human review (about 100 items, blind to model codes):
- 50 stance cells: 15 negative, 15 positive, 10 zero, 10 R1 versus R2 disputes, coded from dossier and passages. Report rejection precision separately, since rejections drive the axes.
- 30 book pairs on the top 2 dimensions: which is closer to pole A.
- 20 pole label checks: does the label fairly summarize its statements?
The stance sample feeds a design-based correction (Egami et al.) of the era and culture regressions. Freeze prompts on a 25-book development set first and report every paraphrase result.

RISKS AND GUARDS
- Bias toward "compatible": no compatibility question is asked. Tension comes from independently coded opposite stances. argued_against entries put opposite poles into the pool. Reversed pairs remove plain agreement bias.
- Hypothesis leakage: prompts mention no axes, poles, trade-offs, era, or culture. Statements come only from book data. Pole labels are written after the fit. Every prompt passes the fresh-model guess test. The pole-naming pairwise prompt is used only after discovery, for validation.
- Memory versus text: dossier lines keep their source, and extreme codes must cite evidence. The 30 dual-source books measure the memory shift per dimension. If it exceeds the standard error, modern books carry a wider error band, and conclusions rest on text-based books.
- Era and translation: raters see normalized modern statements, not original wording. Embeddings only cluster normalized text and never place books. The translation check bounds style effects. Source type is a covariate downstream.
- Silent versus disagreeing: "does not address" is its own code, treated as missing, checked by humans, and mapped separately.
- One end versus the middle: memory loses conditions, so memory books may look extreme. Test polarization on text-based books first; extend to all books only with the measured memory shift applied.

TOKEN COST (about 70k overhead per call)
- Text extraction: 60 old books, 2 per agent, 30 calls at 140k: 4.2M.
- Memory extraction: 150 book runs, 10 per agent, 30 calls at 90k: 2.7M.
- Statement pool and reversals: 6 calls: 0.6M.
- Stance coding: 30 calls per rater at about 105k (4 dossiers of 4k, 300 statements, 1,200 short codes). R1 plus R2: 6.3M. R3 on disputes: 1M. Memory-only: 1M.
- Validation: about 2M.
Total about 18M tokens, 6 to 7 days at 3M per day. The 25-book development run costs about 4M.
