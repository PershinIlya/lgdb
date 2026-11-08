Merged design: one stance matrix, 2 raters, evidence ids, staged rollout

The merge keeps the cheap guards and cuts costly redundancy, mainly B's 4 raters and full negated twins.

1. ELEMENT TABLE

| Element | From | Keep? | Why |
|---|---|---|---|
| argued_against entries become statements, kept if 3 books endorse or 2 reject | A, B | Keep | Main source of opposite poles |
| Criticized views need 3 endorsing books | C | Drop | Deletes poles only critics name |
| Link statements when one's items argue against the other | C | Keep | Free check of two-pole axes |
| Condense dossiers to plain English | B, C | Text books only | Memory output is already plain |
| Title hidden in stance coding | B | Keep | Blocks canonical-book prior, free |
| Evidence id required, else "not addressed" | C | Keep | Best guard against support everywhere |
| Conditional code | B | Merged into 1 and -1 | Separates balanced from silent |
| 4 raters on every cell | B | Drop | About 6M; a 30-book subset shows stability |
| Twin for every statement | B | Drop | Doubles the matrix; 40 pairs suffice |
| Haiku tie-breaker | A, C | Keep | Disputes are about 15% of cells |
| Memory-only coding of 25 text books | A, B | Keep | Measures memory shift |
| Ordinal IRT, k by held-out cells | A, B | Keep | C's pairwise correlations are unstable when sparse |
| Keep rules: 5 books per end, 2 eras or cultures, bootstrap 0.85, source under 10% | A, B, C | Keep | Local, free |
| Bradley-Terry confirmation | A, B | Reduced | 3 dimensions, 1 model |
| Null dossiers, invented books | A | Keep | Cheap negative controls |
| 2 translations | all | 4 books | 8 adds little |
| Egami correction, locked plan | A, B, C | Keep | Guard against LLM hacking |

2. MERGED PIPELINE

Stage 1, text extraction. Existing prompt. Sonnet, 12 passages per book, 24 per agent, 40 new books: 20 calls, 2.8M.

Stage 2, memory recall. Opus and sonnet, 15 books per agent, all books plus 3 invented:
"From your knowledge of [title] by [author], list up to 25 things the book tells the reader to do, value, or avoid. For each give: (1) the advice as one plain modern English sentence, not a quotation; (2) any view or practice the author argues against while giving it, or 'none'; (3) who or when it applies to, if the book says; (4) how central it is: core, recurring, or passing. If you do not know the book well, answer 'unknown'."
17 calls, 1.5M.

Stage 3, condensation of text dossiers. Sonnet, 10 books per agent:
"Merge these notes into at most 40 entries. Write each in plain modern English, keep its conditions and anything the author argues against, and give each a stable id."
5 calls, 0.65M.

Stage 4, statement pool. Local embedding and clustering. Opus, 125 clusters per agent:
"Write one sentence of the form 'A person should ...' that states what these items advise, so that someone could check it against a book. Do not refer to any other group." argued_against clusters get: "...that states the advice these items criticize."
Target 200 statements. For 40 of them, a fresh agent: "Write the sentence a person who disagrees with this would assert." About 0.4M.

Stage 5, stance coding. 5 dossiers per agent, title and author removed, statements shuffled:
"Below are notes on what one book advises, then a list of statements. For each statement, say what this book's author would tell a reader. Codes: 2 = clearly urges it; 1 = urges it only in some circumstances, or leans toward it; 0 = the book does not address it; -1 = advises against it in some circumstances, or leans against it; -2 = clearly advises against it; ? = cannot tell. For any code other than 0 or ?, give the note id that supports it. Most statements will not be addressed by any one book."
R1 sonnet, codes listed 2 to -2. R2 opus, paraphrase, reversed order. Haiku settles sign conflicts. A nonzero code without a valid id becomes 0. On 30 stability books, each model also runs the other paraphrase in new batch company. 25 text books are also coded from memory alone. About 100k per call. Main 4.8M, stability 1.2M, tie-breaks 0.35M, memory-only 0.5M: 6.9M.

Stage 6, dimensions. Subtract each rater's acquiescence. Cell = mean of R1 and R2; 0 and ? are missing. Fit ordinal IRT, k by 10% held-out cells, apply the keep rules plus R1-alone versus R2-alone r >= 0.8. Opus names poles from the top 12 statements per end, no book names: 0.1M.

Stage 7, confirmation. Sonnet, top 3 dimensions, 150 pairs per dimension in both orders, one call each:
"Which of these two books more strongly advises the reader to [pole description]? Answer A, B, equal, or neither addresses this."
Require r >= 0.7 with IRT scores: 0.3M. Leakage: 3 fresh haiku agents answer "What hypothesis might the authors of these prompts be testing?": 0.25M. Translations, 4 books: 0.4M.

3. STAGED ROLLOUT

Pilot: 10 full-text classics (extraction exists) plus 10 modern bestsellers (memory only), plus 3 invented books and 2 null dossiers. Recall 0.2M, condensation 0.13M, 120 statements 0.12M, R1 and R2 on 30 rows (classics also from memory) 1.1M, 2 extra raters on 10 books 0.36M, haiku 0.08M, leakage 0.25M. Total 2.3M, one day.

Go only if all hold:
1. At least 10% of addressed cells negative; at least 20 statements with both a positive and a negative book.
2. On 40 gold cells: reject precision >= 0.7 and reject recall >= 0.6.
3. Krippendorff alpha R1 versus R2 >= 0.7.
4. Both a statement and its reversal endorsed in at most 10% of pairs.
5. Invented books and null dossiers: at least 90% of cells uncoded.
6. Classics: memory versus text sign agreement >= 0.7.
7. No leakage guess mentions poles, opposites, trade-offs, or one-sidedness.

No-go: revise the failing prompt once and recheck on 10 fresh gold cells; a second failure stops the method. Criterion 6 alone only restricts conclusions to text books.

Full run: about 120 books, stages 1 to 7, 13.3M, 5 days at 3M per day. Both stages: 15.6M.

4. RESEARCHER BY HAND (100 items)

1. Before the pilot: write the locked plan; read leakage guesses.
2. After pilot coding, blind to model codes: 40 stance cells (15 predicted reject, 10 endorse, 10 not addressed, 5 disputes).
3. After the full run, blind: 30 cells, 20 random (for Egami) and 10 disputes.
4. After poles are named, before book scores: rank 10 known books on the top 2 dimensions (20 items).
5. Last: check 10 pole labels.
