Full stance-matrix run, locked plan v2 (written 2026-11-22, before any full-run coding)

Changes from the merged design, decided on Nov 15 with the researcher (minimal involvement, run autonomously):
1. Researcher hand checks are replaced by a model gold panel: each gold cell coded alone by fresh opus, sonnet and haiku agents with prompt S4a; majority vote is gold. Conclusions will say the gold is model-based.
2. Dossiers come from memory (S1, opus) for every book. Pilot: memory vs text sign agreement 0.985 on 326 cells. Text dossiers kept for the 9 pilot classics in the corpus as extra rows, to keep checking memory against text. Era effects are reported with that caveat.
3. Corpus: mechanical, from corpus/selection.csv rules 1 and 2 (format match, one cell per title); availability of a public-domain text is no longer required. Per (period, region, format) cell the first 7 eligible candidates by rank: 122 books. Books the model marks "unknown" are dropped. Controls: 3 invented books, 2 null dossiers.
4. S3 v3 (hides advised vs criticized from the writer; criticized practices stated as their supporters would). Clusters need 3 books; at most 100 per threshold (0.40 and 0.50 cosine distance).
5. No reversed statements in the full run (acquiescence was 0% in the pilot).

Coding: R1 sonnet (S4a), R2 opus (S4b), haiku on A vs B disputes. Final cell rule as in the pilot.

Analysis (preregistered):
- Drop near-unanimous statements: keep statements addressed (A or B) by at least 5 books with the minority side at least 10%. Report how many are dropped.
- Dimensions: SVD of the book x kept-statement matrix (A=1, B=-1, else 0), books with at least 10 addressed kept statements. k chosen by parallel analysis against column-permuted matrices (95th percentile). Each dimension kept only if: bootstrap (books) Tucker congruence >= 0.85; at least 5 books at each end (beyond 1 SD); R2 of scores on coverage, B count and source (memory/text) each below 0.3; R1-only vs R2-only score correlation >= 0.8. Unipolar dimensions are reported too.
- Bipolarity: share of kept statements with both sides >= 20%; per dimension, books on both ends.
- Extremity (one-pole hypothesis): bimodality of book scores per dimension (share of books beyond 1 SD vs middle) compared with a null that permutes each statement's codes among the books that address it and repeats the same selection and SVD (200 permutations).
- Era and culture: eta-squared of scores on period and on region, permutation p (1000); memory rows only, text rows as a check.
- Poles named by one opus agent from the top 12 statements per end, no book names (P6, exempt).

Amendment (Nov 22, before statements and coding): clusters with more than 40 members are dropped. At 0.50 a few average-linkage clusters chained unrelated advice (up to 123 items from 70 books), which cannot yield one checkable statement.
