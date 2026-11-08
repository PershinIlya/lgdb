Stance-matrix pilot, locked plan (written 2026-11-08, before any stance coding)

Rows: 10 classics from text dossiers (S2, from work_pilot1 claims), the same 10 from memory (S1, opus), 10 modern books from memory (S1, opus), 3 invented books (S1), 2 null dossiers of random pseudo-words. 35 rows. Titles never shown to coders; rows have opaque ids (rows_key.json).

Pool: classics contribute text entries, modern books memory entries, at most 30 each. Advice and argued_against items are embedded (all-mpnet-base-v2) and clustered separately, average linkage, cosine distance 0.40 (tight) and 0.50 (loose). A cluster is kept if 2 or more books contribute; at most 60 per threshold, ranked by number of books. Opus writes one concrete "A person should" statement per cluster (S3). 40 random statements get a reversal by a separate opus agent (S5).

Coding: R1 sonnet (S4a), R2 opus (S4b, paraphrase, codes listed in reverse order); statement order shuffled per row and rater. Codes A, B, N, U plus flag c. A or B without a valid note id becomes N. Haiku recodes cells where R1 and R2 give A versus B.
Final cell: agreed code; A vs B goes to haiku; one rater A or B and the other N or U counts as that sign (weak); both reported.

Go criteria (critique_2_merged.md section 3): 1 negative share of addressed cells >= 0.10 and >= 20 statements with both an A and a B book; 2 gold reject precision >= 0.7 and recall >= 0.6 (researcher, 40 cells); 3 Krippendorff alpha R1 vs R2 >= 0.7 (interval on -1/0/1, U missing; nominal also reported); 4 statement and its reversal both A in <= 10% of book x pair cells where both addressed; 5 invented and null rows >= 90% N or U; 6 classics memory vs text sign agreement >= 0.7; 7 no leakage guess about poles, opposites, trade-offs, one-sidedness (S5 exempt as construct-inherent).
Also reported (critique_1): negative share without own-source cells, share of statements with >= 2 B books (>= 0.30), median coverage per book (>= 0.25), PC1 R2 against modern/classic, text/memory, coverage, B count; per-threshold negative share.
