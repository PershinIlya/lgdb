# Dilemma axes: locked plan (written 2026-11-28, before any placement was seen)

Why: in the full run (work_full) SVD axes mixed topics, poles were hard to read, and modern books
rarely reached the map. Here each axis is a question with two answers, written from the dossiers'
own "advice || instead of" pairs (A1), and every book is placed on every axis (A2).

Data: same 122 memory rows, 9 text rows and controls as work_full (same blinding, row ids).
Raters: R1 sonnet; R2 opus with answers P and Q swapped (position bias cancels).
Scale: P2 P1 M Q1 Q2 mapped to +2..-2 toward answer x; N, U = not placed.

Consensus per book and axis: both raters placed -> mean; only one placed -> that value (weak);
neither -> not placed. Dispersion = |R1 - R2| where both placed.

Keep an axis if all hold:
- placed (consensus) for >= 20 real books;
- >= 5 books on each side (consensus <= -1 and >= +1);
- rater agreement: Pearson r >= 0.6 on books both placed, and opposite signs (one > 0, other < 0) in <= 10% of them;
- controls: invented and null rows placed on no axis.

Questions answered per kept axis:
- extremity: share of placed books at |v| = 2 in each rater's raw codes, versus M and |v| = 1;
- era and region: eta2 with 1000 permutations, on placed books;
- memory vs text: sign agreement on the 9 classics where both rows are placed.
The map shows kept axes; a book is drawn when it is placed on at least 2 of the 3 shown axes,
a missing coordinate sits at 0 and is marked in the book card.

## Amendment 1 (2026-11-28, before any coding was seen)
The leakage audit failed for A1 (axis writer) and A2 (placement): 2 of 3 models put "opposed advice"
in their top 2 guesses. A2 is dropped. Instead every answer of every axis is one statement (24 in all,
S201..S224), coded with the audited S4a/S4b prompts (R1 sonnet, R2 opus, statements shuffled,
pairs never shown together). Per rater, position = s(x) - s(y) with A=+1, B=-1, other 0, so -2..+2;
not placed if neither answer got A or B. Consensus and keep rules as above.
A1 is used as written (it only words questions; no book is coded with it); its audit failure is
recorded in work_axes/audit_raw and left for the researcher to accept or reject.
Extra titles: 24 well-known books, speeches and essays outside the mechanical sample
(work_axes/extra_books.json), recalled with S1 and coded like the rest. They are drawn on the map,
marked as added, and excluded from every test.

## Amendment 2 (2026-11-28, after the first analysis, at the researcher's request for more modern titles)
Modern books touch few of the 12 axes (median 4). More axes are written with the same A1 prompt, fed with
the 40 tension groups not used by K01..K12 that the existing axes cover worst (mean number of placed axes
of the group's books, lowest first). Period is not used in the selection. New axes are coded like the first
ones (S225 onward, codes_more.jsonl) and analysed with the same keep rules. K01..K12 results are unchanged.
