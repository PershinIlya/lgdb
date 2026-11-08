Critique 1: what can still make the stance matrix fail

Bottom line: the core is sound in principle. But as written, it will most likely produce a matrix that is mostly empty and mostly "endorse". Its first factor would then reflect coverage, polemical style, or source type, not a disagreement in advice. All of this can be tested for about 1M tokens before the main run.

1. The "agree" bias in a new form

Risk: high. The question "what would the author say about this statement?" invites a generous reading. A Stoic text will be coded "endorse" on "Value your relationships" by abstraction. Rejection needs explicit evidence, and the dossiers contain recommendations, not refusals. So reject codes can come almost only from argued_against lines. Two sinks will absorb rejects. The first is "not addressed". A's line "'Does not address' is the expected answer" pushes doubtful rejects there too. The second is "conditional" or "leans" (B's C code, A's +1 and -1). That is the same sink that swallowed real tensions in step 3.

Are argued_against items and twins enough? Not as designed. In A, a statement can enter the pool because 2 books argue against it. Those 2 books then reject it by construction. That is a circular signal, and it makes "who wrote the source line" look like a dimension. Negated twins measure acquiescence, but they rarely create disagreement. The twin of a pleasant statement is usually a strawman ("Avoid long periods alone") that every book rejects.

Recommendations:
Keep: C's evidence rule (no cited id, no code).
Change: use 3 codes plus "unsure". Drop "conditional" and "leans".
Change: write statements as specific claims with a visible cost ("Turn down social invitations to protect time for study"), not abstract virtues.
Change: report the reject share per statement. In NOMINATE style, drop near-unanimous statements (minority under 10% of covered books), and report this rule up front.
Change: mark "own-source" cells (the book contributed to the statement's cluster). Fit the model with and without them.

2. The statement pool as hidden framing

Risk: high. This is now the main place where model framing enters. The clustering threshold sets the level of abstraction. A loose threshold gives universal platitudes that everyone endorses. A tight one gives book-specific statements that only one book addresses. Either way, there is little disagreement. The writer of the canonical statement ("A person should ...") recasts all advice into the model's modern therapeutic vocabulary. The pool also overweights books with many lines: text books give 60 to 100 lines, memory books give 25. Modern books share few topics (step 5). So they will be silent on most statements built from classics.

Recommendations:
Change: cap each book's contribution to the pool (for example, 30 lines per book).
Change: build the pool at 2 thresholds and compare the factors. A dimension that exists at only one threshold is an artifact.
Change: add a human check of 30 statements: does the statement match its cluster, and does it carry a cost?

3. Identifiability with about 120 books

Risk: the most serious one. The Senate analogy breaks down. Senators vote on almost every bill, with under 5% missing. Here, coverage is plausibly 10 to 15%. At 12% coverage, 2 statements share about 120 x 0.12 x 0.12, roughly 2 books. So C's rule of at least 15 co-covered books will almost never be met. The correlation matrix will be undefined or not positive definite. Bayesian IRT will still return numbers, but low-coverage books shrink toward the prior mean. That fakes a "middle" and biases the polarization test against the hypothesis. Silence is also not random: it follows topic and genre.

Likely artifacts for the first factor:
(a) Reject propensity: books with many argued_against lines reject more. This is a "polemical style" factor.
(b) Source type: memory dossiers versus text dossiers.
(c) Difficulty factors: near-ceiling binary items load together because they share endorsement rates, not content.
Religious versus secular vocabulary enters through coverage blocks, not through rejection.

Minimum coverage to require: each kept statement covered by at least 20 books, with at least 4 in the minority stance. Each placed book has at least 25 covered cells, and at least 8 on each dimension where it is placed.
Change: regress factor scores on coverage, argued_against count, source type, and religious/secular. Drop any factor that one of these explains at R2 above 0.3.
Keep: A's separate coverage map. The pattern of silence is a real result about who talks about what.

4. Which dossier conditions are needed

Needed:
- Text dossier for the text books, as the main measure.
- Memory dossier for all books.
- For about 25 text books, both dossiers scored the same way, to measure the shift.
B's title-only condition is the raw memory prior. It adds little. Drop it, or keep it for 5 books only.

"Dossier-blind" is not really blind. A dossier of Meditations is recognizable. So test recognition: ask the model to name the book from its dossier.

The remaining confound cannot be removed: source type is almost perfectly aliased with era. Modern books are memory only, and old books are mostly text. The calibration on famous classics does not transfer to recent bestsellers. Their recall comes from blurbs and online summaries, and is biased toward the central thesis. So era effects on the extremity question are not identifiable from memory books. Only the text-based subset can test them.

5. Quiet loss of "data-driven" or hypothesis leakage

- Keep rules that require at least 4 or 5 books at each end (A, B) select bipolar dimensions. C's "confirmed axis" via argued_against links does the same. That builds the hypothesis into the pipeline. Report unipolar factors too.
- Prompts that say "write the opposite" reveal an interest in opposition to the twin writer. A does not state that twins are batched apart from originals. Make that explicit, as B does.
- Selecting statements on disagreement is defensible. But the extremity test must then use a null that applies the same selection to permuted data.
- The leakage check (guess the hypothesis) tests prompts, not pools. Also run it on a sample batch of statements.

6. The smallest decisive experiment

Use the 20 existing books: 10 text classics with 895 lines, and 10 modern books with 121 lines. Also use the existing memory recall for the classics.

1. Build a pool of about 80 statements: support from 2 books, a cap per book, 2 thresholds.
2. Score stances for both pools. Use sonnet and opus, the evidence rule, and 3 codes. Score the classics from both text and memory dossiers. Cost: about 12 calls, under 1.5M tokens.
3. The researcher codes 40 gold cells, oversampled toward rejects.

Go criteria:
- Rejects are at least 10% of covered cells, and not only own-source cells.
- At least 30% of statements have at least 2 rejecting books.
- Median coverage per book is at least 25%.
- Reject precision is at least 0.7 and reject recall at least 0.5.
- Krippendorff alpha is at least 0.7.
- The first SVD component of the -1/+1 matrix does not simply separate classic from modern, or memory from text.

If rejects stay under 5%, the core fails, as step 3 did.
Fallback: forced choice between 2 data-linked statements, using C's argued_against links: "which would this author advise?" This is a preference question, not a compatibility question.
