Design brief: how to process each book (the core of the method)

GOAL
Place about 70 to 150 influential books that give guidance on how to live (from before 1500 to after 1945; Western, East Asian, and South Asian; aphorism collections and single-thesis books, including modern bestsellers such as Atomic Habits or The Courage to Be Disliked) on data-driven dimensions along which their recommendations differ. The dimensions must emerge from the data, not from a list the researchers supply. Downstream questions: which dimensions exist, where each book sits, whether era and culture predict position, and whether books tend to sit at one end of a dimension rather than in the middle. Later deliverable: an interactive map of books on the dimensions.

HARD CONSTRAINTS
- No API budget. All model calls are made by Claude subagents inside a subscription (opus, sonnet, haiku available). Each subagent call has about 70k tokens of fixed overhead, so work must be batched (tens to a few hundred items per agent). Total budget is roughly a few million tokens per day. Designs needing tens of thousands of separate calls are not feasible; batched designs are.
- Full texts are legally available only for older books (mostly before 1931) plus a few free modern religious texts. Modern bestsellers are available only through model memory (the model's knowledge of the book) or third-party summaries.
- Annotator models must not be told the researchers' expectations, and prompts must not suggest dimensions, poles, or trade-offs, so the dimensions stay data-driven. Each new prompt is checked by asking fresh models to guess the hypothesis behind it.
- The researcher can review roughly 100 items by hand, not thousands.

WHAT WAS TRIED AND WHAT HAPPENED
1. Extraction from full texts (10 classic books, 74 passages of about 1,500 words): 895 recommendations with fields (domain, wording force, conditions, argued_against, and others). Worked fine.
2. Topic clustering of recommendations by embeddings (domain field plus recommendation text): 12 topics at a tuned threshold.
3. Pairwise relation judgments within topics, about 2,300 pairs, batches of 200 per agent, labels duplicate / narrower / compatible / conditional / incompatible / unrelated. Result: 80% compatible, 15% unrelated, 2% conditional, 3 incompatible. No topic split into camps. Real tensions were labeled conditional, for example "be self-sufficient, able to live alone" vs "human nature is made for social acts", and "resist your father when he is wrong" vs "love and respect your parents above all". The literature confirms LLMs overpredict entailment and rarely predict contradiction.
4. Model memory (3 models, 23 books plus 3 invented books): all models said "unknown" for invented books. Same-book agreement between models clearly above different-book agreement. About half of recalled recommendations had a matching passage in the full text by a crude embedding check. Recalled recommendations almost never carry conditions (1 to 9%) and are almost all worded as direct instructions.
5. Model memory on 10 modern bestsellers: 121 recommendations; the argued_against field (what the author criticizes) was very informative, often naming the opposite pole directly. Only 4 topics were shared by at least 4 books: modern books talk about different things.

LITERATURE (see research1_annotation.md and research2_dimensions.md in the same folder)

YOUR TASK
Propose a concrete method for the core: what exactly the models are asked about each book (exact question wording or close to it), in how many variants, by how many models, how answers are aggregated, how dimensions are derived, and how the whole thing is validated, within the constraints above. Be specific enough to implement. Estimate token cost. State the main risks and how your design guards against each, especially: LLM bias toward "compatible", hypothesis leakage, memory versus text, era and translation confounds, and silent versus disagreeing books. Keep it to about 800 to 1200 words. Plain ASCII punctuation, no dashes as punctuation.
