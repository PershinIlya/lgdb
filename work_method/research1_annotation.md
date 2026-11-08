Research summary 1: LLMs as text annotators (2023 to 2026), condensed

- LLM labels are measurements; agreement among LLMs does not validate them. Minimum: accuracy against a held-out human gold set (Tornberg 2024; Ziems et al. 2024; Said and Villaneau 2026, where Dawid-Skene overestimated GPT accuracy 90.7% vs 71.0% on human gold because models shared a bias).
- Prompt stability: treat paraphrased prompts and repeated runs as separate raters, Krippendorff alpha >= 0.8 heuristic (Barrie, Palaiologou, Tornberg 2024/2026). Stability is necessary, not sufficient.
- Temperature 0, recorded prompt codebook, an "uncertain" option, JSON output, multi-model checks; do not tune prompts on the validation set (Tornberg 2024).
- Option order bias: models favor the last listed option; shuffle or reverse options (Rupprecht et al. 2025). Central tendency / neutral drift is common.
- Batching: 6 of 8 models within 2 points of single-item accuracy up to 100 items per batch on tweets (Pipal et al. 2026 preprint); cross-item contamination not well studied.
- Aggregation: majority vote, Dawid-Skene, MACE assume independent errors; LLMs violate this. Correlation-aware aggregators exist (Balasubramanian et al. 2026). Route disagreements to humans; confidence-driven inference with a small targeted human sample keeps valid intervals (Gligoric et al. 2025). Design-based supervised learning corrects downstream estimates with a random human-coded subsample (Egami et al. 2023).
- "LLM hacking": across 37 replicated tasks, top models led to wrong conclusions for about 31% of hypotheses; a few prompt paraphrases could make almost any result significant; human labels are the strongest guard (Baumann et al. 2025).
- Memorization of famous books inflates performance on canonical texts (Chang et al. 2023).
- Pairwise statement judgment: entailment/attestation bias (McKenna et al. 2023); ChatGPT overpredicts entailment and rarely predicts contradiction (Basmov et al. 2023); position bias, run both orders (Wang et al. 2024).
- Pairwise scaling: concept-guided chain of thought plus Bradley-Terry over many A-vs-B judgments beat Wordfish for placing texts on a latent dimension (Wu, Tucker, Nagler, Messing 2023).
