# Day 24 article workflow record

This developer-facing note records the required writing workflow for the
reader-facing article `docs/day24-correct-inference-benchmarking.md`.

- The `technical-blog-writer` skill was read before drafting the article.
- Its writing-style, article-structure, code-guidelines, visualization-policy,
  mermaid-workflow, and review-checklist references were read before the
  corresponding writing and figure actions.
- The article promise was fixed as: explain why batch=1 inference timing is a
  product decision-budget question, then use real native raw samples to
  separate model-only cost, end-to-end cost, synchronization, and provider
  boundaries.
- The benchmark report and figures are generated from benchmark id
  `day24-final-native-v2-20260905-rtx4060`; no chart values are manually entered.
- The Mermaid sources were checked against the shared inference adapter and
  timing implementation, rendered with the pinned helper, and visually
  inspected before embedding.
