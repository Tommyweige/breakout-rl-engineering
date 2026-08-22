# Repository instructions for coding agents

## Technical article writing

Files under `docs/day*.md` are reader-facing technical articles, not README-style implementation reports, PR descriptions, lab notes, or acceptance-checklist prose.

Write from the reader's understanding path: problem → intuition → technical mechanism → real evidence → limitation → next question. Keep technical depth, but do not organize the article around filenames, tests, acceptance criteria, or implementation order unless that ordering genuinely helps explain the concept.

### Heading style

Do **not** turn every section heading into a rhetorical question.

Question-style headings are useful only when the question represents a real conceptual tension, common misconception, or important transition that the reader genuinely needs answered. Do not manufacture obvious questions just to make the article look conversational.

Prefer a natural mix of heading styles:

- concept headings, e.g. `Q-value 不是機率`
- conclusion headings, e.g. `未訓練模型的 argmax 沒有策略意義`
- process headings, e.g. `從 3,136 features 到四個 Q-values`
- evidence headings, e.g. `用真實 forward 檢查輸出`
- occasional question headings when they genuinely improve understanding, e.g. `Q-value 是機率嗎？`

A heading should tell the reader what the section is about or what they will learn from it. If converting a statement into a question makes it longer, more obvious, or more like filler, keep it as a statement.

Avoid repetitive patterns such as multiple consecutive headings beginning with `為什麼...`、`到底...`、`怎麼...`. The article should feel like a coherent technical narrative, not a FAQ generated from every paragraph.

## Images and attachments

For new or modified article attachments, do **not** use parent-relative Markdown paths such as:

```text
../assets/day08/example.png
../../assets/example.svg
```

Use an absolute repository/GitHub reference instead. For image assets that must render from GitHub, prefer a full absolute GitHub `raw` URL pointing to a stable ref. When an article is moved/synchronized to another repository, rewrite the absolute repository URL to that repository rather than leaving a private-repo URL behind.

Generated artifact paths shown as text should also be root-anchored, for example:

```text
/assets/day08/dqn-q-values.png
/assets/day08/dqn-q-values.json
```

## Visualization quality

Visualization is evidence and explanation, not decoration.

- Images must answer a technical question.
- Use real program output, model output, runtime artifact, CSV/JSON/NumPy data, or actual environment observations when the claim depends on them.
- Never hard-code fake values to make a chart look good.
- Do not create decorative HTML/CSS/React/Vue dashboards, fake terminals, or mock UIs for article screenshots.
- If a terminal/output image is included, preserve the underlying text output or metadata when practical so the image is auditable.
- If a Mermaid diagram is useful, storing Mermaid source is allowed, but the article must embed a rendered image (SVG/PNG/screenshot). Do not make raw Mermaid syntax the only visual artifact.
- A diagram that explains architecture may be schematic, but it must match the actual implementation.
- A chart that claims runtime/model behavior must be generated from actual runtime/model data.

For every important image, the article should explain: what the reader is seeing, why it matters, what conclusion it supports, and what it does **not** prove.

## Day-to-day reproducibility

For Day 7 onward, prefer a reproducible visualization/inspection script and preserve seed/run/checkpoint/source metadata when applicable. The article should provide the command needed to regenerate the artifact.
