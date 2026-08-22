# Repository instructions for coding agents

## Technical article writing

Files under `docs/day*.md` are reader-facing technical articles, not README-style implementation reports, PR descriptions, lab notes, or acceptance-checklist prose.

Write from the reader's understanding path: problem → intuition → technical mechanism → real evidence → limitation → next question. Keep technical depth, but do not organize the article around filenames, tests, acceptance criteria, or implementation order unless that ordering genuinely helps explain the concept.

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
