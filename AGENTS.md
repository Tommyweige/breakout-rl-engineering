# Repository instructions for coding agents

## Technical article writing

Files under `docs/day*.md` are reader-facing technical articles, not README-style implementation reports, PR descriptions, lab notes, or acceptance-checklist prose.

Write from the reader's understanding path: problem → intuition → technical mechanism → real evidence → limitation → next step. Keep technical depth, but do not organize the article around filenames, tests, acceptance criteria, or implementation order unless that ordering genuinely helps explain the concept.

### Keep implementation detail proportional

Do not paste code merely because it exists in the implementation.

Keep a code snippet only when seeing the code materially helps the reader understand the concept being taught. Prefer a compact shape/data-flow example or a short representative fragment over reproducing constructors, validation logic, CLI wiring, serialization code, test cases, parameter counts, or file-by-file implementation details.

Tests and engineering checks should normally stay in the repository rather than becoming article sections. Mention them only when the observed result teaches something the reader needs to understand.

If a paragraph can explain the idea more clearly than a code block, use the paragraph.

### Keep authoring mechanics out of the reader-facing article

The article should explain the technical concept, not how Codex produced the article assets.

Do **not** put the following in `docs/day*.md` unless the generation process itself is the concept being taught:

- Mermaid source file locations such as `/assets/dayXX/diagram.mmd`;
- render commands or regeneration commands;
- local machine paths such as `C:\\Users\\...`;
- Codex skill/script locations;
- implementation notes about whether a figure was generated as SVG or PNG;
- standalone labels such as `**svg**`, `**png**`, `Mermaid source`, or similar artifact-maintenance prose.

These details may be preserved in the repository, Issue, asset metadata, comments, a dedicated reproducibility note, or a developer-facing README. They do not belong in the main teaching narrative merely because they are useful to maintainers.

Reproducibility is still required; it should normally be implemented in the repo rather than narrated to the reader.

### Heading style

Do **not** turn every section heading into a rhetorical question.

Question-style headings are useful only when the question represents a real conceptual tension, common misconception, or important transition that the reader genuinely needs answered. Do not manufacture obvious questions just to make the article look conversational.

Prefer a natural mix of heading styles:

- concept headings, e.g. `Q-value 不是機率`
- conclusion headings, e.g. `未訓練模型的 argmax 沒有策略意義`
- process headings, e.g. `從 3,136 features 到四個 Q-values`
- evidence headings, e.g. `用真實 forward 檢查輸出`
- occasional question headings when they genuinely improve understanding

A heading should tell the reader what the section is about or what they will learn from it. If converting a statement into a question makes it longer, more obvious, or more like filler, keep it as a statement.

Avoid repetitive patterns such as multiple consecutive headings beginning with `為什麼...`、`到底...`、`怎麼...`. The article should feel like a coherent technical narrative, not a FAQ generated from every paragraph.

## Images and attachments

For new or modified article attachments, do **not** use parent-relative Markdown paths such as:

```text
../assets/day08/example.png
../../assets/example.svg
```

Use an absolute GitHub URL instead.

For private-repository images, do not use `raw.githubusercontent.com` as the default article link because authenticated browser rendering can be unreliable. Prefer a full GitHub `blob` URL on a stable commit/ref with `?raw=1` for the image source, and wrap the image in a link to the corresponding blob page so the reader can click it.

Example pattern:

```md
[![diagram](https://github.com/OWNER/REPO/blob/COMMIT/assets/dayXX/diagram.svg?raw=1)](https://github.com/OWNER/REPO/blob/COMMIT/assets/dayXX/diagram.svg)
```

If an image is visually too large in the article, do not accept it merely because the source file is valid. Prefer to redesign/render a more compact diagram. When necessary, use GitHub-compatible inline HTML only for image sizing, for example a clickable `<a><img ... width="720"></a>`; do not introduce CSS or decorative HTML layouts.

When an article is synchronized to another repository, rewrite the absolute repository URL to that repository rather than leaving a private-repo URL behind.

Generated artifact paths shown as text should be root-anchored only when the path itself is relevant to the reader. Do not show artifact paths by default.

## Visualization quality

Visualization is evidence and explanation, not decoration.

- Images must answer a technical question.
- Use real program output, model output, runtime artifact, CSV/JSON/NumPy data, or actual environment observations when the claim depends on them.
- Never hard-code fake values to make a chart look good.
- Do not create decorative HTML/CSS/React/Vue dashboards, fake terminals, or mock UIs for article screenshots.
- If a terminal/output image is included, preserve the underlying text output or metadata in the repo when practical so the image is auditable; do not automatically paste that maintenance detail into the article.
- If a Mermaid diagram is useful, write the Mermaid source first, verify it against the real implementation/data flow, render it to SVG/PNG/screenshot, and embed the rendered result. Raw Mermaid syntax is not the final reader-facing visual.
- A diagram that explains architecture may be schematic, but it must match the actual implementation.
- A chart that claims runtime/model behavior must be generated from actual runtime/model data.

### Place diagrams where they teach

A diagram should appear next to the concept it explains.

Do not move every process into one giant overview figure at the beginning of the article. Prefer multiple focused diagrams when the article contains distinct teaching moments.

Do not delete a later diagram merely because an earlier overview contains the same nodes. If a later section introduces an important lifecycle, branch, state transition, or mental model, it may deserve its own smaller focused diagram at that location.

For example, a Day 9 article can reasonably have:

- an early compact diagram for `transition → Replay Buffer → sample → model input`;
- a later focused lifecycle diagram for `new transition → buffer full? → overwrite oldest → continue`.

Prefer a diagram that can be understood without scrolling through a very tall canvas. Split oversized/tall diagrams into smaller focused visuals, remove redundant labels, and keep only the nodes needed for the nearby explanation. Bigger is not automatically clearer.

For every important image, the article should explain what the reader is seeing, why it matters, and the conclusion it supports. Avoid implementation-maintenance commentary unless it changes the technical interpretation.

## Day-to-day reproducibility

For Day 7 onward, prefer a reproducible visualization/inspection script and preserve seed/run/checkpoint/source metadata when applicable.

Keep regeneration commands, Mermaid source, scripts, and metadata in the repository so maintainers can reproduce the artifact. Put a command in the reader-facing article only when running that command is itself useful to understanding the day's technical topic.