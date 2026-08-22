# Repository instructions for coding agents

## Technical article writing

Files under `docs/day*.md` are reader-facing technical articles, not README-style implementation reports, PR descriptions, lab notes, or acceptance-checklist prose.

Write from the reader's understanding path: problem → intuition → technical mechanism → real evidence → limitation → next step. Keep technical depth, but do not organize the article around filenames, tests, acceptance criteria, or implementation order unless that ordering genuinely helps explain the concept.

### Explain terminology at first use

Assume the reader is technically curious but may be seeing the current concept for the first time.

Do not stack unexplained jargon into a sentence. When an important technical term first appears, explain it immediately in plain language before continuing. Prefer Chinese/plain-language meaning first, then the English term when the English name is useful for later reference.

Examples:

- `dtype` → explain that it is the data type that determines how each value is stored and how much memory it uses.
- `observation shape` → explain what each dimension means, not only the tuple itself.
- `frame stack` → explain that several recent frames are grouped together so motion can be inferred.
- `off-policy` → explain that data collected by an older or different behavior can still be used for learning.
- `Tensor` → explain that it is PyTorch's multi-dimensional array used for neural-network computation.

If an English engineering term is not necessary for the reader to understand the idea, replace it with plain language instead of introducing extra vocabulary. For example, prefer `實際量測發現 RAM 成為瓶頸` over `profiling 證明 memory bottleneck`, unless profiling itself is the concept being taught.

Do not assume that a reader who has followed earlier days automatically knows every implementation term such as `slot`, `write_index`, `uint8`, `float32`, `MiB/GiB`, `bootstrap mask`, `compact replay`, or `frame-level storage`. Either define the term when it matters or omit it.

The goal is not to remove technical vocabulary. The goal is that a reader should not need to leave the article to understand a sentence whose main idea is otherwise simple.

### Keep implementation detail proportional

Do not paste code merely because it exists in the implementation.

Keep a code snippet only when seeing the code materially helps the reader understand the concept being taught. Prefer a compact shape/data-flow example or a short representative fragment over reproducing constructors, validation logic, CLI wiring, serialization code, test cases, parameter counts, or file-by-file implementation details.

Tests and engineering checks should normally stay in the repository rather than becoming article sections. Mention them only when the observed result teaches something the reader needs to understand.

If a paragraph can explain the idea more clearly than a code block, use the paragraph.

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

When an article is synchronized to another repository, rewrite the absolute repository URL to that repository rather than leaving a private-repo URL behind.

Generated artifact paths shown as text should be root-anchored, for example:

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

For Day 7 onward, prefer a reproducible visualization/inspection script and preserve seed/run/checkpoint/source metadata when applicable. The article should provide the command needed to regenerate the artifact when the command materially helps reproducibility.
