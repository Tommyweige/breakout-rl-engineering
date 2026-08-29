# Repository instructions for coding agents

## Repository layout and hygiene

Keep responsibilities separated as the project grows:

```text
breakout_rl/   reusable library/runtime implementation
scripts/       executable CLIs grouped by responsibility
configs/       task, training-backend, and experiment definitions
docs/          reader-facing Day articles
assets/        reproducible evidence grouped by Day
tests/         regression and correctness tests
```

Rules for new work:

- reusable implementation imported by multiple tools belongs in `breakout_rl/`;
- executable training CLIs belong in `scripts/training/`;
- evaluation/baseline CLIs belong in `scripts/evaluation/`;
- diagnostics, probes, inspectors, summaries, and report generators belong in `scripts/analysis/`;
- throughput/profiling/system benchmarks belong in `scripts/benchmarks/`;
- plotting, GIF/gameplay recording, and rendered-evidence generators belong in `scripts/visualization/`;
- small educational or interactive demonstrations belong in `scripts/demos/`;
- reader-facing articles belong in `docs/dayXX-*.md`;
- new evidence belongs under `assets/dayXX/`;
- canonical/task/experiment config belongs under the appropriate `configs/` category.

Do not add a new root-level Python CLI unless there is a documented architectural reason. `breakout_env.py` is intentionally retained as the environment module during the Phase 1 cleanup; that exception is not a pattern for new scripts.

Run categorized tools from the repository root with module syntax, for example:

```powershell
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
python -m scripts.analysis.analyze_q_values --help
```

Do not repair moved-script imports with hard-coded local paths or `sys.path.append(...)`. Use package/module imports. Historical JSON/manifests may preserve the command/path that originally produced them; do not rewrite historical provenance merely to match the current layout.

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

All reader-facing article images must use clickable Markdown image syntax. Do not use inline HTML such as `<a><img ...></a>` for normal article images.

Required pattern:

```md
[![diagram](https://github.com/OWNER/REPO/blob/COMMIT/assets/dayXX/diagram.svg?raw=1)](https://github.com/OWNER/REPO/blob/COMMIT/assets/dayXX/diagram.svg)
```

The inner URL is the rendered image source and must end with `?raw=1`. The outer URL points to the corresponding GitHub blob page so the image is clickable.

For private-repository images, do not use `raw.githubusercontent.com` as the default article link because authenticated browser rendering can be unreliable. Prefer a full GitHub `blob` URL on a stable commit/ref with `?raw=1` for the image source.

If an image is visually too large, redesign or rerender the source image to be more compact rather than switching to HTML width controls. Keep the article image syntax consistent.

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

## Canonical Breakout environment contract from Day 16 onward

Day 15 established **Breakout Evaluation / Environment Contract v2**. For all Day 16+ work that creates, trains, evaluates, records, compares, or deploys a Breakout policy, the machine-readable source of truth is:

```text
configs/eval/breakout_contract_v2.json
```

Do not reconstruct this contract from memory, copy values into a new config and let them drift, or silently fall back to the older Day 15 Contract v1. Load and validate the committed contract whenever the runtime can do so directly; otherwise derive an explicit adapter from it and record any unsupported field as a parity limitation.

The canonical semantics currently include:

```text
environment_id = ALE/Breakout-v5
frame_skip = 4
frame_stack = 4
sticky_action_probability = 0.25
fire_reset = true
terminal_on_life_loss = false
TimeLimit source = ale.game_truncated
max raw frames per episode = 108000
agent-step limit = 27000
evaluation epsilon = 0
raw evaluation reward = unclipped Atari reward
fixed concrete evaluation seeds = Contract v2 list
```

### FIRE ownership is part of the RL task definition

From Contract v2 onward, the **environment owns only the mandatory serve FIRE behavior**:

```text
initial serve
+
immediately after an observed life loss
```

The policy action space still contains `FIRE`. Do not remove or remap the four model outputs `NOOP / FIRE / RIGHT / LEFT`. Environment-side FIRE assist is narrowly scoped to serve states; it is not permission to inject arbitrary FIRE actions during normal gameplay.

Whenever the environment overrides a requested policy action with the mandatory serve `FIRE`:

- preserve both the requested action and the actually executed action in diagnostics when practical;
- Replay Buffer transitions must store the **executed environment action**, not the policy's overridden request;
- action counts used as training provenance should make environment-side FIRE visible rather than pretending the requested action was executed;
- vectorized environments must track the serve/life-loss state independently for each sub-environment.

### Training, evaluation, vectorization, and gameplay must agree

Do not compare two trainers as a systems or algorithm A/B if they use different FIRE/reset/termination semantics.

For Day 16+ comparisons, all sides must use the same Contract v2 semantics, including the single-environment reference used against a vectorized candidate. The same rule applies to Day 17 smoke training, Day 18/20 model-family comparisons, Day 21 long training, gameplay recording, and final evaluation.

Day 15 Contract v1 artifacts remain valid **legacy evidence** for the original 100K checkpoint, but v1 and v2 scores must not be presented as if they were obtained under the same environment contract.

### Do not silently change the contract

If later evidence requires changing any task-defining field such as `fire_reset`, frame skip, sticky-action probability, life-loss termination, TimeLimit semantics, evaluation seeds, or raw-reward handling:

1. create a new explicit contract version;
2. document why the task definition changed;
3. update training and evaluation together;
4. rebuild any baseline needed for fair comparison;
5. do not overwrite historical v1/v2 artifacts.

For Browser/ALE-WASM work, if a Contract v2 field cannot be reproduced exactly, mark environment parity as partial and report the mismatch. Do not claim browser rollout scores are directly comparable to Python evaluation until the relevant environment semantics have been verified.
