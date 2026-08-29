# Evidence assets

`assets/` stores reproducible evidence used by the reader-facing Day articles. New work should use `assets/dayXX/` directories.

## Evidence types

- JSON / CSV / NPZ — machine-readable source data or metadata.
- PNG / SVG / GIF — rendered evidence or real recorded gameplay.
- `.mmd` — editable Mermaid source when a structural diagram is appropriate.
- local `README.md` — reproduction commands, provenance, and limits for that Day's evidence.

Figures must come from real code paths or real source artifacts. Do not fabricate terminal output, dashboards, metrics, or experiment results for presentation.

## Stable article links

Reader-facing images should use a stable commit-pinned GitHub blob URL. The embedded image URL uses `?raw=1`, and the image itself links back to the blob page. This keeps an article tied to the exact evidence snapshot it describes.

## Layout

Early project history contains a few flat assets such as `day02-ai-vs-human.gif` and `day04-*.png`. They remain in place in this cleanup to avoid unnecessary link/provenance churn. Day 05 onward already follows the preferred `assets/dayXX/` layout, and all new evidence should do the same.

## Related locations

- `../docs/` — reader-facing articles.
- `../scripts/` — reproduction, analysis, benchmark, and visualization commands.
- `../configs/` — task/training/experiment contracts.
- `../breakout_rl/` — reusable implementation.
