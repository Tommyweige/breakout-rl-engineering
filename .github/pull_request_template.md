## What and why

<!-- One or two sentences: what changes, and the issue it closes (Closes #…). -->

## Checks

- [ ] Tests run and pasted below (`python -m scripts.smoke.run_smoke --units-only`, and `npm test` in `web/` when the web app changed)
- [ ] Training and evaluation still use the same contract (v2 or v3); no silent drift in frame skip, sticky actions, FIRE handling, seeds, or episode limits
- [ ] Any number quoted in README, `docs/final-results/`, or the case study matches `docs/final-results/summary.json`
- [ ] No generated runs, checkpoints, screenshots, or article drafts added to `main`

## Test output

```text
```
