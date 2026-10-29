# Versioning and provenance

Reported numbers must be traceable to one immutable commit. Use **annotated
Git tags**, not a `VERSION` file.

## Recommendation

Tag the exact commit that produced a published result set:

```bash
git tag -a repro-2026-10-29 -m "Reproducible result set: Contract v2 closeout" <full-or-short-sha>
git push origin repro-2026-10-29
```

Why date-stamped `repro-*` tags:

- the repository already uses Day-based tags (`day4`, `day5`, `day6`, `day8`,
  `day9`, `publish/day-02`), so a new `v1`-style tag would collide with that
  history;
- a date identifies the evidence snapshot without implying a semantic API
  guarantee;
- `git describe --tags --always --dirty` then produces a human-readable
  identifier such as `repro-2026-10-29` or `repro-2026-10-29-3-gabc1234`.

If semantic versions are preferred later, tag the same commit twice
(`repro-2026-10-29` and `v1.0.0`); do not move an existing tag.

### Rules

1. Tag the commit, not a branch. Branches move; tags do not.
2. Use annotated (`-a`) tags so the tag records its tagger and message.
3. Never move or delete a published tag. Publish a new one for a new result.
4. Record both the tag and the full 40-character SHA in any report.
5. A dirty worktree is not a release point. Commit first, then tag.

## What the artifacts already record

| Artifact | Field | Notes |
| --- | --- | --- |
| Evaluation `results.json` | `source_provenance.source_commit`, `source_provenance.working_tree_dirty` | written by `breakout_rl/evaluation.py` |
| Evaluation `results.json` | `contract_provenance.contract_sha256` | binds the exact contract bytes |
| Training `summary.json` | `runtime.git_commit_sha`, `runtime.git_dirty` | available only when `--run-dir` is inside a Git worktree |
| Smoke `smoke-manifest.json` | `repository.commit`, `repository.describe`, `repository.dirty` | always written by `scripts/smoke/run_smoke.py` |
| `docs/final-results/summary.json` | `frozen_at` | the commit that added the frozen pack is `77a0c7b` |

The closeout pack currently maps to:

```text
77a0c7bab5dffcee8947313ff4240a756c4c8298  Close out Breakout results ...
```

Verify that the commit still contains the report:

```bash
git log -1 --format='%H %s' -- docs/final-results/summary.json
```

## Suggested release procedure

```bash
git switch main
git pull --ff-only
python -m scripts.smoke.run_smoke --output-dir outputs/smoke
git tag -a repro-2026-10-29 -m "Reproducible result set: <one-line description>"
git push origin repro-2026-10-29
```

After pushing, `git describe --tags --exact-match HEAD` must print the tag, and
`git rev-parse repro-2026-10-29^{commit}` must print the full SHA recorded in
the report.

## If a `VERSION` file is preferred

A root `VERSION` file is acceptable as a human-readable label, but it must not
be the source of truth:

- it can drift from the commit that produced the numbers;
- it does not survive partial checkouts or archive exports;
- it does not tell you whether the worktree was dirty.

If used, make CI read it and print it next to `git rev-parse HEAD`, and still
tag the commit. The smoke manifest already provides the tag and commit
information without an extra file.

## CI and tags

The workflow runs on every push, including tag pushes. A tag push therefore
runs the same unit tests, smoke, and web build as a branch push, which gives the
release point a CI record. `workflow_dispatch` can re-run it later without
changing the tag.
