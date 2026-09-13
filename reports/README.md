# Reports

`reports/` contains **derived engineering summaries** built from committed experiments, evaluations, benchmarks, or deployment evidence.

A report is not a new source of truth. It should make the underlying evidence easy to understand and should point back to the relevant config/run/evaluation whenever practical.

## Use this directory for

- comparison summaries;
- profiling conclusions;
- benchmark tables;
- deployment/parity summaries;
- final engineering reviews generated from existing evidence.

## Do not use this directory for

- executable analysis code — use `scripts/analysis/`;
- raw training runs — use local `runs/` or selected `experiments/`;
- fixed-protocol policy scores — use `evaluations/`;
- reader-facing 30-day articles — use `docs/`;
- charts/images directly embedded in articles — use `assets/dayXX/`.

When a report becomes obsolete, prefer keeping historical provenance clear rather than silently replacing old measurements with numbers from a different contract or code revision.
