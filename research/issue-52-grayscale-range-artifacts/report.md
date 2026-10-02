# Issue 52 operational result

**Classification:** INCONCLUSIVE  
**Round status:** CONTINUE_RESEARCH  
**HVC:** NO  
**Diagnostic:** INCONCLUSIVE; R/N unavailable because N=0.

The single authorized command attempt stopped at the runner's clean-source guard before frozen input loading, environment creation, ALE reset, or action. The output directory had been pre-created in the worktree to capture command output, making the source tree dirty. The runner exited nonzero with `formal run requires clean committed source`.

No formal ALE reset or step occurred: 0 decisions, 0 native frames, and no observations. Raw stdout is preserved as an empty file. The full stderr traceback is preserved in `run_stderr.log`.

The exact command was:

```text
timeout --signal=INT --kill-after=5s 90s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue52_grayscale_range --output-dir research/issue-52-grayscale-range-artifacts
```

Capture shell wrapper: `TIMEFORMAT='outer_wall_seconds=%3R'; time timeout --signal=INT --kill-after=5s 90s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue52_grayscale_range --output-dir research/issue-52-grayscale-range-artifacts > research/issue-52-grayscale-range-artifacts/raw_run_output.json 2> research/issue-52-grayscale-range-artifacts/run_stderr.log`

Outer command wall time: **0.255 seconds**. Pre-run validation had 8 focused tests and a zero-frame preflight; recorded test-plus-preflight use was 0.530 seconds of the 20-second setup/test cap. Formal setup timing was unavailable because the clean-source guard stopped first. No retry or additional collection was performed.

Source commit: `03b40ad633beccfdf6d1d32d3d5eee289fa48ac8`. Source digest: `824353e64942fea53b9a6c6d9499b501bda5e1525ba319eab682c14173c95490`. Completion-source digest: `9444a7955e65f4fe9e1568f5b321d8c9ffc0a105ae0d8549c5c76bc99c08f926`. `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.
