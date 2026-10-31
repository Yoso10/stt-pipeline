WHAT THIS IS
part11_real_bench_c_full_exp_0001: failed test.
Source: data/part11_real/bench_c_full/experiments/exp_0001 (copied: small result files only).
Model: whisper-tiny (a 39M test model, NOT the Hebrew model)
Date: 2026-10-07T23:33:43   Status: failed   Seed: 42
FAILED at benchmark: TrainerConfigError (warmup_steps invalid for 100 steps). Kept as a record; not a result.

KEY NUMBERS
Reason: stage benchmark failed: TrainerConfigError: Invalid training hyperparameters: warmup_steps must be between 0 and max_steps (3), got 10. Fix train.hyperparams (or the searcher's values).

FILES: 11 files, 13 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: none
