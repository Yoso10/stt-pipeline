WHAT THIS IS
part11_real_exp_0001: failed test.
Source: data/part11_real/experiments/exp_0001 (copied: small result files only).
Model: whisper-tiny (a 39M test model, NOT the Hebrew model)
Date: 2026-10-07T23:25:50   Status: failed   Seed: 42
FAILED at prepare: SubsampleConfigError (a Val-slice setting did not fit). Kept as a record; not a result.

KEY NUMBERS
Reason: stage prepare failed: SubsampleConfigError: subsample.val_samples (a fixed Val slice) needs scope train_only: with all_splits every split is reduced by the same fraction.

FILES: 6 files, 8 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: none
