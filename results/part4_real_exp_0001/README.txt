WHAT THIS IS
part4_real_exp_0001: failed test.
Source: data/part4_real/experiments/exp_0001 (copied: small result files only).
Model: whisper-tiny (a 39M test model, NOT the Hebrew model)
Date: 2026-10-06T23:22:47   Status: failed   Seed: 42
FAILED test of the search stage (2 trials of 4 steps): 'No trial completed'. Kept as a record of that failure; it is not a result.

KEY NUMBERS
Reason: stage search failed: SearchError: No trial completed (2 failed, 0 pruned; stopped because of n_trials). Reasons: trial 0: TrainerConfigError: Invalid training hyperparameters: warmup_steps must be between 0 and max_steps (4), got 10. Fix train.hyperparams (or the searcher's values).; trial 1: TrainerConfigError: Invalid training hyperparameters: warmup_steps must be between 0 and max_steps (4), got 10. Fix train.hyperparams (or the searcher's values).. A common cause is a learning rate range that is too high: lower the upper bound of search.parameters.learning_rate.

FILES: 17 files, 17 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: none
