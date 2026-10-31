WHAT THIS IS
work_like_exp_0001: baseline.
Source: data/work_like/experiments/exp_0001 (copied: small result files only).
Model: whisper-large-v3-turbo-ivrit (the real model)
Date: 2026-10-24T23:16:48   Status: completed   Seed: 42
Baseline (untrained) score of the real model on a tiny Val of 8 clips; a first check.

KEY NUMBERS
Baseline on val: WER 0.7974, CER 0.3251, 8 clips, 385 words, decoding 155 s.

FILES: 17 files, 73 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: reports/eval_result_baseline.json, reports/predictions_baseline.csv, reports/review_baseline.txt
