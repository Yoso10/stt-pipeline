WHAT THIS IS
rehearsal_run_exp_0002: REAL tiny rehearsal run.
Source: data/rehearsal_run/experiments/exp_0002 (copied: small result files only).
Model: whisper-tiny (a 39M test model, NOT the Hebrew model)
Date: 2026-10-07T23:47:29   Status: completed   Seed: 42
The full chain (prepare, baseline, search 4 trials x 20 steps, 2 finalists, select, one Test look) on the small test model. It proves that the chain works; it is NOT a quality result.

KEY NUMBERS
Baseline on val: WER 1.1318, CER 0.6925, 101 clips, 4697 words, decoding 228 s.
Winner: trial 2 (params {'learning_rate': 2.3207944e-05}), Val WER 1.0279 (baseline above), Test status confirmed.
test_result_baseline.json: Test WER 1.1728 on 97 clips.
test_result_trial_2.json: Test WER 1.0599 on 97 clips.
Test was opened once: reports/test_once.json.

FILES: 52 files, 1978 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: reports/eval_result_baseline.json, reports/predictions_baseline.csv, reports/report.html, reports/review_baseline.txt, reports/test_result_baseline.json, reports/test_result_trial_2.json, reports/finalists/final_trial_001.val.json, reports/finalists/final_trial_002.val.json, reports/finalists/predictions_final_trial_001.csv, reports/finalists/predictions_final_trial_002.csv, reports/finalists/review_final_trial_001.txt, reports/finalists/review_final_trial_002.txt
