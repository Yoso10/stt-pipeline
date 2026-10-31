HOW TO RUN / HOW TO READ
========================
What this is: the small result files of the finished run exp_0001 (home turbo run), copied
from data\turbo_run\experiments\exp_0001 and from the registry entry model_0001 (metadata only).
Nothing here is run; it is only read.

To read the result:  open  results\turbo_run_exp_0001\reports\report.html  in a browser.
Plain-language explanation (Hebrew):  docs\REPORT_EXPLAINED_exp_0001.md
Error analysis (Hebrew):              docs\ERROR_ANALYSIS_exp_0001.md

KEY NUMBERS
-----------
Command:        python -m stt run --config configs\home_turbo_run.yaml   (config used: config_full.yaml here)
Run time:       13 h 25 min, CPU only, seed 42
Winner:         search trial 1 (final_trial_001), learning_rate 1.58114e-05, 156 steps
Val WER:        0.8224 (baseline) -> 0.4157 (winner), 139 clips, verdict "improved" (90% one-sided)
Test WER:       0.8202 (baseline) -> 0.4375 (winner), 131 clips, CONFIRMED (Test was opened once: reports\test_once.json)
Split:          stage fingerprints in status.json (prepare 87b4aa499459ee89);
                Test fingerprint 213934a0839e21c19bbcce7453362fbe1c2d6be4d5bb449596083120cb59464c
Not reached:    the 0.10-0.30 absolute WER band (Test 0.4375 is 0.1375 above 0.30)

WHAT EACH FILE IS
-----------------
status.json, run_info.json, config_full.yaml, config_diff.yaml
                          run state per stage, library versions + seed, the full config used
logs\run.log.txt          the run log (the original is logs\run.log; renamed here because
                          .gitignore ignores *.log)
metrics\trial_000..002\metrics.csv          loss per step, the 3 search trials (61-clip screening slice)
metrics\final_trial_000..001\metrics.csv    loss per step, the 2 finalists (full Train subset)
search\study.json, *.params.json            the search: learning rates tried and their scores
reports\report.html                         the report (open this)
reports\selection.json, selection_table.csv, selection_reason.txt     why trial 1 won
reports\search_trials.json, search_report.txt, search_screening.txt, finalists_report.txt
reports\test_once.json                      marker: Test opened once (baseline + trial_1)
reports\evaluation_baseline.json            baseline summary on Val
reports\eval_result_baseline.json, finalists\final_trial_00N.val.json      per-clip Val results (reference + hypothesis text)
reports\test_result_baseline.json, test_result_trial_1.json                per-clip Test results (reference + hypothesis text)
reports\predictions_*.csv, review_*.txt, finalists\predictions_*.csv, finalists\review_*.txt   same text, flat / readable
reports\load_report.txt, filter_report.txt, split_report.txt, subsample_report.txt, text_report.txt, rejected.csv   data preparation reports
reports\stage_results\*.json                one summary per stage
registry_model_0001\                        registry entry metadata ONLY: index.csv, exports.json,
                          model_card.json, MODEL_CARD.md, SHA256SUMS (hashes of the weights, to verify a copy),
                          review_list.txt (text)

NOT HERE (on purpose)
---------------------
Model weights (registry\model_0001\model\, 3.2 GB), candidates\, checkpoints\, the encoder cache (3.18 GB),
audio, optimizer state. Back them up separately. The model folder is also left out
(its small tokenizer/config files are in the backup together with the weights).

PRIVACY
-------
Files with transcript text of the dataset (reference / hypothesis): report.html,
eval_result_baseline.json, finalists\final_trial_00N.val.json, test_result_baseline.json,
test_result_trial_1.json, all predictions_*.csv, all review_*.txt, registry_model_0001\review_list.txt.
