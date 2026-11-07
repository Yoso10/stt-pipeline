# Recipes: concrete command sequences, a glossary, and the Test safety facts

## How to run

This page has no script. Each recipe is a list of commands to type in PowerShell, in the project folder, after:

```
$env:PYTHONPATH = "src"
```

The shape of every command (`--config`, `--set`, `--parent`, `--experiment`, `--split`, `--params-from`) is copied from `python -m stt <command> --help`. The keys come from `docs/CONFIG_REFERENCE.md`. How to write the YAML: `docs/YAML_GUIDE.md`.
Checked on 2026-11-08. "(run)" = I ran it on the tiny synthetic data, in a scratch folder outside the repo (it proves the mechanics, not any quality). "(from code reading, not run)" = read in the code only.
Three rules for every recipe: (1) `--experiment exp_N` needs EXACTLY the same config as the first call, or it stops with exit 2 (run); (2) every command prints one JSON at the end and writes into `experiments\exp_NNNN\`; (3) read `docs/YAML_GUIDE.md` section 2 for the PowerShell 5.1 quoting trap before you put a list in `--set`.

## Glossary

- **Experiment (exp_NNNN):** one run folder. Everything of that run is inside it.
- **Trial:** one attempt inside the search: one set of hyperparameters, trained briefly on a small slice, scored by Val loss. Many per experiment.
- **Finalist:** one of the best `search.finalists.k` trials, retrained on the full Train and scored by WER on Val.
- **Val:** data used to DECIDE (rank trials and finalists, compare with the baseline).
- **Test:** data kept sealed for one final confirmation. Today the program enforces "once" only partly: see "Test safety (current behaviour)" below.
- **Baseline:** the original model, before fine-tuning.

## Commands (copied from --help)

| Command | What it does |
|---|---|
| `python -m stt prepare --config my.yaml` | load, split, normalize, filter; write the reports |
| `python -m stt train --config my.yaml --set train.hyperparams.max_steps=50` | prepare, then ONE training run (Val loss only, no WER) |
| `python -m stt evaluate --config my.yaml --experiment exp_0003` | score the baseline and the trained model of that experiment on Val |
| `python -m stt search --config my.yaml --set search.n_trials=4` | prepare, then the search (Val loss on a small slice) |
| `python -m stt run --config my.yaml --set subsample.val_samples=100` | prepare, baseline on Val, search, finalists, selection (Test), registry |
| `python -m stt learning-curve --config my.yaml --params-from exp_0003` | the same settings on growing Train sizes |
| `python -m stt check --config configs/smoke.yaml` | validate a config and the machine; changes nothing |
| `python -m stt final-test --config my.yaml --experiment exp_0003` | score the Test once (refused if any earlier Test look exists) |
| `python -m stt report --experiment exp_0004` | rebuild `reports\report.html` from stored files |

## Recipe 1: use only a specific amount of data (no code change)

Two ways.

**A. By size, with the Subsampler** (keys `subsample.*`). The choice is deterministic (same seed, same data, same subset) and nested (a smaller subset sits inside a bigger one when strategy, seed and `repeat_index` are unchanged). It comes AFTER the split and the filter.

| Key | Default | Meaning |
|---|---|---|
| `subsample.enabled` | false | off = all the data |
| `subsample.preset` | none | `smoke` = 20 Train clips; `tiny` = 30 minutes of Train; `work_like` = 10% of the audio, whole lessons, in Train AND Val AND Test; `half` = 50% of Train; `full` = all. Your own: `subsample.presets` |
| `subsample.size` | `all` | instead of a preset: exactly one of `minutes`, `samples`, `fraction` (for example `--set 'subsample.size={\"minutes\": 30}'` in PowerShell 5.1) |
| `subsample.strategy` | `by_sample` | `by_sample` or `by_group` (whole lessons) |
| `subsample.scope` | `train_only` | `all_splits` reduces Train, Val and Test by the same fraction (rehearsal only) |
| `subsample.val_samples` | none | a FIXED number of Val clips (a Val slice), whatever the Train size; scope `train_only`; not with `subsample.val: follow` |
| `subsample.val` | `fixed` | `follow` reduces Val by the same share as Train |
| `subsample.repeats`, `subsample.repeat_index` | 1, 0 | another draw of the same size |
| `subsample.min_train_samples` | 10 | fewer Train clips FAIL the run (exit 3) |

Example: `python -m stt run --config my.yaml --set subsample.enabled=true --set subsample.preset=tiny --set subsample.val_samples=100`.
Errors: a size larger than the data (exit 3, both numbers are named); `minutes` or `samples` with `all_splits` (exit 2); a split that would be empty (exit 3).

**The gate caveat.** The split gate (`split.min_samples_per_split` 30, `split.min_groups_per_split` 2) runs BEFORE the Subsampler (`wiring.py:130-143`). After subsampling the only size checks are Train >= `subsample.min_train_samples` and "no empty split". So `all_splits` and `work_like` can leave Val or Test below the gate without any error. Then the Comparator may answer `insufficient_groups` (fewer than `eval.significance.min_groups` = 5 lessons), and `select.on_insufficient_groups` decides (`plain_difference` = judge by the plain difference, flagged; `reject`). Check `reports\subsample_report.txt` after `prepare`.

**B. By files.** Make a small folder with a small `manifest.csv` that holds exactly the rows you want, and point `data.source.path` at it. The gate still applies, so a handful of files will not pass. To watch only the mechanics, `python -m stt smoke` uses synthetic data and the tiny test model; `configs\smoke.yaml` lowers the gate on purpose. Lowering the gate is for mechanics tests, never for a result you report.

## Recipe 2: one training run (no search), fixed hyperparameters

```
python -m stt train --config my.yaml --set train.hyperparams.learning_rate=3e-5 --set train.hyperparams.grad_accum_steps=4
python -m stt evaluate --config my.yaml --set train.hyperparams.learning_rate=3e-5 --set train.hyperparams.grad_accum_steps=4 --experiment exp_0005
```

(`exp_0005` = the number that `train` printed.) Notes:

- `train` computes the Val LOSS only, never WER (run). It prints `stages.train.summary` with `trial`, `stopped_reason`, `best_step`, `steps`, `mean_seconds_per_step`, `model_dir`, `best_val_loss`.
- Files of `train`: the model folder `candidates\exp_0005_trial_001\exp_0005_trial_001_step_NNNNN\`, `metrics\trial_001\metrics.csv`, the prepare reports (`load_report.txt`, `split_report.txt`, `text_report.txt`, `filter_report.txt`, `rejected.csv`, plus `subsample_report.txt` when on), `stage_results\prepare.json`, `train.json`.
- `evaluate --experiment exp_0005` needs the SAME `--config` and `--set` values (a changed key is refused, exit 2). To avoid retyping, use the experiment's own file: `--config experiments\exp_0005\config_full.yaml` (run).
- Files of `evaluate` (run): `evaluation_baseline.json`, `evaluation_trained.json`, `comparison.json`, `eval_result_baseline.json`, `predictions_baseline.csv`, `predictions_trained.csv`, `review_baseline.txt`, `review_trained.txt`. The printed `key_numbers` hold `baseline_wer`, `trained_wer`, `verdict`.
- `evaluate` scores the Val split only. It decodes the baseline too, so it takes about twice the time of one scoring.
- `--parent exp_0003` only sets the BASE of `config_diff.yaml` (the header says `# difference from exp_0003`) and writes `parent_id` into `run_info.json`. It copies NO values and NO results (`store.py:119-130`, run). To reuse a past run's values, use its `config_full.yaml` as `--config` (run).

## Recipe 3: take a reference run, change ONE thing

1. Reference: Recipe 2 with the values you want as the reference, for example experiment `exp_0005` (after `train` and `evaluate`). To take a past run's whole setup: `--config experiments\exp_0003\config_full.yaml`.
2. Variant: the same command with ONE key changed, linked with `--parent`:
   `python -m stt train --config experiments\exp_0005\config_full.yaml --parent exp_0005 --set train.hyperparams.max_steps=300`, then `python -m stt evaluate --config experiments\exp_0005\config_full.yaml --set train.hyperparams.max_steps=300 --experiment exp_0006`.
   One key at a time, for example `train.hyperparams.max_steps`, `train.hyperparams.warmup_steps`, `train.hyperparams.weight_decay`, `train.freeze_policy`.
3. Compare both on the SAME Val set (same `text.eval_version`, same split and subsample). Each experiment compares its model with the BASELINE only (`comparison.json`); there is no command that compares two trained models. What exists is `tools\compare_evals.py`, used like this:
   - score the REFERENCE model as if it were the baseline, in its own experiment: `python -m stt evaluate --config experiments\exp_0005\config_full.yaml --set model.path=<the model folder of exp_0005>` (this makes `exp_0007`; Val only);
   - then: `.\.venv\Scripts\python.exe tools\compare_evals.py --dir experiments --baseline-from exp_0007 --experiment exp_0006`.
   It prints both WERs, the relative reduction and the paired-bootstrap verdict. **Tested mechanically on tiny data only; the reference is scored as baseline and still labeled "Baseline" in the output.** On real data it is UNTESTED. It costs one more full Val scoring.
4. A difference without a confidence statement is not evidence: one seed, a small Val. Say so when you report it.

A decoding-only change (`eval.generation.num_beams`, `eval.generation.max_new_tokens`, `eval.batch_size`) needs NO training: run `evaluate` only, on the same trained model, with the new value, in a new experiment: `python -m stt evaluate --config my.yaml --set eval.generation.num_beams=5 --set model.path=<the model folder>` (the model path form: run; the `num_beams` value itself: from code reading, not run).

## Recipe 4: one run on a finalist

**A standalone `finalists` command: NOT SUPPORTED.** `finalists` is a stage inside `run`. Asking for it alone with `--experiment` and `--set pipeline.stages=...` is refused (exit 2, config differs; run). In a NEW experiment it would stop with exit 2 because there is no `search_trials.json` yet (from code reading, not run).

Two ways that do work.

**(i) Manual single training with the finalist's values**

1. Read the values. Searched values: `experiments\exp_N\reports\finalists\final_trial_NNN.json`, key `params` (full precision, for example `1.5811388e-05`). Also in `reports\search_trials.json` (`trials[].params`), `reports\selection.json` (`table[].params`) and the registry `model_card.json` (`searched_params`, `finalists`). The HTML report shows them ROUNDED (`1.58114e-05`); copy from the JSON.
2. The other hyperparameters are NOT in those files. They are in `experiments\exp_N\config_full.yaml`, section `train.hyperparams`.
3. Run Recipe 2 with `--config experiments\exp_N\config_full.yaml --set train.hyperparams.learning_rate=1.5811388e-05`. Add `--set select.test_confirmation=false` only if the config is used for `run` (it is harmless otherwise).
4. Only `learning-curve` can take values by experiment: `--params-from exp_N` (the winner's searched values, or the best finalist's when the baseline was kept; `curve.py:56-80`).

**(ii) Look at the search first, then continue with `run`**

```
python -m stt search --config my.yaml --set select.test_confirmation=false
python -m stt run --config my.yaml --set select.test_confirmation=false --experiment exp_0008
```

(`exp_0008` = the number `search` printed. Both calls need the SAME config, so put every `--set` in both, or better in the YAML.)
What happens (run): the second call skips `prepare`, runs `evaluate` (the baseline on Val), runs `search` again (it only reads the finished study and trains nothing new, because the later stage invalidates it), then runs ALL finalists, the selection and the registration. **It always runs the finalists and the selection, and it opens the Test unless `select.test_confirmation=false`.** You cannot stop after one finalist. To stop in the middle, press Ctrl+C once (a checkpoint is saved) and continue with the same command.

## Recipe 5: full training of every config

Allowed (nothing in the code refuses it), but it means double training in `run`:

- `search.screening.preset: full` makes every search trial train on the whole Train (run: `search --set search.screening.preset=full` used 48 of 48 Train clips).
- With `search.finalists.k` >= `search.n_trials` every trial becomes a finalist (run: 4 trials and `k=5` gave finalists 3, 2, 1, 0).
- The finalists stage retrains each finalist from scratch on the same full Train (`finalists.py:139-159`, from code reading, not run). So each trial is trained twice, with the same data and values.

Cost, from real runs of this project: the tiny rehearsal searched 4 trials in 833 s (about 3.5 minutes each; a finalist about 7 minutes with its Val scoring, a benchmark estimate). The real turbo run (exp_0001, 156 steps) searched 3 trials in 18,368 s (about 5.1 h) and retrained and scored 2 finalists in 6 h 37 min.

**Recommended:** one `train` and one `evaluate` per config (Recipe 2). It trains each config once and gives a WER for each. Compare with Recipe 3.

Stage names. Default `pipeline.stages` (the stages of `run`): `prepare, evaluate, search, finalists, select_and_register`. Registered stages (run): `benchmark, evaluate, final_test, finalists, learning_curve, prepare, search, select_and_register, train`. The other commands run their own fixed stages: `train` = prepare + train; `evaluate` = prepare + evaluate; `search` = prepare + search; `final-test` = prepare + final_test (`cli.py:75`).

## Test safety (current behaviour)

Facts only. Documented here, NOT changed. Sources: `stagehelpers.py:67-78`, `stages.py:93-100,154-162`, `testlook.py`, `finaltest.py:42-71`, `selection.py:103`.

1. **Which commands guard the Test.** `evaluate` refuses the split named in `pipeline.test_split` (default `test`), also through `--split` (exit 2). The helper that scores a model refuses it unless the selection stage calls it. The guard compares NAMES only.
2. **Which commands do not guard it.** `train` and `search` use the split named in `pipeline.eval_split` (default `val`) for their Val loss without comparing it with `pipeline.test_split`. `--set pipeline.eval_split=test` on them would use the Test split (from code reading, not run). Changing `pipeline.test_split` to another name would also lift the `evaluate` guard (from code reading, not run).
3. **Where the look is recorded.** Per EXPERIMENT: `reports\test_once.json` lists the models whose Test scoring started (`baseline`, `trial_N`; written BEFORE the scoring), and `reports\test_result_<model>.json` holds the result. Within one experiment a model is scored once; a stored result is returned instead of a new look.
4. **A new experiment scores the Test again.** `run` checks only its own experiment. A second `run` in a new experiment of the same data scored the same Test split again (run: exp_0005 after exp_0001).
5. **Only `final-test` looks at all experiments.** It refuses (exit 3) when any experiment of the same `run.experiments_dir` has `test_once.json` or `test_result_*.json`, or when the marker `test_split_scored.json` exists. After a `run` it refused (run, exit 3). So after a `run`, do not use `final-test`.
6. **`select.test_confirmation: false`** means the Test is never opened by `run` (`test_status` `not_evaluated`).

**Rule for exploration runs: always add `--set select.test_confirmation=false`** (or put it in the YAML), so that only the one run you decide to report opens the Test.

## Open questions for the owner

- The Test is protected per experiment in `run`, and `train`/`search` do not guard `pipeline.eval_split`. Do you want a change (a cross-experiment check in `run`, a guard in `train`/`search`)? Not done; the brief forbids code changes.
- Should `docs/CONFIG_REFERENCE.md` stop showing `text.profiles.*.steps` as "(required)"? It is generated from the code, so this is a code or generator change.
