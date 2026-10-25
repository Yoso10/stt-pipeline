# Deck cards: material for a slide author

## How to run (use)

This file has no program to run. Hand it to the person who builds the slides. Each card is one slide or one box on a slide. Three groups: A, the 18 components in pipeline order; B, one run from start to end (10 cards); C, the user guide (9 cards). Every card has the same fields, in this order: Title, Job, Takes in, Gives out, Talks to, Keys, Status, Number. The detail is in `docs/COMPONENT_GUIDE.md`; the commands are in `docs/USER_GUIDE.md`.

How to read the numbers: "measured" = read from a stored report or log of the home laptop; "estimated" = computed from measured numbers; "never run" = never done for real. The home laptop: 8 cores, 15.77 GB RAM, CPU only. The rehearsal = whisper-tiny on one tenth of the Daf Yomi data. Turbo = `ivrit-ai/whisper-large-v3-turbo`.

## Group A: the 18 components, in pipeline order

**1. Config**
- Job: reads every setting, checks it, freezes it.
- Takes in: a YAML file, `--set` words, environment.
- Gives out: one read-only settings object.
- Talks to: every component gets its own section from it.
- Keys: `runtime.seed`, `runtime.online`, `extends` (in YAML).
- Status: tested (34 tests); ran for real in every command.
- Number: 21 sections, 184 keys.

**2. DataSource**
- Job: reads labeled audio into one uniform list and logs what it rejected.
- Takes in: parquet files or a folder with `manifest.csv` plus audio.
- Gives out: samples, `rejected.csv`, a load report.
- Talks to: hands the samples to the Splitter.
- Keys: `data.source.path`, `data.columns.group`, `data.max_rejected_ratio`.
- Status: tested (35); ran for real.
- Number: 8,723 clips loaded, 0 rejected, 710 lessons (measured).

**3. Splitter**
- Job: divides clips into Train, Val, Test; a lesson never sits in two parts.
- Takes in: the sample list.
- Gives out: three splits, a report, fingerprints.
- Talks to: feeds the TextNormalizer.
- Keys: `split.mode`, `split.ratios`, `split.seed`.
- Status: tested (41); ran for real (one mode).
- Number: Train 6,973 / Val 875 / Test 875 clips (measured).

**4. TextNormalizer**
- Job: makes a training text and a scoring text from each label.
- Takes in: raw labels with time markers.
- Gives out: `train_text`, `eval_text`, flags, a character inventory.
- Talks to: the SampleFilter reads its texts; the Evaluator reuses its scoring recipe.
- Keys: `text.eval_version`, `text.profiles.eval.steps`, `text.bracket.mode`.
- Status: tested (36); ran for real.
- Number: all 8,723 labels carry time markers (measured).

**5. SampleFilter**
- Job: decides which clips are good enough.
- Takes in: normalized clips of one split.
- Gives out: kept clips and a "would reject" list.
- Talks to: passes kept clips to the Subsampler.
- Keys: `filter.mode`, `filter.min_kept_ratio`.
- Status: tested (29); ran for real (report mode); apply mode never run.
- Number: 0 of 6,973 Train clips would be rejected (measured).

**6. Subsampler**
- Job: picks a smaller, repeatable part of the data.
- Takes in: the filtered splits and a size or preset.
- Gives out: smaller splits and a report.
- Talks to: later stages see only the smaller data.
- Keys: `subsample.enabled`, `subsample.preset`, `subsample.val_samples`.
- Status: tested (26); ran for real.
- Number: `work_like`: 702 / 101 / 97 clips (measured).

**7. ModelLoader**
- Job: loads a Whisper model and tokenizer from a local folder.
- Takes in: a model folder.
- Gives out: model, extractor, tokenizer, a model description.
- Talks to: the FeaturePipeline, TrainerCore and Evaluator use what it loads.
- Keys: `model.path`, `model.dtype`, `model.language`.
- Status: tested (28); ran for real; sharded weights never run.
- Number: turbo: 808,878,080 parameters, load 17 s (measured).

**8. FeaturePipeline**
- Job: turns clips into model input and label numbers, and makes batches.
- Takes in: kept clips, the model.
- Gives out: a lazy dataset and a batch maker.
- Talks to: the TrainerCore and Evaluator read the batches.
- Keys: `features.cache.enabled`, `features.max_dropped_ratio`, `features.shuffle`.
- Status: tested (32); ran for real.
- Number: 21 ms per clip (measured earlier); cache off.

**9. TrainerCore**
- Job: runs one training, returns the model folder.
- Takes in: model, Train and Val data, settings.
- Gives out: model folder, `metrics.csv`.
- Talks to: the Searcher and Orchestrator start it.
- Keys: `train.hyperparams.learning_rate`, `train.freeze_policy`, `train.save_optimizer_state`.
- Status: tested (51); ran for real (turbo: 3-step benchmarks only).
- Number: turbo, frozen encoder: 171.9M of 808.9M parameters trained; 106 s per step (measured).

**10. Evaluator**
- Job: transcribes clips, scores WER and CER, compares two results.
- Takes in: clips, references, a model.
- Gives out: WER, CER, per-clip scores, review list.
- Talks to: the Selector uses its comparisons.
- Keys: `eval.batch_size`, `eval.significance.confidence`, `eval.generation.num_beams`.
- Status: tested (23 + 7); ran for real.
- Number: turbo baseline WER 0.8335 on 101 Val clips; 16.6 s per clip (measured).

**11. HyperparamSearcher**
- Job: tries settings and ranks them by Val loss.
- Takes in: a search space, a trial runner.
- Gives out: ranked trials, the best few as finalists.
- Talks to: each trial calls the TrainerCore.
- Keys: `search.n_trials`, `search.parameters`, `search.finalists.k`.
- Status: tested (28); ran with tiny; never with turbo.
- Number: rehearsal: 4 trials, 195 s each, best Val loss 1.46 (measured).

**12. Selector**
- Job: picks the winner, or says the baseline stays.
- Takes in: Val results, a one-time Test callable.
- Gives out: a verdict, a comparison table, a plain reason.
- Talks to: the ModelRegistry stores its decision.
- Keys: `select.min_verdict`, `select.tie_epsilon`, `select.test_confirmation`.
- Status: tested (17); ran for real once (rehearsal).
- Number: Val WER 1.1318 to 1.0279; Test 1.1728 to 1.0599 (measured).

**13. ModelRegistry**
- Job: stores the chosen model as a numbered, never-edited entry.
- Takes in: the selection result and the model folder.
- Gives out: `registry\model_NNNN\`, model card, checksums, `index.csv`.
- Talks to: the ModelExporter adds export records.
- Keys: `registry.dir`, `registry.transfer`, `registry.cleanup`.
- Status: tested (24); ran for real once.
- Number: `model_0001`, 149 MB, Test set looked at 1 time (measured).

**14. ModelExporter**
- Job: converts a registered model to a target format and checks it.
- Takes in: a registry entry id.
- Gives out: `exports.json`, a verification result.
- Talks to: reads from the ModelRegistry; nothing follows.
- Keys: `export.formats`, `export.verify`, `export.max_wer_diff`.
- Status: tested (23); ran for real (one format); others never built.
- Number: export `ok`, 37,760,640 parameters matched (rehearsal, measured).

**15. RunManager**
- Job: creates numbered experiment folders and keeps their records.
- Takes in: the settings of a run.
- Gives out: `exp_NNNN\` with config, status, logs, lock.
- Talks to: every command writes its files through it.
- Keys: `run.experiments_dir`, `run.parent`, `run.lock_stale_minutes`.
- Status: tested (29); ran for real.
- Number: baseline experiment `exp_0002`: 17 report files, 0.56 MB (measured).

**16. Orchestrator and command line**
- Job: wires components, runs a stage or everything.
- Takes in: a command and `--config`.
- Gives out: an experiment, a JSON summary, an exit code.
- Talks to: calls all other components in order.
- Keys: `pipeline.stages`, `benchmark.steps`.
- Status: tested (175 tests); ran for real; learning-curve on real data never run.
- Number: rehearsal 38 min against 37 planned (measured).

**17. ReportBuilder**
- Job: builds one offline HTML page: verdict, charts, listening list.
- Takes in: the stored result files.
- Gives out: `reports\report.html`.
- Talks to: reads what the other stages stored; the `report` command writes it.
- Keys: `report.target_relative_wer_reduction`, `report.hints.enabled_rules`, `report.absolute_audio_paths`.
- Status: tested (55); ran for real on the rehearsal.
- Number: rehearsal report 116 KB, 12 sections (measured).

**18. Tools**
- Job: home-side scripts for models, data, packages.
- Takes in: a repo name; network allowed here only.
- Gives out: model and data folders, `wheels\`, `MANIFEST.sha256`.
- Talks to: nothing at run time.
- Keys: none; options `--repo`, `--out`.
- Status: tested; ran for real; closed-machine wheels never run.
- Number: turbo download 3.24 GB, 11 files (measured).

## Group B: one run from start to end (10 cards)

**B1. Get the ingredients (home)**
- Job: download model, data and packages while the network exists.
- Takes in: repo names.
- Gives out: `models\`, `data\daf_yomi_raw`, `wheels\`.
- Talks to: Tools only.
- Keys: none.
- Status: ran for real.
- Number: turbo 3.24 GB; command `python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit`.

**B2. Check and plan**
- Job: prove the machine is ready and know the cost.
- Takes in: the config.
- Gives out: `PROBLEM` lines; `benchmark.txt` with the planned time.
- Talks to: `check`, then `benchmark`.
- Keys: `benchmark.steps`, `train.cpu_threads`.
- Status: ran for real.
- Number: turbo plan 10 h 24 min to 14 h 49 min for 4 trials and 2 finalists (measured plan).

**B3. Prepare the data**
- Job: load, split, clean, filter, shrink.
- Takes in: raw clips and labels.
- Gives out: five reports in `reports\`.
- Talks to: DataSource, Splitter, TextNormalizer, SampleFilter, Subsampler.
- Keys: `subsample.preset`, `split.ratios`, `filter.mode`.
- Status: ran for real.
- Number: 8,723 clips become 702 / 101 / 97 in 73 s (measured, turbo run).

**B4. Score the untrained model**
- Job: set the reference point.
- Takes in: the Val clips and the untrained model.
- Gives out: `eval_result_baseline.json`, review list.
- Talks to: ModelLoader, Evaluator.
- Keys: `eval.batch_size`, `model.path`.
- Status: ran for real.
- Number: turbo WER 0.8335, 1,681 s (measured).

**B5. Search**
- Job: try settings on a small screening set.
- Takes in: a tiny Train set, a Val slice.
- Gives out: ranked trials, finalists.
- Talks to: Searcher calls the TrainerCore per trial.
- Keys: `search.n_trials`, `search.finalists.k`, `search.screening.preset`.
- Status: ran for real with whisper-tiny; never run with turbo.
- Number: tiny: 4 trials in 832.8 s (measured).

**B6. Retrain the finalists**
- Job: train the best settings on all Train data and score on Val.
- Takes in: finalists, full Train.
- Gives out: finalist models and Val WERs.
- Talks to: TrainerCore, Evaluator.
- Keys: `search.finalists.k`, `train.hyperparams.max_steps`.
- Status: ran for real with tiny; never run with turbo.
- Number: tiny: 2 finalists in 14 min 17 s (measured).

**B7. Choose the winner**
- Job: decide on Val alone.
- Takes in: baseline and finalist Val results.
- Gives out: a verdict and a table.
- Talks to: Selector, Comparator.
- Keys: `select.min_verdict`, `eval.significance.confidence`.
- Status: ran for real once.
- Number: improvement 0.1039 WER, lower bound 0.0759 (measured, rehearsal).

**B8. The one-time Test look**
- Job: confirm the winner on unseen clips, once.
- Takes in: the winner, the baseline, the Test clips.
- Gives out: `test_once.json`, Test status.
- Talks to: the Selector asks, the Orchestrator guards.
- Keys: `select.test_confirmation`, `pipeline.test_split`.
- Status: ran for real once; never with turbo.
- Number: Test WER 1.1728 to 1.0599, status confirmed (measured).

**B9. Register and export**
- Job: store the model with its facts and check it loads.
- Takes in: the winner folder.
- Gives out: `registry\model_NNNN\`, `exports.json`.
- Talks to: ModelRegistry, ModelExporter.
- Keys: `registry.dir`, `export.formats`.
- Status: ran for real once.
- Number: checksums cover 15 files (measured, rehearsal).

**B10. Read the report**
- Job: see the verdict and listen.
- Takes in: the stored files.
- Gives out: `report.html`.
- Talks to: ReportBuilder.
- Keys: `report.target_relative_wer_reduction`.
- Status: ran for real.
- Number: target 10% reduction; rehearsal reached 9.2% (measured).

## Group C: the user guide (9 cards)

**C1. Install**
- Job: set up the project once on a machine.
- Takes in: the project at a short path, `wheels\`.
- Gives out: a working `.venv`.
- Talks to: Python and pip.
- Keys: none.
- Status: ran at home; closed machine never run.
- Number: path 40 characters or fewer. Command: `python -m venv .venv`, then `pip install --no-index --find-links wheels -r requirements.lock`.

**C2. Check**
- Job: list everything wrong at once.
- Takes in: the config.
- Gives out: OK and `PROBLEM` lines.
- Talks to: reads folders and the machine.
- Keys: `--config`.
- Status: ran for real.
- Number: command `python -m stt check --config configs\work_like.yaml` (changes nothing).

**C3. Prepare**
- Job: read the data and inspect the reports, no training.
- Takes in: the data folder.
- Gives out: five reports.
- Talks to: the data components.
- Keys: `data.source.path`, `data.columns.group`.
- Status: ran for real.
- Number: command `python -m stt prepare --config configs\my_work.yaml`; 73 s for the Daf Yomi data (measured).

**C4. Benchmark**
- Job: measure speed, memory and disk.
- Takes in: the config and the model.
- Gives out: `benchmark.txt`.
- Talks to: TrainerCore and Evaluator.
- Keys: `benchmark.steps`, `train.hyperparams.batch_size`.
- Status: ran for real with turbo.
- Number: command `python -m stt benchmark --config configs\work_like.yaml` (run it twice, trust the larger plan); it keeps a 7.4 GiB folder.

**C5. Evaluate**
- Job: score the untrained model on Val.
- Takes in: the config, a model.
- Gives out: WER, review list.
- Talks to: Evaluator.
- Keys: `model.path`, `model.dtype`, `eval.batch_size`.
- Status: ran for real.
- Number: command `python -m stt evaluate --config configs\work_like.yaml --set model.path=models/whisper-large-v3-turbo-ivrit --set model.dtype=float32`; 1,681 s (measured).

**C6. Run**
- Job: the whole pipeline.
- Takes in: the config.
- Gives out: verdict, registry entry.
- Talks to: all stages.
- Keys: `pipeline.stages`, `search.n_trials`, `train.hyperparams.max_steps`.
- Status: ran for real with tiny; never with turbo.
- Number: command `python -m stt run --config configs\work_like.yaml`; 38 min with tiny. Ctrl+C once stops safely.

**C7. Report**
- Job: build the HTML page from stored files.
- Takes in: an experiment number.
- Gives out: `report.html`.
- Talks to: ReportBuilder.
- Keys: `report.target_relative_wer_reduction`.
- Status: ran for real.
- Number: command `python -m stt report --experiment exp_0002`; nothing is recomputed.

**C8. Export**
- Job: export a registered model and check it.
- Takes in: a registry entry.
- Gives out: `exports.json`.
- Talks to: ModelExporter.
- Keys: `export.formats`.
- Status: ran for real (one format).
- Number: command `python -m stt export model_0001 --config configs\work_like.yaml`; exit code 5 means the check failed.

**C9. Troubleshoot**
- Job: find the cause fast.
- Takes in: the exit code and the log.
- Gives out: the fix.
- Talks to: `logs\run.log`, `status.json`.
- Keys: `train.rename_retries`, `train.save_optimizer_state`.
- Status: tables exist; most fixes tested only in unit tests.
- Number: exit codes 0 ok, 2 settings, 3 data, 4 training or scoring, 5 export or checksums, 130 stopped. Command: `Get-Content <experiment>\logs\run.log -Tail 40`.
