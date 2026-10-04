# Troubleshooting

## How to use this file (for the owner and for any agent, including the weaker internal one)
1. Read the error message first: it names the key, the path or the stage, and the fix.
2. Find the component in the tables below and use the "Minimal fix" column. Prefer a Config change over a code change, and a change inside one adapter over a change in shared logic.
3. Never change a LOCKED interface, config key, folder structure or dependency to fix a problem. State what the fix could affect and apply it only with the owner's approval.
4. Keep the system portable: no hard-coded paths, no network, no machine-specific or dataset-specific assumptions.
5. Details are always in `experiments/exp_NNNN/logs/run.log` and `status.json`.

## Exit codes
| Code | Meaning |
|---|---|
| 0 | success (including "the baseline stays") |
| 2 | config error |
| 3 | data error |
| 4 | failure in a training or evaluation stage |
| 5 | export or verification failure |
| 130 | interrupted by the user (resume is possible) |

## Reproducibility expectation
The same seed gives the same result on the same machine. Results are not guaranteed to be bit-identical across machines, CPUs or core counts, so exact numbers are not compared between home and work.

## Config (C0)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL unknown key | typo in YAML or CLI `--set` | the message lists the valid keys; fix the name |
| FAIL wrong type or range | value of the wrong type | fix the value in YAML |
| YAML parse error | indentation or a tab character | the message gives the line; fix it |
| an environment value is ignored | wrong prefix | environment keys use the `STT__SECTION__KEY` form (to be confirmed in implementation) |
| the program tries to use the network | `runtime.online` is true | set it to false; online is for the home machine only |

## DataSource (C1)
| Symptom | Likely cause | Minimal fix (does not change code structure) |
|---|---|---|
| FAIL column not found | work data uses other column names | change `data.columns.*` in YAML |
| FAIL rejected ratio too high | wrong audio path or mapping | check `rejected.csv` reasons; fix `data.source.path` or columns; do NOT raise the ratio just to pass |
| many `unreadable_audio` for mp3 on Windows | audio library cannot decode mp3 | run the home install test again; convert audio to wav before ingestion; do not rewrite the loader |
| FAIL no group column | work data has no speaker/lesson id | set `data.columns.group`, or consciously set `group_fallback` to `parent_dir` / `file_stem` |
| extraction interrupted | crash during extract | delete the extract folder (no manifest means incomplete) and rerun |

## Splitter (C3)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL too few groups | group key too coarse for the data size | use a finer key (e.g. one lesson/recording) or `time_block`; do not just lower `min_groups_per_split` |
| FAIL oversized group | one recording dominates the data | choose another key or `time_block`; raise the tolerance only consciously |
| FAIL fingerprint mismatch | the data changed since the split was saved | create a new `split.version`; keep the old one so past experiments stay interpretable |
| test WER suspiciously good | group key too fine (neighbouring content in train) or text duplicated across splits | try a coarser key; check text overlap (O19) |
| test WER jumps between experiments | test has too few groups/samples | enlarge the test share or the number of groups; keep the test fixed |
| FAIL regex did not match | work data has other file/column naming | change `split.group_key.regex` in YAML only |

## TextNormalizer (C2)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| WER high for BOTH baseline and fine-tuned | reference text contains symbols that were not spoken | read the character inventory; add them to `text.punctuation.chars` or `text.bracket.pairs` in YAML (new `eval_version`); do not edit code |
| WER looks good but outputs look wrong | the eval profile is too aggressive | compare with a stricter profile as a new `eval_version`; remember that versions are not comparable |
| the model outputs timestamp tokens | training labels still contain them | check `text.train.timestamps` and the train profile steps |
| the new dataset has markers like `[noise]` | work data uses its own notation | set `text.bracket.mode: with_content` for those pairs, or add the pair |
| the report flags `had_digits` | digits in the work data | ask the owner; a number-normalization step would be a NEW step class, not an edit |
| many `empty_after_normalization` | transcripts hold only timestamp tokens or symbols | check the inventory; let SampleFilter remove them via `min_text_chars` |

## SampleFilter (C1b)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL required field missing | work data has no `quality_score` etc. | set that rule `enabled: false` in YAML, or map the column in DataSource config; do not edit the rule |
| FAIL kept ratio too low | thresholds too strict for this dataset | run in `report` mode, read the per-rule counts, relax the dominant rule; do not just lower `min_kept_ratio` |
| model weak on noisy audio after training | quality filter removed the noisy kind of audio | compare an experiment with the filter off (config diff only); see OPEN issue on filter thresholds |
| a split becomes empty | filter applied to a small split | restrict the rule's `applies_to` to `train` |

## Subsampler (C3b)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL size larger than available | requested minutes exceed the filtered train set | choose a smaller size, or `all` |
| FAIL fewer than minimum samples | size too small for a meaningful run | raise the size or lower `subsample.min_train_samples` knowingly |
| WER very different between repeats | small data, high variance | use `repeats` and report mean and spread; do not trust one draw |
| overshoot is large | `by_group` with long lessons | use `by_sample` or accept it as reported |
| learning curve crosses | different Test sets used | use `scope: train_only` for curves |
| Val too small to stop early reliably | `val: follow` on small data | use `val: fixed` or accept noisy early stopping |

## ModelLoader (C5)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL folder or files missing | model not copied to the machine | copy the full folder from home, or run `fetch_model` at home; do not enable online mode as a fix |
| FAIL language not supported | English-only model chosen | change `model.path` to a multilingual model |
| FAIL old weights format | community model published as `.bin` only | convert once at home to safetensors, then copy |
| garbage output after loading a fine-tuned folder | wrong or incomplete folder | check `pipeline_model_info.json`, compare with the experiment config |
| out of memory on load | model too large for the machine | use a smaller model; do not change loader code |
| output language is wrong (not Hebrew) | language not set | check `model.language`; the ivrit-ai cards require it to be set explicitly |

## FeaturePipeline (C4)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL audio longer than the window | SampleFilter is in `report` mode or `max_duration` disabled | set `filter.mode: apply` and enable `max_duration` in YAML; do not edit this component |
| FAIL dropped ratio too high | many long transcripts | read the label-length distribution; check that timestamps are stripped; raise nothing without a decision |
| model learns to output timestamps by accident | keep mode on | check `text.train.timestamps` |
| out of memory while building | cache on or too many workers | set `features.cache.enabled: false` and `features.num_workers: 0` |
| very slow training | short clips cost the same as long ones | see O33; reduce the subsample or use a smaller model; no code change |
| FAIL missing audio library | home install differs from work machine | repeat the home install test (O9) |

## TrainerCore (C6)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL NaN loss | learning rate too high, bad batch | lower `learning_rate`; check the first live metrics lines |
| training very slow | too many or too few threads, short clips cost like long ones (O33) | adjust `train.cpu_threads`; use a smaller model or fewer steps |
| out of memory | batch too large for the machine | lower `batch_size`, raise `grad_accum_steps`, or use `freeze_encoder` |
| Val loss does not move | learning rate too low, too few steps | read the live trend; increase `max_steps` or the rate |
| Val loss goes up while train loss goes down | overfitting on small data | rely on early stopping and the best checkpoint; use more data or fewer steps |
| FAIL output folder not empty | rerun without resume | choose a new experiment folder (RunManager does this) or set `resume: true` |
| machine unusable during the run | too many threads | lower `train.cpu_threads` |

## Evaluator (C7)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| WER close to 1.0 for baseline and trained | wrong language setting, or references contain unspoken symbols | check `model.language`; read the character inventory and the review list |
| WER improves but the review list looks wrong | profile too aggressive or too gentle | compare under a new `eval_version`; versions are not comparable |
| comparison says `not_comparable` | different `eval_version` | re-evaluate the baseline with the current profile |
| comparison says `insufficient_groups` | few groups in the evaluated set | evaluate on a larger set or choose a smaller group key; read the plain difference only |
| many `hit_length_cap` outputs | model loops | lower `max_new_tokens`, check the training labels; do not hide the flag |
| decoding too slow | beams or batch size | `num_beams: 1`, lower batch size, evaluate a subset in smoke runs |
| FAIL all outputs empty | features or language wrong | check the FeaturePipeline report and `model.language` |

## HyperparamSearcher (C8)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL too many failures | learning rate range too high | lower the upper bound in `search.parameters.learning_rate` |
| FAIL all trials pruned | pruning too aggressive early | raise `search.pruning.min_evals` or disable pruning |
| FAIL study space differs | config changed after the first run | use a new experiment folder, or restore the old space |
| search too slow | trials too long | use a smaller screening preset, fewer trials, or a time budget |
| machine sluggish in parallel mode | too many workers | lower `search.parallel_trials` |
| FAIL Optuna missing | package not installed on this machine | install it in the virtual environment, or set `search.strategy: grid` |
| best trial at the edge of the range | range too narrow | widen that range in YAML for the next experiment |

## Selector (C9)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL no candidates | search produced no completed trial | read the search report and its failure reasons |
| FAIL different eval_version | baseline evaluated with an old profile | re-evaluate the baseline with the current profile |
| always `baseline_retained` | thresholds too strict, or training does not help | read the comparison table; check `min_verdict` and the live training curves |
| `not_confirmed` on Test | Val is small or the candidate overfit Val | read the Test numbers beside it, listen to the review list |
| FAIL test called twice | orchestration bug | fix the caller; never bypass the guard |

## ModelRegistry (C10)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL entry exists | two runs raced, or a leftover temp folder | remove the leftover temp folder named in the message; never delete a finished entry |
| verify says files changed | corrupted copy between machines | copy the entry again from the source |
| index does not match entries | index edited by hand or a crash | run `rebuild_index()` |
| disk full on registration | large model, move across drives | free space, or set `registry.transfer: copy` knowingly |
| `best` returns nothing | different `eval_version` or test set | compare only like with like; re-evaluate under the current profile |

## ReportBuilder (C14)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| a chart is empty or missing | input not available in this run | read the "not available" note; check that the search or the learning curve was run |
| Hebrew text appears reversed or garbled | page opened without UTF-8 | the page declares UTF-8; open it in a normal browser; do not edit the file by hand |
| correlation rows say "too few points" | small evaluation set | evaluate a larger set, or lower `report.min_points_for_correlation` knowingly |
| the live page does not refresh | the file is opened through a viewer that does not reload | open it in a browser; the refresh is built into the page |
| a hint looks wrong | threshold not calibrated | change the threshold in YAML; hints are advisory |
| the file is large | many points or long lists | lower `report.max_points_per_chart` |

## ModelExporter (C11)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL missing package | converter not installed on the closed machine | bring the package in advance with the other wheels; do not enable online mode |
| verification differs too much | conversion changed the outputs | inspect the difference first; raise `max_wer_diff` only knowingly |
| FAIL target exists | repeated export | export to a new target folder or remove the failed one |
| FAIL nothing to export | the winner was "baseline retained" | no export is needed |

## RunManager (C12)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL experiment is locked | another run is active | wait, or stop that run |
| FAIL stale lock | a run died without releasing | check that no run is active, then delete the `run.lock` file named in the message |
| FAIL config differs on resume | config edited after the run started | restore the stored config, or start a new experiment with `extends` |
| warning path too long | experiments folder deep in the tree | move `run.experiments_dir` closer to the drive root |
| no console output from components | console level too high | lower `run.log.console_level`; details are always in `logs/run.log` |

## Orchestrator and CLI (C13)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| exit code 2 | invalid or missing config key | read the message (it names the key); fix the YAML |
| exit code 3 | data path or column mapping wrong | read `rejected.csv` and the `prepare` report; fix `data.*` in YAML |
| exit code 4 | training or evaluation failed | read `logs/run.log`, `status.json` and the failing stage; consult the component tables in TROUBLESHOOTING |
| exit code 5 | export dependency missing or verification failed | see the ModelExporter table |
| rerun does nothing | all stages completed | use a new experiment (`--parent`), or change the config |
| rerun repeats a stage | its config slice changed | intended; check the diff in `config_diff.yaml` |
| the machine went to sleep | sleep prevention off or blocked by policy | check `run.prevent_sleep`; ask the machine owner about power policy |

## Standalone tools (T)
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| fetch FAIL not found | wrong repository name | check the name on the Hub; do not guess a similar one |
| fetch FAIL login required | repository is gated | obtain the files another way; the pipeline expects only local folders |
| wheel missing for Windows | package has no build for this Python | try another Python version (O1), or ask before replacing the package |
| offline install test fails | a dependency was not downloaded | rerun `fetch_wheels` (it includes dependencies); read the named package |
| `check` reports a manifest mismatch | a file was changed or the copy is damaged | copy the release again from home |

