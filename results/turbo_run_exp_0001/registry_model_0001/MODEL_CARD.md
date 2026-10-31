# Model card: model_0001 (experiment exp_0001)

**Result: WINNER; confirmed by the Test.**

## In plain words

Trial 1 (learning_rate 1.58114e-05) is the winner: its Val WER is 0.4157 against 0.8224 for the baseline (improvement +0.4068; The candidate is better than the baseline at the 90% one-sided level (WER difference +0.4068, lower bound +0.3777, 11 groups)). The Test confirms it: Test WER 0.4375 for the winner against 0.8202 for the baseline (improvement +0.3827; The candidate is better than the baseline at the 90% one-sided level (WER difference +0.3827, lower bound +0.3602, 10 groups)).

## Numbers

### Val (the choice was made on this set)

| item | value |
|---|---|
| split | val |
| files / words | 139 / 6397 |
| WER (all words together) | 0.4157 |
| CER | 0.1844 |
| mean of the per-file WER | 0.4666 |
| files with a WER ... | under 10%: 1.4%, under 20%: 2.9% |
| baseline WER | 0.8224 |
| difference from the baseline (positive = better) | 0.4068 |
| interval of the difference | 0.3777 to 0.4337 (90% level) |
| verdict against the baseline | improved (11 groups) |

### Test (looked at once, as the final confirmation)

| item | value |
|---|---|
| split | test |
| files / words | 131 / 6258 |
| WER (all words together) | 0.4375 |
| CER | 0.2041 |
| mean of the per-file WER | 0.5084 |
| files with a WER ... | under 10%: 0.8%, under 20%: 5.3% |
| baseline WER | 0.8202 |
| difference from the baseline (positive = better) | 0.3827 |
| interval of the difference | 0.3602 to 0.4061 (90% level) |
| verdict against the baseline | improved (10 groups) |

## All finalists on Val

| rank | trial | parameters | Val WER | difference | verdict | Val loss | steps | note |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | learning_rate = 1.58114e-05 | 0.4157 | 0.4068 | improved | 0.4550 | 156 | WINNER |
| 2 | 0 | learning_rate = 5e-06 | 0.5166 | 0.3058 | improved | 0.5290 | 156 | eligible |

## How it was made

- base model: whisper-large-v3-turbo-ivrit (`models\whisper-large-v3-turbo-ivrit`)
- trial: 1, searched parameters: learning_rate = 1.58114e-05, steps: 156, Val loss: 0.4550
- seed: 42; evaluation profile (eval_version): 1
- decoding: num_beams 1, max_new_tokens 448, batch_size 2, max_tokens_per_audio_sec 8.0
- data / split / subsample fingerprints: 349a08f9a17b140052fd0be1c23d6be61b5ce78e1a92894551f0f4325ae57b42 / e5889fa9e5e85a00679f975f55df63e34d488add8af7c376c23e13dde2489de7 / 7fc4e3b230f6e17f84efa5f5f0b22d9c6ba2cfb263d4533298cc4602d19c779b
- configuration of the experiment: `exp_0001/config_full.yaml` (relative to the experiments folder)
- code version: 0.0.0; written at: 2026-10-31T14:56:56
- libraries: torch 2.14.1, transformers 5.18.0, numpy 2.5.3, pydantic 2.13.5, PyYAML 6.0.3, soundfile 0.14.0, scipy 1.18.1, pyarrow 25.0.1, accelerate 1.15.0, safetensors 0.8.0

## How often the Test set was looked at

This Test set (fingerprint `213934a0839e21c19bbcce7453362fbe1c2d6be4d5bb449596083120cb59464c`) has been looked at 1 time(s) by registered entries, this one included. The count covers registered entries only, so it is a lower bound. Comparisons between experiments are made on Val; the Test numbers are reported but not used to pick among experiments.

## Files of this entry

- `model/`: the model folder. To use it, set `model.path` to this folder.
- `model_card.json`: the same facts as data (read by the report builder).
- `MODEL_CARD.md`: this file.
- `review_list.txt`: the files to listen to (worst first, then random), with the word-level differences and the audio paths at the time of the run.
- `SHA256SUMS`: the checksum of every file of this entry; `ModelRegistry.verify('model_0001')` checks them (use it after copying the registry to another machine).
