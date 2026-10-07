# Data onboarding checklist (new dataset)

## How to run
```
Copy-Item configs\work.yaml configs\my_work.yaml                 # the template stays unchanged; you edit the COPY (O110)
python -m stt check   --config configs\my_work.yaml        # lists every problem at once; the `<<< FILL IN ... >>>` markers show up here
python -m stt prepare --config configs\my_work.yaml        # prepare only, no training: reads, validates, splits, normalizes, filters
```
(PowerShell, in the project folder, after `.\.venv\Scripts\Activate.ps1` and `$env:PYTHONPATH = "src"`.)
Then read `experiments/exp_NNNN/reports/` (load report, rejected list, split report, character inventory, filter report).

## 1. Collect before mapping
- [ ] What the data is: audio folder, transcript file(s), format (CSV, parquet, other)
- [ ] Audio: format, sampling rate, mono or stereo, typical and maximum length
- [ ] Transcripts: one per clip? Are timestamps or markup inside? Punctuation? Digits or words? Latin letters? Non-speech markers?
- [ ] A group identifier per clip (recording, lesson, topic) that must not be split between Train and Test
- [ ] How many hours, files and distinct groups
- [ ] Which labels are verified (gold) and whether a quality score exists

## 2. Map the columns in your copy `configs/my_work.yaml`
```
data:
  source: {type: csv, path: <folder>}
  columns: {audio: <audio column>, text: <text column>, group: <group column>}
  group_fallback: fail          # leaving this means a missing group column stops the run, on purpose
```
For a Hugging Face parquet folder use `type: hf_parquet`; nested `metadata` fields appear with the prefix `meta_`. A column name that does not exist is reported by `prepare` with the list of the columns that do exist. The Daf Yomi mapping is a working example: `configs/data_gmara.yaml`.

## 3. Run `prepare` and read
1. Rejected clips and reasons (`rejected.csv`): fix the mapping, do not raise the gate just to pass.
2. The character inventory: every character that is not a Hebrew letter, a digit or a space, with counts. Decide whether each one is punctuation, markup or a real symbol, and set `text.*` in YAML (never in code).
3. Duration distribution against the model window (30 seconds) and `filter` thresholds. Set the filter mode to `apply` for the long-clip rule before training.
4. Splits: the number of groups per split must meet the minimums; with few groups, read the Splitter troubleshooting table.
5. Non-speech clips: how they are labeled; use the marker recipe in the TextNormalizer brief when needed.

## 4. Questions for the data owner
- Who transcribed and how accurate is it? Are there known label errors?
- Are there recurring names, terms or abbreviations that matter for the result?
- Will more data of the same kind arrive later?
