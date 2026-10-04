# Build order

How to use: the agent builds ONE component per task, in this order. Every task ends with: unit tests passing offline, the component's entry in `docs/COMPONENTS.md`, `docs/DECISIONS.md` updated if a decision was made, and a git commit. Every wave ends with a real end-to-end smoke run that produces its artifacts only to prove that the chain works.

## Step 0: Bootstrap (one task, before any component)
- `git init`, `.gitignore`, first commit.
- `requirements.txt` (pipeline) and `requirements-tools.txt` (home-side tools only), environment files, folder skeleton (`src/`, `tests/`, `configs/`, `docs/`, `tools/`).
- Clean-environment install test at home with Python 3.12 (O1), including mp3 decoding on Windows (O9).
- `tools/fetch_wheels.py`: offline wheel bundle and a clean offline install test; `requirements.lock`.
- Copy the docs from the project into `docs/`.

## Wave 1: first end-to-end run
Order: C0 Config, C1 DataSource, C3 Splitter, C2 TextNormalizer, C1b SampleFilter, C5 ModelLoader (with `tools/make_test_model.py`), C4 FeaturePipeline, C6 TrainerCore, C7 Evaluator, C12 RunManager, C13 minimal Orchestrator (`smoke`, `prepare`, `train`, `evaluate`, `check`).
Exit check: `smoke` runs the whole chain on synthetic data and the tiny test model; then one real small run on the Hub data with the small OpenAI model (baseline against trained, simple report).

## Wave 2: search and selection
Order: C3b Subsampler, C8 HyperparamSearcher, C9 Selector, C10 ModelRegistry, Orchestrator commands `search`, `run`, `learning-curve`, `benchmark`.
Exit check: `smoke` of the full `run` (search, selection, registration); a real search on a screening subset.

## Wave 3: reporting, export and the rest
Order: C14 ReportBuilder (full), C11 ModelExporter, Orchestrator commands `report`, `export`, `registry`, `cleanup`, `config-docs`, `tools/make_manifest.py`, `tools/fetch_model.py`, `tools/fetch_data.py`, final documents check (`docs/COMPONENTS.md` complete, `docs/CONFIG_REFERENCE.md` generated, `docs/RUNBOOK.md` verified against the real CLI).
Exit check: the owner reviews the first real report; attributes, look and hint thresholds are adjusted together with the agent (O54).

## Rules for every task
- One component at a time. Do not start the next component; report using the project format (what was built, how it meets the Definition of Done, decisions or deviations, open questions, suggested next step) and stop.
- Do not change an existing interface, file name, folder structure, config key or dependency without proposing it first.
- Every change, however small, is tested and verified to work before it is reported as done.
