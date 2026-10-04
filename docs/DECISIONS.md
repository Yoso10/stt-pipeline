# Decisions

How to use: the decisions of each component are the "Rules" and "Decisions" sections of its brief in `docs/PRD.md` (each rule carries its status tag). This file holds the decisions that span components, the dependency status, and the list of proposals the owner rejected. The agent reads this file and the component's brief before coding, and updates this file whenever a decision is made. A DEFAULT or OPEN item is never changed on the agent's own initiative (see `docs/OPEN_ISSUES.md`).

## 1. Cross-cutting requirements (owner)
1. STAGE 1 = A LOT OF DATA: all the Daf Yomi data on the Hub is used at home. SMALL-DATA READINESS is a LATER rehearsal stage (the work data is about one tenth of it): gates and defaults must stay sensible on a small set, rehearsed at home with the Subsampler `work_like` preset, but it is not a blocker for stage 1.
2. UNATTENDED RUNS: no operator during a run. Resume, logs to file, graceful time cap, exit codes.
3. NO ADMIN RIGHTS on the closed machine: virtual environment and per-user installs only, no system changes.
4. ONE OPERATOR: the owner (a programmer) runs everything alone on the closed machine. Messages, reports and docs must be clear and easy.
5. OUTSIDE FIRST: everything must work at home at the level of the real system before the move inside.
6. TWO LENSES FOR SUCCESS: the automatic test decides; the owner's own listening can reveal practical failure.
7. TASK FACTS: Hebrew only (language fixed); only the domain jargon changes between datasets; labels are treated as gold; the metric is WER; timestamps and special punctuation do not count in the final result.
8. FINAL MODEL FORMAT UNKNOWN: an export layer is required.
9. Report sensitivity is not a concern for now.
10. AGENT VERIFICATION (owner): the coding agent must run the unit tests of every component and, after that, run an end-to-end smoke run on tiny synthetic data with the tiny model. The artifacts of that run (reports, metrics file, model folder) are produced only to prove that the whole chain works.
11. Parallel runs must be a simple, TESTED switch, sequential by default.
12. EVERY change, however small, must be tested and verified to work before it is reported as done. Changes to interfaces, keys or files are proposed first (CLAUDE.md rule 3).
13. The report content (attributes, look, hint thresholds) is expected to be adjusted together with the agent after the owner sees the first real output.
14. USAGE HEADER (owner): every script and document the agent writes starts with a concise "How to run" section with an exact command and an example (CLAUDE.md section 9).
15. COMPONENT MAP (owner): a concise summary document of every component and its use, kept up to date by the agent (docs/COMPONENTS.md, CLAUDE.md section 10).
16. INTERNAL AGENT AT THE WORK SITE (owner): there is an internal agent on the closed machine; it is much weaker than Claude but can help. All documents (CLAUDE.md, COMPONENTS.md, TROUBLESHOOTING.md, RUNBOOK.md) must therefore be explicit, short and self-contained, with exact commands, so that a weaker agent can follow them. The same rules (diagnose first, smallest fix, no interface changes) apply to it.
17. REPRODUCIBILITY EXPECTATION: the same seed gives the same result on the same machine; results are not guaranteed bit-identical across machines, CPUs or core counts, so exact numbers are not compared between home and work. This is written into TROUBLESHOOTING.md.
18. BOOTSTRAP FIRST (owner, definite): before any component is built, the agent creates the requirements and environment files, initializes the git repository (`git init`) for version control, and runs the clean-environment install test at home. Every component task then ends with a commit.

## 2. Global decisions (owner-locked unless marked DEFAULT)
1. Pipeline order: DataSource, Splitter, TextNormalizer, SampleFilter, Subsampler, ModelLoader and FeaturePipeline, TrainerCore, Evaluator, then search, selection, registry, export, report. RunManager and Orchestrator wrap everything.
2. Offline by default. The pipeline never downloads anything. Only the standalone home-side tools use the network (DECISION: the ModelLoader loads local folders only, even when online mode is on).
3. CPU only. The machine must stay usable: `train.cpu_threads` is limited (DEFAULT half of the cores).
4. Data: at home all of the Daf Yomi data on the Hub is used (stage 1). The work data is about one tenth of that (rehearsed later with the Subsampler `work_like` preset).
5. Splitting is by group (content, for example the lesson `entry_id`), not by speaker. Test is never touched by subsampling in `train_only` scope. Ratios 80/10/10 are DEFAULT.
6. Two texts per sample: `train_text` (gentle cleaning, punctuation kept) and `eval_text` (aggressive cleaning, punctuation removed). WER is computed on `eval_text`. Each eval profile has a version; results of different versions are not comparable.
7. Timestamps: v1 strips timestamp tokens only. There is no need to deal with timestamps in the transcript. `keep` is a documented later extension.
8. SampleFilter: structural rules apply to all splits, quality rules to Train only (DEFAULT). Report mode is the default. Clips longer than the Whisper window are removed by the filter; the FeaturePipeline never truncates audio and FAILS if a longer clip arrives. Labels longer than the model limit are removed with a report and a gate.
9. Models: phase 1 validates the pipeline with a small OpenAI Whisper model; the ivrit-ai large-family models are a later phase run on the owner's own machine. The language is Hebrew only and is set from Config.
10. Training: unit is steps; only Val loss during training; checkpoints, resume, early stopping and an optional time cap; full fine-tuning by default; no LoRA and no `peft` in v1.
11. Search: Optuna is approved only as long as it helps; DEFAULT search over `learning_rate` and `grad_accum_steps` only; Val loss as the objective with pruning; finalists retrained on the full data and judged by WER on Val; sequential by default with a simple, tested switch for parallel trials.
12. Evaluation: corpus WER primary, CER secondary, our own implementation (no `jiwer`, never `evaluate.load`); paired bootstrap over groups with a lenient level of evidence (DEFAULT 90 percent, one-sided); a listening list for the owner; no weighting of error types.
13. Selection: Val decides; Test is evaluated once for the winner; "the baseline stays" is a legitimate outcome (exit code 0); a winner that Test does not confirm is flagged, not dropped.
14. Registry: numbered immutable entries, model card, checksums, index, a test-set usage counter. Exporter v1: Hugging Face format only plus the interface.
15. RunManager: numbered experiment folders (no timestamps in names), full config stored read-only, parent and diff, nothing deleted automatically, heartbeat lock, all reports written by the RunManager.
16. Orchestrator: one `run` command, stage commands, completion markers, stop at the first failure, fixed exit codes, graceful interrupt, sleep prevention on Windows, `smoke`, `benchmark`, `check`, `report`, `config-docs`.
17. Python 3.12 is the DEFAULT version, to be confirmed by the clean-environment install test.
18. Version control: `git init` at the bootstrap step; every component task ends with a commit.

## 3. Dependencies
| Dependency | Status | Used by |
|---|---|---|
| pydantic, PyYAML | approved | Config |
| torch, transformers, safetensors, accelerate | approved | ModelLoader, FeaturePipeline, TrainerCore, Evaluator |
| optuna | approved only if it helps (the `grid` strategy needs no dependency) | HyperparamSearcher adapter |
| huggingface_hub | installed with transformers; used by the home-side tools only | tools |
| audio header and decoding library (candidate soundfile; mp3 on Windows to be verified) | PENDING, needs approval (O9) | DataSource, FeaturePipeline |
| parquet reader (candidate pyarrow) | PENDING, needs approval (O9) | DataSource |
| resampling and mono conversion (candidates scipy, torchaudio) | PENDING, needs approval (O36) | FeaturePipeline |
| numpy (installed with torch) | approval requested for explicit use (O36) | FeaturePipeline |
| datasets | NOT used (owner approved a plain loop) | none |
| peft | NOT approved, not needed in phase 1 | none |
| matplotlib, jiwer, TensorBoard | NOT used (own code) | none |

## 4. Proposals the owner rejected or did not need (do not propose again without new evidence)
- A diagnostics bundle command (an internal agent exists at the work site).
- A file-lock retry helper for Windows.
- Timestamp training (`keep` mode) in v1.
- Weighting of error types in the metric.
- Picking among experiments by Test results.
