# PRD Component 14: ReportBuilder

Status legend: LOCKED = explicitly decided by the owner (binding). DEFAULT = proposed starting value
(not binding, configurable, may change by evidence from experiments). OPEN = undecided: the agent
presents 2-3 options with trade-offs and a recommendation, and does not choose.
Everything not tagged below is LOCKED.
NOTE: numbered 14 only because it was added late; it will be renumbered in the final PRD. The owner wants to review the first real output and then adjust it with the agent (the attribute list and the look are deliberately open to change). Every change must be tested and verified.

## Priority / order and dependencies
- Depends on Config (`report.*`), on `SelectionResult` (Selector), `EvalResult` and `FileScore` with attributes (Evaluator), `StepMetrics` and the `MetricsSink` protocol (TrainerCore), `SearchResult` (HyperparamSearcher), and optionally on a learning-curve result (Orchestrator, type OPEN).
- It is the last stage in the pipeline: it only presents what the other components produced.

## Responsibility
Turn the results of a run into a self-contained visual report, and into a live progress page while training runs, that show at a glance whether the model succeeded, where it failed and what could be changed.

## Interface and implementations
- `ReportBuilder.build(data: ReportData) -> ReportOutput`: the final report as one HTML text.
- `HtmlProgressSink` (implements the TrainerCore `MetricsSink` protocol): writes the live progress page and regenerates it as metric records arrive.
- `DiagnosisRule` (typing.Protocol): `evaluate(data: ReportData) -> DiagnosisHint | None`. Rules are selected by name through a registry; a new rule is a new class. Several implementations exist from the start, so the protocol is justified.
- Chart drawing (line, bars with intervals, scatter, histogram) is internal pure code producing inline SVG. No interface is created for it.
- Correlation is a pure function (rank correlation), in its own module.
- Writing the report to disk is done by the caller (RunManager, O14), except for the progress sink, which writes its own live file like the CSV sink does.

## Inputs -> Outputs (typed, dataclasses)
- `ReportData`: `selection: SelectionResult`, `baseline_val: EvalResult`, `test_results: Mapping[str, EvalResult]`, `histories: Mapping[int, list[StepMetrics]]` (trial id to training history), `search: SearchResult | None`, `learning_curve: LearningCurve | None` (type OPEN), `run_info: RunInfo` (experiment number, config reference, model info, data and split fingerprints, timings, library versions, generation time passed in), `settings: ReportSettings`.
- `DiagnosisHint`: `rule: str`, `severity` (`info` | `warning`), `message: str`, `evidence: Mapping[str, float | str]`, `caveat: str`.
- `CorrelationRow`: `attribute: str`, `n: int`, `rho: float | None`, `note: str` (for example "too few points" or "constant").
- `ReportOutput`: `html: str`, `sections: list[str]` (names of sections that were produced), `warnings: list[str]`.

## Rules
1. LOCKED (owner): the report is ONE self-contained HTML file that opens offline in a browser. No network, no external scripts, fonts or images. Charts are inline SVG drawn by our own code. No new dependency (no matplotlib).
2. LOCKED (owner): the top of the page is a plain-language summary suitable for presenting: the verdict (winner or "the baseline stays"), WER of the baseline and of the winner with the interval, and the Test confirmation. The details follow below it.
3. LOCKED (owner): sections: (a) comparison table of all finalists; (b) WER bars: baseline against each finalist with intervals; (c) training curves (train and Val loss) of the winner and of the other trials; (d) distribution of per-file WER and the share of files under the thresholds; (e) WER by group; (f) search: each searched parameter against Val loss, with rank correlation; (g) learning curve when it was run; (h) correlations between per-file errors and file attributes; (i) diagnosis hints; (j) the listening list with word-level differences and audio paths; (k) run information (experiment number, config reference, fingerprints, versions, timings, and how many times this test set has already been evaluated, passed in by the Orchestrator from the registry).
4. LOCKED (owner): correlations between errors and attributes are shown so that the owner can see where the problem is. They are rank correlations computed in our own code, always shown with the number of points, and always accompanied by the note that a correlation is not a cause.
5. LOCKED: diagnosis hints come from rules whose thresholds are in Config. Every hint shows its evidence and a caveat, and is worded as a hint, never as a stated fact.
6. LOCKED: Hebrew text (references, hypotheses) is displayed right-to-left correctly; labels and headings are English (owner approved).
7. LOCKED: the live progress page refreshes itself, shows the trend of the losses, learning rate, step time and estimated time left, and works while the training is still running.
8. LOCKED: a section whose input is missing (no search, no learning curve, no attributes) says "not available in this run". It is never an error and never an empty chart.
9. LOCKED: the builder never recomputes WER, never changes a result and never decides. It presents and correlates what it is given.
10. LOCKED: the same inputs give the same output, including the generation time (passed in, not read from the clock).
11. LOCKED: all text from the data is HTML-escaped (a transcript may contain markup characters).
12. LOCKED: the page prints to PDF cleanly from the browser (print style).
13. DEFAULT: color-blind-safe palette.
14. DEFAULT: the listening list shows audio paths as text and `file:` links (no audio is embedded, to keep the file small).

## Config keys it reads (section `report`)
- `report.attributes` [DEFAULT: `duration_sec`, `label_tokens`, `ref_words`, `meta_quality_score` (only if present), `had_digits`, `had_latin`, `had_brackets`]. Open to change after the owner sees the first output.
- `report.min_points_for_correlation` [DEFAULT: 30]
- `report.max_points_per_chart` [DEFAULT: 2000; deterministic thinning]
- `report.hints.enabled_rules` [DEFAULT: all registered]
- `report.hints.*` thresholds [DEFAULT, to calibrate on real runs]:
  - overfitting: Val loss rising over the last K evaluations while the training loss falls
  - no learning: relative Val loss improvement below a threshold
  - unstable training: loss spikes
  - gain concentrated in few groups
  - weakness on short clips
  - many outputs hit the length cap or are empty
  - best trial at the edge of a searched range
  - many failed or pruned trials
- `report.progress.refresh_seconds` [DEFAULT: 10], `report.progress.every_n_records` [DEFAULT: 1]
- `report.title` [DEFAULT: experiment name]

## Out of scope (must NOT do)
- No computing of WER, CER or significance (Evaluator).
- No choosing winners (Selector).
- No training, no data processing, no network.
- No external libraries for charts, no JavaScript from outside.
- No embedding of audio.
- No claims about causes.

## Edge cases and failure behavior
- Very many files: scatter points are thinned deterministically to `max_points_per_chart` and the page says so.
- Missing values (NaN) or constant attributes: the correlation row says "too few points" or "constant".
- A history without any Val loss: the curves section shows the training loss only and a note.
- Paths with spaces or Hebrew characters: links are URL-encoded; the plain text path is shown too.
- Unknown rule name in Config: FAIL listing the registered names.
- A rule raises an exception: the hint is replaced by a warning in the report (the report is still produced) and the error is logged.
- No finalists at all: the summary says so; the report still shows what exists.
- Progress sink cannot write the file: log the error and continue training (the live page must never stop a run).

## If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| a chart is empty or missing | input not available in this run | read the "not available" note; check that the search or the learning curve was run |
| Hebrew text appears reversed or garbled | page opened without UTF-8 | the page declares UTF-8; open it in a normal browser; do not edit the file by hand |
| correlation rows say "too few points" | small evaluation set | evaluate a larger set, or lower `report.min_points_for_correlation` knowingly |
| the live page does not refresh | the file is opened through a viewer that does not reload | open it in a browser; the refresh is built into the page |
| a hint looks wrong | threshold not calibrated | change the threshold in YAML; hints are advisory |
| the file is large | many points or long lists | lower `report.max_points_per_chart` |

## Acceptance tests (offline, synthetic results)
1. The summary panel shows the right verdict, numbers and interval for a winner case and for a "baseline retained" case.
2. The page contains no reference to an external address (no remote script, stylesheet, font or image).
3. The rank correlation matches hand-computed values, including ties; constant data gives a note and no number.
4. Each default hint rule fires on a synthetic history built to trigger it and stays silent on a healthy one.
5. Hebrew text cells are marked right-to-left; markup characters in a transcript are escaped.
6. A missing section input produces the "not available" note and no error.
7. Same inputs give byte-identical output.
8. The progress sink regenerates the page as records arrive, the page contains a refresh instruction, and a write failure does not stop the caller.
9. Thinning is deterministic and keeps the cap.
10. The print style is present.
11. Nothing imports a plotting library; no network.

## OPEN items (agent presents options, owner decides)
- The type of the learning-curve data (defined with the Orchestrator).
- The final list of attributes and the look of the report: to be changed together with the agent after the owner sees the first real output.
- Whether to add a second, shorter "one page" print layout for presentations.
- Calibration of the hint thresholds on real runs.
