# PRD Component 10: ModelRegistry

Status legend: LOCKED = explicitly decided by the owner (binding). DEFAULT = proposed starting value
(not binding, configurable, may change by evidence from experiments). OPEN = undecided: the agent
presents 2-3 options with trade-offs and a recommendation, and does not choose.
Everything not tagged below is LOCKED.
NOTE: the owner will send more answers (manager questions) and the PRD will be revised then.

## Priority / order and dependencies
- Depends on Config (`registry.*`), on `SelectionResult` (Selector) and on `RunInfo` (Orchestrator / RunManager).
- Runs after the Selector and before the ModelExporter and the ReportBuilder.

## Responsibility
Store the selected model together with its metadata as an immutable, numbered entry, and keep an index of all entries.

## Interface and implementations
- `ModelRegistry.register(request: RegisterRequest) -> RegistryEntry`
- `ModelRegistry.get(entry_id: str) -> RegistryEntry`, `ModelRegistry.list_entries() -> list[RegistryEntry]`
- `ModelRegistry.best(eval_version: int, test_fingerprint: str) -> RegistryEntry | None`
- `ModelRegistry.verify(entry_id: str) -> VerifyReport` (checks the stored files against `SHA256SUMS`).
- `ModelRegistry.rebuild_index() -> None` (the index is derived from the entry cards).
- One implementation, on the file system, with `pathlib` only. No interface is created. No Hugging Face imports.

## Inputs -> Outputs (typed, dataclasses)
- `ModelCard`: base model name and path, data / split / subsample / test-set fingerprints, config or diff reference, `eval_version`, Val and Test summary (WER, CER, interval, verdict), decode settings, searched parameters, library versions, seed, generation time (passed in), the Selector's reason.
- `RegisterRequest`: `selection: SelectionResult`, `card: ModelCard`, `experiment_id: str`, `finalist_dirs: Sequence[Path]`.
- `RegistryEntry`: `entry_id` (for example `model_0001`), `experiment_id`, `kind` (`model` | `baseline_retained`), `model_dir: Path | None`, `card_path: Path`, `checksums_path: Path | None`, `test_evaluations: int` (how many registered entries, this one included, were evaluated on the same test set).
- `VerifyReport`: files checked, missing files, mismatching files, `ok: bool`.

## Rules
1. LOCKED (owner): the registry keeps the WINNER and its metadata. The folders of the other finalists and the training checkpoints are deleted after a successful registration (`registry.cleanup`), unless `registry.keep_finalists` is true.
2. LOCKED (owner): numbering, not timestamps, in all names: entries are `model_0001`, `model_0002`, and so on, each linked to its experiment (`exp_0001`) and to the config or diff file of that experiment.
3. LOCKED (owner): `model_card.json` plus a short human-readable `MODEL_CARD.md`.
4. LOCKED (owner): entries are immutable. An existing entry is never overwritten; a new run creates a new entry.
5. LOCKED (owner): an index file (`index.csv`) lists all entries with their metrics. "Best so far" is computed only among entries with the same `eval_version` and the same test-set fingerprint, because other comparisons are invalid.
6. LOCKED (owner): when the Selector returns "baseline retained", an entry of kind `baseline_retained` is still recorded (card and index row, no model folder), so the history is complete.
7. LOCKED (owner): a `SHA256SUMS` file is written for the stored model files, to detect a corrupted copy after moving the registry between machines on a removable drive. This is a transfer-integrity check and is independent of the model identification by name and path in the ModelLoader.
8. LOCKED: registration is crash-safe: files are written into a temporary folder and renamed into place at the end; the cleanup of finalists and checkpoints happens only after the checksums are written and verified. A crash leaves either a complete entry or none.
9. DEFAULT: the winner folder is MOVED into the registry (a copy when the experiment folder is on another drive), so a large model does not take twice the disk space.
10. LOCKED: no network, no model loading, no evaluation.
11. LOCKED (owner): test-set usage counter. For every test-set fingerprint the registry counts how many entries were evaluated on it and shows the count in the index and the card ("this test set has been looked at N times"). Comparison between experiments is made on Val; the Test numbers are reported but not used to pick among experiments. The count covers registered entries only, so it is a lower bound.

## Config keys it reads (section `registry`)
- `registry.dir`
- `registry.keep_finalists` [DEFAULT: false]
- `registry.cleanup` [DEFAULT: true]
- `registry.transfer` [DEFAULT: `move`; alternative `copy`]
- `registry.checksums` [DEFAULT: true]

## Out of scope (must NOT do)
- No choosing the winner (Selector), no evaluation, no training.
- No format conversion (ModelExporter).
- No loading of models or any Hugging Face / torch import.
- No deletion of anything except the finalist folders and checkpoints named in the request, and only after a successful registration.
- No network; no interactive prompts.

## Edge cases and failure behavior
- Registry folder missing: created. Entry number already exists on disk: FAIL (never overwrite).
- Winner model folder missing or empty: FAIL naming the path.
- Not enough disk space or a crash during registration: the temporary folder is removed and the registry is left unchanged; FAIL with the reason.
- `verify` finds a missing or changed file: FAIL listing the files.
- Index missing or corrupted: FAIL naming the problem, with the fix `rebuild_index()`; the index is never silently rebuilt.
- `best()` with no matching entries: returns `None`.
- `baseline_retained` with no Test result: recorded, flagged `test_not_evaluated`.
- Paths with spaces or Hebrew characters work.

## If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL entry exists | two runs raced, or a leftover temp folder | remove the leftover temp folder named in the message; never delete a finished entry |
| verify says files changed | corrupted copy between machines | copy the entry again from the source |
| index does not match entries | index edited by hand or a crash | run `rebuild_index()` |
| disk full on registration | large model, move across drives | free space, or set `registry.transfer: copy` knowingly |
| `best` returns nothing | different `eval_version` or test set | compare only like with like; re-evaluate under the current profile |

## Acceptance tests (offline, temporary folders, small fake model files)
1. A registration creates `model_0001` with the model files, `model_card.json`, `MODEL_CARD.md` and `SHA256SUMS`, and adds an index row.
2. A second registration creates `model_0002`; the first is unchanged byte for byte.
3. A crash simulated before the final rename leaves no entry and no stray files; cleanup does not run.
4. After success, the finalist folders and checkpoints are removed; with `keep_finalists: true` they stay.
5. `baseline_retained` creates an entry without a model folder and an index row.
6. `verify` passes on an intact entry and FAILS on a changed file, naming it.
7. `best` ignores entries with another `eval_version` or test fingerprint.
8. `rebuild_index` reproduces the index from the cards; a corrupted index FAILS and is not rebuilt silently.
9. Move and copy modes both work; paths with spaces and Hebrew characters work.
10. No network; no Hugging Face or torch import.

## OPEN items (agent presents options, owner decides)
- Whether a registry entry should also store the review list (listening samples) next to the card, or only reference the report.
- Retention of old entries (never deleted automatically in v1).
