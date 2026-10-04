# CLAUDE.md: STT Fine-Tuning Pipeline (Whisper / ivrit-ai)

This file is read by the coding agent (and by the weaker internal agent at the work site). Keep to the rules literally. When in doubt, ask. Sections 0 to 6 are the original project instructions; sections 7 to 12 are additions agreed with the owner.

## 0. Context (never forget)
- Goal: a generic, clean "factory" that ingests labeled audio+text, splits it, and fine-tunes
  a Hebrew Whisper model, then evaluates it (WER).
- TARGET ENVIRONMENT: a closed (air-gapped) machine, CPU only, NO agent present, possibly NO
  Hugging Face access. Everything must run OFFLINE and NON-INTERACTIVELY.
- The owner must be able to understand the system without reading the code.
  Therefore: no surprises, no unrequested changes.

## 1. Working protocol (strict)
1. ONE component per task. Never implement a second component "while you're at it".
2. Before coding a component: read docs/DECISIONS.md and the component's brief in docs/PRD.md.
   If anything is ambiguous or missing -> ASK. Do not guess.
3. Do NOT change an existing interface, file name, folder structure, config key or dependency
   without explicit approval. Propose the change first, wait for a yes.
4. Do NOT add new dependencies. If one seems necessary, ask and explain why.
5. Do NOT refactor or "improve" code outside the current component.
6. A component is DONE only when its Definition of Done (section 4) is fully met.
7. After finishing, report using the format in section 6. Then STOP and wait.

## 2. SOLID rules (how every component must be built)
- SRP: one responsibility per module. Loader != trainer != evaluator != selector.
  If you need the word "and" to describe a class, split it.
- OCP: extend by adding a new class implementing an existing interface, never by editing
  pipeline logic or adding if/else on type names. Use a registry/factory for selection by name.
- LSP: every implementation must be fully substitutable for its interface, with no extra
  preconditions, no surprising exceptions, same return types.
- ISP: small, task-specific interfaces (typing.Protocol / ABC). No interface method that an
  implementer would leave empty. Create an interface ONLY where >1 implementation exists
  or is planned.
- DIP: high-level code (Orchestrator) depends only on interfaces. Concrete classes are wired
  in ONE place (composition root). Hugging Face / Optuna / scipy are imported only inside
  their own adapter modules.
- Components communicate through typed data objects (dataclasses), not dicts or globals.
- No global state, no hidden singletons, no side effects at import time.

## 3. Coding standards
- Python, 100% type hints, checked clean with mypy --strict (if available offline).
- pathlib for all paths. Zero hard-coded paths, ratios, sample rates, model names, thresholds,
  or hyperparameters: everything comes from Config.
- Fail fast with clear, actionable error messages (what is wrong, which key/path, how to fix).
- Use `logging`, not print (except the final JSON report). Log to console AND file.
- Deterministic: a seed comes from Config and is applied everywhere.
- No network calls at runtime. Never call anything that downloads (e.g. evaluate.load,
  from_pretrained with a hub name) unless Config explicitly allows online mode.
- No interactive prompts ever (no input(), no confirmations). Exit codes: 0 success, non-zero failure.
- Each component ships with unit tests that run offline on tiny synthetic data.

## 4. Definition of Done (per component)
1. Implements exactly the responsibility in its brief, nothing more.
2. Public interface matches the brief (names, signatures, types).
3. All parameters come from Config; defaults documented.
4. Unit tests pass offline; edge cases from the brief are covered.
5. Works with no network access.
6. Docstring on every public class/function: purpose, inputs, outputs, errors.
7. DECISIONS.md updated if a decision was made; no undocumented decisions.
8. docs/COMPONENTS.md entry written or updated (section 10).
9. Every script and document written starts with a "How to run" header (section 9).
10. The change is committed to git (section 12).

## 5. Component brief format (every component in PRD.md follows it)
- Priority / order and dependencies
- Responsibility (one sentence)
- Inputs -> Outputs (typed)
- Config keys it reads
- Out of scope (what it must NOT do)
- Edge cases and failure behavior
- If it fails: minimal safe fixes (table: symptom, likely cause, minimal fix)
- Acceptance tests

## 6. Report format after each component
1. What was built (files created/changed)
2. How it satisfies each Definition of Done item
3. Decisions or deviations (should be none; if any, flagged)
4. Open questions
5. Suggested next step (do not start it)

## 7. Decision status (binding rule)
Every item in PRD.md and DECISIONS.md carries one tag:
- LOCKED: explicitly decided by the owner. Binding. Changing it needs approval.
- DEFAULT: a starting value or choice proposed to get going. NOT binding. Keep it configurable,
  never hard-code it.
- OPEN (also written PENDING): undecided.
Anything the owner did not explicitly confirm is NOT locked.
For DEFAULT and OPEN items the agent must not silently choose: it presents 2-3 options with
trade-offs and a recommendation, and waits for the owner.
When real run results exist (reports, WER, rejected-sample statistics, timings), the agent proposes
changes to DEFAULT and OPEN items based on that evidence: it shows the evidence, records the outcome
in docs/OPEN_ISSUES.md, and never applies such a change on its own.

## 8. When something fails
Diagnose first. Propose the smallest fix that changes no LOCKED interface, config key, folder
structure or dependency. State what the fix could affect and apply it only after approval.
Prefer a Config change over a code change, and a change inside one adapter over a change in shared
logic. Keep the system portable: no hard-coded paths, no network, no machine-specific or
dataset-specific assumptions, so it works on another computer and with the next dataset.
Use docs/TROUBLESHOOTING.md first; each component has an "If it fails" table.

## 9. Usage header
Every script and every document the agent writes begins with a short "How to run" section:
what it does in one sentence, the exact command to run it, and one concrete example with
example output where useful. Keep it concise. The owner runs everything alone on the closed
machine and must be able to start any script from its first lines.

## 10. Component map (docs/COMPONENTS.md)
The agent maintains docs/COMPONENTS.md. For every component it holds a concise entry: purpose in
one sentence, how to use it (the command or call and one example), inputs and outputs, config keys,
known limits, and where its tests are. The entry is written or updated at the end of every
component task, and the whole file is checked at the end of the project, so that the owner knows
exactly what is in the project without reading the code. Before coding a component, also read its
entry and the entries of the components it depends on.

## 11. Verification
- Run the unit tests of the component, then run the end-to-end smoke run when the wave in docs/BUILD_ORDER.md says so. The artifacts of a smoke run exist only to prove the chain works.
- Every change, however small, is tested and verified to work before it is reported as done.
- Parallel runs are a simple switch and must be tested before being relied on.

## 12. Process
- Bootstrap first (docs/BUILD_ORDER.md, Step 0): `git init`, requirements and environment files, the clean-environment install test, then the first component.
- Build in the order of docs/BUILD_ORDER.md. Commit at the end of every component task.
- The owner may also use the weaker internal agent at the work site. The same rules apply to it: diagnose first, smallest fix, no interface changes. Write explicit, short, self-contained instructions in every document.
- Reproducibility: the same seed gives the same result on the same machine; do not expect bit-identical numbers across machines.
