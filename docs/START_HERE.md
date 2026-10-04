# Start here: opening the project and giving the agent its first task

## How to run (the short version)
```
mkdir stt-pipeline ; cd stt-pipeline
git init -b main
# copy CLAUDE.md to this folder and the other documents into docs\ (section 2), then:
git add . ; git commit -m "Add project documents"
claude          # start the coding agent here, then paste the first prompt from section 4
```

## 1. What you need on the home machine (one time)
- Git installed (`git --version`).
- Python 3.12 installed for your user, no admin needed (`py -3.12 --version`). If it is missing, install it from python.org with "Install for me only". If you cannot get 3.12, tell the agent which version you have; it will test that one (O1).
- Claude Code (the coding agent) installed.
- Internet access at home. The pipeline itself is offline, but installing packages and downloading the model and data need the network, and only the home-side tools use it.
- Free disk space of at least 10 GB (the data is about 800 MB, packages and models add more).
- Note down your RAM and number of CPU cores (Task Manager, Performance tab). You will give them to the agent in the prompt.

## 2. Create the repository
Use PowerShell. Replace names as you like.
```
mkdir stt-pipeline
cd stt-pipeline
git init -b main
git config user.name  "Your Name"        # only if not set already
git config user.email "you@example.com"  # only if not set already
mkdir docs
```
Copy the files you downloaded from the chat:
- `CLAUDE.md` goes in the repository ROOT (the agent reads it automatically).
- These go into `docs\`: `PRD.md`, `DECISIONS.md`, `OPEN_ISSUES.md`, `COMPONENTS.md`, `BUILD_ORDER.md`, `TROUBLESHOOTING.md`, `RUNBOOK.md`, `DATA_ONBOARDING.md`, `START_HERE.md`.

Create a file named `.gitignore` in the root with exactly this (large files must never enter the repository):
```
.venv/
__pycache__/
*.pyc
.env
data/
models/
wheels/
experiments/
registry/
*.egg-info/
.pytest_cache/
.mypy_cache/
```
First commit:
```
git add .
git commit -m "Add project documents"
```

## 3. Create the remote repository (optional, recommended as a backup)
Option A, with the GitHub command line (`gh`) installed and logged in:
```
gh repo create stt-pipeline --private --source . --remote origin --push
```
Option B, in the browser: create an EMPTY private repository on GitHub (no README), then:
```
git remote add origin https://github.com/<your-user>/stt-pipeline.git
git push -u origin main
```
Rules: keep it private; never commit data, models or wheels (the `.gitignore` already blocks them). The agent commits locally after every component; YOU decide when to push. For the closed machine you copy a release folder, not the repository history.

## 4. The first prompt (paste it into the agent as is, after filling the brackets)
```
You are starting a new project in this git repository (already initialized, one commit with the
project documents). Work only inside this repository folder.

READ FIRST, in this order: CLAUDE.md, docs/DECISIONS.md, docs/BUILD_ORDER.md,
docs/OPEN_ISSUES.md. Then skim docs/PRD.md (the index, the Config brief and the
Orchestrator brief). CLAUDE.md is binding. The PRD and DECISIONS are the source of truth.
Do not change them; propose any change to me first.

YOUR TASK NOW is ONLY "Step 0: Bootstrap" from docs/BUILD_ORDER.md. Do not start any component.

MY ENVIRONMENT (home machine): Windows [10/11], PowerShell, no GPU, CPU only, RAM [__] GB,
[__] CPU cores, free disk [__] GB, Python versions installed: [for example 3.12.x].
Internet is available here. The TARGET machine (later) is Windows, CPU only, no admin rights,
no network, Python version unknown yet.

DELIVER:
1. Folder skeleton (src/, tests/, configs/, docs/, tools/), .gitignore (already exists, extend only
   if needed), requirements.txt (pipeline) and requirements-tools.txt (home-side tools only).
   Pin versions after testing. Use only the dependencies marked approved in docs/DECISIONS.md.
2. A virtual environment and a CLEAN-ENVIRONMENT INSTALL TEST at home with Python 3.12:
   report what installs, what does not, and whether mp3 files can be read on Windows (O9).
3. tools/fetch_wheels.py: downloads all wheels for the target Python and Windows into wheels/,
   tests an offline install into a clean venv (--no-index --find-links) and writes
   requirements.lock. Follow docs/PRD.md section "Standalone home-side tools".
4. tools/fetch_model.py and tools/fetch_data.py (same section). Test fetch_model on
   openai/whisper-tiny. Write and unit-test fetch_data but DO NOT download the Daf Yomi dataset
   yet; tell me its size first.
5. Unit tests for the tools, with the network faked. Everything starts with a "How to run" header
   (CLAUDE.md section 9).
6. Commit to git at the end (locally; do not push).

DO NOT: add dependencies that are not approved (ask first); install anything globally or use
admin rights; change the PRD or DECISIONS; start any component; download the dataset.

WHEN DONE: report in the format of CLAUDE.md section 6, and in the "Open questions" part ask me
to decide, with 2 to 3 options and a recommendation each: (a) the Python version result (O1),
(b) the audio header and decoding library and the parquet reader (O9), (c) the resampling library
and numpy (O36), (d) the package and command-line name (O62), (e) the source folder layout.
Then STOP and wait for me.
```

## 5. The prompt for every next task (template)
After you review the report, give one task at a time. Wave 1 order (from docs/BUILD_ORDER.md): C0 Config, C1 DataSource, C3 Splitter, C2 TextNormalizer, C1b SampleFilter, C5 ModelLoader, C4 FeaturePipeline, C6 TrainerCore, C7 Evaluator, C12 RunManager, then the minimal C13 Orchestrator.
```
TASK: build component [ID Name] (wave [N], docs/BUILD_ORDER.md).
Before coding, read: CLAUDE.md, docs/DECISIONS.md, the brief of [ID] in docs/PRD.md, and the
docs/COMPONENTS.md entries of the components it depends on.
Build exactly this component and nothing else. Meet the Definition of Done (CLAUDE.md section 4).
For anything ambiguous, DEFAULT or OPEN, ask me with 2 to 3 options and a recommendation.
When done: run the unit tests, write or update the entry in docs/COMPONENTS.md (with a working
"How to use" example), update docs/DECISIONS.md if a decision was made, commit locally, report in
the CLAUDE.md section 6 format, and STOP.
```
At the end of a wave, add: "Then run the wave's end-to-end smoke run as described in docs/BUILD_ORDER.md and report the result."

## 6. What the agent must always receive (checklist)
- [ ] `CLAUDE.md` in the repository root and the `docs/` folder committed (the agent has no other source of truth).
- [ ] A single, bounded task (one component or Step 0) and the instruction to stop after the report.
- [ ] The facts it cannot know: operating system, RAM, cores, free disk, Python versions, which folders hold the data and models, and that the target machine is closed, CPU only, without admin rights.
- [ ] What it must read before coding (the list above) and what it must not change without asking.
- [ ] The approved and pending dependencies (they are in `docs/DECISIONS.md`, section 3).
- [ ] The commit rule: commit locally after each task; you push.

## 7. What you check after each task
1. `git log --stat`: only the expected files changed.
2. The report: each Definition of Done item, any deviation, and the open questions. Answer the questions; the agent must not decide DEFAULT or OPEN items.
3. Run the new component's example from `docs/COMPONENTS.md` yourself once.
4. Tests: `python -m unittest discover -s tests` passes offline (disconnect the network once to be sure).

## 8. Keeping the documents in sync
The agent works from the copies in the repository. The copies in the Claude project and in the repository must not drift apart: when a decision changes, change it in the repository documents (through the agent, after your approval) and tell me, so that the project copy is updated too.

## 9. If the setup goes wrong
- `py -3.12` not found: install Python 3.12 for your user, or tell the agent the version you have.
- `git commit` asks for your identity: run the two `git config` lines in section 2.
- The agent starts changing documents or building more than the task: stop it, and point it to CLAUDE.md sections 1 and 7.
- The install test fails for a package: do not replace the package yourself; the agent reports it and proposes options (O1, O9).
