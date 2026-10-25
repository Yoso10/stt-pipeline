#!/usr/bin/env bash
# How to run: the Linux twin of tools/run_work.ps1 (same steps, same state file). It calls ONLY existing commands, stops at the first error and says what to look at.
#   bash tools/run_work.sh full                          # THE WHOLE RUN, no questions: probe, check, prepare, baseline, train, evaluate, compare, report
#   bash tools/run_work.sh full --epochs 3               # 3 epochs (default 2)
#   bash tools/run_work.sh full --with-test              # ... and, after compare, the ONE-TIME Test scoring (only on the final chosen run!)
#   bash tools/run_work.sh full --dry                    # print what it would run and the estimated hours, run nothing
# Run the SAME command again to RESUME after a stop or a crash (finished steps are skipped; an unfinished training continues from its checkpoint). The state is in
# <experiments-dir>/full_run_state.json (rename it to start a fresh run).
# Single steps: probe | check | prepare | benchmark | baseline | train | evaluate | compare | report | final-test, with --experiment exp_NNNN where needed.
# Options: --config FILE (default configs/work_cpu.yaml), --threads N (default: physical cores from tools/probe_env.py), --experiments-dir DIR (default data/work_cpu/experiments),
#          --model-path DIR, --baseline-from exp_NNNN, --set key=value (repeatable, given to every step).
# UNTESTED on Linux and on the work machine (written to be the same as the PowerShell script; syntax checked with `bash -n`).
set -u
STEP="${1:-}"; [ $# -gt 0 ] && shift
CONFIG="configs/work_cpu.yaml"; THREADS=0; EPOCHS=2; WITH_TEST=0; DRY=0; EXPERIMENT=""; BASELINE_FROM=""
EXPERIMENTS_DIR="data/work_cpu/experiments"; MODEL_PATH="models/whisper-large-v3-turbo-ivrit"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --with-test) WITH_TEST=1; shift;;
    --dry) DRY=1; shift;;
    --experiment) EXPERIMENT="$2"; shift 2;;
    --baseline-from) BASELINE_FROM="$2"; shift 2;;
    --experiments-dir) EXPERIMENTS_DIR="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --set) EXTRA+=("--set" "$2"); shift 2;;
    *) echo "STOP: unknown option $1"; exit 2;;
  esac
done
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
CONST_STEP=26.7; CONST_VALLOSS=0.67; CONST_DECODE=2.9; CONST_ENCODE=14.8; CONST_EXTRAS=600
STATE="$EXPERIMENTS_DIR/full_run_state.json"

say() { echo "$@"; }
run() {  # run a python command, stop at the first failure
  echo; echo "RUNNING: python $*"; echo
  [ "$DRY" = 1 ] && { echo "(dry run: not started)"; return 0; }
  "$PY" "$@"; code=$?
  if [ $code -ne 0 ]; then echo; echo "STOP: the step failed with exit code $code. Read the last lines above, then docs/WORK_RUNBOOK.md (failure guide). Run the same command again to continue."; exit $code; fi
}
experiment_names() { [ -d "$EXPERIMENTS_DIR" ] && ls -1 "$EXPERIMENTS_DIR" | grep -E '^exp_[0-9]+$' | sort; }
latest_experiment() { experiment_names | tail -1; }
status_of() { "$PY" - "$EXPERIMENTS_DIR/$1/status.json" <<'PYEOF'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
except Exception:
    print("missing")
PYEOF
}
need_threads() {
  [ "$THREADS" -ge 1 ] 2>/dev/null && return 0
  THREADS=4
  if [ "$DRY" = 0 ]; then
    probe_text="$("$PY" tools/probe_env.py --model "$MODEL_PATH")"
    phys="$(echo "$probe_text" | sed -n 's/.*, \([0-9]*\) physical cores.*/\1/p')"
    logical="$(echo "$probe_text" | sed -n 's/.* \([0-9]*\) logical threads.*/\1/p')"
    if [ -n "$phys" ]; then THREADS="$phys"; elif [ -n "$logical" ]; then THREADS=$(( logical / 2 > 0 ? logical / 2 : 1 )); fi
  fi
  echo "Threads: $THREADS (the physical cores found by the probe; give --threads N to choose)."
}
state_get() { "$PY" - "$STATE" "$1" "$2" <<'PYEOF'
import json, sys
path, key, default = sys.argv[1:4]
try:
    print(json.load(open(path, encoding="utf-8")).get(key, default))
except Exception:
    print(default)
PYEOF
}
state_set() {  # state_set key value [key value ...]
  [ "$DRY" = 1 ] && return 0
  mkdir -p "$EXPERIMENTS_DIR"
  "$PY" - "$STATE" "$@" <<'PYEOF'
import json, sys
path, pairs = sys.argv[1], sys.argv[2:]
try:
    state = json.load(open(path, encoding="utf-8"))
except Exception:
    state = {}
for key, value in zip(pairs[::2], pairs[1::2]):
    state[key] = int(value) if value.lstrip("-").isdigit() else (True if value == "true" else (False if value == "false" else value))
json.dump(state, open(path, "w", encoding="utf-8"), indent=1)
PYEOF
}
run_capture() {  # run a step that creates an experiment; sets NEW_ID (even on failure) and LAST_CODE
  local before after
  before="$(experiment_names)"
  echo; echo "RUNNING: python $*"; echo
  if [ "$DRY" = 1 ]; then echo "(dry run: not started)"; NEW_ID="<new experiment>"; LAST_CODE=0; return 0; fi
  "$PY" "$@"; LAST_CODE=$?
  after="$(experiment_names)"
  NEW_ID="$(comm -13 <(echo "$before") <(echo "$after") | tail -1)"
}

case "$STEP" in
  probe) run tools/probe_env.py --model "$MODEL_PATH"; say "LOOK AT: RAM free (at least about 8 GB), Disk free, the model line (must say OK), the physical cores.";;
  check) run -m stt check --config "$CONFIG" "${EXTRA[@]}"; say "LOOK AT: the last line must be 'Everything checked is fine.'";;
  prepare) run -m stt prepare --config "$CONFIG" "${EXTRA[@]}"; say "LOOK AT (in $EXPERIMENTS_DIR/$(latest_experiment)/reports): load_report.txt, rejected.csv, split_report.txt, filter_report.txt, text_report.txt.";;
  benchmark) need_threads; run -m stt benchmark --config "$CONFIG" --set "train.cpu_threads=$THREADS" --set train.hyperparams.warmup_steps=3 "${EXTRA[@]}"; say "LOOK AT: $EXPERIMENTS_DIR/$(latest_experiment)/reports/benchmark.txt";;
  baseline) need_threads; run -m stt evaluate --config "$CONFIG" --set "train.cpu_threads=$THREADS" "${EXTRA[@]}"; say "LOOK AT: $EXPERIMENTS_DIR/$(latest_experiment)/reports/evaluation_baseline.json";;
  train) need_threads; args=(-m stt train --config "$CONFIG" --set "train.cpu_threads=$THREADS" "${EXTRA[@]}"); [ -n "$EXPERIMENT" ] && args+=(--experiment "$EXPERIMENT"); run "${args[@]}"; say "LOOK AT: $EXPERIMENTS_DIR/$(latest_experiment)/metrics/trial_001/metrics.csv";;
  evaluate) need_threads; [ -z "$EXPERIMENT" ] && { echo "STOP: give --experiment exp_NNNN (the train experiment)"; exit 2; }; run -m stt evaluate --config "$CONFIG" --set "train.cpu_threads=$THREADS" --experiment "$EXPERIMENT" "${EXTRA[@]}";;
  compare) [ -z "$EXPERIMENT" ] && { echo "STOP: give --experiment exp_NNNN"; exit 2; }; args=(tools/compare_evals.py --dir "$EXPERIMENTS_DIR" --experiment "$EXPERIMENT"); [ -n "$BASELINE_FROM" ] && args+=(--baseline-from "$BASELINE_FROM"); run "${args[@]}";;
  report) exp="${EXPERIMENT:-$(latest_experiment)}"; run -m stt report --config "$CONFIG" --experiment "$exp" "${EXTRA[@]}"; say "THE REPORT: $EXPERIMENTS_DIR/$exp/reports/report.html";;
  final-test) need_threads; [ -z "$EXPERIMENT" ] && { echo "STOP: give --experiment exp_NNNN"; exit 2; }; say "The Test split is scored ONCE. A second call is refused (exit code 3)."; run -m stt final-test --config "$CONFIG" --set "train.cpu_threads=$THREADS" --experiment "$EXPERIMENT" "${EXTRA[@]}";;
  full)
    say "=== FULL RUN: probe, check, prepare, baseline, train, evaluate, compare$([ "$WITH_TEST" = 1 ] && echo ', final-test'), report ==="
    need_threads
    if [ -f "$STATE" ]; then EPOCHS="$(state_get epochs "$EPOCHS")"; say "A state file was found ($STATE): this run CONTINUES it (epochs $EPOCHS). To start fresh, rename that file."; else state_set epochs "$EPOCHS" threads "$THREADS"; fi
    run tools/probe_env.py --model "$MODEL_PATH"
    run -m stt check --config "$CONFIG" "${EXTRA[@]}"
    if [ "$(state_get prepare "")" = "" ]; then
      run_capture -m stt prepare --config "$CONFIG" "${EXTRA[@]}"
      if [ "$DRY" = 0 ]; then
        [ "$LAST_CODE" -ne 0 ] && { echo "STOP: prepare failed with exit code $LAST_CODE."; exit $LAST_CODE; }
        read -r TRAIN_CLIPS VAL_CLIPS < <("$PY" - "$EXPERIMENTS_DIR/$NEW_ID/reports/stage_results/prepare.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
print(s["kept_train"], s["kept_val"])
PYEOF
)
        PER_EPOCH=$(( (TRAIN_CLIPS + 15) / 16 )); MAX_STEPS=$(( EPOCHS * PER_EPOCH )); EVAL_EVERY=$(( MAX_STEPS / 4 )); [ "$EVAL_EVERY" -lt 1 ] && EVAL_EVERY=1
        WARMUP=$(( MAX_STEPS / 20 )); [ "$WARMUP" -gt 5 ] && WARMUP=5
        state_set prepare "$NEW_ID" train_clips "$TRAIN_CLIPS" val_clips "$VAL_CLIPS" max_steps "$MAX_STEPS" eval_every "$EVAL_EVERY" warmup "$WARMUP"
        say "Read the five reports in $EXPERIMENTS_DIR/$NEW_ID/reports before trusting the run (split_report.txt: at least 5 lessons in Val and in Test)."
      fi
    fi
    if [ "$DRY" = 1 ]; then TRAIN_CLIPS=800; VAL_CLIPS=100; MAX_STEPS=$(( EPOCHS * 50 )); EVAL_EVERY=$(( MAX_STEPS / 4 )); WARMUP=5
    else TRAIN_CLIPS="$(state_get train_clips 0)"; VAL_CLIPS="$(state_get val_clips 0)"; MAX_STEPS="$(state_get max_steps 0)"; EVAL_EVERY="$(state_get eval_every 1)"; WARMUP="$(state_get warmup 0)"; fi
    EVALS=$(( (MAX_STEPS + EVAL_EVERY - 1) / EVAL_EVERY ))
    CACHE_ON=0; grep -Pzo 'encoder_cache:\s*\n\s*enabled:\s*true' "$CONFIG" >/dev/null 2>&1 && CACHE_ON=1
    HOURS="$("$PY" - "$EXPERIMENTS_DIR" "$TRAIN_CLIPS" "$VAL_CLIPS" "$MAX_STEPS" "$EVALS" "$CACHE_ON" "$CONST_STEP" "$CONST_VALLOSS" "$CONST_DECODE" "$CONST_ENCODE" "$CONST_EXTRAS" <<'PYEOF'
import glob, json, os, sys
root, tc, vc, steps, evals, cache = sys.argv[1], *(int(x) for x in sys.argv[2:6]), int(sys.argv[6])
s, l, d, e, extras = (float(x) for x in sys.argv[7:12])
files = sorted(glob.glob(os.path.join(root, "*", "reports", "benchmark.json")), key=os.path.getmtime)
if files:
    m = json.load(open(files[-1], encoding="utf-8"))["measured"]
    s, l, d = m["seconds_per_step"], m["seconds_per_val_loss_example"], m["seconds_per_decode_clip"]
total = steps * s + evals * vc * l + 3 * vc * d + extras + ((tc + vc) * e if cache else 0)
print(round(total / 3600, 1))
PYEOF
)"
    say; say "CHOSEN: $TRAIN_CLIPS Train clips, $VAL_CLIPS Val clips -> $EPOCHS epochs = $MAX_STEPS steps (16 clips per step), a Val-loss point every $EVAL_EVERY steps ($EVALS points), warmup $WARMUP."
    say "ESTIMATED TIME: about $HOURS hours (encoder cache $([ "$CACHE_ON" = 1 ] && echo ON || echo OFF)). The computer must stay on."
    COMMON=(--set "train.cpu_threads=$THREADS" --set "train.hyperparams.max_steps=$MAX_STEPS" --set "train.eval_every_steps=$EVAL_EVERY" --set "train.hyperparams.warmup_steps=$WARMUP" "${EXTRA[@]}")
    BASE_ID="$(state_get baseline "")"
    if [ -z "$BASE_ID" ] || [ "$(status_of "$BASE_ID")" != "completed" ]; then
      run_capture -m stt evaluate --config "$CONFIG" "${COMMON[@]}"
      if [ "$DRY" = 0 ]; then [ -n "$NEW_ID" ] && state_set baseline "$NEW_ID"; [ "$LAST_CODE" -ne 0 ] && { echo "STOP: the baseline evaluation failed (exit code $LAST_CODE). Run the same command again."; exit $LAST_CODE; }; say "BASELINE: look at $EXPERIMENTS_DIR/$(state_get baseline "")/reports/evaluation_baseline.json and listen to 3 clips of review_baseline.txt. Stop (Ctrl+C) now if the WER is above 1.5 or the outputs are empty."; fi
    fi
    TRAIN_ID="$(state_get train "")"
    if [ -z "$TRAIN_ID" ] || [ "$(status_of "$TRAIN_ID")" != "completed" ]; then
      args=(-m stt train --config "$CONFIG" "${COMMON[@]}"); [ -n "$TRAIN_ID" ] && args+=(--experiment "$TRAIN_ID")
      run_capture "${args[@]}"
      if [ "$DRY" = 0 ]; then [ -z "$TRAIN_ID" ] && [ -n "$NEW_ID" ] && state_set train "$NEW_ID"; [ "$LAST_CODE" -ne 0 ] && { echo "STOP: the training stopped (exit code $LAST_CODE). Run the same command again to continue it."; exit $LAST_CODE; }; fi
    fi
    if [ "$DRY" = 1 ]; then TRAIN_ID="<train experiment>"; BASE_ID="<baseline experiment>"; else TRAIN_ID="$(state_get train "")"; BASE_ID="$(state_get baseline "")"; fi
    if [ "$(state_get evaluated false)" != "True" ] && [ "$(state_get evaluated false)" != "true" ]; then run -m stt evaluate --config "$CONFIG" --experiment "$TRAIN_ID" "${COMMON[@]}"; state_set evaluated true; fi
    run tools/compare_evals.py --dir "$EXPERIMENTS_DIR" --experiment "$TRAIN_ID" --baseline-from "$BASE_ID"
    if [ "$WITH_TEST" = 1 ] && [ "$(state_get tested false)" != "True" ]; then say "The Test split is scored ONCE now (--with-test). A second call is refused."; run -m stt final-test --config "$CONFIG" --experiment "$TRAIN_ID" "${COMMON[@]}"; state_set tested true; fi
    run -m stt report --config "$CONFIG" --experiment "$TRAIN_ID"
    say; say "DONE. OPEN THE REPORT: $EXPERIMENTS_DIR/$TRAIN_ID/reports/report.html (in a browser; no network needed)."
    ;;
  *) echo "usage: bash tools/run_work.sh full|probe|check|prepare|benchmark|baseline|train|evaluate|compare|report|final-test [options]  (see the top of this file)"; exit 2;;
esac
