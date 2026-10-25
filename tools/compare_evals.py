r"""compare_evals: print the untrained model against the trained model from the files that `python -m stt evaluate` stored (read-only; nothing is recomputed by a model).

How to run (PowerShell, project folder, after `$env:PYTHONPATH = "src"`):
    .\.venv\Scripts\python.exe tools\compare_evals.py --dir data\work_cpu\experiments --experiment exp_0003
    .\.venv\Scripts\python.exe tools\compare_evals.py --dir data\work_cpu\experiments --baseline-from exp_0002 --experiment exp_0003
Meaning: `--experiment` is the experiment in which `evaluate` scored the TRAINED model (it holds `reports\evaluation_trained.json` and `predictions_trained.csv`); the baseline is read
from the same experiment unless `--baseline-from` names another one (for example the early baseline-only evaluation). Both must be on the same Val clips.
Example output:
    Baseline : WER 0.8335  CER 0.3889  (101 clips, 4697 words)   exp_0002
    Trained  : WER 0.7400  CER 0.3300  (101 clips, 4697 words)   exp_0003
    Relative reduction of the WER: 11.2%   (target 10%: reached)
    Verdict (paired bootstrap over lessons, 90% one-sided, 7 lessons): improved ...
It calls the project's own code (`eval_result_from_json` for the stored baseline, `Comparator` for the verdict). The trained model's per-clip errors are read back from
`predictions_trained.csv` (word errors = WER x reference words, exact). Exit code 0 when the files were read, 2 when a file is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trained_files(csv_path: Path) -> list[SimpleNamespace]:
    """The per-clip scores of the trained model from `predictions_trained.csv` (id, lesson, reference words, word errors)."""
    files = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            words = len(row["reference"].split())
            files.append(SimpleNamespace(sample_id=row["id"], group_id=row["group"], ref_words=words, word_errors=round(float(row["wer"]) * words)))
    return files


def main() -> int:
    """Print the comparison. Returns 2 when a stored file is missing."""
    parser = argparse.ArgumentParser(description="Baseline against the trained model, from stored evaluation files.")
    parser.add_argument("--dir", type=Path, required=True, help="the experiments folder (run.experiments_dir)")
    parser.add_argument("--experiment", required=True, help="the experiment where `evaluate` scored the trained model, for example exp_0003")
    parser.add_argument("--baseline-from", default=None, help="another experiment that holds the baseline evaluation (default: the same one)")
    parser.add_argument("--target", type=float, default=0.10, help="the target relative reduction of the WER (default 0.10 = 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="the seed of the bootstrap (runtime.seed)")
    args = parser.parse_args()
    base_dir = args.dir / (args.baseline_from or args.experiment) / "reports"
    new_dir = args.dir / args.experiment / "reports"
    needed = [base_dir / "evaluation_baseline.json", base_dir / "eval_result_baseline.json", new_dir / "evaluation_trained.json", new_dir / "predictions_trained.csv"]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        print("A stored file is missing: " + "; ".join(str(p) for p in missing))
        print("Fix: run `python -m stt evaluate --config <your config> --experiment <the experiment of the train step>` first (it scores the baseline AND the trained model).")
        return 2
    from stt.evaluator import Comparator, EvalConfig, eval_result_from_json

    base_summary, new_summary = read_json(needed[0]), read_json(needed[2])
    baseline = eval_result_from_json(needed[1].read_text(encoding="utf-8"))
    trained = SimpleNamespace(eval_version=new_summary["eval_version"], files=trained_files(needed[3]))
    rebuilt = sum(f.word_errors for f in trained.files) / max(1, sum(f.ref_words for f in trained.files))
    if abs(rebuilt - new_summary["wer"]) > 0.002:
        print(f"WARNING: predictions_trained.csv gives WER {rebuilt:.4f} but evaluation_trained.json says {new_summary['wer']:.4f}: the files do not belong together.")
    print(f"Baseline : WER {base_summary['wer']:.4f}  CER {base_summary['cer']:.4f}  ({base_summary['n_files']} clips, {base_summary['n_words']} words)   {args.baseline_from or args.experiment}")
    print(f"Trained  : WER {new_summary['wer']:.4f}  CER {new_summary['cer']:.4f}  ({new_summary['n_files']} clips, {new_summary['n_words']} words)   {args.experiment}")
    if base_summary["n_files"] != new_summary["n_files"] or base_summary["n_words"] != new_summary["n_words"]:
        print("WARNING: the two evaluations are not on the same clips; the numbers are not comparable.")
    reduction = (base_summary["wer"] - new_summary["wer"]) / base_summary["wer"] if base_summary["wer"] else None
    if reduction is None:
        print("Relative reduction of the WER: not defined (the baseline WER is 0)")
    else:
        state = "reached" if reduction >= args.target else "not reached"
        print(f"Relative reduction of the WER: {reduction * 100:.1f}%   (target {args.target * 100:g}%: {state})")
    try:
        comparison = Comparator(EvalConfig(), args.seed).compare(baseline, trained)  # type: ignore[arg-type]
        print(f"Verdict (paired bootstrap over lessons): {comparison.verdict}. {comparison.explanation}")
        print("(improved = clearly better; no_clear_difference = could be chance; worse; insufficient_groups = too few lessons for a statement.)")
    except Exception as exc:  # noqa: BLE001  the comparison needs the same clips in both files
        print(f"The verdict could not be computed ({type(exc).__name__}: {exc})")
    stored = new_dir / "comparison.json"
    if stored.is_file():
        print(f"The `evaluate` stage stored its own verdict in {stored.name}: {read_json(stored).get('verdict')}")
    print("The Test split was not used by this tool (nor by `evaluate`).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
