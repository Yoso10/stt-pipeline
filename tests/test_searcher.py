"""Unit tests of the HyperparamSearcher component (stt.searcher): the acceptance tests of the C8 brief, offline, with fake trial runners (no training).

How to run (project root, virtual environment active):
    python -m unittest tests.test_searcher -v
The fake runners (tests/search_fakes.py) return a synthetic objective. The parallel test starts real worker processes (spawn), about 10 seconds.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))  # worker processes inherit sys.path, so search_fakes can be imported there too

from search_fakes import BowlRunner, ClockRunner, DyingRunner, InterruptingRunner, RecordingRunner, ScriptedRunner, bowl, metrics, read_trial_log  # noqa: E402

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.searcher import (  # noqa: E402
    SEARCHABLE,
    GridStrategy,
    HyperparamSearcher,
    MedianRule,
    ParamRange,
    PruningSpec,
    SearchConfig,
    SearchConfigError,
    SearchError,
    SearchInterrupted,
    SearchSpec,
    StudyStore,
    apply_params,
    build_spec,
    default_registry,
    grid_points,
)
from stt.trainercore import TrainHyperparams  # noqa: E402

SPACE = {"learning_rate": ParamRange("log_float", 5e-6, 5e-5, points=3), "grad_accum_steps": ParamRange("categorical", choices=(2, 4))}


def make_spec(**overrides: object) -> SearchSpec:
    base: dict[str, object] = dict(
        parameters=SPACE, strategy="grid", n_trials=6, time_budget_minutes=None, pruning=PruningSpec(False, 3), parallel_trials=1, max_failed_ratio=0.5,
        min_trials_before_abort=4, finalists_k=3, study_dir=None, seed=42, total_threads=4,
    )
    base.update(overrides)
    return SearchSpec(**base)  # type: ignore[arg-type]


def searcher(**kwargs: object) -> HyperparamSearcher:
    return HyperparamSearcher(**kwargs)  # type: ignore[arg-type]


class StrategyTests(unittest.TestCase):
    def test_1_grid_enumerates_the_expected_combinations_in_a_fixed_order(self) -> None:
        result = searcher().search(make_spec(n_trials=10), BowlRunner())
        combos = [(round(float(r.trial.params["learning_rate"]), 9), r.trial.params["grad_accum_steps"]) for r in result.trials]
        lrs = [round(x, 9) for x in (5e-6, (5e-6 * 5e-5) ** 0.5, 5e-5)]
        self.assertEqual(combos, [(lr, acc) for lr in lrs for acc in (2, 4)])  # the product, the last parameter varies fastest
        self.assertEqual(result.stopped_reason, "space_exhausted")  # 6 combinations, budget 10
        self.assertEqual([r.trial.trial_id for r in result.trials], [0, 1, 2, 3, 4, 5])

    def test_grid_points_for_every_kind(self) -> None:
        self.assertEqual(grid_points(ParamRange("log_float", 1e-5, 1e-3, points=3)), [1e-5, 1e-4, 1e-3])
        self.assertEqual(grid_points(ParamRange("float", 0.0, 1.0, points=5)), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(grid_points(ParamRange("int", 2, 5)), [2, 3, 4, 5])
        self.assertEqual(grid_points(ParamRange("categorical", choices=("a", 3))), ["a", 3])
        self.assertEqual(GridStrategy(make_spec()).size(), 6)

    def test_2_the_optuna_strategy_finds_the_better_region_of_a_synthetic_objective_within_the_budget(self) -> None:
        spec = make_spec(strategy="optuna_tpe", n_trials=15, parameters={"learning_rate": ParamRange("log_float", 1e-6, 1e-3), "grad_accum_steps": ParamRange("categorical", choices=(1, 2, 4, 8))})
        result = searcher().search(spec, BowlRunner())
        random_start = sorted(r.outcome.val_loss for r in result.trials[:3])  # type: ignore[type-var]
        self.assertLess(result.report.best_val_loss, random_start[0])  # the later trials beat the random start
        self.assertLess(result.report.best_val_loss, 0.5 + 0.1)  # close to the optimum of 0.5
        best_lr = float(result.report.best_params["learning_rate"])  # type: ignore[index]
        self.assertTrue(5e-6 < best_lr < 1e-4)  # the better region around 2.2e-5

    def test_7_the_same_seed_gives_the_same_sequence_in_sequential_mode(self) -> None:
        spec = make_spec(strategy="optuna_tpe", n_trials=8, parameters={"learning_rate": ParamRange("log_float", 1e-6, 1e-3), "grad_accum_steps": ParamRange("categorical", choices=(1, 2, 4, 8))})
        a = [dict(r.trial.params) for r in searcher().search(spec, BowlRunner()).trials]
        b = [dict(r.trial.params) for r in searcher().search(spec, BowlRunner()).trials]
        c = [dict(r.trial.params) for r in searcher().search(replace(spec, seed=7), BowlRunner()).trials]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_10_unknown_strategy_fails_listing_the_names_and_a_missing_optuna_names_grid(self) -> None:
        with self.assertRaises(SearchConfigError) as ctx:
            searcher().search(make_spec(strategy="random"), BowlRunner())
        self.assertIn("grid, optuna_tpe", str(ctx.exception))
        with mock.patch.dict(sys.modules, {"stt.searcher.optuna_strategy": None}):
            with self.assertRaises(SearchConfigError) as ctx2:
                searcher().search(make_spec(strategy="optuna_tpe"), BowlRunner())
        self.assertIn("optuna", str(ctx2.exception))
        self.assertIn("'grid'", str(ctx2.exception))
        self.assertEqual(default_registry().names(), ["grid", "optuna_tpe"])


class BudgetAndFailureTests(unittest.TestCase):
    def test_3_n_trials_and_the_time_budget_each_stop_the_search_with_the_right_reason(self) -> None:
        result = searcher().search(make_spec(n_trials=4), BowlRunner())
        self.assertEqual((result.stopped_reason, len(result.trials)), ("n_trials", 4))
        clock = [0.0]
        runner = ClockRunner(clock, step=10.0)  # a trial takes 40 fake seconds
        limited = HyperparamSearcher(clock=lambda: clock[0]).search(make_spec(n_trials=6, time_budget_minutes=1.0), runner)
        self.assertEqual(limited.stopped_reason, "time_budget")
        self.assertLess(len(limited.trials), 6)

    def test_a_trial_stopped_by_the_time_budget_is_a_failure_with_that_reason_and_never_a_result(self) -> None:
        clock = [0.0]
        result = HyperparamSearcher(clock=lambda: clock[0]).search(make_spec(n_trials=6, time_budget_minutes=1.0), ClockRunner(clock, step=10.0))
        reasons = [r.outcome.error for r in result.trials if r.outcome.status == "failed"]
        self.assertEqual(reasons, ["time_budget"])
        stopped = next(r for r in result.trials if r.outcome.error == "time_budget")
        self.assertIsNone(stopped.outcome.val_loss)  # the value the runner returned after being told to stop is NOT used
        self.assertNotIn(stopped.trial.trial_id, [r.trial.trial_id for r in result.ranked])
        self.assertEqual(result.stopped_reason, "time_budget")  # and it does not count towards the failure ratio

    def test_4_a_failing_trial_is_recorded_the_search_continues_and_the_failure_ratio_aborts(self) -> None:
        result = searcher().search(make_spec(n_trials=6), ScriptedRunner({1: "raise", 2: "fail"}))
        statuses = [r.outcome.status for r in result.trials]
        self.assertEqual(statuses, ["complete", "failed", "failed", "complete", "complete", "complete"])
        self.assertEqual(result.trials[1].outcome.error, "RuntimeError: boom in trial 1")
        self.assertEqual(result.trials[2].outcome.error, "the loss became NaN at step 3")
        self.assertEqual((result.report.n_failed, result.report.n_complete), (2, 4))
        spec = make_spec(n_trials=6, min_trials_before_abort=4, max_failed_ratio=0.5)
        aborted = None
        with self.assertRaises(SearchError):  # no trial completed: the run also fails after the abort
            aborted = searcher().search(spec, ScriptedRunner({i: "fail" for i in range(6)}))
        self.assertIsNone(aborted)
        # a mixed run that aborts but has results: failures 4 of 5 > 0.5 after 4 trials
        result2 = searcher().search(make_spec(n_trials=6, min_trials_before_abort=4), ScriptedRunner({1: "fail", 2: "fail", 3: "fail"}))
        self.assertEqual(result2.stopped_reason, "too_many_failures")
        self.assertEqual(len(result2.trials), 4)  # the check happens after 4 finished trials: 3 of 4 failed

    def test_zero_completed_trials_fails_with_the_reasons_and_carries_the_result(self) -> None:
        with self.assertRaises(SearchError) as ctx:
            searcher().search(make_spec(n_trials=3), ScriptedRunner({0: "fail", 1: "fail", 2: "raise"}))
        message = str(ctx.exception)
        self.assertIn("No trial completed", message)
        self.assertIn("the loss became NaN", message)
        self.assertIn("boom in trial 2", message)
        self.assertIn("learning_rate", message)
        self.assertEqual(len(ctx.exception.result.trials), 3)  # type: ignore[attr-defined]


class PruningTests(unittest.TestCase):
    def test_5_the_pruning_callback_stops_a_clearly_worse_trial_and_the_outcome_is_pruned(self) -> None:
        spec = make_spec(n_trials=6, pruning=PruningSpec(True, 2))
        result = searcher().search(spec, BowlRunner(offset={3: 5.0}))  # trial 3 is much worse than the others
        by_id = {r.trial.trial_id: r.outcome for r in result.trials}
        self.assertEqual(by_id[3].status, "pruned")
        self.assertEqual(len(by_id[3].reports), 2)  # stopped at its second evaluation (min_evals 2)
        self.assertEqual(by_id[3].steps, 100)
        self.assertNotIn(3, [r.trial.trial_id for r in result.ranked])
        self.assertEqual(sum(o.status == "pruned" for o in by_id.values()), 1)
        self.assertEqual(result.report.n_pruned, 1)
        self.assertTrue(any("sequence" in n for n in result.report.notes))  # the report says that pruning can change the sequence

    def test_the_first_trials_are_never_pruned_and_min_evals_is_respected(self) -> None:
        rule = MedianRule(min_evals=3)
        peers = [[(50, 1.0), (100, 0.9), (150, 0.8)], [(50, 1.1), (100, 1.0), (150, 0.9)]]
        self.assertFalse(rule.should_prune([(50, 9.0), (100, 9.0)], peers))  # only 2 evaluations
        self.assertTrue(rule.should_prune([(50, 9.0), (100, 9.0), (150, 9.0)], peers))
        self.assertFalse(rule.should_prune([(50, 9.0), (100, 9.0), (150, 9.0)], peers[:1]))  # fewer than two peers
        self.assertFalse(rule.should_prune([(50, 0.5), (100, 0.5), (150, 0.5)], peers))  # better than the median
        disabled = searcher().search(make_spec(n_trials=6, pruning=PruningSpec(False, 1)), BowlRunner(offset={3: 5.0}))
        self.assertEqual(disabled.report.n_pruned, 0)

    def test_all_trials_pruned_fails_with_the_advice_to_relax_pruning(self) -> None:
        class AlwaysPruned:
            def __call__(self, trial, report):  # type: ignore[no-untyped-def]
                from stt.searcher import TrialOutcome

                report(metrics(50, 1.0))
                return TrialOutcome("pruned", 1.0, 50, 0.1)

        with self.assertRaises(SearchError) as ctx:
            searcher().search(make_spec(n_trials=3), AlwaysPruned())
        self.assertIn("pruned", str(ctx.exception))
        self.assertIn("search.pruning.min_evals", str(ctx.exception))


class StudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "search"

    def test_6_a_rerun_continues_the_study_the_counts_add_up_and_a_changed_space_fails(self) -> None:
        first = searcher().search(make_spec(n_trials=4, study_dir=self.dir), BowlRunner())
        self.assertEqual(len(first.trials), 4)
        stored = json.loads((self.dir / "study.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stored["trials"]), 4)
        second = searcher().search(make_spec(n_trials=6, study_dir=self.dir), BowlRunner())
        self.assertEqual((len(second.trials), second.report.n_resumed, second.stopped_reason), (6, 4, "n_trials"))
        self.assertEqual([r.trial.trial_id for r in second.trials], [0, 1, 2, 3, 4, 5])
        self.assertEqual(len({tuple(sorted((k, str(v)) for k, v in r.trial.params.items())) for r in second.trials}), 6)  # no combination was repeated
        self.assertTrue(any("earlier run" in n for n in second.report.notes))
        third = searcher().search(make_spec(n_trials=6, study_dir=self.dir), BowlRunner())  # nothing left to run
        self.assertEqual((len(third.trials), third.report.n_resumed), (6, 6))
        wider = dict(SPACE, learning_rate=ParamRange("log_float", 1e-6, 5e-5))
        with self.assertRaises(SearchConfigError) as ctx:
            searcher().search(make_spec(parameters=wider, study_dir=self.dir), BowlRunner())
        self.assertIn("Stored:", str(ctx.exception))
        self.assertIn("Now:", str(ctx.exception))
        with self.assertRaises(SearchConfigError):
            searcher().search(make_spec(seed=1, study_dir=self.dir), BowlRunner())
        with self.assertRaises(SearchConfigError):
            searcher().search(make_spec(strategy="optuna_tpe", study_dir=self.dir), BowlRunner())

    def test_an_interrupt_keeps_the_finished_trials_in_the_study_and_a_rerun_continues(self) -> None:
        with self.assertRaises(SearchInterrupted):
            searcher().search(make_spec(n_trials=6, study_dir=self.dir), InterruptingRunner(at=2))
        stored = json.loads((self.dir / "study.json").read_text(encoding="utf-8"))
        self.assertEqual([t["trial_id"] for t in stored["trials"]], [0, 1])  # the interrupted trial is not recorded as a result or a failure
        again = searcher().search(make_spec(n_trials=6, study_dir=self.dir), BowlRunner())
        self.assertEqual((len(again.trials), again.report.n_resumed, again.report.n_failed), (6, 2, 0))
        self.assertEqual(len({tuple(sorted((k, str(v)) for k, v in r.trial.params.items())) for r in again.trials}), 6)

    def test_the_optuna_strategy_resumes_from_the_stored_trials(self) -> None:
        space = {"learning_rate": ParamRange("log_float", 1e-6, 1e-3), "grad_accum_steps": ParamRange("categorical", choices=(1, 2, 4, 8))}
        a = searcher().search(make_spec(strategy="optuna_tpe", n_trials=4, parameters=space, study_dir=self.dir), BowlRunner())
        b = searcher().search(make_spec(strategy="optuna_tpe", n_trials=7, parameters=space, study_dir=self.dir), BowlRunner())
        self.assertEqual((len(b.trials), b.report.n_resumed), (7, 4))
        self.assertEqual([dict(r.trial.params) for r in b.trials[:4]], [dict(r.trial.params) for r in a.trials])

    def test_a_damaged_study_file_fails_and_is_not_touched(self) -> None:
        self.dir.mkdir(parents=True)
        (self.dir / "study.json").write_text("{ nope", encoding="utf-8")
        with self.assertRaises(SearchError) as ctx:
            searcher().search(make_spec(study_dir=self.dir), BowlRunner())
        self.assertIn("cannot be read", str(ctx.exception))
        self.assertEqual((self.dir / "study.json").read_text(encoding="utf-8"), "{ nope")

    def test_the_store_round_trip(self) -> None:
        result = searcher().search(make_spec(n_trials=3, study_dir=self.dir), BowlRunner())
        loaded = StudyStore().load(self.dir, make_spec(study_dir=self.dir))
        self.assertEqual([(r.trial.trial_id, dict(r.trial.params), r.outcome.status, r.outcome.val_loss, r.outcome.reports) for r in loaded], [(r.trial.trial_id, dict(r.trial.params), r.outcome.status, r.outcome.val_loss, r.outcome.reports) for r in result.trials])


class ResultTests(unittest.TestCase):
    def test_9_the_result_is_ranked_by_val_loss_and_limited_to_the_finalists_k(self) -> None:
        result = searcher().search(make_spec(n_trials=6, finalists_k=3), BowlRunner())
        losses = [r.outcome.val_loss for r in result.ranked]
        self.assertEqual(len(result.ranked), 3)
        self.assertEqual(losses, sorted(losses))  # type: ignore[type-var]
        everything = sorted(r.outcome.val_loss for r in result.trials)  # type: ignore[type-var]
        self.assertEqual(losses, everything[:3])
        self.assertEqual(result.report.best_val_loss, losses[0])
        self.assertIn("best Val loss", result.report.summary())
        one = searcher().search(make_spec(n_trials=6, finalists_k=1), BowlRunner())
        self.assertEqual(len(one.ranked), 1)

    def test_the_report_warns_when_the_best_trial_is_at_the_edge_of_a_range(self) -> None:
        result = searcher().search(make_spec(n_trials=6), BowlRunner(best_lr=5e-5))
        self.assertTrue(any("edge of the range of 'learning_rate'" in n for n in result.report.notes))

    def test_the_searcher_never_declares_a_winner(self) -> None:
        result = searcher().search(make_spec(n_trials=3), BowlRunner())
        self.assertFalse(hasattr(result, "winner"))


class ParallelTests(unittest.TestCase):
    def test_8_parallel_trials_run_in_worker_processes_record_each_trial_once_and_divide_the_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = str(Path(tmp) / "trials.jsonl")
            result = searcher().search(make_spec(n_trials=4, parallel_trials=2, total_threads=4), RecordingRunner(log, sleep=0.2))
            rows = read_trial_log(log)  # one file per worker process (O108)
        self.assertEqual(sorted(r["trial"] for r in rows), [0, 1, 2, 3])  # every trial ran exactly once
        self.assertEqual(sorted(r.trial.trial_id for r in result.trials), [0, 1, 2, 3])  # and is recorded once
        self.assertEqual({r["threads"] for r in rows}, {2})  # 4 threads divided by 2 parallel trials
        self.assertEqual((result.report.parallel_trials, result.report.threads_per_trial), (2, 2))
        import os

        self.assertNotIn(os.getpid(), {r["pid"] for r in rows})  # in worker processes
        self.assertGreaterEqual(len({r["pid"] for r in rows}), 2)
        self.assertTrue(all(r.outcome.status == "complete" for r in result.trials))
        self.assertEqual(result.stopped_reason, "n_trials")
        self.assertTrue(any("can vary" in n for n in result.report.notes))

    def test_sequential_mode_gives_every_trial_all_the_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = str(Path(tmp) / "t.jsonl")
            searcher().search(make_spec(n_trials=2, parallel_trials=1, total_threads=4), RecordingRunner(log))
            rows = read_trial_log(log)
        self.assertEqual({r["threads"] for r in rows}, {4})

    def test_a_worker_that_dies_is_recorded_as_a_failed_trial_and_more_parallel_trials_than_cores_fail(self) -> None:
        with self.assertRaises(SearchError) as ctx:
            searcher().search(make_spec(n_trials=2, parallel_trials=2, total_threads=2), DyingRunner())
        self.assertIn("ended without a result", str(ctx.exception))
        self.assertEqual(len(ctx.exception.result.trials), 2)  # type: ignore[attr-defined]
        with self.assertRaises(SearchConfigError) as ctx2:
            HyperparamSearcher(cores=lambda: 1).search(make_spec(parallel_trials=2), BowlRunner())
        self.assertIn("2", str(ctx2.exception))
        self.assertIn("1 cores", str(ctx2.exception))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_the_yaml_example(self) -> None:
        c = SearchConfig()
        self.assertEqual((c.strategy, c.n_trials, c.time_budget_minutes, c.parallel_trials, c.max_failed_ratio, c.min_trials_before_abort, c.finalists.k, c.finalist_repeats, c.screening.preset), ("grid", 6, None, 1, 0.5, 4, 3, 1, "tiny"))
        self.assertEqual((c.pruning.enabled, c.pruning.min_evals), (True, 3))
        self.assertEqual(sorted(c.parameters), ["grad_accum_steps", "learning_rate"])  # the owner's default space: only these two
        self.assertEqual((c.parameters["learning_rate"].kind, c.parameters["learning_rate"].low, c.parameters["learning_rate"].high), ("log_float", 5e-6, 5e-5))
        self.assertEqual(c.parameters["grad_accum_steps"].choices, [2, 4])
        loaded = load_config(ROOT / "configs" / "search_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"search": SearchConfig}))
        self.assertEqual(loaded.section("search", SearchConfig), c)

    def test_invalid_values_fail_at_config_validation_naming_the_key(self) -> None:
        bad = [
            {"parameters": {"learning_rate": {"kind": "log_float", "low": 1e-3, "high": 1e-5}}},
            {"parameters": {"grad_accum_steps": {"kind": "categorical", "choices": []}}},
            {"parameters": {"learning_rate": {"kind": "log_float", "low": 0.0, "high": 1e-5}}},
            {"parameters": {"dropout": {"kind": "float", "low": 0.0, "high": 0.5}}},
            {"parameters": {"max_steps": {"kind": "int", "low": 10, "high": 100}}},
            {"parameters": {}},
            {"n_trials": 0},
            {"time_budget_minutes": 0},
            {"parallel_trials": 0},
            {"max_failed_ratio": 1.5},
            {"unknown": 1},
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SearchConfig.model_validate(values)
        with self.assertRaises(ValueError) as ctx:
            SearchConfig.model_validate({"parameters": {"max_steps": {"kind": "int", "low": 10, "high": 100}}})
        self.assertIn("max_steps", str(ctx.exception))
        self.assertEqual(sorted(SEARCHABLE), sorted(["learning_rate", "grad_accum_steps", "batch_size", "warmup_steps", "weight_decay", "max_grad_norm"]))

    def test_build_spec_and_apply_params(self) -> None:
        spec = build_spec(SearchConfig(), 42, 6)
        self.assertEqual((spec.strategy, spec.n_trials, spec.finalists_k, spec.seed, spec.total_threads, spec.pruning), ("grid", 6, 3, 42, 6, PruningSpec(True, 3)))
        self.assertEqual(spec.parameters["grad_accum_steps"].choices, (2, 4))
        base = TrainHyperparams(1e-5, 100, 4, 4, 10, 0.0, 1.0, 42)
        applied = apply_params(base, {"learning_rate": 3e-5, "grad_accum_steps": 2.0})
        self.assertEqual((applied.learning_rate, applied.grad_accum_steps, applied.max_steps, applied.batch_size, applied.seed), (3e-5, 2, 100, 4, 42))
        with self.assertRaises(SearchConfigError):
            apply_params(base, {"max_steps": 5})


class ImportRuleTests(unittest.TestCase):
    def test_10_optuna_is_imported_only_inside_its_adapter_and_no_network_is_used(self) -> None:
        offenders = []
        for path in sorted((ROOT / "src" / "stt" / "searcher").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(m.split(".")[0] in {"optuna", "torch", "transformers", "socket", "urllib", "requests"} for m in modules) and path.name != "optuna_strategy.py":
                    offenders.append((path.name, modules))
        self.assertEqual(offenders, [])
        code = "import sys; sys.path.insert(0, 'src'); import stt.searcher; print('optuna' in sys.modules, 'torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            searcher().search(make_spec(strategy="optuna_tpe", n_trials=3, parameters={"learning_rate": ParamRange("log_float", 1e-6, 1e-3)}), BowlRunner())

    def test_the_bowl_has_its_minimum_at_the_documented_point(self) -> None:
        self.assertAlmostEqual(bowl({"learning_rate": 2.2e-5, "grad_accum_steps": 4}), 0.5)
        self.assertGreater(bowl({"learning_rate": 1e-4, "grad_accum_steps": 4}), 0.5)


if __name__ == "__main__":
    unittest.main()
