"""Tests of the Orchestrator side of the report (Wave 3, task 2): the loader of the stored files, the `report` command, the `report` section and the live progress page in the training.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_report -v
The experiments are made with FAKE services (tests/search_stage_fakes.py: the full `run` chain, a learning curve and a benchmark, in temporary folders, with the REAL Selector, registry and Orchestrator);
the `report` command is then run against their stored files. The command must never call the trainer or the evaluator: the tests make that fail loudly. One real test trains the tiny test model with
the `html` progress sink. Nothing is downloaded; the Test split is never evaluated.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from search_stage_fakes import fake_services  # noqa: E402
from test_orchestrator_curve import curve_config  # noqa: E402

from stt.config import AppConfig, load_config  # noqa: E402
from stt.orchestrator import InterruptFlag, NoopSleepGuard, Orchestrator, RunPlan, SmokeConfig, cli, default_stage_registry, write_smoke_data  # noqa: E402
from stt.orchestrator.reportcmd import REPORT_SECTIONS, latest_experiment  # noqa: E402
from stt.orchestrator.reportdata import load_report_data, read_metrics_csv  # noqa: E402
from stt.orchestrator.wiring import RegistryWiring, TrainingWiring, build_services, exit_code_for, section_registry  # noqa: E402
from stt.registry import RegistryConfig  # noqa: E402
from stt.reportbuilder import ReportConfig, ReportConfigError, ReportError  # noqa: E402
from stt.runmanager import ExperimentSpec, ExperimentStore, RunConfig  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"
RUN_STAGES = ["prepare", "evaluate", "search", "finalists", "select_and_register"]
CSV_HEADER = "step,epoch_fraction,train_loss,val_loss,learning_rate,grad_norm,seconds_per_step,elapsed_sec,eta_sec\n"


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ReportCase(unittest.TestCase):
    """exp_0001: the fake `run` chain (registered); exp_0002: a learning curve; exp_0003: a benchmark; all in one temporary folder."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.experiments = cls.tmp / "experiments"
        cls.store = ExperimentStore(RunConfig(experiments_dir=cls.experiments))
        cls.run_config = curve_config(cls.tmp)
        cls.execute(RUN_STAGES, cls.run_config)
        cls.execute(["prepare", "learning_curve"], cls.run_config)
        cls.execute(["prepare", "benchmark"], cls.run_config)

    @classmethod
    def execute(cls, stages: list[str], config: AppConfig) -> None:
        handle = cls.store.create(ExperimentSpec(config, None, "0.0.0", {"torch": "x"}, "2026-10-07T09:00:00"))
        outcome = Orchestrator(config, fake_services(config), default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle))
        assert outcome.exit_code == 0, outcome.message
        for folder in sorted(handle.paths.metrics.glob("*trial_*")):  # the fake training writes no metrics.csv: write one like the real CSV sink does
            rows = "".join(f"{i},{i / 10},{2.0 - 0.2 * i},{'' if i % 2 else 1.9 - 0.15 * i},1e-05,1.0,0.5,{i * 0.5},{9 - i}\n" for i in range(1, 6))
            (folder / "metrics.csv").write_text(CSV_HEADER + rows, encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        import os
        import shutil
        import stat

        shutil.rmtree(cls.tmp, onexc=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE), func(path)))  # registry entries are read-only
        if cls.tmp.exists():
            cls._tmp.cleanup()

    def report(self, *args: str) -> tuple[int, dict[str, Any] | None, str]:
        code, out, err = run_cli("report", "--set", f"run.experiments_dir={self.experiments.as_posix()}", *args)
        return code, (json.loads(out) if out.strip().startswith("{") else None), err

    def snapshot(self, experiment: str) -> dict[str, bytes]:
        root = self.experiments / experiment
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != "report.html" and p.name != "run.lock"}


class ReportCommandTests(ReportCase):
    def test_the_command_writes_the_report_of_an_existing_experiment_with_every_section_and_says_where_it_is(self) -> None:
        code, summary, err = self.report("--experiment", "exp_0001", "--curve-from", "exp_0002", "--benchmark-from", "exp_0003")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertEqual(summary["sections_with_data"], [key for key, _ in REPORT_SECTIONS])
        self.assertEqual((summary["sections_without_data"], summary["input_problems"], summary["curve_from"], summary["benchmark_from"]), ([], [], "exp_0002", "exp_0003"))
        path = self.experiments / "exp_0001" / "reports" / "report.html"
        self.assertEqual(Path(summary["report_file"]), path)
        html = path.read_text(encoding="utf-8")
        for part in ("WINNER: trial 1", "Learning curve", "Planned time (benchmark", "registered entries, this one included", "model_0001"):
            self.assertIn(part, html)
        self.assertIn("The report is in:", err)
        self.assertIn("exp_0002", summary["inputs"]["learning_curve"])

    def test_without_the_options_the_curve_and_the_benchmark_are_named_as_missing(self) -> None:
        code, summary, err = self.report("--experiment", "exp_0001")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertEqual(summary["sections_without_data"], ["Learning curve: Train size against Val WER"])
        self.assertIn("pass --curve-from", summary["inputs"]["learning_curve"])
        self.assertIn("pass --benchmark-from", summary["inputs"]["benchmark"])
        self.assertIn("Sections WITHOUT data in this experiment: Learning curve", err)
        html = (self.experiments / "exp_0001" / "reports" / "report.html").read_text(encoding="utf-8")
        self.assertIn("Not available in this run: the learning curve", html)

    def test_the_history_chart_reads_the_registry_cards_and_the_target_adds_the_dashed_line_and_the_verdict_line(self) -> None:
        """Wave 3 task 7: one point per registered experiment from the stored cards; the target comes from report.target_relative_wer_reduction."""
        code, summary, err = self.report("--experiment", "exp_0001", "--set", "report.target_relative_wer_reduction=0.1")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertEqual(summary["inputs"]["history"], "1 registered experiment(s)")
        html = (self.experiments / "exp_0001" / "reports" / "report.html").read_text(encoding="utf-8")
        part = html.split('<h2 id="history">')[1].split("<h2")[0]
        self.assertIn("model_0001", part)
        self.assertIn("target: 10% below the baseline", part)
        self.assertEqual(part.count("stroke-dasharray"), 1)
        self.assertIn("Relative reduction of the Val WER against the baseline:", html.split("<h2")[0])  # in the verdict section at the top
        self.assertIn("(target: 10%)", html.split("<h2")[0])
        code, no_target, err = self.report("--experiment", "exp_0001")
        self.assertIn("(no target set)", (self.experiments / "exp_0001" / "reports" / "report.html").read_text(encoding="utf-8").split("<h2")[0])

    def test_an_experiment_without_a_selection_takes_the_history_from_the_configured_registry_or_names_it_missing(self) -> None:
        code, summary, err = self.report("--experiment", "exp_0003", "--set", f"registry.dir={(self.tmp / 'registry').as_posix()}")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertEqual(summary["inputs"]["history"], "1 registered experiment(s)")
        empty = self.tmp / "no_registry_here"
        code, summary, err = self.report("--experiment", "exp_0003", "--set", f"registry.dir={empty.as_posix()}")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertIn("missing", summary["inputs"]["history"])
        self.assertIn("History: the Val WER of each registered experiment", summary["sections_without_data"])  # named, never an empty chart
        self.assertFalse(empty.exists())  # reading a registry that is not there creates nothing

    def test_an_experiment_with_its_own_curve_file_uses_it_and_the_default_experiment_is_the_latest(self) -> None:
        code, summary, err = self.report("--experiment", "exp_0002")
        self.assertEqual(code, 0, err)
        assert summary is not None
        self.assertIn("learning_curve", summary["sections_with_data"])
        code, latest, err = self.report()
        self.assertEqual(code, 0, err)
        assert latest is not None
        self.assertEqual(latest["experiment_id"], "exp_0003")  # no --experiment: the newest one
        self.assertEqual(latest_experiment(ExperimentStore(RunConfig(experiments_dir=self.experiments))), "exp_0003")

    def test_the_command_reads_only_it_never_calls_the_trainer_or_the_evaluator_and_changes_nothing_else(self) -> None:
        before = self.snapshot("exp_0001")
        status = (self.experiments / "exp_0001" / "status.json").read_bytes()
        registry_before = sorted(str(p.relative_to(self.tmp / "registry")) for p in (self.tmp / "registry").rglob("*"))
        with mock.patch("stt.orchestrator.wiring.build_services", side_effect=AssertionError("no services may be built")), mock.patch("stt.orchestrator.wiring.EvaluationWiring.evaluate", side_effect=AssertionError("no evaluation")), mock.patch(
            "stt.orchestrator.wiring.TrainingWiring.train", side_effect=AssertionError("no training")
        ):
            code, _, err = self.report("--experiment", "exp_0001", "--curve-from", "exp_0002")
        self.assertEqual(code, 0, err)
        after = self.snapshot("exp_0001")
        self.assertEqual(after, before)  # not a byte of any other file changed, nothing was added or removed
        self.assertEqual((self.experiments / "exp_0001" / "status.json").read_bytes(), status)  # no status, no marker
        self.assertEqual(sorted(str(p.relative_to(self.tmp / "registry")) for p in (self.tmp / "registry").rglob("*")), registry_before)
        self.assertEqual(len(list(self.experiments.glob("exp_*"))), 3)  # no experiment was made

    def test_a_rerun_replaces_the_page_and_the_settings_come_from_the_command_line(self) -> None:
        self.report("--experiment", "exp_0001")
        code, summary, err = self.report("--experiment", "exp_0001", "--set", "report.title=My own title", "--set", 'report.hints.enabled_rules=["edge_of_range"]')
        self.assertEqual(code, 0, err)
        html = (self.experiments / "exp_0001" / "reports" / "report.html").read_text(encoding="utf-8")
        self.assertIn("<title>My own title</title>", html)
        block = html.split('id="hints"')[1].split("<h2")[0]
        self.assertNotIn("overfitting", block)

    def test_an_unknown_rule_an_unknown_experiment_and_an_unknown_source_are_config_errors_that_list_what_exists(self) -> None:
        code, _, err = self.report("--experiment", "exp_0001", "--set", 'report.hints.enabled_rules=["nope"]')
        self.assertEqual(code, 2)
        self.assertIn("Registered rules:", err)
        code, _, err = self.report("--experiment", "exp_0099")
        self.assertEqual(code, 2)
        self.assertIn("Existing experiments: exp_0001, exp_0002, exp_0003", err)
        code, _, err = self.report("--experiment", "exp_0001", "--curve-from", "exp_0098")
        self.assertEqual(code, 2)
        self.assertIn("--curve-from names 'exp_0098'", err)
        with tempfile.TemporaryDirectory() as empty:
            code, out, err = run_cli("report", "--set", f"run.experiments_dir={Path(empty).as_posix()}")
            self.assertEqual((code, out), (2, ""))
            self.assertIn("no experiment to report on", err)

    def test_an_experiment_in_use_is_refused_not_overwritten(self) -> None:
        handle = ExperimentStore(RunConfig(experiments_dir=self.experiments)).open("exp_0001")
        with handle.lock():
            code, _, err = self.report("--experiment", "exp_0001")
        self.assertEqual(code, 2)
        self.assertIn("locked", err)

    def test_the_command_has_help_with_an_example_and_the_report_section_has_its_defaults(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["report", "--help"])
        for part in ("python -m stt report", "--curve-from", "--benchmark-from"):
            self.assertIn(part, out.getvalue())
        self.assertIn("report", section_registry().sections)
        loaded = load_config(SMOKE_YAML, overrides=[f"data.source.path={TINY.as_posix()}", f"model.path={TINY.as_posix()}"], registry=section_registry(), env_file=None, target_environ={})
        self.assertEqual(loaded.section("report", ReportConfig), ReportConfig())
        self.assertEqual((exit_code_for(ReportConfigError("x")), exit_code_for(ReportError("x"))), (2, 4))


class LoaderTests(ReportCase):
    def load(self, experiment: str = "exp_0001", **kw: Any) -> Any:
        registry_for = lambda folder: RegistryWiring(RegistryConfig(dir=Path(folder)))  # noqa: E731
        return load_report_data(self.experiments / experiment, ReportConfig(), "2026-10-07T12:00:00", registry_for, **kw)

    def test_a_full_experiment_becomes_all_the_views_with_the_card_and_the_test_count_passed_in(self) -> None:
        loaded = self.load(curve_root=self.experiments / "exp_0002", benchmark_root=self.experiments / "exp_0003")
        d = loaded.data
        self.assertEqual(loaded.problems, ())
        self.assertIsNotNone(d.selection)
        self.assertEqual((d.selection.verdict, d.selection.winner_trial, len(d.selection.table)), ("winner", 1, 4))  # type: ignore[union-attr]
        self.assertEqual((d.baseline_val.split_name, sorted(d.finalist_val), sorted(d.test_results)), ("val", [0, 1, 2, 3], ["baseline", "trial_1"]))  # type: ignore[union-attr]
        self.assertEqual((sorted(d.final_histories), sorted(d.search_histories)), ([0, 1, 2, 3], [0, 1, 2, 3]))
        self.assertEqual((len(d.search.trials), len(d.learning_curve.points), d.benchmark.model), (4, 3, "baseline"))  # type: ignore[union-attr]
        info = d.run_info
        self.assertEqual((info.experiment_id, info.registry_entry, info.generated_at, info.test_evaluations), ("exp_0001", "model_0001", "2026-10-07T12:00:00", 1))
        self.assertEqual((info.data_fingerprint, info.split_fingerprint, info.subsample_fingerprint), ("d" * 64, "s" * 64, None))
        self.assertEqual(info.test_fingerprint and len(info.test_fingerprint), 64)
        self.assertEqual(d.selection.confidence, 0.9)  # type: ignore[union-attr]
        for name in ("selection", "search", "baseline", "learning_curve", "benchmark", "registry_card"):
            self.assertNotIn("missing", loaded.found[name], name)

    def test_a_missing_file_is_none_never_an_error_and_is_named(self) -> None:
        loaded = self.load("exp_0003")  # the benchmark experiment: no search, no selection, no results
        d = loaded.data
        self.assertEqual((d.selection, d.search, d.baseline_val, d.learning_curve, dict(d.finalist_val), dict(d.test_results)), (None, None, None, None, {}, {}))
        self.assertIsNotNone(d.benchmark)  # its own benchmark file
        self.assertEqual(loaded.found["selection"], "missing")
        self.assertEqual(loaded.found["finalists"], "missing")
        self.assertEqual(loaded.problems, ())

    def test_a_damaged_file_is_left_out_and_named_in_the_problems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "exp_0001"
            import shutil

            shutil.copytree(self.experiments / "exp_0001", copy, ignore=shutil.ignore_patterns("run.lock", "candidates"))
            (copy / "reports" / "search_trials.json").write_text("{not json", encoding="utf-8")
            (copy / "reports" / "eval_result_baseline.json").write_text(json.dumps({"codec": "stt.eval_result", "schema_version": 99}), encoding="utf-8")
            loaded = load_report_data(copy, ReportConfig(), "x")
            self.assertIsNone(loaded.data.search)
            self.assertIsNone(loaded.data.baseline_val)
            self.assertEqual(len(loaded.problems), 2)
            self.assertTrue(any("search_trials.json" in p and "JSONDecodeError" in p for p in loaded.problems))
            self.assertTrue(any("eval_result_baseline.json" in p for p in loaded.problems))
            self.assertIn("unreadable", loaded.found["search"])

    def test_a_registry_that_cannot_be_read_is_a_named_problem_and_the_count_is_unknown(self) -> None:
        loaded = load_report_data(self.experiments / "exp_0001", ReportConfig(), "x", lambda folder: RegistryWiring(RegistryConfig(dir=Path(folder) / "moved_away")))
        self.assertIsNone(loaded.data.run_info.test_evaluations)
        self.assertTrue(any("registry card model_0001" in p for p in loaded.problems))
        self.assertIsNotNone(loaded.data.selection)  # the rest of the report is not affected

    def test_the_metrics_file_is_read_into_records_a_resume_step_cuts_and_a_damaged_one_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            self.assertIsNone(read_metrics_csv(path))
            path.write_text(CSV_HEADER + "1,0.1,2.0,,1e-05,1.0,0.5,0.5,5.0\n2,0.2,1.9,1.8,1e-05,1.0,0.5,1.0,4.0\n3,0.3,1.8,,1e-05,1.0,0.5,1.5,3.0\n", encoding="utf-8")
            records = read_metrics_csv(path) or []
            self.assertEqual([(r.step, r.train_loss, r.val_loss) for r in records], [(1, 2.0, None), (2, 1.9, 1.8), (3, 1.8, None)])
            self.assertEqual([r.step for r in read_metrics_csv(path, 2) or []], [1, 2])
            path.write_text(CSV_HEADER + "x,0.1\n", encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                read_metrics_csv(path)

    def test_the_data_folder_is_the_extraction_folder_when_there_is_one_else_the_source(self) -> None:
        import os
        import shutil
        import stat

        import yaml

        loaded = self.load()
        self.assertEqual(Path(loaded.data.run_info.data_folder or "").name, "tiny_whisper")  # no extraction: data.source.path (the fake config points at the tiny model folder)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "exp_0001"
            shutil.copytree(self.experiments / "exp_0001", copy, ignore=shutil.ignore_patterns("run.lock", "candidates"))
            config = copy / "config_full.yaml"
            os.chmod(config, stat.S_IWRITE)
            tree = yaml.safe_load(config.read_text(encoding="utf-8"))
            tree["data"]["extract"] = {"dir": "data/extracted/daf_yomi", "overwrite": False}
            config.write_text(yaml.safe_dump(tree), encoding="utf-8")
            self.assertEqual(load_report_data(copy, ReportConfig(), "x").data.run_info.data_folder, "data/extracted/daf_yomi")

    def test_the_section_list_of_the_command_is_the_builders(self) -> None:
        loaded = self.load(curve_root=self.experiments / "exp_0002", benchmark_root=self.experiments / "exp_0003")
        from stt.reportbuilder import ReportBuilder

        self.assertEqual(ReportBuilder().build(loaded.data).sections, [key for key, _ in REPORT_SECTIONS])


class ProgressWiringTests(unittest.TestCase):
    def config(self, tmp: Path, *extra: str) -> AppConfig:
        return load_config(SMOKE_YAML, overrides=[f"data.source.path={tmp.as_posix()}", f"model.path={TINY.as_posix()}", *extra], registry=section_registry(), env_file=None, target_environ={})

    def test_the_html_sink_is_registered_by_name_writes_next_to_the_metrics_and_preloads_a_resumed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "experiments" / "exp_0007" / "metrics" / "trial_003"
            folder.mkdir(parents=True)
            (folder / "metrics.csv").write_text(CSV_HEADER + "".join(f"{i},0.{i},2.0,,1e-05,1.0,0.5,{i}.0,9.0\n" for i in range(1, 6)), encoding="utf-8")
            wiring = TrainingWiring(self.config(Path(tmp)))
            sink = wiring._progress_sink(folder, 3)  # a resume from step 3: the records up to step 3 are fed in first
            page = (folder / "progress.html").read_text(encoding="utf-8")
            self.assertIn("exp_0007 / trial_003", page)
            self.assertIn("Step <b>3</b>", page)
            self.assertNotIn("Step <b>5</b>", page)
            self.assertEqual(sink.path, folder / "progress.html")
            fresh = wiring._progress_sink(Path(tmp) / "experiments" / "exp_0007" / "metrics" / "trial_004", None)
            self.assertFalse(fresh.path.exists())  # nothing yet: the page appears with the first record

    def test_an_unreadable_history_does_not_stop_the_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "experiments" / "exp_0007" / "metrics" / "trial_001"
            folder.mkdir(parents=True)
            (folder / "metrics.csv").write_text(CSV_HEADER + "x,y\n", encoding="utf-8")
            TrainingWiring(self.config(Path(tmp)))._progress_sink(folder, 2)  # must not raise

    def test_the_real_training_with_the_html_sink_leaves_a_live_page_and_the_default_sinks_do_not_change(self) -> None:
        self.assertEqual(self.config(Path(".")).section("train", __import__("stt.trainercore", fromlist=["TrainConfig"]).TrainConfig).sinks, ["csv", "console"])  # opt in, no default change
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
            args = ["--config", str(SMOKE_YAML), "--set", f"data.source.path={(work / 'smoke').as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(work / 'experiments').as_posix()}",
                    "--set", 'train.sinks=["csv","console","html"]']
            code, out, err = run_cli("train", *args)
            self.assertEqual(code, 0, err)
            page = work / "experiments" / "exp_0001" / "metrics" / "trial_001" / "progress.html"
            self.assertTrue(page.is_file())
            html = page.read_text(encoding="utf-8")
            self.assertIn('http-equiv="refresh"', html)
            self.assertIn("exp_0001 / trial_001", html)
            self.assertIn("estimated time left", html)
            self.assertTrue((page.parent / "metrics.csv").is_file())  # the csv sink is unchanged
            code, out, err = run_cli("report", "--set", f"run.experiments_dir={(work / 'experiments').as_posix()}")  # the report of a training experiment: no verdict, named as missing
            self.assertEqual(code, 0, err)
            self.assertIn("training_curves", json.loads(out)["sections_with_data"])

    def test_the_services_still_build_without_the_report_settings_being_used(self) -> None:
        self.assertIsNotNone(build_services(self.config(Path("."))).training)


if __name__ == "__main__":
    unittest.main()
