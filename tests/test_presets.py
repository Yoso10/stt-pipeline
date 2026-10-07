"""Tests of the ready-made presets (Wave 3 task 8; DECISIONS 82): configs/home_full.yaml, configs/work_like.yaml and configs/work.yaml, built with `extends` from the two validated files of the first real runs.

How to run (project root, virtual environment active):
    python -m unittest tests.test_presets -v
Offline: no data and no model are read (the paths are overridden with folders of the repository or a temporary folder). The presets use ONLY documented keys, so loading them against the schemas proves that no key was invented
and no default changed.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, load_config  # noqa: E402
from stt.config.sources import load_yaml_with_extends  # noqa: E402
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.config import BenchmarkConfig, ProgressConfig  # noqa: E402
from stt.orchestrator.wiring import section_registry  # noqa: E402
from stt.reportbuilder import ReportConfig  # noqa: E402
from stt.searcher import SearchConfig  # noqa: E402
from stt.trainercore import TrainConfig  # noqa: E402

CONFIGS = ROOT / "configs"
TINY = (ROOT / "tests" / "assets" / "tiny_whisper").as_posix()
MARKER = re.compile(r'"<<<[^"]*>>>"')


def raw(name: str) -> dict[str, Any]:
    """The YAML of a preset with its `extends` chain merged, NOT validated (so a template with markers can be read)."""
    return load_yaml_with_extends(CONFIGS / name)


def load(name: str, *overrides: str) -> Any:
    with tempfile.TemporaryDirectory() as tmp:  # the data folder only has to exist
        return load_config(CONFIGS / name, overrides=[f"data.source.path={Path(tmp).as_posix()}", f"model.path={TINY}", *overrides], registry=section_registry(), env_file=None, target_environ={})


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class EveryPresetTests(unittest.TestCase):
    def test_the_four_presets_of_the_prd_exist_and_each_starts_with_a_how_to_run_header(self) -> None:
        for name in ("smoke.yaml", "home_full.yaml", "work_like.yaml", "work.yaml"):
            with self.subTest(preset=name):
                first = (CONFIGS / name).read_text(encoding="utf-8").splitlines()[0]
                self.assertTrue(first.startswith("# How to run"), first)

    def test_the_warmup_never_exceeds_the_steps_of_the_benchmark_and_the_target_and_interval_are_set(self) -> None:
        for name in ("home_full.yaml", "work_like.yaml", "work.yaml"):
            with self.subTest(preset=name):
                tree = raw(name)
                self.assertLessEqual(tree["train"]["hyperparams"]["warmup_steps"], BenchmarkConfig().steps)  # O119: the benchmark trains only 3 steps
                self.assertEqual(tree["report"]["target_relative_wer_reduction"], 0.10)  # the owner's target
                self.assertEqual(tree["progress"]["interval_seconds"], 30)
                self.assertEqual(tree["train"]["cpu_threads"], 2)
                self.assertFalse(tree["search"]["pruning"]["enabled"])


class HomeFullTests(unittest.TestCase):
    def test_it_loads_against_the_schemas_with_the_values_of_the_validated_full_run(self) -> None:
        config = load("home_full.yaml")  # a key that does not exist, or a wrong value, fails here
        train, search = config.section("train", TrainConfig), config.section("search", SearchConfig)
        self.assertEqual((train.hyperparams.max_steps, train.hyperparams.warmup_steps, train.hyperparams.batch_size, train.hyperparams.grad_accum_steps), (100, 3, 2, 2))
        self.assertEqual((train.log_every_steps, train.eval_every_steps, train.save_every_steps), (10, 50, 100))
        self.assertEqual((search.n_trials, search.finalists.k, list(search.parameters)), (4, 2, ["learning_rate"]))
        self.assertEqual(config.section("report", ReportConfig).target_relative_wer_reduction, 0.10)
        self.assertEqual(config.section("progress", ProgressConfig).interval_seconds, 30.0)
        tree = raw("home_full.yaml")
        self.assertEqual(tree["model"]["path"], "models/whisper-tiny")
        self.assertEqual(tree["subsample"], {"val_samples": 100})  # all the data: the subsampler stays off, only the Val-loss slice is set
        self.assertEqual(tree["data"]["source"]["path"], "data/daf_yomi_raw")  # inherited from data_gmara.yaml
        self.assertNotIn("run", tree)  # the standard folders `experiments` and `registry`

    def test_the_search_grid_has_exactly_as_many_points_as_trials(self) -> None:
        points = raw("home_full.yaml")["search"]["parameters"]["learning_rate"]["points"]
        self.assertEqual(points, raw("home_full.yaml")["search"]["n_trials"])  # a grid of 4 points for 4 trials: no trial is cut off


class WorkLikeTests(unittest.TestCase):
    def test_it_is_home_full_with_only_the_rehearsal_changes(self) -> None:
        full, like = raw("home_full.yaml"), raw("work_like.yaml")
        self.assertEqual(like["subsample"], {"enabled": True, "preset": "work_like", "val_samples": None})
        self.assertEqual((like["train"]["hyperparams"]["max_steps"], like["train"]["hyperparams"]["warmup_steps"]), (20, 2))
        self.assertEqual((like["train"]["log_every_steps"], like["train"]["eval_every_steps"], like["train"]["save_every_steps"]), (5, 10, 20))
        self.assertEqual((like["run"]["experiments_dir"], like["registry"]["dir"]), ("data/work_like/experiments", "data/work_like/registry"))  # its own folders: not in the history of the real runs
        for key in ("batch_size", "grad_accum_steps"):
            self.assertEqual(like["train"]["hyperparams"][key], full["train"]["hyperparams"][key])
        for section in ("model", "eval", "search", "progress", "report", "data"):
            self.assertEqual(like[section], full[section], section)  # everything else is inherited unchanged

    def test_it_loads_and_the_val_slice_is_switched_off_because_the_preset_reduces_every_split(self) -> None:
        from stt.subsampler import SubsampleConfig

        config = load("work_like.yaml")
        sub = config.section("subsample", SubsampleConfig)
        self.assertEqual((sub.enabled, sub.preset, sub.val_samples), (True, "work_like", None))  # val_samples with all_splits is refused by the Subsampler (found by the real check)


class WorkTemplateTests(unittest.TestCase):
    def test_the_template_has_exactly_five_marked_placeholders_and_inherits_nothing_of_daf_yomi(self) -> None:
        text = (CONFIGS / "work.yaml").read_text(encoding="utf-8")
        markers = MARKER.findall(text)
        self.assertEqual(len(markers), 5, markers)  # the data folder, the three columns, the model folder
        self.assertTrue(all("FILL IN" in m for m in markers))
        tree = raw("work.yaml")
        self.assertNotIn("extra_types", tree["data"])  # extends base.yaml only: none of the Daf Yomi specifics
        self.assertNotIn("extract", tree["data"])
        self.assertEqual(tree["data"]["group_fallback"], "fail")  # the leakage protection stays
        self.assertIn("COPY it to a NEW name", text)  # O110: the manifest lists work.yaml itself

    def test_unfilled_it_is_refused_with_a_clear_message_that_shows_the_marker_and_does_not_crash(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(CONFIGS / "work.yaml", registry=section_registry(), env_file=None, target_environ={})
        self.assertIn("<<< FILL IN", str(ctx.exception))
        self.assertIn("data.source.path", str(ctx.exception))
        code, out, err = run_cli("check", "--config", str(CONFIGS / "work.yaml"))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("ERROR: Invalid configuration", err)
        self.assertIn("<<< FILL IN: the folder with the audio files", err)

    def test_with_the_data_folder_filled_in_the_model_marker_is_a_problem_line_and_when_all_are_filled_it_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "data").mkdir()
            text = (CONFIGS / "work.yaml").read_text(encoding="utf-8").replace("extends: base.yaml", f"extends: {(CONFIGS / 'base.yaml').as_posix()}")
            values = iter([(work / "data").as_posix(), "audio", "text", "group", "<<< FILL IN: the folder of the base model on this machine, for example models\\\\whisper-tiny >>>"])
            half = MARKER.sub(lambda m: '"' + next(values) + '"', text)  # the model marker is put back, only the data and the columns are filled in
            partial = work / "my_work.yaml"
            partial.write_text(half, encoding="utf-8")
            code, out, err = run_cli("check", "--config", str(partial))
            self.assertEqual(code, 2, out + err)
            self.assertRegex(out, r"PROBLEM +model\.path does not exist: <<< FILL IN")
            self.assertIn("1 problem(s) found", out)
            (work / "model").mkdir()
            done = half.replace(MARKER.findall(text)[-1].strip('"').replace("\\\\", "\\"), (work / "model").as_posix())
            filled = work / "filled.yaml"
            filled.write_text(done, encoding="utf-8")
            config = load_config(filled, registry=section_registry(), env_file=None, target_environ={})
            self.assertEqual(config.section("train", TrainConfig).hyperparams.warmup_steps, 3)


if __name__ == "__main__":
    unittest.main()
