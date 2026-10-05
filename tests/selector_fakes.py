"""Fake EvalResults and a fake one-time Test evaluation for the tests and the readable scenarios of the Selector.

How to run: this module is not a script; tests/test_selector.py imports it. The results are made by the REAL Evaluator (and so can be compared by the REAL
Comparator) from a scripted fake transcriber: every reference has 10 words and a file with `k` errors has its first `k` words replaced.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from stt.evaluator import DecodeSettings, EvalConfig, EvalItem, EvalRequest, EvalResult, Evaluator, RawHypothesis  # noqa: E402
from stt.featurepipeline import FeatureExample  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, create_loader  # noqa: E402
from stt.selector import Candidate  # noqa: E402

_MODEL: LoadedModel | None = None
WORDS = [f"w{i}" for i in range(10)]


def model() -> LoadedModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = create_loader(ModelConfig(path=ROOT / "tests" / "assets" / "tiny_whisper")).load()
    return _MODEL


class _Data:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> FeatureExample:
        return FeatureExample(self.ids[i], np.zeros((80, 3000), dtype=np.float32), [1])


class _Fake:
    def __init__(self, texts: Mapping[str, str]) -> None:
        self.texts = texts

    def transcribe(self, model, dataset, settings):  # type: ignore[no-untyped-def]
        return [RawHypothesis(dataset[i].sample_id, self.texts[dataset[i].sample_id]) for i in range(len(dataset))]


def make_result(split: str, errors: int | Sequence[int], groups: int = 10, files_per_group: int = 2, version: int = 1, name: str = "model", id_prefix: str = "f") -> EvalResult:
    """An EvalResult on `split`: `groups` groups of `files_per_group` files; `errors` is the number of wrong words (of 10) in every file, or a list per file."""
    n = groups * files_per_group
    per_file = [errors] * n if isinstance(errors, int) else list(errors)
    items = [EvalItem(f"{id_prefix}{g}_{f}", f"g{g}", " ".join(WORDS), Path(f"audio/{id_prefix}{g}_{f}.mp3")) for g in range(groups) for f in range(files_per_group)]
    texts = {item.sample_id: " ".join(["x"] * k + WORDS[k:]) for item, k in zip(items, per_file, strict=True)}
    request = EvalRequest(items, _Data([i.sample_id for i in items]), model(), split, version, str, DecodeSettings(1, 100, 4))
    result = Evaluator(EvalConfig(), _Fake(texts), 42).evaluate(request)  # type: ignore[arg-type]
    return replace(result, model_info=replace(result.model_info, name=name))


def candidate(trial_id: int, val_errors: int | Sequence[int], val_loss: float = 1.0, params: Mapping[str, float | int | str] | None = None, steps: int = 100, **kw: object) -> Candidate:
    return Candidate(trial_id, params or {"learning_rate": 1e-5 * (trial_id + 1), "grad_accum_steps": 4}, Path(f"models/trial_{trial_id}"), make_result("val", val_errors, name=f"trial_{trial_id}", **kw), val_loss, steps)  # type: ignore[arg-type]


class FakeTest:
    """The one-time Test evaluation: `errors` maps a trial id (or `None` for the baseline) to the wrong words per file; counts the calls; may raise."""

    def __init__(self, errors: Mapping[int | None, int], split: str = "test", fail: Exception | None = None, **kw: object) -> None:
        self.errors = errors
        self.split = split
        self.fail = fail
        self.kw = kw
        self.calls: list[int | None] = []

    def __call__(self, c: Candidate | None) -> EvalResult:
        key = None if c is None else c.trial_id
        self.calls.append(key)
        if self.fail is not None:
            raise self.fail
        return make_result(self.split, self.errors[key], name="baseline" if c is None else f"trial_{c.trial_id}", id_prefix="t", **self.kw)  # type: ignore[arg-type]


def must_not_be_called(c: Candidate | None) -> EvalResult:
    raise AssertionError("the Test evaluation must not be called")


CallableT = Callable[[Candidate | None], EvalResult]
