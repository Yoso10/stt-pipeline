"""Tests of the console hygiene of Wave 3 task 7b (DECISIONS 81): the log filter that hides TWO known harmless transformers messages and nothing else.

How to run (project root, virtual environment active):
    python -m unittest tests.test_console_hygiene -v
The unit tests build log records by hand. Two tests start a fresh Python process each (a few seconds, the tiny test model, no network): one WITHOUT the filter (the two messages are there: the control) and one WITH it
(they are gone, the other messages and the decoded tokens are the same).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.evaluator.hf_transcriber import GENERATION_LOGGER, HARMLESS_PREFIXES, HfTranscriber, KnownHarmlessWarnings, hide_known_harmless_warnings  # noqa: E402

TINY = (ROOT / "tests" / "assets" / "tiny_whisper").as_posix()
BOTH = "Both `max_new_tokens` (=240) and `max_length`(=448) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
MASK = "The attention mask is not set with a batched input, and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
OTHER = "Passing `generation_config` together with generation-related arguments=({'begin_suppress_tokens'}) is deprecated and will be removed in future versions."


def record(message: str, name: str = GENERATION_LOGGER) -> logging.LogRecord:
    return logging.LogRecord(name, logging.WARNING, __file__, 1, message, None, None)


class FilterTests(unittest.TestCase):
    def test_the_two_known_messages_are_dropped_and_every_other_message_passes(self) -> None:
        f = KnownHarmlessWarnings()
        self.assertFalse(f.filter(record(BOTH)))
        self.assertFalse(f.filter(record(BOTH.replace("=240", "=4"))))  # the numbers differ per call
        self.assertFalse(f.filter(record(MASK)))
        for message in (OTHER, "Both versions are installed", "The attention mask is fine", "Something else entirely", "out of memory", "", "  Both `max_new_tokens` with a space first"):
            with self.subTest(message=message):
                self.assertTrue(f.filter(record(message)))
        self.assertEqual(len(HARMLESS_PREFIXES), 2)  # exactly two known messages, no more

    def test_the_filter_is_installed_once_on_the_generation_logger_only(self) -> None:
        HfTranscriber(1)
        HfTranscriber(2)
        hide_known_harmless_warnings()
        logger = logging.getLogger(GENERATION_LOGGER)
        self.assertEqual(sum(isinstance(x, KnownHarmlessWarnings) for x in logger.filters), 1)
        self.assertFalse(any(isinstance(x, KnownHarmlessWarnings) for x in logging.getLogger("transformers").filters))  # not the whole library
        self.assertFalse(any(isinstance(x, KnownHarmlessWarnings) for x in logging.getLogger("stt").filters))  # and not our own logs

    def test_a_message_through_the_real_logger_is_dropped_but_another_one_arrives(self) -> None:
        hide_known_harmless_warnings()
        seen: list[str] = []

        class Spy(logging.Handler):
            def emit(self, rec: logging.LogRecord) -> None:
                seen.append(rec.getMessage())

        logger = logging.getLogger(GENERATION_LOGGER)
        spy = Spy(logging.WARNING)
        logger.addHandler(spy)
        try:
            logger.warning(BOTH)
            logger.warning(MASK)
            logger.warning(OTHER)
        finally:
            logger.removeHandler(spy)
        self.assertEqual(seen, [OTHER])


SCRIPT = """
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, {src!r})
from stt.modelloader import ModelConfig, create_loader
USE = {use!r}
if USE:
    from stt.evaluator.hf_transcriber import HfTranscriber
    HfTranscriber(1)  # the constructor installs the filter
model = create_loader(ModelConfig(path=Path({tiny!r}))).load()
native = model.model
native.eval()
torch.manual_seed(0)
feats = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 80, 3000)).astype(np.float32) * 0.1)
with torch.no_grad():
    for _ in range(2):
        out = native.generate(input_features=feats, num_beams=1, max_new_tokens=4, do_sample=False)
print("TOKENS", out.tolist())
"""


def run_script(use_filter: bool) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "decode.py"
        script.write_text(SCRIPT.format(src=str(ROOT / "src"), tiny=TINY, use=use_filter), encoding="utf-8")
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, cwd=tmp, timeout=300)
    assert done.returncode == 0, done.stderr[-600:]
    return done.stdout, done.stderr


class ProcessTests(unittest.TestCase):
    def test_without_the_filter_the_two_messages_are_there_and_with_it_they_are_gone_and_the_tokens_are_the_same(self) -> None:
        out_plain, err_plain = run_script(False)
        out_quiet, err_quiet = run_script(True)
        self.assertIn("Both `max_new_tokens`", err_plain)  # the control: the messages really are written without the filter
        self.assertIn("The attention mask is not set with a batched input", err_plain)
        self.assertNotIn("Both `max_new_tokens`", err_quiet)
        self.assertNotIn("The attention mask is not set", err_quiet)
        self.assertEqual(out_plain, out_quiet)  # the decoded token ids are identical: the filter changes no result
        self.assertTrue(out_quiet.startswith("TOKENS"))

    def test_the_real_transcriber_on_the_tiny_model_writes_neither_message(self) -> None:
        import numpy as np

        from stt.evaluator import DecodeSettings
        from stt.featurepipeline import FeatureExample
        from stt.modelloader import ModelConfig, create_loader

        model = create_loader(ModelConfig(path=Path(TINY))).load()
        rng = np.random.default_rng(0)
        examples = [FeatureExample(f"s{i}", rng.standard_normal((80, 3000)).astype(np.float32) * 0.1, [1]) for i in range(4)]

        class Data:
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> FeatureExample:
                return examples[index]

        seen: list[str] = []

        class Spy(logging.Handler):
            def emit(self, rec: logging.LogRecord) -> None:
                seen.append(rec.getMessage())

        logger = logging.getLogger("transformers")
        spy = Spy(logging.WARNING)
        logger.addHandler(spy)
        try:
            out = HfTranscriber(1).transcribe(model, Data(), DecodeSettings(1, 4, 2))
        finally:
            logger.removeHandler(spy)
        self.assertEqual(len(out), 4)
        self.assertFalse([m for m in seen if m.startswith(HARMLESS_PREFIXES)], seen)


if __name__ == "__main__":
    unittest.main()
