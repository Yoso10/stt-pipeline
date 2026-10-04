"""Unit tests of the TextNormalizer component (stt.textnormalizer): the acceptance tests of the C2 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_textnormalizer -v
The tests use small synthetic Hebrew texts in memory; no file is read and nothing is downloaded.
"""

from __future__ import annotations

import ast
import random
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.textnormalizer import (  # noqa: E402
    FLAGS,
    TextConfig,
    TextNormalizer,
    TextNormalizerError,
    build_profile,
    default_registry,
    parse_segments,
)
from stt.textnormalizer.steps import (  # noqa: E402
    CollapseWhitespace,
    LowercaseLatin,
    RemoveBracketMarkup,
    RemoveNikud,
    RemovePunctuation,
    StripTimestampTokens,
    UnicodeClean,
    UnifyQuotes,
)

TOKEN = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")


def sample(i: str, text: str) -> AudioSample:
    return AudioSample(i, Path(f"{i}.wav"), text, "g", 3.0, None, {})


def conf(**overrides: object) -> TextConfig:
    return TextConfig.model_validate(overrides)


class StepTests(unittest.TestCase):
    """Test 1: each step, table-driven, with Hebrew examples."""

    def table(self, step: object, cases: list[tuple[str, str]]) -> None:
        for given, expected in cases:
            with self.subTest(given=given):
                self.assertEqual(step.apply(given), expected)  # type: ignore[attr-defined]

    def test_unicode_clean(self) -> None:
        step = UnicodeClean(TextConfig().unicode.invisible_chars)
        self.table(step, [("שלום‎עולם", "שלוםעולם"), ("﻿א‏ב‬ג", "אבג"), ("é", "é"), ("ללא שינוי", "ללא שינוי")])

    def test_strip_timestamp_tokens(self) -> None:
        step = StripTimestampTokens(TOKEN)
        self.table(step, [("<|0.44|>אמר ליה. <|6.52|><|7.02|>כן <|8.4|>", "אמר ליה. כן "), ("אין טוקנים", "אין טוקנים"), ("<|abc|>x", "<|abc|>x")])

    def test_remove_punctuation_and_maqaf(self) -> None:
        c = TextConfig().punctuation
        step = RemovePunctuation(c.chars, c.space_chars)
        self.table(step, [("אמר, רבא. כו׳ \"טוב\" – כן…", "אמר רבא כו טוב  כן"), ("בית־המקדש", "בית המקדש"), ("a-b", "ab"), ("שלום?!", "שלום")])

    def test_unify_quotes_geresh_variants(self) -> None:
        step = UnifyQuotes(TextConfig().quotes.map)
        self.table(step, [("כו'", "כו׳"), ("כו’", "כו׳"), ("רש\"י", "רש״י"), ("“טוב”", "״טוב״"), ("כו׳", "כו׳")])

    def test_remove_nikud_keeps_letters_and_maqaf(self) -> None:
        self.table(RemoveNikud(), [("שָׁלוֹם", "שלום"), ("בְּרֵאשִׁית", "בראשית"), ("א־ב", "א־ב"), ("ללא", "ללא")])

    def test_brackets_in_all_three_modes(self) -> None:
        pairs = (("[", "]"), ("(", ")"))
        text = "אמר [אשר] לא (כו׳) יבאו"
        self.assertEqual(RemoveBracketMarkup(pairs, "chars_only").apply(text), "אמר אשר לא כו׳ יבאו")
        self.assertEqual(RemoveBracketMarkup(pairs, "with_content").apply(text), "אמר  לא  יבאו")
        self.assertEqual(RemoveBracketMarkup(pairs, "keep").apply(text), text)
        self.assertEqual(RemoveBracketMarkup(pairs, "with_content").apply("a [b [c] d] e"), "a  e")
        self.assertEqual(RemoveBracketMarkup(pairs, "chars_only").apply("a [b [c"), "a b c")  # unbalanced: characters removed, text kept
        self.assertEqual(RemoveBracketMarkup(pairs, "with_content").apply("a [b ] ] c ["), "a   c ")

    def test_lowercase_latin_only(self) -> None:
        self.table(LowercaseLatin(), [("Hello שלום ABC", "hello שלום abc"), ("ÉCOLE", "école"), ("אבג", "אבג")])

    def test_collapse_whitespace(self) -> None:
        self.table(CollapseWhitespace(), [("  א   ב \n ג\t", "א ב ג"), ("א ב", "א ב"), ("", ""), ("   ", "")])


class ProfileTests(unittest.TestCase):
    def test_default_profiles_on_a_realistic_label(self) -> None:
        raw = "<|0.44|>אמר ליה. כי קאמר שמואל, [אשר] (כו׳) ‎בלי… <|6.52|><|7.02|>הלכה? <|8.00|>"
        self.assertEqual(build_profile(TextConfig(), "train")(raw), "אמר ליה. כי קאמר שמואל, אשר כו׳ בלי… הלכה?")
        self.assertEqual(build_profile(TextConfig(), "eval")(raw), "אמר ליה כי קאמר שמואל אשר כו בלי הלכה")

    def test_geresh_is_kept_in_train_and_dropped_in_eval(self) -> None:
        self.assertEqual(build_profile(TextConfig(), "train")("גמ׳ כו'"), "גמ׳ כו׳")
        self.assertEqual(build_profile(TextConfig(), "eval")("גמ׳ כו'"), "גמ כו")

    def test_2_idempotence_on_fixed_and_random_texts(self) -> None:
        alphabet = list("אבגדהוזחטיכלמנסעפצקרשת ךםןףץ,.?!…׳״'\"()[]־-<|>0123456789abcXYZ‎ְּ\n\t")
        rng = random.Random(0)
        texts = ["", "   ", "<|1.0|>", "([a]) [b (c) d]", "a <|0.1|> b", *("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(300))]
        for profile in ("train", "eval"):
            fn = build_profile(TextConfig(), profile)
            for text in texts:
                once = fn(text)
                self.assertEqual(fn(once), once, (profile, text))
        for mode in ("with_content", "keep"):
            fn = build_profile(conf(bracket={"mode": mode}), "eval")
            for text in texts:
                self.assertEqual(fn(fn(text)), fn(text), (mode, text))

    def test_3_same_config_same_results(self) -> None:
        raw = "<|0.1|>שלום, [עולם] <|1.0|>"
        a = [build_profile(TextConfig(), p)(raw) for p in ("train", "eval")]
        b = [build_profile(TextConfig(), p)(raw) for p in ("train", "eval")]
        self.assertEqual(a, b)

    def test_8_eval_profile_gives_equal_text_for_equal_words(self) -> None:
        eval_profile = build_profile(TextConfig(), "eval")
        reference = "אמר רבא: שלום, עולם. [כו׳]"
        hypothesis = "אמר רבא שלום עולם כו"
        self.assertEqual(eval_profile(reference), eval_profile(hypothesis))
        self.assertEqual(eval_profile("שָׁלוֹם"), eval_profile("שלום"))

    def test_bracket_mode_per_profile_recipe_for_non_speech_marker(self) -> None:
        config = conf(profiles={"train": {"steps": ["strip_timestamps", "remove_bracket_markup", "collapse_whitespace"], "bracket_mode": "keep"},
                                "eval": {"steps": ["strip_timestamps", "remove_bracket_markup", "collapse_whitespace"], "bracket_mode": "with_content"}})  # fmt: skip
        raw = "שלום [רעש] עולם"
        self.assertEqual(build_profile(config, "train")(raw), "שלום [רעש] עולם")
        self.assertEqual(build_profile(config, "eval")(raw), "שלום עולם")

    def test_10_step_order_matters_timestamps_before_punctuation(self) -> None:
        punct = RemovePunctuation(TextConfig().punctuation.chars, TextConfig().punctuation.space_chars)
        strip = StripTimestampTokens(re.compile(r"<\|(\d+\.\d+)\|>"))  # a token with a decimal point
        raw = "<|0.44|>שלום<|6.52|>"
        wrong_order = strip.apply(punct.apply(raw))  # punctuation first removes the '.', the token no longer matches
        self.assertEqual(wrong_order, "<|044|>שלום<|652|>")  # residue left in the text
        self.assertEqual(punct.apply(strip.apply(raw)), "שלום")  # the configured order is clean

    def test_custom_eval_steps_and_changed_chars_come_from_config_only(self) -> None:
        config = conf(punctuation={"chars": "!", "space_chars": ""}, profiles={"eval": {"steps": ["remove_punctuation"]}})
        self.assertEqual(build_profile(config, "eval")("שלום! עולם."), "שלום עולם.")

    def test_unknown_step_and_profile_fail_listing_names(self) -> None:
        with self.assertRaises(TextNormalizerError) as ctx:
            build_profile(conf(profiles={"eval": {"steps": ["unicode_clean", "magic"]}}), "eval")
        message = str(ctx.exception)
        self.assertIn("magic", message)
        self.assertIn("collapse_whitespace", message)
        self.assertIn("text.profiles.eval.steps", message)
        with self.assertRaises(TextNormalizerError) as ctx2:
            build_profile(TextConfig(), "test")
        self.assertIn("train, eval", str(ctx2.exception))
        self.assertEqual(len(default_registry().names()), 8)


class TimestampTests(unittest.TestCase):
    def parse(self, text: str) -> tuple[bool, object]:
        return parse_segments(text, TOKEN)

    def test_4_segments_are_extracted(self) -> None:
        had, segments = self.parse("<|0.44|>אמר ליה. <|6.52|><|7.02|>כן <|8.42|><|9.00|>")
        self.assertTrue(had)
        self.assertEqual([(s.start_sec, s.end_sec, s.text) for s in segments], [(0.44, 6.52, "אמר ליה. "), (7.02, 8.42, "כן ")])  # type: ignore[union-attr]

    def test_4_final_token_without_text_is_normal(self) -> None:
        had, segments = self.parse("<|0.0|>שלום <|2.0|>")
        self.assertEqual((had, len(segments)), (True, 1))  # type: ignore[arg-type]

    def test_4_no_tokens(self) -> None:
        self.assertEqual(self.parse("שלום"), (False, None))

    def test_4_malformed_tokens_give_none(self) -> None:
        cases = [
            "טקסט לפני <|0.0|>שלום <|1.0|>",  # text before the first token
            "<|0.0|>שלום",  # missing end token
            "<|2.0|>שלום <|1.0|>",  # end before start
            "<|0.0|>א <|5.0|><|3.0|>ב <|6.0|>",  # next start before previous end
            "<|0.0|>א <|1.0|> אמצע <|2.0|>ב <|3.0|>",  # text between an end and the next start
            "<|0.0|>א <|1.0|> זנב",  # text after the last end
        ]
        for text in cases:
            with self.subTest(text=text):
                had, segments = self.parse(text)
                self.assertTrue(had)
                self.assertIsNone(segments)

    def test_4_malformed_tokens_flagged_and_still_stripped(self) -> None:
        result = TextNormalizer(TextConfig()).normalize([sample("a", "<|2.0|>שלום <|1.0|>")])
        item = result.samples[0]
        self.assertIsNone(item.timestamp_segments)
        self.assertIn("bad_timestamps", item.flags)
        self.assertEqual(item.eval_text, "שלום")
        self.assertEqual(item.train_text, "שלום")


class NormalizeTests(unittest.TestCase):
    def data(self) -> list[AudioSample]:
        return [
            sample("a", "<|0.0|>אמר, [אשר] <|2.0|>"),
            sample("b", "<|0.0|>hello 123 (x) <|2.0|>"),
            sample("c", "<|0.0|><|1.0|>"),
            sample("d", "שָׁלוֹם כו׳‎"),
            sample("e", "..."),
        ]

    def test_5_flags_counts_and_inventory(self) -> None:
        result = TextNormalizer(TextConfig()).normalize(self.data())
        flags = {s.sample.id: set(s.flags) for s in result.samples}
        self.assertEqual(flags["a"], {"had_timestamps", "had_brackets"})
        self.assertEqual(flags["b"], {"had_timestamps", "had_brackets", "had_digits", "had_latin"})
        self.assertEqual(flags["c"], {"had_timestamps", "empty_after_normalization"})
        self.assertEqual(flags["d"], {"had_nikud"})
        self.assertEqual(flags["e"], {"empty_after_normalization"})
        self.assertTrue(set().union(*flags.values()) <= set(FLAGS))
        report = result.report
        self.assertEqual(report.flag_counts["had_timestamps"], 3)
        self.assertEqual(report.flag_counts["empty_after_normalization"], 2)
        self.assertEqual(report.samples, 5)
        self.assertEqual(report.eval_version, 1)
        self.assertEqual(report.profile_steps["train"][0], "unicode_clean")

    def test_5_character_inventory_has_names_and_counts_per_profile(self) -> None:
        report = TextNormalizer(TextConfig()).normalize(self.data()).report
        by_char = {c.char: c for c in report.inventory}
        self.assertNotIn("1", by_char)  # digits are not listed (flagged separately)
        self.assertNotIn("א", by_char)  # Hebrew letters are not listed
        self.assertNotIn("<", by_char)  # timestamp tokens are masked
        dot = by_char["."]
        self.assertEqual((dot.raw, dot.train, dot.eval, dot.samples), (3, 3, 0, 1))
        self.assertEqual(dot.name, "FULL STOP")
        self.assertEqual(dot.codepoint, "U+002E")
        geresh = by_char["׳"]
        self.assertEqual((geresh.raw, geresh.train, geresh.eval), (1, 1, 0))
        self.assertIn("HEBREW PUNCTUATION GERESH", geresh.name)
        lrm = by_char["‎"]
        self.assertEqual((lrm.raw, lrm.train, lrm.eval), (1, 0, 0))
        self.assertIn("LEFT-TO-RIGHT MARK", lrm.name)
        self.assertIn("HEBREW POINT", [c.name for c in report.inventory if c.char == "ָ"][0])
        self.assertEqual(report.inventory_total, len(report.inventory))
        self.assertIn("FULL STOP", report.summary())

    def test_inventory_top_n_limits_the_listing_but_not_the_total(self) -> None:
        report = TextNormalizer(conf(inventory={"top_n": 2})).normalize(self.data()).report
        self.assertEqual(len(report.inventory), 2)
        self.assertGreater(report.inventory_total, 2)

    def test_changed_counts_per_profile(self) -> None:
        report = TextNormalizer(TextConfig()).normalize([sample("a", "שלום עולם"), sample("b", "שלום, עולם")]).report
        self.assertEqual(dict(report.changed), {"train": 0, "eval": 1})

    def test_6_empty_samples_are_flagged_and_not_dropped(self) -> None:
        result = TextNormalizer(TextConfig()).normalize(self.data())
        self.assertEqual(len(result.samples), 5)
        self.assertEqual(result.report.empty_ids["eval"], ("c", "e"))
        self.assertEqual(result.report.empty_ids["train"], ("c",))
        self.assertEqual(result.samples[4].eval_text, "")
        self.assertEqual(result.samples[4].train_text, "...")

    def test_9_samples_are_unchanged_and_order_is_preserved(self) -> None:
        data = self.data()
        before = [(s.id, s.text, s.group_id, s.duration_sec, dict(s.extra)) for s in data]
        result = TextNormalizer(TextConfig()).normalize(data)
        self.assertEqual(before, [(s.id, s.text, s.group_id, s.duration_sec, dict(s.extra)) for s in data])
        self.assertEqual([n.sample.id for n in result.samples], [s.id for s in data])
        self.assertTrue(all(n.sample is s for n, s in zip(result.samples, data)))
        self.assertEqual(result.samples[0].sample.text, "<|0.0|>אמר, [אשר] <|2.0|>")

    def test_timestamp_segments_are_kept_in_the_normalized_sample(self) -> None:
        item = TextNormalizer(TextConfig()).normalize([sample("a", "<|0.5|>שלום <|2.0|><|2.5|>")]).samples[0]
        self.assertEqual([(s.start_sec, s.end_sec) for s in item.timestamp_segments or ()], [(0.5, 2.0)])

    def test_flags_ignore_the_digits_and_brackets_inside_timestamp_tokens(self) -> None:
        item = TextNormalizer(TextConfig()).normalize([sample("a", "<|0.44|>שלום <|6.52|>")]).samples[0]
        self.assertEqual(set(item.flags), {"had_timestamps"})

    def test_11_no_file_access_and_no_network_imports(self) -> None:
        with mock.patch("builtins.open", side_effect=AssertionError("no file access")), mock.patch(
            "pathlib.Path.open", side_effect=AssertionError("no file access")
        ):
            TextNormalizer(TextConfig()).normalize(self.data())
        for path in (ROOT / "src" / "stt" / "textnormalizer").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotIn(
                        name.split(".")[0],
                        {"requests", "urllib3", "httpx", "huggingface_hub", "socket", "tools", "soundfile", "pyarrow", "scipy", "torch", "transformers"},
                        f"{path} imports {name}",
                    )


class ConfigTests(unittest.TestCase):
    def load(self, *overrides: str) -> TextConfig:
        config = load_config(overrides=list(overrides), registry=SectionRegistry({"text": TextConfig}), environ={}, env_file=None, target_environ={})
        return config.section("text", TextConfig)

    def test_defaults_match_the_brief(self) -> None:
        c = self.load()
        self.assertEqual(c.eval_version, 1)
        self.assertEqual(c.profiles.train.steps, ("unicode_clean", "strip_timestamps", "unify_quotes", "remove_bracket_markup", "collapse_whitespace"))
        self.assertEqual(c.profiles.eval.steps[:3], ("unicode_clean", "strip_timestamps", "remove_punctuation"))
        self.assertEqual((c.bracket.mode, c.bracket.pairs, c.train.timestamps, c.inventory.top_n), ("chars_only", (("[", "]"), ("(", ")")), "strip", 50))

    def test_timestamps_keep_fails_with_explanation(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load("text.train.timestamps=keep")
        self.assertIn("not implemented in v1", str(ctx.exception))

    def test_partial_profile_override_keeps_the_other_profile(self) -> None:
        c = self.load('text.profiles.eval.steps=["unicode_clean"]')
        self.assertEqual(c.profiles.eval.steps, ("unicode_clean",))
        self.assertEqual(c.profiles.train.steps[0], "unicode_clean")
        self.assertEqual(len(c.profiles.train.steps), 5)

    def test_invalid_values_fail(self) -> None:
        for bad, expected in (
            ("text.bracket.mode=erase", "bracket.mode"),
            ('text.bracket.pairs=[["[[","]"]]', "single characters"),
            ("text.timestamp_pattern=(", "regular expression"),
            ("text.timestamp_pattern=<a>", "capture group"),
            ("text.eval_version=0", "eval_version"),
            ("text.profiles.eval.bracket_mode=x", "bracket_mode"),
            ("text.nonsense=1", "Unknown key"),
        ):
            with self.assertRaises(ConfigError, msg=bad) as ctx:
                self.load(bad)
            self.assertIn(expected, str(ctx.exception))

    def test_shipped_example_config_equals_the_defaults(self) -> None:
        config = load_config(ROOT / "configs" / "text_default.yaml", registry=SectionRegistry({"text": TextConfig}), environ={}, env_file=None, target_environ={})
        self.assertEqual(config.section("text", TextConfig), TextConfig())


if __name__ == "__main__":
    unittest.main()
