"""The `select` configuration section, owned by the Selector component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.selector.config import SelectConfig
    config = load_config(path, registry=SectionRegistry({"select": SelectConfig}))
    select = config.section("select", SelectConfig)
Example YAML with every key and its DEFAULT: configs/select_default.yaml. The level of evidence of the comparisons (90 percent, one-sided) is set in the
`eval.significance` keys of the Evaluator, not here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from stt.config import SectionModel


class SelectConfig(SectionModel):
    """The `select` section."""

    min_verdict: Literal["improved", "no_clear_difference"] = Field(
        default="improved", description="DEFAULT improved. The lowest Val verdict against the baseline that makes a candidate eligible (the lenient level of evidence is set in eval.significance)."
    )
    tie_epsilon: float = Field(default=0.001, ge=0.0, description="DEFAULT 0.001 (absolute WER). Val WERs closer than this are a tie; the lower Val loss wins the tie.")
    on_insufficient_groups: Literal["plain_difference", "reject"] = Field(
        default="plain_difference", description="DEFAULT plain_difference. With too few groups for a confidence statement: judge by the plain WER difference (flagged), or reject the candidate."
    )
    test_confirmation: bool = Field(default=True, description="DEFAULT true. Evaluate the Test ONCE, for the winner and the baseline. false: test_status not_evaluated.")
