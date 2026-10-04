"""A plain token counter built from a loaded tokenizer, for the composition root to give to other components.

How to run: this module is not a script. Example (PowerShell, project root; `loaded` is a LoadedModel):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.modelloader import ModelConfig, create_loader, make_token_counter; c = make_token_counter(create_loader(ModelConfig(path=Path('models/whisper-tiny'))).load().tokenizer); print(c('שלום עולם'))"
The result is a `Callable[[str], int]` that counts text tokens without special tokens. The SampleFilter (rule
`max_label_tokens`) takes such a callable and imports no tokenizer itself.
"""

from __future__ import annotations

from collections.abc import Callable

from stt.modelloader.types import LabelTokenizer


def make_token_counter(tokenizer: LabelTokenizer) -> Callable[[str], int]:
    """Return `text -> number of tokens (no special tokens)` using `tokenizer`."""
    return tokenizer.count_tokens
