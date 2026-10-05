"""stt.registry: numbered, immutable model entries with a card, checksums and an index (C10 ModelRegistry).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C10):
    $env:PYTHONPATH = "src"
    python -c "from stt.registry import ModelRegistry, RegistryConfig; print(ModelRegistry(RegistryConfig()).list_entries())"
Example output (no registry folder yet):
    []
Importing this package imports no torch, no transformers and no network code.
"""

from stt.registry.card import (
    card_from_json,
    card_from_selection,
    card_to_json,
    render_markdown,
    render_review,
    summary_from_eval,
    test_set_fingerprint,
)
from stt.registry.config import RegistryConfig
from stt.registry.errors import RegistryError, RegistryVerifyError
from stt.registry.hashing import check_sums, sha256_file, write_sums
from stt.registry.index import COLUMNS, IndexStore
from stt.registry.registry import ModelRegistry
from stt.registry.types import (
    EvalSummary,
    FinalistRow,
    ModelCard,
    RegisterRequest,
    RegistryEntry,
    VerifyReport,
)

__all__ = [
    "COLUMNS",
    "EvalSummary",
    "FinalistRow",
    "IndexStore",
    "ModelCard",
    "ModelRegistry",
    "RegisterRequest",
    "RegistryConfig",
    "RegistryEntry",
    "RegistryError",
    "RegistryVerifyError",
    "VerifyReport",
    "card_from_json",
    "card_from_selection",
    "card_to_json",
    "check_sums",
    "render_markdown",
    "render_review",
    "sha256_file",
    "summary_from_eval",
    "test_set_fingerprint",
    "write_sums",
]
