"""Select a ModelLoader by name (registry); the Hugging Face adapter is imported only when it is used.

How to run: this module is not a script; the composition root calls `create_loader`. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelloader import default_registry; print(default_registry().names())"
Example output:
    ['hf_whisper']
Importing this module does not import torch or transformers (the adapter is imported inside the factory).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from stt.modelloader.config import ModelConfig
from stt.modelloader.errors import ModelLoaderError
from stt.modelloader.types import ModelLoader

LoaderFactory = Callable[[ModelConfig], ModelLoader]


def _hf_whisper(config: ModelConfig) -> ModelLoader:
    from stt.modelloader.hf_whisper import HfWhisperLoader  # imported only when this loader is chosen

    return HfWhisperLoader(config)


class LoaderRegistry:
    """A read-only name -> factory mapping."""

    def __init__(self, factories: Mapping[str, LoaderFactory]) -> None:
        self._factories: Mapping[str, LoaderFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered loader names, sorted."""
        return sorted(self._factories)

    def create(self, config: ModelConfig) -> ModelLoader:
        """Create the loader named by `config.loader`. Errors: ModelLoaderError listing the registered names."""
        factory = self._factories.get(config.loader)
        if factory is None:
            raise ModelLoaderError(f"Unknown model.loader '{config.loader}'. Registered loaders: {', '.join(self.names())}.")
        return factory(config)


def default_registry() -> LoaderRegistry:
    """The registry with the built-in loader `hf_whisper`."""
    return LoaderRegistry({"hf_whisper": _hf_whisper})


def create_loader(config: ModelConfig, registry: LoaderRegistry | None = None) -> ModelLoader:
    """Create the ModelLoader selected by `model.loader` (default registry unless one is given)."""
    return (registry or default_registry()).create(config)
