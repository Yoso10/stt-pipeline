"""Typed, frozen configuration objects: SectionModel, RuntimeConfig, SectionRegistry, AppConfig.

How to run: this module is not a script. Example of use (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.config import load_config; print(load_config(env_file=None).runtime)"
Example output:
    online=False seed=42

Each later component owns the schema of its own config section: it subclasses `SectionModel`
and the composition root registers it in a `SectionRegistry`. The loader never has to be edited
when a component is added (decision 7 of the Config brief).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stt.config.errors import ConfigError

RESERVED_KEYS = frozenset({"runtime", "extends"})

S = TypeVar("S", bound="SectionModel")


class SectionModel(BaseModel):
    """Base class of every config section: frozen (read-only) and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RuntimeConfig(SectionModel):
    """The `runtime` section, defined by the Config component.

    Keys: `online` (explicit switch for running at home; the pipeline is offline by default)
    and `seed` (applied everywhere for determinism).
    """

    online: bool = Field(default=False, description="True only at home: allows online mode. Offline by default.")
    seed: int = Field(default=42, description="DEFAULT 42. Seed applied everywhere for deterministic runs.")


class SectionRegistry:
    """The set of config sections known to the loader: name -> schema class (read-only).

    Created by the composition root; there is no global registry.
    Errors: ConfigError when a name is reserved or a class is not a SectionModel.
    """

    def __init__(self, sections: Mapping[str, type[SectionModel]] | None = None) -> None:
        checked: dict[str, type[SectionModel]] = {}
        for name, model in (sections or {}).items():
            if name in RESERVED_KEYS or not name.isidentifier() or name != name.lower():
                raise ConfigError(
                    f"Cannot register config section '{name}': use a lower-case identifier "
                    f"that is not reserved ({sorted(RESERVED_KEYS)})."
                )
            if not (isinstance(model, type) and issubclass(model, SectionModel)):
                raise ConfigError(f"Config section '{name}' must be a subclass of SectionModel (frozen, no unknown keys).")
            checked[name] = model
        self._sections: Mapping[str, type[SectionModel]] = MappingProxyType(checked)

    @property
    def sections(self) -> Mapping[str, type[SectionModel]]:
        """The registered sections (name -> class)."""
        return self._sections

    def with_section(self, name: str, model: type[SectionModel]) -> SectionRegistry:
        """Return a NEW registry that also contains `name`."""
        return SectionRegistry({**self._sections, name: model})

    def __iter__(self) -> Iterator[str]:
        return iter(self._sections)


class AppConfig(BaseModel):
    """The whole resolved configuration: read-only.

    `runtime` is always present. `sections` holds the validated section objects of the
    registered components. Use `section(name, SchemaClass)` to read one with its type.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    runtime: RuntimeConfig = RuntimeConfig()
    sections: Mapping[str, SectionModel] = MappingProxyType({})

    @field_validator("sections", mode="after")
    @classmethod
    def _freeze_sections(cls, value: Mapping[str, SectionModel]) -> Mapping[str, SectionModel]:
        return MappingProxyType(dict(value))

    def section(self, name: str, schema: type[S]) -> S:
        """Return section `name` as an instance of `schema`.

        Errors: ConfigError when the section is not loaded or has another type.
        """
        found = self.sections.get(name)
        if found is None:
            raise ConfigError(f"Config section '{name}' is not loaded. Register its schema in the SectionRegistry.")
        if not isinstance(found, schema):
            raise ConfigError(f"Config section '{name}' has type {type(found).__name__}, not {schema.__name__}.")
        return found
