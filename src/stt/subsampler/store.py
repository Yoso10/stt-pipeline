"""SubsampleStore: saves and loads a subset as JSON (the Subsampler never touches files).

How to run: this module is not a script; `resolve_subsample` uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.subsampler import SubsampleStore, SubsampleSize, SubsampleSpec; s = SubsampleSpec(SubsampleSize('minutes', 30.0), 'by_sample', 'train_only', 'fixed', 42, 0, 'train', 'val', 10); print(SubsampleStore().path(Path('subsets'), s))"
Example output:
    subsets/minutes_30_by_sample_train_only_fixed_s42_r0.json   (printed with the separator of your system)
File format (JSON, UTF-8): schema_version, the spec, the data fingerprint, the fingerprint of the result, and the ids of every split. One file per spec.
A stored file is NEVER overwritten and NEVER reused when the spec or the data changed (the fingerprint of the current spec and data differs): the
load then FAILS and says what to do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from stt.subsampler.errors import SubsampleError
from stt.subsampler.subsampler import Splits, data_fingerprint, result_fingerprint, spec_document
from stt.subsampler.types import SubsampleSpec
from stt.textnormalizer import NormalizedSample

SCHEMA_VERSION = 1


def _label(spec: SubsampleSpec) -> str:
    size = "all" if spec.size.kind == "all" else f"{spec.size.kind}_{spec.size.value:g}"
    slice_part = f"_v{spec.val_samples}" if spec.val_samples is not None else ""
    return f"{size}_{spec.strategy}_{spec.scope}_{spec.val}{slice_part}_s{spec.seed}_r{spec.repeat_index}"


class SubsampleStore:
    """Persistence of a subset: `save`, `load`, `path`."""

    def path(self, directory: Path, spec: SubsampleSpec) -> Path:
        """The file of this spec inside `directory`."""
        return directory / f"{_label(spec)}.json"

    def exists(self, directory: Path, spec: SubsampleSpec) -> bool:
        """True when a file of this spec exists."""
        return self.path(directory, spec).is_file()

    def save(self, directory: Path, spec: SubsampleSpec, splits: Splits, chosen: Mapping[str, Sequence[str]]) -> Path:
        """Write the file (temp file and rename). Errors: SubsampleError when the file exists (never overwritten)."""
        target = self.path(directory, spec)
        if target.exists():
            raise SubsampleError(f"A subset is already stored in {target}. A stored subset is never overwritten: use another subsample.dir.")
        data_fp = data_fingerprint(splits)
        document = {
            "schema_version": SCHEMA_VERSION,
            "spec": spec_document(spec),
            "data_fingerprint": data_fp,
            "fingerprint": result_fingerprint(spec, data_fp, chosen),
            "splits": {name: list(ids) for name, ids in chosen.items()},
        }
        directory.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, directory: Path, spec: SubsampleSpec, splits: Splits) -> dict[str, list[NormalizedSample]]:
        """Read the stored subset of this spec and return the splits made from the CURRENT samples.

        Errors: SubsampleError when the file is damaged, the spec or the data differ from the stored ones (stale), or a stored id does not exist.
        """
        path = self.path(directory, spec)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            ids: dict[str, list[str]] = document["splits"]
            stored_spec, stored_fp, stored_data = document["spec"], document["fingerprint"], document["data_fingerprint"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SubsampleError(f"The stored subset {path} cannot be read ({type(exc).__name__}: {exc}). Delete the file to make a new one, or use another subsample.dir.") from exc
        data_fp = data_fingerprint(splits)
        if stored_spec != spec_document(spec) or stored_data != data_fp or stored_fp != result_fingerprint(spec, data_fp, ids):
            raise SubsampleError(
                f"The stored subset {path} is STALE: the spec or the data changed since it was made (the fingerprints differ). "
                "It is never reused silently: delete the file to make a new subset, or use another subsample.dir."
            )
        result: dict[str, list[NormalizedSample]] = {}
        for name, samples in splits.items():
            wanted = set(ids.get(name, [n.sample.id for n in samples]))
            present = {n.sample.id for n in samples}
            if not wanted <= present:
                raise SubsampleError(f"The stored subset {path} names {len(wanted - present)} samples that are not in the split '{name}' (for example {sorted(wanted - present)[:3]}). Make a new subset.")
            result[name] = [n for n in samples if n.sample.id in wanted]
        return result
