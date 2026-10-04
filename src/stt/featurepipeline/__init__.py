"""stt.featurepipeline: model-ready examples (input features and label ids) from the kept samples (C4 FeaturePipeline).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C4):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline import FeaturesConfig; print(FeaturesConfig().max_dropped_ratio)"
Example output:
    0.05
Importing this package does not import torch, soundfile or scipy (the adapters are imported when they are needed).
"""

from stt.featurepipeline.cache import FeatureCache, extractor_signature
from stt.featurepipeline.config import CacheConfig, FeaturesConfig, resolve_seed
from stt.featurepipeline.dataset import Item, LazyFeatureDataset
from stt.featurepipeline.errors import AudioReadError, FeaturePipelineError
from stt.featurepipeline.pipeline import FeaturePipeline
from stt.featurepipeline.types import (
    AudioHeader,
    AudioReader,
    FeatureDataset,
    FeatureExample,
    FeatureReport,
    FeatureResult,
    LengthStats,
    LoadedAudio,
)

__all__ = [
    "AudioHeader",
    "AudioReadError",
    "AudioReader",
    "CacheConfig",
    "FeatureCache",
    "FeatureDataset",
    "FeatureExample",
    "FeaturePipeline",
    "FeaturePipelineError",
    "FeatureReport",
    "FeatureResult",
    "FeaturesConfig",
    "Item",
    "LazyFeatureDataset",
    "LengthStats",
    "LoadedAudio",
    "extractor_signature",
    "resolve_seed",
]
