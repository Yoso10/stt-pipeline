"""stt.modelloader: load one Whisper model, its feature extractor and its tokenizer from a LOCAL folder (C5 ModelLoader).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C5):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelloader import default_registry; print(default_registry().names())"
Example output:
    ['hf_whisper']
Importing this package does not import torch or transformers.
"""

from stt.modelloader.config import ModelConfig
from stt.modelloader.errors import ModelLoaderError
from stt.modelloader.registry import LoaderRegistry, create_loader, default_registry
from stt.modelloader.token_counter import make_token_counter
from stt.modelloader.types import FeatureExtractor, LabelTokenizer, LoadedModel, ModelInfo, ModelLoader
from stt.modelloader.whisper_folder import check_model_folder

__all__ = [
    "FeatureExtractor",
    "LabelTokenizer",
    "LoadedModel",
    "LoaderRegistry",
    "ModelConfig",
    "ModelInfo",
    "ModelLoader",
    "ModelLoaderError",
    "check_model_folder",
    "create_loader",
    "default_registry",
    "make_token_counter",
]
