"""The images the steps run in, as recipes. Every version is pinned: these are the versions the recipes were measured with."""
from __future__ import annotations

from inscribe.backends.base import ImageSpec, PipLayer

_ENV = {"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
_OWN = ("pydantic==2.13.5", "pyyaml==6.0.3")  # what this package itself imports inside a step

SMOKE = ImageSpec(pip=(PipLayer(_OWN),))

# Training: torch built for CUDA 12.6, then the Hugging Face stack.
TRAIN = ImageSpec(
    pip=(PipLayer(("torch==2.14.1",), index_url="https://download.pytorch.org/whl/cu126"),
         PipLayer(("transformers==5.18.0", "peft==0.21.2", "accelerate==1.15.0", "safetensors==0.8.0", "tokenizers==0.23.2",
                   "huggingface-hub==1.33.0", "numpy==2.2.6", *_OWN))),
    env=_ENV,
)
