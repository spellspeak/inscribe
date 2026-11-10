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

# Export: llama.cpp's own prebuilt image at one build, with the converter from source at the commit that build was
# made from. The converter keeps its own, older Python stack.
LLAMA_BUILD = "b11146"
LLAMA_COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"
_LLAMA = "/opt/llama.cpp"
EXPORT = ImageSpec(
    base=f"ghcr.io/ggml-org/llama.cpp:full-cuda-{LLAMA_BUILD}",
    clear_entrypoint=True,
    apt=("git",),
    run=(f"git init {_LLAMA}",
         f"cd {_LLAMA} && git remote add origin https://github.com/ggml-org/llama.cpp && git fetch --depth 1 origin {LLAMA_COMMIT} "
         f'&& git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "{LLAMA_COMMIT}"',
         f"python -m pip install --no-cache-dir -r {_LLAMA}/requirements/requirements-convert_hf_to_gguf.txt numpy==2.2.6 sentencepiece==0.2.1 "
         "protobuf==4.25.8 tokenizers==0.22.2 huggingface-hub==0.36.2 safetensors==0.8.0 requests==2.34.2 " + " ".join(_OWN),
         f"python -m pip install --no-cache-dir --no-deps {_LLAMA}/gguf-py",
         "test -x /app/llama-server && test -x /app/llama-quantize && test -x /app/llama-imatrix"),
    env={"HF_HUB_DISABLE_TELEMETRY": "1", "LD_LIBRARY_PATH": "/app:/usr/local/cuda/lib64"},
)
