# GLM-5.3-Flash on SM120 (RTX PRO 6000 Blackwell) — vLLM fork

<!-- markdownlint-disable MD060 -->

> Chinese version: [`README.md`](README.md)
>
> Upstream vLLM README: [`README_UPSTREAM.md`](README_UPSTREAM.md)

This repository is based on
[vllm-project/vllm](https://github.com/vllm-project/vllm). It selectively
integrates the GLM-5.3-Flash changes from PR #53906 and pairs them with the
FlashInfer NoPE sparse MLA implementation for SM120.

The validated configuration uses **4× RTX PRO 6000 Blackwell**, TP4, FP8 KV
cache, five MTP tokens, CUDA Graph, prefix caching, and long-context serving.

## Changelog

### 2026-08-28

- Integrated the GLM-5.3-Flash model entry, hybrid cache, kpool sparse
  indexer, MTP, and serving changes from PR #53906.
- Preserved and tested the FlashInfer SM120 GLM NoPE sparse MLA path,
  including `FLASHINFER_MLA_SPARSE_SM120` and its capability checks.
- Validated 256K and 784K inputs with 512 output tokens, prefix-cache reuse,
  MTP with five tokens, CUDA Graph capture, and structured tool calls.
- Selected `max-num-batched-tokens=4096`, `block-size=2304`,
  `gpu-memory-utilization=0.97`, `max-num-seqs=4`, and
  `max-model-len=auto` as the stable tested configuration.
- This release is source-only. It does not include wheels or model weights
  and must be paired with the
  [`glm53-flash-nope-sm120`](https://github.com/yhfgyyf/flashinfer/tree/glm53-flash-nope-sm120)
  FlashInfer branch.

### Source provenance

| Component | Commit |
| --- | --- |
| vLLM PR #53906 | `933876c388` |
| Validated vLLM SM120 integration | `da2f75cdd9` |
| FlashInfer SM120 NoPE kernel | `b338a943` |
| FlashInfer TP8/H8 operator tests | `def89fa6` |

## Why this fork exists

GLM-5.3-Flash NoPE / sparse MLA serving requires these constraints to hold
together:

- The GLM NoPE path with `qk_rope_head_dim=0` must not fall through to the
  regular MLA / RoPE implementation.
- FlashInfer sparse MLA on SM120 needs explicit backend selection and warmup.
- The kpool sparse indexer, hybrid-cache page size, and FlashInfer / DeepGEMM
  page geometry must agree.
- MTP and prefix caching affect long-context throughput and usable KV capacity.

Important code paths include:

- `vllm/platforms/cuda.py` — SM120 backend selection and GLM NoPE routing
- `vllm/utils/flashinfer.py` — FlashInfer capability checks
- `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` — kpool indexer
- `vllm/models/glm5next/nvidia/attention.py` — GLM-5.3-Flash attention
- `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py` — API tests

## Validated environment

| Item | Version |
| --- | --- |
| GPU | 4× RTX PRO 6000 Blackwell |
| Architecture | SM120 |
| Python | 3.12.14 |
| CUDA toolkit | 13.0 |
| PyTorch / Triton | 2.13.0+cu130 / 3.7.1 |
| FlashInfer | flashinfer-python 0.6.18 source; cubin 0.6.17 |
| vLLM runtime | 0.27.2rc1.dev53906+precompiled plus this source tree |

## Download the paired source release

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download --repo yhfgyyf/vllm-GLM-5.3-Flash-sm120 \
  --pattern 'vllm-*-source-*.tar.gz' \
  --pattern 'flashinfer-*-source-*.tar.gz' \
  --pattern MANIFEST.json \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-glm53-release

cd /tmp/vllm-glm53-release
sha256sum -c SHA256SUMS
tar -xzf vllm-*-source-*.tar.gz
tar -xzf flashinfer-*-source-*.tar.gz
```

The attached tarballs are source archives, not prebuilt wheels. GitHub's
automatically generated “Source code” archive contains only vLLM; use both
attached tarballs for the tested vLLM / FlashInfer pairing.

## Install from source

### Create the environment

```bash
uv venv --python 3.12
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install torch==2.13.0 --torch-backend=cu130
```

### Clone and install the paired branches

```bash
git clone --branch glm53-flash-nope-sm120 --recursive \
  https://github.com/yhfgyyf/flashinfer.git flashinfer-glm53-sm120
git clone https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120.git

cd flashinfer-glm53-sm120
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install -r requirements.txt
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install --no-build-isolation -e . -v

cd ../vllm-GLM-5.3-Flash-sm120
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

export VLLM_GLM53_SRC="$PWD"
export FLASHINFER_GLM53_SRC="$(cd ../flashinfer-glm53-sm120 && pwd)"
export PYTHONPATH="$FLASHINFER_GLM53_SRC:$VLLM_GLM53_SRC"
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

For a full vLLM CUDA-extension build, follow
[`docs/contributing/incremental_build.md`](docs/contributing/incremental_build.md).
This repository does not contain a `build_wheel.sh` script.

## Operator sanity check

```python
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_sparse_mla_sm120,
    has_flashinfer_sparse_mla_sm120_glm_nope,
)

print("cap:", current_platform.get_device_capability())
print("flashinfer sparse MLA SM120:", has_flashinfer_sparse_mla_sm120())
print("flashinfer GLM NoPE sparse MLA:", has_flashinfer_sparse_mla_sm120_glm_nope())
```

## Serve the model

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/path/to/flashinfer-glm53-sm120:/path/to/vllm-GLM-5.3-Flash-sm120
export FLASHINFER_JIT_DIR=/path/to/writable/flashinfer-jit
export FLASHINFER_WORKSPACE_BASE=/path/to/writable/flashinfer-jit

vllm serve /path/to/GLM-5.3-Flash \
  --host 127.0.0.1 --port 8000 \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 2304 \
  --max-model-len auto \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

The startup log should select `FLASHINFER_MLA_SPARSE_SM120`, complete CUDA
Graph capture, and report a healthy endpoint.

## Validation results (4× RTX PRO 6000)

- 256K input / 512 output: 4/4 successful requests.
- 784K input / 512 output: 4/4 successful requests.
- The 8K tool-call case returned structured tool-call data with valid UTF-8.
- Full and piecewise prefill/decode CUDA Graph capture completed, and
  `/health` returned HTTP 200.

### Long-context throughput

| Input / output | 8192-token baseline | 4096-token steady state | Prefill change |
| --- | ---: | ---: | ---: |
| 256K / 512 | 27.865 s; 9,407.50 tok/s | 31.649 s; 8,282.96 tok/s | -11.95% |
| 784K / 512 | 101.473 s; 7,911.64 tok/s | 112.685 s; 7,124.45 tok/s | -9.95% |

The 4096-token configuration increased automatic KV capacity from 822,528 to
**912,384 tokens**. Repeating the same prompt produced:

| Input | Prefix-cache hit rate | Repeat-request TTFT |
| --- | ---: | ---: |
| 256K | 98.4375% | 1.308 s |
| 784K | 99.5855% | 3.117 s |

### TP8 / H8 status

The model has 64 global heads, so TP4 uses 16 heads per rank and TP8 would use
8. The paired FlashInfer branch passes H8 and H16 prefill/decode numerical
tests (`4 passed, 2 warnings in 29.01s`). The available host has four GPUs, so
this validates the H8 operator/API path, not an end-to-end TP8 deployment.

### Why `block-size=2304` is required for the tested TP4 setup

- vLLM raises `--block-size 256` to 1792, but `1792 × 528 = 946176` bytes is
  smaller than the 1,146,880-byte hybrid-cache state page, so KV-cache
  initialization fails.
- 2176 fits the state page, but `2176 / index_kpool(4) = 544` maps to
  `block_kv=32`; the SM120 DeepGEMM FP8 paged-MQA path accepts 64, so profiling
  fails.
- The current path needs a manager block divisible by `index_kpool × 64 = 256`.
  The smallest such value at or above `ceil(1146880 / 528) = 2173` is 2304.

The “mamba state page” wording comes from vLLM's generic cache abstraction and
does not mean that Kimi-K3-specific KDA state is present in GLM's MLA KV cache.
The 2304 result applies to this TP4/FP8 geometry and must be revalidated when
the tensor-parallel size changes.

## Release contents and limitations

- Release assets contain the paired vLLM and FlashInfer source tarballs,
  `MANIFEST.json`, and `SHA256SUMS`; they do not contain model weights, wheels,
  or a compiled JIT cache.
- `gpu-memory-utilization=0.98` hit OOM during long-context sparse-indexer
  profiling/requests, so 0.97 is the stable recommendation.
- DCP and end-to-end TP8 serving were not tested in this run.
- AI assistance was used for code and test work. A human submitter must still
  understand and review every change before proposing it upstream.

## License and upstream

This fork is based on
[vllm-project/vllm](https://github.com/vllm-project/vllm), licensed under
Apache-2.0, with the GLM-5.3-Flash integration layered on top.
