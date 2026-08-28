# GLM-5.3-Flash on SM120 (RTX PRO 6000 Blackwell) — vLLM fork

> Chinese version: [`README.md`](README.md)
> This repository is based on [vllm-project/vllm](https://github.com/vllm-project/vllm). It selectively integrates the GLM-5.3-Flash / PR #53906 changes and keeps the SM120 FlashInfer NoPE sparse MLA path.

This fork extends **GLM-5.3-Flash** serving to **SM120** (RTX PRO 6000 Blackwell and similar Blackwell server platforms). The current branch has been validated with a **4-GPU RTX PRO 6000** setup using TP4, FP8 KV cache, 5-token MTP, CUDA Graph, long-context serving, prefix caching, and operator-level checks.

## Changelog

### 2026-08-28

- Integrated the GLM-5.3-Flash / PR #53906 model entry, hybrid cache, kpool sparse indexer, MTP, and serving updates.
- Kept the SM120 FlashInfer NoPE sparse MLA path, including `FLASHINFER_MLA_SPARSE_SM120`, `has_flashinfer_sparse_mla_sm120()`, and `has_flashinfer_sparse_mla_sm120_glm_nope()`.
- The stable configuration is `max-num-batched-tokens=4096`, `enable-prefix-caching`, `block-size=2304`, `gpu-memory-utilization=0.97`, `max-num-seqs=4`, and `max-model-len=auto`.
- Verified 256K / 784K long-context runs, prefix-cache hits, MTP with 5 tokens, and stable 4-GPU serving.

## Why this fork exists

GLM-5.3-Flash NoPE / sparse MLA serving needs the following constraints to hold at the same time:

- The `qk_rope_head_dim=0` GLM NoPE path must not fall back to the regular MLA / RoPE path.
- SM120 requires backend selection and warmup specifically for FlashInfer sparse MLA.
- The kpool sparse indexer, state page sizing, and FlashInfer / DeepGEMM page alignment must all be consistent.
- MTP and prefix caching materially affect long-context throughput and usable capacity, so the README must document a reproducible serving setup.

### Main code paths

- `vllm/platforms/cuda.py` - SM120 backend selection and GLM NoPE routing
- `vllm/utils/flashinfer.py` - FlashInfer capability checks
- `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` - kpool sparse indexer
- `vllm/models/glm5next/nvidia/attention.py` - GLM-5.3-Flash attention path
- `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py` - backend/API coverage

## Verified environment

| Item | Version |
| --- | --- |
| GPU | 4× RTX PRO 6000 Blackwell |
| Platform | SM120 |
| Python | 3.12 |
| CUDA toolkit | 13.x |
| torch | cu130 series |
| FlashInfer | SM120 NoPE sparse MLA fork |
| vLLM | current GLM-5.3-Flash integration branch |

## Quick install

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download --repo yhfgyyf/vllm-GLM-5.3-Flash-sm120 \
  --archive tar.gz \
  --dir /tmp/vllm-glm53-release

tar -xzf /tmp/vllm-glm53-release/*.tar.gz -C /tmp/vllm-glm53-release
cd /tmp/vllm-glm53-release/vllm-GLM-5.3-Flash-sm120-*

UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

## Source install

```bash
uv venv --python 3.12
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install torch --torch-backend=cu130
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130

git clone https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120.git
cd vllm-GLM-5.3-Flash-sm120

export CUDA_HOME=/usr/local/cuda
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8

./build_wheel.sh
uv pip install --force-reinstall --no-deps dist/*.whl
```

## Sanity checks

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

## Serving

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm serve /path/to/GLM-5.3-Flash \
  --served-model-name glm53-flash \
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --port 8000
```

## Validation summary

| Input | Configuration | Result |
| --- | --- | --- |
| 256K | prefix caching on, chunked prefill 4096 | Stable startup, ~98.44% cache hit on repeat requests |
| 784K | prefix caching on, chunked prefill 4096 | Stable startup, ~99.59% cache hit on repeat requests |

## License / source

This fork is based on [vllm-project/vllm](https://github.com/vllm-project/vllm) under Apache-2.0, with GLM-5.3-Flash integration work layered on top.
