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
- Selected `max-num-batched-tokens=8192`, `block-size=2304`,
  `gpu-memory-utilization=0.97`, `max-num-seqs=4`, and
  `max-model-len=auto` as the stable tested configuration.
- This release provides paired vLLM and FlashInfer wheels. It does not include
  model weights and must be paired with the
  [`glm53-flash-nope-sm120`](https://github.com/yhfgyyf/flashinfer/tree/glm53-flash-nope-sm120)
  FlashInfer branch.

### Source provenance

| Component | Commit |
| --- | --- |
| vLLM PR #53906 | `933876c388` |
| Validated vLLM SM120 integration | `da2f75cdd9` |
| FlashInfer SM120 NoPE kernel | `b338a943` |

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
| FlashInfer | flashinfer-python 0.6.18 wheel; cubin 0.6.17 |
| vLLM runtime | 0.27.2rc1.dev53906 wheel |

## Install from GitHub Release wheels

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install torch==2.13.0 --torch-backend=cu130

gh release download --repo yhfgyyf/vllm-GLM-5.3-Flash-sm120 \
  --pattern 'flashinfer_python-*.whl' \
  --pattern 'vllm-*.whl' \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-glm53-release

cd /tmp/vllm-glm53-release
sha256sum -c SHA256SUMS
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install ./vllm-*.whl \
  --torch-backend=cu130
```

## Serve the model

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
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
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

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

## Release contents and limitations

- Release assets contain paired vLLM and FlashInfer wheels and `SHA256SUMS`;
  they do not contain model weights or a compiled JIT cache.
- `gpu-memory-utilization=0.98` hit OOM during long-context sparse-indexer
  profiling/requests, so 0.97 is the stable recommendation.
- AI assistance was used for code and test work. A human submitter must still
  understand and review every change before proposing it upstream.

## License and upstream

This fork is based on
[vllm-project/vllm](https://github.com/vllm-project/vllm), licensed under
Apache-2.0, with the GLM-5.3-Flash integration layered on top.
