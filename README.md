# GLM-5.3-Flash on SM120 (RTX PRO 6000 Blackwell) — vLLM fork

<!-- markdownlint-disable MD060 -->

> English version: [`README_EN.md`](README_EN.md)
>
> Upstream vLLM README: [`README_UPSTREAM.md`](README_UPSTREAM.md)
>
> 本仓库基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)，选择性合入 GLM-5.3-Flash / PR #53906 相关改动，并保留 SM120 上的 FlashInfer NoPE sparse MLA 适配。

把 vLLM 的 **GLM-5.3-Flash** 推理扩展到 **SM120**（RTX PRO 6000 Blackwell / 服务器 Blackwell 平台）。当前代码已按 **4 张 RTX PRO 6000**、TP4、FP8 KV cache、MTP 5 tokens、CUDA Graph 的配置验证过长上下文、prefix caching、serving 和数值测试。

## Changelog

### 2026-08-28

- 选择性合入 GLM-5.3-Flash / PR #53906 的模型入口、hybrid cache、kpool sparse indexer、MTP 以及 serving 相关更新。
- 保留并验证 SM120 FlashInfer NoPE sparse MLA 路径，包括 `FLASHINFER_MLA_SPARSE_SM120`、`has_flashinfer_sparse_mla_sm120()` 和 `has_flashinfer_sparse_mla_sm120_glm_nope()`。
- 当前启动配置为 `max-num-batched-tokens=8192`、`enable-prefix-caching`、`block-size=2304`、`gpu-memory-utilization=0.97`、`max-num-seqs=4`、`max-model-len=auto`。
- 已验证 256K / 784K 长上下文、prefix cache 命中、MTP 5 tokens、以及 4 卡 serve 的启动与稳定运行。
- 本 Release 提供配套的 vLLM 与 FlashInfer wheel，不包含模型权重。

### 代码来源

| 内容 | 提交 |
|---|---|
| vLLM PR #53906 | `933876c388` |
| 本 fork 的 SM120 集成 | `da2f75cdd9` |
| FlashInfer SM120 NoPE kernel | `b338a943` |

## 背景:为什么需要这个 fork

GLM-5.3-Flash 的 NoPE / sparse MLA 路径需要同时满足以下约束:

- GLM NoPE 的 `qk_rope_head_dim=0` 分支不能误走普通 MLA / RoPE 路径。
- SM120 上的 FlashInfer sparse MLA 需要专门的 backend 选择与 warmup。
- kpool sparse indexer、状态页大小、以及 FlashInfer / DeepGEMM 的 page 对齐必须同时成立。
- MTP 与 prefix caching 会影响实际的长上下文吞吐和可用容量，需要在 README 中明确可复现配置。

### 关键代码路径

- `vllm/platforms/cuda.py` - SM120 backend 选择和 GLM NoPE 分发
- `vllm/utils/flashinfer.py` - FlashInfer 能力检测
- `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` - kpool sparse indexer
- `vllm/models/glm5next/nvidia/attention.py` - GLM-5.3-Flash attention path
- `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py` - backend / API coverage

## 已验证环境

| 项 | 版本 |
|---|---|
| GPU | 4× RTX PRO 6000 Blackwell |
| 平台 | SM120 |
| Python | 3.12.14 |
| CUDA toolkit | 13.0 |
| torch / Triton | 2.13.0+cu130 / 3.7.1 |
| FlashInfer | flashinfer-python 0.6.18 wheel；cubin 0.6.17 |
| vLLM | GLM-5.3-Flash SM120 wheel |

## Wheel 安装

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

## 部署（vllm serve）

### 启动参数

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

## 测试结果（4× RTX PRO 6000）

### 稳定性与工具调用

- 256K 输入、512 输出：4/4 成功。
- 784K 输入、512 输出：4/4 成功。
- 8K 工具调用返回结构正确的 tool call，UTF-8 输出无乱码。
- 最终服务完成 full/piecewise prefill/decode CUDA Graph capture，`/health=200`，启动日志中 `ERROR=0`。

### 长上下文与 prefix caching

| 输入/输出 | 8192 chunk 基线 | 4096 chunk 稳态 | Prefill 变化 |
|---|---:|---:|---:|
| 256K / 512 | 27.865s；9,407.50 tok/s | 31.649s；8,282.96 tok/s | -11.95% |
| 784K / 512 | 101.473s；7,911.64 tok/s | 112.685s；7,124.45 tok/s | -9.95% |

4096 配置把自动 KV 容量从 822,528 提高到 **912,384 tokens**。重复相同 prompt 时：

| 输入 | Prefix 命中率 | 重复请求 TTFT |
|---|---:|---:|
| 256K | 98.4375% | 1.308s |
| 784K | 99.5855% | 3.117s |

### 启动配置

| 参数 | 值 |
|---|---|
| `max-num-batched-tokens` | `8192` |
| `enable-prefix-caching` | `true` |
| `block-size` | `2304` |
| `gpu-memory-utilization` | `0.97` |
| `max-num-seqs` | `4` |
| `max-model-len` | `auto` |

## Release 内容与限制

- Release 附件包含配套的 vLLM 与 FlashInfer wheel、`MANIFEST.json` 和
  `SHA256SUMS`；不包含模型权重或 JIT 编译缓存。
- `gpu-memory-utilization=0.98` 在长上下文 sparse-indexer profile/请求中出现过
  OOM，因此推荐 0.97，而不是把更高 auto capacity 当作稳定容量。
- 本轮未验证 DCP。
- 代码和测试包含 AI 辅助；发布者仍需理解并审查全部改动后再向上游提交 PR。

## 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)（Apache-2.0）及其 GLM-5.3-Flash 相关集成。
