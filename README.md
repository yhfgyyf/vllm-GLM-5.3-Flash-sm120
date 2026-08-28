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
- 推荐稳定配置为 `max-num-batched-tokens=4096`、`enable-prefix-caching`、`block-size=2304`、`gpu-memory-utilization=0.97`、`max-num-seqs=4`、`max-model-len=auto`。
- 已验证 256K / 784K 长上下文、prefix cache 命中、MTP 5 tokens、以及 4 卡 serve 的启动与稳定运行。
- 本 Release 只发布源码，不包含 wheel 或模型权重；必须搭配 [`yhfgyyf/flashinfer` 的 `glm53-flash-nope-sm120` 分支](https://github.com/yhfgyyf/flashinfer/tree/glm53-flash-nope-sm120)。

### 代码来源

| 内容 | 提交 |
|---|---|
| vLLM PR #53906 | `933876c388` |
| 本 fork 的 SM120 集成 | `da2f75cdd9` |
| FlashInfer SM120 NoPE kernel | `b338a943` |
| FlashInfer TP8/H8 测试 | `def89fa6` |

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
| FlashInfer | 0.6.18 源码分支；cubin 0.6.17 |
| vLLM | 0.27.2rc1.dev53906+precompiled + 本源码树 |

## 下载最新源码 Release

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

源码包不等于预编译 wheel；下载后仍需按下一节安装。GitHub 自动生成的
`Source code (zip/tar.gz)` 只包含 vLLM，Release 附件中的两个源码包才是本次验证的配套组合。

## 源码安装

### Python 环境

```bash
uv venv --python 3.12
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install torch==2.13.0 --torch-backend=cu130
```

### Clone 配套源码

```bash
git clone --branch glm53-flash-nope-sm120 --recursive \
  https://github.com/yhfgyyf/flashinfer.git flashinfer-glm53-sm120
git clone https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120.git
```

### 安装 FlashInfer 和 vLLM

```bash
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

完整编译 vLLM CUDA 扩展时，请遵循
[`docs/contributing/incremental_build.md`](docs/contributing/incremental_build.md)，本仓库不存在 `build_wheel.sh`。

## 算子级自检（无需起完整模型）

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

## 部署（vllm serve）

### 启动参数

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

### 关键启动信号

- 日志里应能看到 `FLASHINFER_MLA_SPARSE_SM120`
- prefix caching 应正常命中
- CUDA Graph 应完成 capture
- MTP 5 tokens 应保持稳定

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

### 稳定配置

| 参数 | 值 |
|---|---|
| `max-num-batched-tokens` | `4096` |
| `enable-prefix-caching` | `true` |
| `block-size` | `2304` |
| `gpu-memory-utilization` | `0.97` |
| `max-num-seqs` | `4` |
| `max-model-len` | `auto` |

### TP8 / H8

模型全局 heads 为 64，因此 TP4 每 rank 16 heads，TP8 每 rank 8 heads。配套
FlashInfer 分支的 prefill/decode H8/H16 数值测试为 `4 passed, 2 warnings in 29.01s`。
当前机器只有 4 张 GPU，所以这证明 H8 算子/API 路径正确，不等于完成了 TP8 端到端 serving 认证。

### 为什么当前 TP4 使用 `block-size=2304`

- `--block-size 256` 会先被 vLLM 提升到 1792；`1792 × 528 = 946176`
  bytes，小于 1,146,880-byte 状态型 hybrid-cache 页，KV cache 初始化失败。
- 2176 足以容纳状态页，但 `2176 / index_kpool(4) = 544`，只能虚拟切成
  `block_kv=32`；SM120 DeepGEMM FP8 paged-MQA 只接受 64，因此 profile 失败。
- 当前路径要求 manager block 是 `index_kpool × 64 = 256` 的倍数；大于等于
  `ceil(1146880 / 528) = 2173` 的最小 256 倍数正好是 **2304**。

这里的 “mamba state page” 是 vLLM 通用 cache 抽象/错误文本，并不表示 GLM MLA
KV cache 混入了 Kimi-K3 专属 KDA state。2304 的最小值结论只适用于当前 TP4/FP8
配置；更换 TP 数后需要重新验证。

## Release 内容与限制

- Release 附件包含 vLLM、FlashInfer 两个源码 tarball、`MANIFEST.json` 和
  `SHA256SUMS`；不包含模型权重、wheel 或 JIT 编译缓存。
- `gpu-memory-utilization=0.98` 在长上下文 sparse-indexer profile/请求中出现过
  OOM，因此推荐 0.97，而不是把更高 auto capacity 当作稳定容量。
- 本轮未验证 DCP，也未在 8 卡机器上做 TP8 端到端测试。
- 代码和测试包含 AI 辅助；发布者仍需理解并审查全部改动后再向上游提交 PR。

## 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)（Apache-2.0）及其 GLM-5.3-Flash 相关集成。
