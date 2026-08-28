# GLM-5.3-Flash on SM120 (RTX PRO 6000 Blackwell) — vLLM fork

<!-- markdownlint-disable MD060 -->

> English version: [`README_EN.md`](README_EN.md)
> 本仓库基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)，选择性合入 GLM-5.3-Flash / PR #53906 相关改动，并保留 SM120 上的 FlashInfer NoPE sparse MLA 适配。

把 vLLM 的 **GLM-5.3-Flash** 推理扩展到 **SM120**（RTX PRO 6000 Blackwell / 服务器 Blackwell 平台）。当前代码已按 **4 张 RTX PRO 6000**、TP4、FP8 KV cache、MTP 5 tokens、CUDA Graph 的配置验证过长上下文、prefix caching、serving 和数值测试。

## Changelog

### 2026-08-28

- 选择性合入 GLM-5.3-Flash / PR #53906 的模型入口、hybrid cache、kpool sparse indexer、MTP 以及 serving 相关更新。
- 保留并验证 SM120 FlashInfer NoPE sparse MLA 路径，包括 `FLASHINFER_MLA_SPARSE_SM120`、`has_flashinfer_sparse_mla_sm120()` 和 `has_flashinfer_sparse_mla_sm120_glm_nope()`。
- 推荐稳定配置为 `max-num-batched-tokens=4096`、`enable-prefix-caching`、`block-size=2304`、`gpu-memory-utilization=0.97`、`max-num-seqs=4`、`max-model-len=auto`。
- 已验证 256K / 784K 长上下文、prefix cache 命中、MTP 5 tokens、以及 4 卡 serve 的启动与稳定运行。

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
| Python | 3.12 |
| CUDA toolkit | 13.x |
| torch | cu130 系列 |
| FlashInfer | SM120 NoPE sparse MLA fork |
| vLLM | 当前 GLM-5.3-Flash integration branch |

## 3. 快速安装(最新 Release，免手工拼装)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download --repo yhfgyyf/vllm-GLM-5.3-Flash-sm120 \
  --pattern '*.tar.gz' \
  --pattern 'MANIFEST.json' \
  --pattern 'SHA256SUMS' \
  --dir /tmp/vllm-glm53-release

tar -xzf /tmp/vllm-glm53-release/*.tar.gz -C /tmp/vllm-glm53-release
cd /tmp/vllm-glm53-release/vllm-GLM-5.3-Flash-sm120-*

UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

如果你想直接从源码树编译，也可以跳到下一节。

## 4. 源码安装(clone 本仓库编译)

### 4.1 Python 环境

```bash
uv venv --python 3.12
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install torch --torch-backend=cu130
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

### 4.2 clone 本仓库

```bash
git clone https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120.git
cd vllm-GLM-5.3-Flash-sm120
```

### 4.3 编译 / 打包

```bash
export CUDA_HOME=/usr/local/cuda
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8

./build_wheel.sh
uv pip install --force-reinstall --no-deps dist/*.whl
```

## 5. 算子级自检(无需起完整模型)

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

## 6. 部署(vllm serve)

### 6.1 源模型

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

### 6.2 关键启动信号

- 日志里应能看到 `FLASHINFER_MLA_SPARSE_SM120`
- prefix caching 应正常命中
- CUDA Graph 应完成 capture
- MTP 5 tokens 应保持稳定

## 7. 测试结果(4× RTX PRO 6000)

### 7.1 推理正确性

```text
Q: 用一句话介绍长城。
A: 长城是中国古代为抵御北方游牧民族入侵而修筑的、横跨多个朝代、绵延数千公里的军事防御工程。
```

### 7.2 长上下文与 prefix caching

| 输入 | 配置 | 结果 |
|---|---|---|
| 256K | `prefix caching on`, `chunked prefill 4096` | 稳定启动，重复请求命中率约 98.44% |
| 784K | `prefix caching on`, `chunked prefill 4096` | 稳定启动，重复请求命中率约 99.59% |

### 7.3 稳定配置

| 参数 | 值 |
|---|---|
| `max-num-batched-tokens` | `4096` |
| `enable-prefix-caching` | `true` |
| `block-size` | `2304` |
| `gpu-memory-utilization` | `0.97` |
| `max-num-seqs` | `4` |
| `max-model-len` | `auto` |

## 8. 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)（Apache-2.0）及其 GLM-5.3-Flash 相关集成。
