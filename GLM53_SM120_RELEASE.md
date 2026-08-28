# vLLM GLM-5.3-Flash SM120 Wheel Release

## Install

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
  uv pip install ./flashinfer_python-*.whl ./vllm-*.whl \
  --torch-backend=cu130
```

## Serve

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
