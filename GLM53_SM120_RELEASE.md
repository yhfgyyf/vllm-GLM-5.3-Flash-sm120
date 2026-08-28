# vLLM GLM-5.3-Flash SM120 Source Release

## 中文

这是面向 **RTX PRO 6000 Blackwell / SM120** 的 GLM-5.3-Flash 配套源码发布。
Release 同时锁定 vLLM 与 FlashInfer 源码，避免只下载 vLLM 后缺少 GLM NoPE
sparse MLA 算子的情况。

### 主要内容

- vLLM：GLM-5.3-Flash / PR #53906 模型、hybrid cache、kpool sparse indexer、
  MTP 和 SM120 backend 集成。
- FlashInfer：SM120 GLM NoPE sparse MLA JIT kernel，以及 H8/H16
  prefill/decode 数值测试。
- 推荐的稳定配置：TP4、FP8 KV cache、MTP 5 tokens、CUDA Graph、prefix
  caching、`max-num-seqs=4`、`max-num-batched-tokens=4096`、
  `block-size=2304`、`gpu-memory-utilization=0.97`、`max-model-len=auto`。

### 下载与校验

```bash
gh release download --repo yhfgyyf/vllm-GLM-5.3-Flash-sm120 \
  --pattern 'vllm-*-source-*.tar.gz' \
  --pattern 'flashinfer-*-source-*.tar.gz' \
  --pattern MANIFEST.json \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-glm53-release

cd /tmp/vllm-glm53-release
sha256sum -c SHA256SUMS
```

附件包括：

- 锁定当前 Release 提交的 vLLM 源码包；
- 锁定 `yhfgyyf/flashinfer:glm53-flash-nope-sm120` 的 FlashInfer 源码包；
- 记录提交、环境、验证结果和附件哈希的 `MANIFEST.json`；
- 可直接执行 `sha256sum -c` 的 `SHA256SUMS`。

GitHub 自动生成的 `Source code (zip/tar.gz)` 只包含 vLLM；要复现已验证组合，
请下载上述两个附件源码包。

### 验证结果

- 服务器：4× RTX PRO 6000 Blackwell，Python 3.12.14，CUDA 13.0，
  PyTorch 2.13.0+cu130，Triton 3.7.1。
- 256K 输入 / 512 输出：4/4 成功；4096 chunk 稳态 prefill
  8,282.96 tok/s。
- 784K 输入 / 512 输出：4/4 成功；4096 chunk 稳态 prefill
  7,124.45 tok/s。
- 重复 prompt 的 prefix-cache 命中率：256K 为 98.4375%，784K 为
  99.5855%。
- H8/H16 FlashInfer prefill/decode 数值测试：`4 passed, 2 warnings`。
- 8K tool-call 返回结构正确且无乱码；CUDA Graph capture 完成；
  `/health` 返回 HTTP 200。

### 注意事项

- 本 Release **仅包含源码**，不包含 wheel、模型权重或 FlashInfer JIT 缓存。
- 需要设置 `FLASHINFER_DISABLE_VERSION_CHECK=1` 并确保配套 FlashInfer 源码
  在 `PYTHONPATH` 中。
- `gpu-memory-utilization=0.98` 在长上下文场景出现过 OOM，稳定值为 0.97。
- 当前 TP4/FP8 路径需要 `block-size=2304`；256 和 2176 分别因状态页容量与
  DeepGEMM page geometry 不匹配而失败。
- 本轮没有进行 DCP 或 8 卡 TP8 端到端 serving；H8 结论来自算子/API 测试。

完整安装、启动参数、性能数据和 block-size 原因见仓库
[`README.md`](https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120/blob/main/README.md)。

## English

This is the paired **GLM-5.3-Flash source release for RTX PRO 6000 Blackwell /
SM120**. It pins both vLLM and FlashInfer so the GLM NoPE sparse MLA kernel is
not omitted from the deployment.

### Highlights

- vLLM integration for GLM-5.3-Flash / PR #53906, hybrid cache, the kpool
  sparse indexer, MTP, and the SM120 backend.
- FlashInfer SM120 GLM NoPE sparse MLA JIT kernel with H8/H16 prefill/decode
  numerical coverage.
- Stable tested setup: TP4, FP8 KV cache, five MTP tokens, CUDA Graph, prefix
  caching, `max-num-seqs=4`, `max-num-batched-tokens=4096`,
  `block-size=2304`, `gpu-memory-utilization=0.97`, and `max-model-len=auto`.

### Assets and verification

The attached assets contain paired vLLM and FlashInfer source tarballs,
`MANIFEST.json`, and `SHA256SUMS`. Run the download and checksum commands in
the Chinese section above. GitHub's automatic source archive contains only
vLLM, while the attached archives reproduce the validated source pair.

Validation used 4× RTX PRO 6000 GPUs, Python 3.12.14, CUDA 13.0, PyTorch
2.13.0+cu130, and Triton 3.7.1. Four 256K/512 and four 784K/512 requests
completed. With a 4096-token chunk, steady-state prefill measured 8,282.96
tok/s at 256K and 7,124.45 tok/s at 784K. H8/H16 operator tests reported
`4 passed, 2 warnings`; the structured tool call, UTF-8 output, CUDA Graph
capture, and HTTP health check also passed.

### Limitations

- This is a **source-only** release: no wheels, model weights, or JIT cache are
  included.
- Set `FLASHINFER_DISABLE_VERSION_CHECK=1` and put the paired FlashInfer source
  tree on `PYTHONPATH`.
- `gpu-memory-utilization=0.98` encountered OOM in long-context testing; 0.97
  is the stable tested value.
- The tested TP4/FP8 geometry requires `block-size=2304`.
- DCP and end-to-end eight-GPU TP8 serving were not tested. H8 is covered at
  the operator/API level only.

See the repository
[`README.md`](https://github.com/yhfgyyf/vllm-GLM-5.3-Flash-sm120/blob/main/README.md)
for full installation steps, serving arguments, benchmark data, and the
block-size analysis.
