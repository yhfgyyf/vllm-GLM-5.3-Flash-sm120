# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
    _kv_scale_format_for_model,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(
    model_type: str,
    *,
    qk_nope_head_dim: int = 128,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                index_topk=2048,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            ),
        ),
    )


def test_glm_nope_capability_checks_with_kv_cache_mla_signature(monkeypatch) -> None:
    flashinfer_module = ModuleType("flashinfer")
    decode_module = ModuleType("flashinfer.decode")
    mla_module = ModuleType("flashinfer.mla")
    cache_module = ModuleType("flashinfer.mla._sparse_mla_sm120_cache")

    def trtllm_batch_decode_sparse_mla_dsv4(*args, **kwargs):
        pass

    def trtllm_batch_decode_with_kv_cache_mla(*args, kv_scale_format=None, **kwargs):
        pass

    decode_module.trtllm_batch_decode_sparse_mla_dsv4 = (
        trtllm_batch_decode_sparse_mla_dsv4
    )
    decode_module.trtllm_batch_decode_with_kv_cache_mla = (
        trtllm_batch_decode_with_kv_cache_mla
    )
    cache_module.glm_nope_gather_and_dequantize = lambda *args, **kwargs: None
    cache_module.glm_nope_quantize_and_cache = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode_module)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", mla_module)
    monkeypatch.setitem(
        sys.modules, "flashinfer.mla._sparse_mla_sm120_cache", cache_module
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope()


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )
    assert issubclass(FlashInferMLASparseSM120Impl, FlashInferMLASparseImpl)


def test_glm_nope_uses_native_scale_format_and_528_byte_cache() -> None:
    assert (
        _kv_scale_format_for_model("glm5_next", 256, 0, 512)
        == "arbitrary_fp32_nope"
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        shape = FlashInferMLASparseSM120Backend.get_kv_cache_shape(
            num_blocks=7,
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            cache_dtype_str="fp8_ds_mla",
        )

    assert shape == (7, 64, 528)


def test_glm_nope_shape_requires_exact_text_config() -> None:
    with set_current_vllm_config(
        _fake_vllm_config("glm4_moe", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        shape = FlashInferMLASparseSM120Backend.get_kv_cache_shape(
            num_blocks=7,
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            cache_dtype_str="fp8_ds_mla",
        )

    assert shape == (7, 64, 656)


def test_rope64_packed_cache_remains_656_bytes() -> None:
    with set_current_vllm_config(_fake_vllm_config("deepseek_v3")):
        shape = FlashInferMLASparseSM120Backend.get_kv_cache_shape(
            num_blocks=7,
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            cache_dtype_str="fp8_ds_mla",
        )

    assert shape == (7, 64, 656)


def test_non_glm_sm120_backend_accepts_block_size_64(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


@pytest.mark.parametrize("block_size", [128, 256])
def test_glm_nope_sm120_backend_accepts_manager_block_size_multiples(
    monkeypatch, block_size: int
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: True
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=block_size,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_backend_rejects_unsupported_block_size(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=96,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert "block_size not supported" in invalid_reasons


def test_glm_nope_requires_new_flashinfer_helpers(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=128,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert any("GLM NoPE" in reason for reason in invalid_reasons)


def test_non_glm_sparse_mla_does_not_require_glm_helpers(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    with set_current_vllm_config(_fake_vllm_config("deepseek_v3")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []
