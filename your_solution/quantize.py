"""Offline weight quantization.

Packed format convention:
  - Two signed INT4 values per uint8 byte
  - Low nibble = even element, high nibble = odd element
  - Scales are FP16, one per group
"""

import torch


def quantize_weights(weight: torch.Tensor, group_size: int = 64) -> dict:
    """Quantize a FP16 weight tensor to packed INT4 format.

    Args:
        weight: [N, K] float16 weight tensor.
        group_size: Number of elements per quantization group.

    Returns:
        dict with:
            "weight_packed": [N, K//2] uint8 tensor (packed INT4)
            "weight_scales": [N, K//group_size] float16 tensor (per-group scales)
            "group_size": int
    """
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, "group_size must be even"

    num_groups = K // group_size

    # Compute scales in fp32 for numerical stability.
    w = weight.float().reshape(N, num_groups, group_size)
    abs_w = w.abs()

    # Conservative percentile clipping to reduce sensitivity to extreme outliers
    # while staying close to standard symmetric INT4 behavior.
    # kthvalue is a partial select (O(n)) vs quantile's full sort (O(n log n))
    k = max(1, int(0.999 * group_size))
    clip = torch.kthvalue(abs_w, k, dim=-1, keepdim=True).values

    # Handle all-zero groups safely.
    nonzero = clip > 0
    scale = torch.where(nonzero, clip / 7.0, torch.ones_like(clip))
    rscale = torch.where(nonzero, 7.0 / clip, torch.zeros_like(clip))

    # Symmetric signed INT4 quantization.
    q = torch.round(w * rscale).clamp(-8, 7).to(torch.int8).reshape(N, K)

    # Pack two INT4 values per byte.
    even = (q[:, 0::2] & 0xF).to(torch.uint8)
    odd = ((q[:, 1::2] & 0xF) << 4).to(torch.uint8)
    packed = odd | even

    scales = scale.squeeze(-1).half()

    return {
        "weight_packed": packed,
        "weight_scales": scales,
        "group_size": group_size,
    }