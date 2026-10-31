"""Per-example hybrid-loss reduction; no change to an active recognizer or schedule."""

import torch


def reduce_components(ctc_sum, attention_sum, target_tokens, attention_positions):
    """Normalize CTC by target tokens and attention by nonignored positions including EOS."""
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("A positive native target-token count is required")
    if type(attention_positions) is not int or attention_positions != target_tokens + 1:
        raise ValueError("Single-example attention positions must include exactly one EOS")
    for value in (ctc_sum, attention_sum):
        if not isinstance(value, torch.Tensor) or value.ndim != 0 or not value.is_floating_point():
            raise ValueError("Live scalar loss tensors are required")
        if not torch.isfinite(value.detach()).item():
            raise ValueError("Nonfinite native component")
    if ctc_sum.device != attention_sum.device:
        raise ValueError("Loss components must be on the same device")
    return 0.1 * (ctc_sum / target_tokens) + 0.9 * (attention_sum / attention_positions)
