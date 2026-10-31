"""Check scalar values and gradients against native alternative reductions on small CPU tensors."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from reduction import reduce_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT / "Research_Phase_05_2026-10-13/vendor/auto_avsr/espnet/nets/pytorch_backend"
CTC_PATH = ROOT / "Research_Phase_05_2026-10-13/phase5/ctc_loss.py"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    output = HERE / "CPU_01"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    smoothing_path = NATIVE / "transformer/label_smoothing_loss.py"
    smoothing = module("independent_native_smoothing", smoothing_path).LabelSmoothingLoss
    cpu_ctc = module("independent_native_ctc", CTC_PATH).cpu_ctc
    native_attention = smoothing(9, -1, 0.1, False)
    token_attention = smoothing(9, -1, 0.1, True)
    cases = []
    patterns = [(1, [1]), (2, [1, 2]), (3, [2, 2, 3]), (7, [1, 2, 1, 1, 3, 2, 4]),
                (12, [1, 2, 3, 4] * 3)]
    for seed, (count, tokens) in enumerate(patterns, 701):
        torch.manual_seed(seed)
        length = count * 3 + 4
        ctc_logits = torch.randn(length, 1, 9, requires_grad=True)
        att_logits = torch.randn(1, count + 3, 9, requires_grad=True)
        target = torch.tensor(tokens, dtype=torch.long)
        attention_target = torch.tensor([tokens + [8, -1, -1]], dtype=torch.long)
        ctc_sum = cpu_ctc(ctc_logits.log_softmax(-1), target, [length], [count], zero_infinity=False)
        att_sum = native_attention(att_logits, attention_target)
        reduced = reduce_components(ctc_sum, att_sum, count, count + 1)
        reference = 0.1 * torch.nn.functional.ctc_loss(
            ctc_logits.log_softmax(-1), target, [length], [count],
            reduction="mean", blank=0, zero_infinity=False)
        reference = reference + 0.9 * token_attention(att_logits, attention_target)
        gradients = torch.autograd.grad(reduced, (ctc_logits, att_logits), retain_graph=True)
        expected = torch.autograd.grad(reference, (ctc_logits, att_logits), retain_graph=True)
        torch.testing.assert_close(reduced, reference, rtol=1e-6, atol=1e-6)
        differences = []
        for actual, wanted in zip(gradients, expected):
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)
            differences.append(float((actual - wanted).abs().max()))
        assert torch.count_nonzero(gradients[1][:, -2:, :]).item() == 0
        blanket = (0.1 * ctc_sum + 0.9 * att_sum) / (count + 1)
        assert abs(float(reduced - blanket)) > 1e-5
        cases.append({"target_tokens": count, "ctc_frames": length, "includes_repeated_labels":
                      any(left == right for left, right in zip(tokens, tokens[1:])),
                      "attention_positions": count + 1, "ignored_positions": 2,
                      "loss": float(reduced.detach()), "reference_loss": float(reference.detach()),
                      "gradient_max_abs_differences": differences,
                      "blanket_division_loss": float(blanket.detach())})
    rejected = 0
    for first, second in [(0, 1), (True, 2), (2, 2), (2, 4)]:
        try:
            reduce_components(torch.tensor(1.0), torch.tensor(2.0), first, second)
        except ValueError:
            rejected += 1
    assert rejected == 4
    for bad in [torch.tensor(float("nan")), torch.tensor([1.0]), 1.0]:
        try:
            reduce_components(bad, torch.tensor(2.0), 2, 3)
        except ValueError:
            rejected += 1
    assert rejected == 7 and not torch.cuda.is_initialized()
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "status": "component_reduction_values_and_logit_gradients_match_native_reference",
              "torch": str(torch.__version__), "device": "cpu", "cases": cases,
              "invalid_inputs_rejected": rejected, "full_recognizer_checked": False,
              "native_optimizer_update_checked": False, "training_admitted": False,
              "recognition_benefit_proven": False,
              "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                              for path in [HERE / "reduction.py", Path(__file__), smoothing_path, CTC_PATH]}}
    with (output / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
