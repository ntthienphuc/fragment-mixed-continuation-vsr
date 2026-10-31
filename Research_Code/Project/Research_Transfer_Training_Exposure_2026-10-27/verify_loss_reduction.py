"""Check the actual attention loss against independent KL and repeated-token examples."""

import ast
import importlib.util
import math

import torch

from common import HERE, PHASE5, ROOT, save, sha, utc


def main():
    torch.set_num_threads(1)
    output = HERE / "Loss_Check_01"
    output.mkdir(exist_ok=False)
    native = PHASE5 / "vendor/auto_avsr/espnet/nets/pytorch_backend"
    criterion_path = native / "transformer/label_smoothing_loss.py"
    model_path = native / "e2e_asr_conformer.py"
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "LabelSmoothingLoss"]
    assert len(constructors) == 1
    assert isinstance(constructors[0].args[3], ast.Constant)
    assert constructors[0].args[3].value is False
    spec = importlib.util.spec_from_file_location("archived_label_loss", criterion_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator().manual_seed(20261027)
    logits = torch.randn(1, 7, 2002, dtype=torch.float64, generator=generator)
    labels = torch.tensor([[1, 9, 13, 8, 29, 2001, -1]], dtype=torch.long)
    probabilities = logits.softmax(-1).tolist()[0]
    manual = 0.0
    valid = 0
    for frame, target in enumerate(labels[0].tolist()):
        if target == -1:
            continue
        valid += 1
        for token, probability in enumerate(probabilities[frame]):
            truth = 0.9 if token == target else 0.1 / 2001
            manual += truth * (math.log(truth) - math.log(probability))
    results = {}
    for normalize in (False, True):
        criterion = module.LabelSmoothingLoss(2002, -1, 0.1, normalize)
        actual = float(criterion(logits, labels))
        repeated = float(criterion(logits.repeat(1, 2, 1), labels.repeat(1, 2)))
        expected = manual / valid if normalize else manual
        assert abs(actual - expected) < 1e-10
        expected_repeat = actual if normalize else 2 * actual
        assert abs(repeated - expected_repeat) < 1e-10
        results[str(normalize)] = {"loss": actual, "repeated_loss": repeated,
                                   "repeated_over_original": repeated / actual,
                                   "manual_kl_absolute_error": abs(actual - expected)}
    save(output / "Verification.json", {
        "status": "actual_criterion_and_independent_kl_passed", "utc": utc(),
        "native_normalize_length": False, "classes": 2002, "valid_positions": valid,
        "padding_ignored": True, "results": results,
        "code_sha256": sha(HERE / "verify_loss_reduction.py"),
        "criterion_path": criterion_path.relative_to(ROOT).as_posix(),
        "criterion_sha256": sha(criterion_path), "constructor_sha256": sha(model_path),
        "new_training": False, "actual_gradient_ratio_estimated": False,
        "interpretation": "Repeated identical token losses scale with token count in the archived objective; "
                          "this synthetic check does not infer gradient magnitude or an effective learning rate.",
    })
    print({"status": "passed", "native_repeated_loss_ratio": results["False"]["repeated_over_original"],
           "normalized_control_ratio": results["True"]["repeated_over_original"]}, flush=True)


if __name__ == "__main__":
    main()
