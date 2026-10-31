"""Native update arithmetic with identity-bound, atomic resumable checkpoints."""

from pathlib import Path
import time
import torch


def update(model, optimizer, scaler, objective, stream, step, precision, device, deadline=None):
    from phase5.runtime import autocast
    from phase5.training import learning_rate

    if not 1 <= step <= 6000:
        raise ValueError("Step is outside the fixed training budget")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(step, 6000, 2e-6)
    for group in optimizer.param_groups:
        group["lr"] = rate
    total = 0.0
    consumed = []
    for micro in range(8):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Update reached its finite execution deadline")
        position = (step - 1) * 8 + micro
        row, pixels = next(stream)
        with autocast(precision, device):
            loss, details = objective.loss(model, pixels, row, position, 17, "both")
        scaler.scale(loss / 8).backward()
        total += float(loss.detach()) / 8
        consumed.append({"sample": position, "clip_id": row["clip_id"], **details})
        del loss, pixels
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return {"step": step, "loss": total, "lr": rate, "gradient_norm": float(norm), "samples": consumed}


def save_checkpoint(path, model, optimizer, scaler, step, identity):
    from phase5.runtime import rng_state

    path = Path(path)
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise FileExistsError("An interrupted checkpoint write must be investigated first")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "rng": rng_state(), "step": step, "identity": identity}
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scaler, identity):
    from phase5.runtime import restore_rng

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"] != identity or not 0 <= payload["step"] <= 6000:
        raise ValueError("Checkpoint identity or step changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])
    return int(payload["step"])
