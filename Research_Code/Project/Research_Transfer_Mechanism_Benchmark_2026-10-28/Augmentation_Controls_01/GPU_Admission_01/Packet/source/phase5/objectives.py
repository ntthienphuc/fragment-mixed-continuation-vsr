"""Matched-sample training objectives, including explicit full-to-window distillation."""

import torch
from .views import context_for, span, target_text
from .common import rank


def sample_loss(model, frames, row, sample_index, seed, budget):
    is_full = model.method == 'T0' or sample_index % 2 == 0
    if is_full:
        left, right, begin, end = 0, len(frames), 0, len(frames)
        text = row['text']
    else:
        choices = row['train_starts']
        position = int(rank(f"{seed}|{row['clip_id']}|{sample_index}")[:16], 16) % len(choices)
        start = choices[position]
        core_length = row.get('core_frames', 125)
        left, right, begin, end = span(len(frames), start, context_for(model.method, budget), core_length)
        text = target_text(row['words'], start, start + core_length)
    encoded = model.encode(frames[left:right])
    intermediate = model.intermediate
    core = model.representation(encoded, begin, end)
    loss, details = model.asr_loss(core, text)
    if model.method == 'T3':
        auxiliary = model.auxiliary_loss(intermediate[:, begin:end], text)
        loss = loss + .1 * auxiliary
        details['intermediate_ctc'] = float(auxiliary.detach())
    if model.method == 'T2' and not is_full:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            teacher = model.encode(frames)[:, left + begin:left + end]
            teacher_logits = model.native.ctc.ctc_lo(teacher).float() / 2
            teacher_probability = teacher_logits.softmax(-1)
        model.train(was_training)
        logp = (model.native.ctc.ctc_lo(core).float() / 2).log_softmax(-1)
        kl = torch.nn.functional.kl_div(logp, teacher_probability, reduction='none').sum(-1).mean() * 4
        loss = loss + .15 * kl
        details['consistency'] = float(kl.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective')
    details.update(full=is_full, input_frames=right-left, target_frames=end-begin)
    return loss, details
