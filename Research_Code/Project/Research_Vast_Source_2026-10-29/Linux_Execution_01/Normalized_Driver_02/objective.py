"""Apply the prospectively declared per-component reduction to matched source views."""

from adapter import MatchedObjective
from reduction import reduce_components


class NormalizedObjective(MatchedObjective):
    def __init__(self, dataset, arm):
        if arm not in ('full', 'fixed_time'):
            raise ValueError('This objective is only the declared full/fixed-time normalization control')
        super().__init__(dataset, arm)

    def loss(self, model, frames, row, sample_index, seed, budget):
        import torch
        from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
        from espnet.nets.pytorch_backend.transformer.mask import target_mask
        from phase5.model import cpu_ctc

        if budget not in ('past', 'both'):
            raise ValueError('Unsupported native budget')
        pixels, text, numerator, denominator, full = self.select(model, frames, row, sample_index, seed)
        if numerator != 1 or denominator != 1 or model.native.criterion.normalize_length:
            raise ValueError('Expected original unweighted native component sums')
        encoded = model.representation(model.encode(pixels), 0, len(pixels))
        ids = model.tokenizer.encode(text)
        target = torch.tensor([ids], dtype=torch.long, device=model.device)
        lengths = torch.tensor([encoded.size(1)], device=model.device)
        with torch.autocast(model.device.type, enabled=False):
            logits = model.native.ctc.ctc_lo(model.native.ctc.dropout(encoded.float())).transpose(0, 1)
            ctc_sum = cpu_ctc(logits.log_softmax(2).to('cpu'), target.flatten(), lengths, [len(ids)]) / logits.size(1)
        ys_in, ys_out = add_sos_eos(target, model.native.sos, model.native.eos, -1)
        mask = torch.ones(1, 1, encoded.size(1), dtype=torch.bool, device=model.device)
        prediction = model.native.decoder(ys_in, target_mask(ys_in, -1), encoded, mask)[0]
        attention_sum = model.native.criterion(prediction, ys_out)
        attention_positions = int(ys_out.ne(-1).sum().item())
        loss = reduce_components(ctc_sum, attention_sum, len(ids), attention_positions)
        details = {'ctc': float(ctc_sum.detach()), 'attention': float(attention_sum.detach()),
                   'full': full, 'input_frames': len(pixels), 'target_frames': len(pixels),
                   'matched_arm': self.arm, 'weight_numerator': 1, 'weight_denominator': 1,
                   'target_tokens': len(ids), 'attention_positions': attention_positions,
                   'reduction': 'ctc_per_target_token_attention_per_valid_position'}
        return loss, details
