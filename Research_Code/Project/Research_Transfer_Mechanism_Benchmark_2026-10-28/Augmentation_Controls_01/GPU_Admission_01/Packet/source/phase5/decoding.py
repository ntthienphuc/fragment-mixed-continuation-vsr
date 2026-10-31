"""Predeclared reference-free decoder controls on identical target representations."""

import math
import torch


def decode(model, encoded, name='joint01'):
    if name == 'greedy':
        return model.greedy(encoded)
    from espnet.nets.batch_beam_search import BatchBeamSearch

    class BoundedSearch(BatchBeamSearch):
        def post_process(self, i, maxlen, ratio, running, ended):
            return super().post_process(i, min(maxlen, ceiling), ratio, running, ended)

    weights = {'joint01': .1, 'joint03': .3, 'joint07': .7, 'ctc_only': 1., 'bounded': .3}
    if name not in weights:
        raise ValueError('Unknown decoding control')
    native = model.native
    posterior = native.ctc.log_softmax(encoded)[0]
    ids = posterior.argmax(-1).unique_consecutive()
    count = int(((ids > 0) & (ids < native.eos)).sum())
    ceiling = min(encoded.size(1), max(8, math.ceil(1.5 * count) + 4))
    weight = weights[name]
    constructor = BoundedSearch if name == 'bounded' else BatchBeamSearch
    beam = constructor(beam_size=3, vocab_size=native.odim,
                       weights={'decoder': 1 - weight, 'ctc': weight},
                       scorers=native.scorers(), sos=native.sos, eos=native.eos,
                       token_list=model.tokenizer.tokens, pre_beam_ratio=1.5,
                       pre_beam_score_key=None if name == 'ctc_only' else 'decoder')
    hypotheses = beam(encoded[0])
    if not hypotheses:
        raise RuntimeError('No ended hypothesis')
    return model.tokenizer.decode(hypotheses[0].yseq[1:].tolist())
