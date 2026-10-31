"""Evaluation retains failure coverage and never chooses a decoder with test references."""

import time
import torch
from .decoding import decode
from .text import score, reference_text
from .views import span


def evaluate(model, data, context='both', decoder='greedy', view='window', limit=0):
    if view not in {'window', 'full'}:
        raise ValueError('Unknown evaluation view')
    rows = data.selection(limit)
    selected = [r for r in rows if view == 'full' or r['window_eligible']]
    was_training = model.training
    model.eval()
    output = []
    with torch.inference_mode():
        for row in selected:
            reference = reference_text(row['text'] if view == 'full' else row['window_reference'],
                                       model.tokenizer.language)
            began = time.perf_counter()
            result = {'clip_id': row['clip_id'], 'source_id': row['source_id'],
                      'reference': reference, 'method': model.method, 'context': context,
                      'decoder': decoder, 'view': view, 'status': 'ok', 'error': None}
            try:
                frames = data.frames(row['clip_id'])
                if view == 'full':
                    left, right, start, end = 0, len(frames), 0, len(frames)
                else:
                    left, right, start, end = span(len(frames), row['eval_start'], context,
                                                   row.get('core_frames', 125))
                encoded = model.encode(frames[left:right])
                core = model.representation(encoded, start, end)
                result['text'] = decode(model, core, decoder)
                if model.device.type == 'cuda':
                    torch.cuda.synchronize()
                result['input_frames'] = right - left
            except (RuntimeError, ValueError, OSError) as error:
                result.update(text='', status='error', error=f'{type(error).__name__}: {error}')
            result.update(scores=score(reference, result['text']), seconds=time.perf_counter() - began)
            output.append(result)
    # Preparation failures have unknown eligible window targets. Keep them in explicit coverage,
    # and score as empty hypotheses for the full-utterance endpoint when not taking a subset.
    if view == 'full' and not limit:
        for row in data.failures:
            reference = reference_text(row['text'], model.tokenizer.language)
            output.append({**row, 'reference': reference, 'text': '', 'status': 'preparation_error',
                           'scores': score(reference, ''), 'method': model.method,
                           'context': context, 'decoder': decoder, 'view': view, 'seconds': 0})
    model.train(was_training)
    totals = {}
    for metric in ['word', 'character', 'no_punctuation']:
        counts = {key: sum(r['scores'][metric][key] for r in output)
                  for key in ['errors', 'units', 'substitutions', 'deletions', 'insertions']}
        counts['rate'] = counts['errors'] / counts['units'] if counts['units'] else None
        totals[metric] = counts
    summary = {'planned_preparation': data.receipt['planned'], 'preparation_failed': len(data.failures),
               'view': view, 'context': context, 'decoder': decoder, 'requested_rows': len(rows),
               'window_ineligible': sum(not r['window_eligible'] for r in rows),
               'scored_rows': len(output), 'inference_failed': sum(r['status'] != 'ok' for r in output),
               'metrics': totals, 'subset_limit': limit}
    return output, summary
