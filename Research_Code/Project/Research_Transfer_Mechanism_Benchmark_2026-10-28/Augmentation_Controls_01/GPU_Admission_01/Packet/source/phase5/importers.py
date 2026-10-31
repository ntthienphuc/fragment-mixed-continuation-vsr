"""Read only explicitly requested splits. No automatic test discovery."""

import csv
import json
import math
from pathlib import Path
from .text import normalize
from .common import digest


def validate_words(words):
    if not words:
        raise ValueError('No word timing rows')
    for word in words:
        if not word['text'] or not math.isfinite(word['start']) or not math.isfinite(word['end']):
            raise ValueError('Empty text or nonfinite timestamp')
        if word['start'] < 0 or word['end'] < word['start']:
            raise ValueError('Negative or reversed timestamp')
    return words


def vlr_records(root, split):
    source = root / 'manifests' / f'{split}_strict.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != split:
                raise ValueError('Split column disagrees with requested manifest')
            with (root / row['label_path']).open(encoding='utf-8-sig', newline='') as labels:
                words = [{'text': w['Word'].strip(), 'start': float(w['Start']), 'end': float(w['End'])}
                         for w in csv.DictReader(labels)]
            blanks = [w for w in words if not w['text']]
            if blanks:
                if any(not math.isfinite(w['start']) or w['start'] != w['end'] or w['start'] < 0 for w in blanks):
                    raise ValueError('Blank annotation has nonzero or invalid duration')
                words = [w for w in words if w['text']]
                if normalize(' '.join(w['text'] for w in words), True) != normalize(row['transcript'], True):
                    raise ValueError('Removing blank timing row changes the transcript')
            yield {'clip_id': row['clip_id'], 'source_id': row['source_id'], 'split': split,
                   'video': str(root / row['video_path']), 'text': row['transcript'],
                   'expected_video_blake2b_128': row['video_blake2b_128'],
                   'words': validate_words(words), **({'ignored_blank_zero_rows': len(blanks)} if blanks else {})}


def lrs2_label(path):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    header = next((line.split(':', 1)[1].strip() for line in lines if line.startswith('Text:')), '')
    words = []
    for line in lines:
        if line.startswith(('Text:', 'Conf:')):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            start, end = float(fields[1]), float(fields[2])
        except ValueError:
            continue
        words.append({'text': fields[0], 'start': start, 'end': end})
    if not header:
        raise ValueError('LRS2 Text header missing')
    return header, validate_words(words) if words else []


def lrs2_records(root, split, list_path):
    for line in list_path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        relative = Path(line.split()[0])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid official list path')
        video = root / 'main' / relative.with_suffix('.mp4')
        text, words = lrs2_label(video.with_suffix('.txt'))
        source = 'official'
        alignment_status = 'ok'
        if not words:
            sidecar = video.with_suffix('.alignment.json')
            alignment = json.loads(sidecar.read_text())
            if alignment['transcript'] != text or alignment['label_sha256'] != digest(video.with_suffix('.txt')):
                raise ValueError('Alignment does not match original LRS2 transcript')
            if alignment['video_sha256'] != digest(video):
                raise ValueError('Alignment video identity changed')
            words = alignment['words']
            source = 'audio_ctc_forced_alignment'
            alignment_status = alignment['status']
            if words:
                validate_words(words)
        yield {'clip_id': relative.as_posix(), 'source_id': relative.parent.as_posix(),
               'split': split, 'video': str(video), 'text': text, 'words': words,
               'word_timing_source': source, 'alignment_status': alignment_status}
