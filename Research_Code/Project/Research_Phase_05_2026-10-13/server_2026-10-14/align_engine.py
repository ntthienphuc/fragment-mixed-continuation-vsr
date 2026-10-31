"""Audio is used only to derive timing labels; original visual transcripts remain intact."""
import re
import av
import numpy as np
import torch
import torchaudio
from num2words import num2words


def decode_audio(path):
    arrays = []
    with av.open(str(path)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError('Expected exactly one audio stream')
        audio = container.streams.audio[0]
        video = container.streams.video[0]
        offset = float((audio.start_time or 0) * audio.time_base - (video.start_time or 0) * video.time_base)
        if abs(offset) > .04:
            raise ValueError(f'Audio/video start offset requires manual review: {offset}')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        for frame in container.decode(audio):
            arrays.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(frame))
        arrays.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(None))
    if not arrays:
        raise ValueError('No audio samples')
    return torch.from_numpy(np.concatenate(arrays).astype(np.float32)), offset


def alignment_spelling(word):
    text = word.upper().replace('’', "'")
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())).upper(), text)
    return re.sub(r"[^A-Z'|]", '', text.replace('-', '|'))


class Aligner:
    def __init__(self, device='cuda'):
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(device).eval()
        self.labels = self.bundle.get_labels()
        self.mapping = {label: index for index, label in enumerate(self.labels)}
        self.device = device

    def align(self, waveform, offset, transcript):
        original = transcript.split()
        spellings = [alignment_spelling(word) for word in original]
        if any(not word for word in spellings):
            raise ValueError('An original word cannot be represented by acoustic vocabulary')
        targets, ownership = [], []
        for index, spelling in enumerate(spellings):
            if index:
                targets.append(self.mapping['|'])
                ownership.append(None)
            for letter in spelling:
                targets.append(self.mapping[letter])
                ownership.append(index if letter != '|' else None)
        with torch.inference_mode():
            emission, _ = self.model(waveform[None].to(self.device))
            emission = emission.float().log_softmax(-1).cpu()
        aligned, probabilities = torchaudio.functional.forced_align(
            emission, torch.tensor([targets], dtype=torch.int64), blank=0)
        spans = torchaudio.functional.merge_tokens(aligned[0], probabilities[0].exp(), blank=0)
        if [span.token for span in spans] != targets:
            raise ValueError('CTC alignment does not preserve target sequence')
        ratio = (len(waveform) / 16000) / emission.shape[1]
        words = []
        for index, word in enumerate(original):
            group = [span for span, owner in zip(spans, ownership) if owner == index]
            if not group:
                raise ValueError('Original word has no aligned acoustic frames')
            words.append({'text': word, 'start': max(0., group[0].start * ratio + offset),
                          'end': max(0., group[-1].end * ratio + offset),
                          'alignment_score': float(sum(s.score * (s.end - s.start) for s in group) /
                                                   sum(s.end - s.start for s in group))})
        return words
