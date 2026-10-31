"""Decode without resampling; use the archived mouth-crop implementation for raw LRS2."""

import sys
import av
import numpy as np


def read_video(path, raw=False):
    images = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected one video stream')
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - 25) > .001:
            raise ValueError('Expected 25 fps; timestamps must not be silently resampled')
        stream.thread_count = 1
        for frame in container.decode(stream):
            images.append(frame.to_ndarray(format='rgb24' if raw else 'gray'))
    if not images:
        raise ValueError('No decoded video frames')
    array = np.stack(images)
    if not raw and array.shape[1:] != (96, 96):
        raise ValueError('VLR processed artifact must be native gray96, not a raw face-video replacement')
    return array


class MouthCropper:
    def __init__(self, source):
        sys.path.insert(0, str(source / 'preparation'))
        from .detector import Detector
        from detectors.mediapipe.video_process import VideoProcess
        self.detector = Detector()
        self.processor = VideoProcess(convert_gray=True)

    def __call__(self, frames):
        crop = self.processor(frames, self.detector(frames))
        if crop is None:
            raise ValueError('Mouth tracking failed')
        crop = np.asarray(crop)
        if crop.shape != (len(frames), 96, 96) or crop.dtype != np.uint8:
            raise ValueError('Unexpected native mouth-crop output')
        return crop
