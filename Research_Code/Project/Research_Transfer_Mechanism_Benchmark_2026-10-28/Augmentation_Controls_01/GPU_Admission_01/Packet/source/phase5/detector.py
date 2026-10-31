"""Mouth landmarks with largest-face selection fixed relative to the archived recipe."""

import numpy as np


def largest_detection(detections):
    def area(item):
        box = item.location_data.relative_bounding_box
        return max(0., box.width) * max(0., box.height)
    return max(detections, key=area)


class Detector:
    def __init__(self):
        import mediapipe as mp
        factory = mp.solutions.face_detection.FaceDetection
        self.full = factory(min_detection_confidence=.5, model_selection=1)
        self.short = factory(min_detection_confidence=.5, model_selection=0)

    def detect(self, frames, detector):
        points = []
        for frame in frames:
            detections = detector.process(frame).detections
            if not detections:
                points.append(None)
                continue
            face = largest_detection(detections)
            height, width = frame.shape[:2]
            points.append(np.array([[int(p.x * width), int(p.y * height)]
                                    for p in face.location_data.relative_keypoints[:4]]))
        return points

    def __call__(self, frames):
        points = self.detect(frames, self.full)
        if all(p is None for p in points):
            points = self.detect(frames, self.short)
        if all(p is None for p in points):
            raise ValueError('No face detected in clip')
        return points
