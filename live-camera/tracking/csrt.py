"""Box-initialized CSRT visual tracking. All calls run on one CPU executor."""
import math

from protocol import TrackedBox


class CSRTTracker:
    def __init__(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        if not hasattr(cv2, 'TrackerCSRT_create'):
            raise ValueError('CSRT requires opencv-contrib-python-headless; run setup-tracking.sh')
        self.reset()

    def reset(self):
        self.tracks = []
        self.size = None
        self.next_id = 1

    def frame(self, image):
        return self.cv2.cvtColor(self.np.asarray(image.convert('RGB')), self.cv2.COLOR_RGB2BGR)

    def initialize(self, image, detections):
        """A reacquisition receives fresh IDs; no claim of persistent identity."""
        frame = self.frame(image)
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.tracks = []
        boxes = []
        for detection in detections[:8]:
            x1, y1, x2, y2 = detection.box
            x, y = round(x1*w), round(y1*h)
            right, bottom = min(w, round(x2*w)), min(h, round(y2*h))
            if right-x < 4 or bottom-y < 4:
                continue
            tracker = self.cv2.TrackerCSRT_create()
            # Modern OpenCV init returns None on success, not a boolean.
            if tracker.init(frame, (x, y, right-x, bottom-y)) is False:
                continue
            ident = self.next_id
            self.next_id += 1
            self.tracks.append((ident, detection.label, tracker))
            boxes.append(TrackedBox(track_id=ident, label=detection.label,
                box=[x/w, y/h, right/w, bottom/h]))
        return boxes

    def update(self, image):
        if not self.tracks:
            return []
        frame = self.frame(image)
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            self.tracks = []
            return []
        alive, boxes = [], []
        for ident, label, tracker in self.tracks:
            ok, (x, y, width, height) = tracker.update(frame)
            if not ok or not all(math.isfinite(v) for v in (x, y, width, height)):
                continue
            coords = [max(0., x/w), max(0., y/h), min(1., (x+width)/w), min(1., (y+height)/h)]
            try:
                boxes.append(TrackedBox(track_id=ident, label=label, box=coords))
                alive.append((ident, label, tracker))
            except ValueError:
                continue
        self.tracks = alive
        return boxes
