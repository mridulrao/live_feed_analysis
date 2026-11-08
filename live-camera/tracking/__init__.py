"""Visual tracking plugins: seed with detections, then follow subsequent images."""
from typing import Protocol
from protocol import DetectionBox, TrackedBox


class Tracker(Protocol):
    def reset(self): ...
    def initialize(self, image, detections: list[DetectionBox]) -> list[TrackedBox]: ...
    def update(self, image) -> list[TrackedBox]: ...


def create_tracker(name) -> Tracker:
    if name == 'csrt':
        from .csrt import CSRTTracker
        return CSRTTracker()
    raise ValueError(f'Unknown visual tracking plugin: {name}; use TRACKING_PLUGIN=csrt')
