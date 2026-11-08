"""Detection plugin registry. Plugins load once and return normalized labeled boxes."""
from typing import Protocol
from protocol import DetectionBox


class Detector(Protocol):
    async def start(self): ...
    async def detect(self, image, query) -> list[DetectionBox]: ...
    async def close(self): ...


def create_detector(name):
    if name == 'locate_anything':
        from .locate_anything import LocateAnythingDetector
        return LocateAnythingDetector()
    raise ValueError(f'Unknown detection plugin: {name}')
