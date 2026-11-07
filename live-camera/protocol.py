"""Shared, bounded wire protocol. Never includes credentials or saved image files."""
import io
import json
import math
import struct
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVENT_TOPIC = 'onestreamer.events.v1'
EVIDENCE_TOPIC = 'onestreamer.evidence.v1'
MAX_EVIDENCE_BYTES = 2_000_000


class Crop(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    x: float = Field(ge=0, le=0.8)
    y: float = Field(ge=0, le=0.8)
    width: float = Field(ge=0.2, le=1)
    height: float = Field(ge=0.2, le=1)

    @model_validator(mode='after')
    def fits(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError('Crop must fit inside the image')
        return self


class Start(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=2000)
    image_budget: Literal[224, 336, 448] = 448
    crop: Crop | None = None
    verify: bool = False

    @model_validator(mode='after')
    def nonempty(self):
        self.question = self.question.strip()
        if not self.question:
            raise ValueError('Enter a question or instruction')
        return self


class SessionID(BaseModel):
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')


class WorkerStart(Start, SessionID):
    room: str = Field(pattern=r'^camera-[a-f0-9]{32}$')
    publisher: str = Field(pattern=r'^camera-[a-f0-9]{32}$')

    @model_validator(mode='after')
    def matches_session(self):
        if self.room != 'camera-' + self.session_id or self.publisher != self.room:
            raise ValueError('Room and publisher must match session')
        return self


class Observation(SessionID):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    event_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    state: Literal['response', 'standby', 'silence', 'error', 'snapshot']
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    response: str | None = Field(default=None, max_length=8000)
    inference_seconds: float = Field(default=0, ge=0)
    skipped_seconds: float = Field(default=0, ge=0)
    image_budget: int = 448
    input_size: list[int] = Field(default_factory=list, max_length=2)
    evidence_status: str = Field(default='not_requested', max_length=100)


def crop_frame(image, crop):
    if crop is None:
        return image.copy()
    w, h = image.size
    result = image.crop((round(crop.x*w), round(crop.y*h),
                         round((crop.x+crop.width)*w), round((crop.y+crop.height)*h)))
    if min(result.size) < 32:
        raise ValueError('The selected crop is too small')
    return result


def encode_evidence(observation, images, timestamps):
    """Length-prefixed manifest followed by the exact two selected cropped JPEGs."""
    selected = [0, len(images)-1]
    blobs = []
    for i in selected:
        out = io.BytesIO()
        images[i].save(out, format='JPEG', quality=90)
        blobs.append(out.getvalue())
    header = json.dumps({'observation': observation.model_dump(),
                         'timestamps': [timestamps[i] for i in selected],
                         'lengths': [len(b) for b in blobs]}, allow_nan=False).encode()
    packet = struct.pack('!I', len(header)) + header + b''.join(blobs)
    if len(packet) > MAX_EVIDENCE_BYTES:
        raise ValueError('Evidence exceeds byte limit')
    return packet


def decode_evidence(packet):
    if not 4 <= len(packet) <= MAX_EVIDENCE_BYTES:
        raise ValueError('Invalid evidence size')
    size, = struct.unpack('!I', packet[:4])
    if not 0 < size <= 20_000 or 4+size > len(packet):
        raise ValueError('Invalid evidence manifest')
    manifest = json.loads(packet[4:4+size])
    observation = Observation.model_validate(manifest['observation'])
    lengths, timestamps = manifest['lengths'], manifest['timestamps']
    if (len(lengths) != 2 or any(type(n) is not int or n <= 0 for n in lengths)
            or sum(lengths)+4+size != len(packet)
            or len(timestamps) != 2
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in timestamps)
            or timestamps[0] > timestamps[1]
            or timestamps[0] < observation.start or timestamps[-1] > observation.end):
        raise ValueError('Invalid evidence framing or timestamps')
    images, offset = [], 4+size
    for length in lengths:
        with Image.open(io.BytesIO(packet[offset:offset+length])) as image:
            if image.format != 'JPEG' or min(image.size) < 32 or max(image.size) > 1280:
                raise ValueError('Invalid evidence image')
            images.append(image.convert('RGB'))
        offset += length
    return observation, images, timestamps
