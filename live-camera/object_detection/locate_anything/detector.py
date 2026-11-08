"""Event-triggered grounding and bounded, box-initialized CPU tracking."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import re

from protocol import DetectionBox


def parse_boxes(answer, fallback):
    """NVIDIA coordinates are integer xyxy in [0, 1000]; reject malformed boxes."""
    boxes = []
    label = fallback[:80] or 'Object'
    for match in re.finditer(r'<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', answer, re.S):
        if match.group(1) is not None:
            label = match.group(1).strip()[:80] or label
            continue
        coords = [int(v)/1000 for v in match.groups()[1:]]
        try:
            box = DetectionBox(label=label, box=coords)
        except ValueError:
            continue
        boxes.append(box)
        if len(boxes) == 8:
            break
    return boxes


class LocateAnythingDetector:
    def __init__(self):
        self.process = None
        self.lock = asyncio.Lock()

    async def start(self):
        source = os.environ.get('LOCATE_ANYTHING_SOURCE') or '/workspace/Eagle/Embodied'
        python = os.environ.get('LOCATE_ANYTHING_PYTHON') or '/workspace/locate-anything-env/bin/python'
        if not Path(python).is_file() or not Path(source, 'locateanything_worker.py').is_file():
            raise ValueError('Run setup-tracking.sh or configure LOCATE_ANYTHING_SOURCE and LOCATE_ANYTHING_PYTHON')
        environment = {**os.environ, 'LOCATE_ANYTHING_SOURCE': source,
            'LOCATE_ANYTHING_MODEL': os.environ.get('LOCATE_ANYTHING_MODEL') or '/workspace/models/LocateAnything-3B'}
        self.process = await asyncio.create_subprocess_exec(
            python, str(Path(__file__).with_name('process.py')),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=131072, env=environment)
        try:
            ready = await self.read(300)
            if not ready.get('ready'):
                raise RuntimeError('LocateAnything failed to load')
        except BaseException:
            await self.close()
            raise

    async def read(self, timeout):
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            raise RuntimeError('LocateAnything process exited; restart the GPU worker')
        return json.loads(line)

    async def detect(self, image, query):
        async with self.lock:
            data = io.BytesIO()
            image.save(data, 'JPEG', quality=90)
            request = json.dumps({'image': base64.b64encode(data.getvalue()).decode(), 'query': query})
            try:
                self.process.stdin.write(request.encode()+b'\n')
                await self.process.stdin.drain()
                reply = await self.read(60)
            except BaseException:
                # A timed-out reply cannot be mistaken for a later request's result.
                await self.close()
                raise
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            return parse_boxes(reply['answer'], query)

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
