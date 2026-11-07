"""Real local LiveKit transport test, synthetic GPU/model and fake OpenAI client.

Start `livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1`, then run:
    uv run python live-camera/validate_api.py
No webcam, CUDA GPU, cloud credentials or paid API calls are used.
"""
import argparse
import asyncio
from contextlib import suppress
import io
import json
import logging
import os
import tempfile
import time
from types import SimpleNamespace
import uuid

import httpx
from livekit import rtc
from PIL import Image

from gpu_worker import GPUWorker
from protocol import Observation, crop_frame
from server import Coordinator, create_app
from settings import Settings
from verification import Assessment, Reviewer


class SyntheticEngine:
    def __init__(self):
        self.options = None
        self.steps = 0
        self.last_end = 0

    def start(self, options):
        self.options, self.steps, self.last_end = options, 0, 0

    def stop(self):
        self.options = None

    def step(self, batch):
        time.sleep(1.4)  # Reception must continue while inference runs in another thread.
        times = [t for t, _ in batch]
        images = [crop_frame(im, self.options.crop) for _, im in batch]
        result = Observation(session_id=self.options.session_id, event_id=uuid.uuid4().hex,
            state='response', start=times[0], end=times[-1]+.25,
            response='A blue object', inference_seconds=1.4,
            skipped_seconds=max(0, times[0]-self.last_end), input_size=list(images[-1].size))
        self.steps += 1
        self.last_end = result.end
        return result, images, times


async def until(predicate, timeout=15):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(.1)
    raise AssertionError('Timed out waiting for integration condition')


async def validate(args):
    os.environ.update(LIVEKIT_URL=args.url, LIVEKIT_API_KEY='devkey', LIVEKIT_API_SECRET='secret',
        LIVEKIT_CONTROL_ROOM='test-control-'+uuid.uuid4().hex,
        LIVEKIT_BACKEND_IDENTITY='test-backend', LIVEKIT_WORKER_IDENTITY='test-gpu', OPENAI_API_KEY='')
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        for part in kwargs['input'][0]['content']:
            if part['type'] == 'input_image':
                import base64
                with Image.open(io.BytesIO(base64.b64decode(part['image_url'].split(',')[1]))) as image:
                    assert image.size == (160, 120), image.size
        return SimpleNamespace(output_parsed=Assessment(verdict='agree', object_label='blue object',
            certainty='medium', evidence=['Blue surface'], explanation='Synthetic review.'))
    settings = Settings()
    worker = GPUWorker(settings, engine_factory=SyntheticEngine)
    publisher = rtc.Room()
    pump_task = None
    with tempfile.TemporaryDirectory() as directory:
        reviewer = Reviewer(directory, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), cooldown=0)
        c = Coordinator(settings=settings, workspace=directory, reviewer=reviewer)
        app = create_app(c)
        try:
            await worker.connect()
            async with app.router.lifespan_context(app):
                await until(lambda: c.worker_ready)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    headers = {'X-Access-Key': c.access_key}
                    response = await client.post('/api/start', headers=headers, json={
                        'question': 'Describe objects', 'verify': True,
                        'crop': {'x': .25, 'y': .25, 'width': .5, 'height': .5}})
                    assert response.status_code == 200, response.text
                    result = response.json()
                    sid = result['session_id']
                    await publisher.connect(result['url'], result['token'], rtc.RoomOptions(auto_subscribe=False))
                    source = rtc.VideoSource(320, 240)
                    track = rtc.LocalVideoTrack.create_video_track('synthetic-camera', source)
                    publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
                    await publisher.local_participant.publish_track(track, publish_options)
                    async def pump():
                        while True:
                            frame = rtc.VideoFrame(320, 240, rtc.VideoBufferType.RGBA, bytes([0, 0, 255, 255])*320*240)
                            source.capture_frame(frame)
                            await asyncio.sleep(1/15)
                    pump_task = asyncio.create_task(pump())
                    await until(lambda: len(c.session.events) >= 2 and len(reviewer.get_results(sid)) >= 2, 25)
                    rows = list(c.session.events.values())
                    assert all(r['input_size'] == [160, 120] for r in rows), rows
                    assert len(worker.active.frames) <= 4
                    reviews = reviewer.get_results(sid)
                    assert all(r['event_id'] in c.session.events for r in reviews)
                    assert any(r['skipped_seconds'] > 0 for r in rows[1:]), rows
                    # Manual review comes from the GPU's fresh buffer, not a browser upload.
                    await until(lambda: not worker.active.sending and worker.active.evidence.empty())
                    response = await client.post('/api/verify', headers=headers, json={'session_id': sid})
                    assert response.status_code == 200, response.text
                    manual_id = response.json()['event_id']
                    await until(lambda: any(r.get('event_id') == manual_id and r['status'] == 'complete'
                                          for r in reviewer.get_results(sid)))
                    events = (await client.post('/api/events', headers=headers,
                                              json={'session_id': sid})).json()
                    assert events['active'] and events['reviews']
                    assert (await client.post('/api/start', headers=headers,
                                              json={'question': 'second'})).status_code == 409
                    response = await client.post('/api/stop', headers=headers, json={'session_id': sid})
                    assert response.status_code == 200, response.text
                    await until(lambda: worker.active is None)
                    assert not c.session.active
                    # Another session must have an independent room and model history.
                    response = await client.post('/api/start', headers=headers, json={'question': 'new'})
                    assert response.status_code == 200, response.text
                    next_sid = response.json()['session_id']
                    assert sid != next_sid
                    assert worker.engine.steps == 0
                    assert not c.session.events
                    assert (await client.post('/api/events', headers=headers,
                                              json={'session_id': sid})).status_code == 409
                    await client.post('/api/stop', headers=headers, json={'session_id': next_sid})
                    await until(lambda: worker.active is None)
                    print(json.dumps({'result': 'PASS', 'observations': len(rows), 'reviews': len(calls),
                        'checks': ['real WebRTC video', 'GPU-side sampling during slow inference',
                                   'bounded frame buffer', 'cropped evidence byte streams',
                                   'event/review correlation', 'manual evidence request',
                                   'single-session exclusion', 'stop and room cleanup',
                                   'new-session isolation']}, indent=2))
        finally:
            if pump_task:
                pump_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pump_task
            await publisher.disconnect()
            await worker.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:7880', choices=['ws://127.0.0.1:7880'])
    asyncio.run(validate(parser.parse_args()))
