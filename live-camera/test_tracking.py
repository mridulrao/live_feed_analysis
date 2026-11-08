"""Tracking protocol, time alignment, isolation and local API tests without GPU weights."""
import asyncio
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from gpu_worker import GPUWorker, VideoSession
from protocol import Crop, TrackingUpdate, TrackedBox, WorkerStart
from server import CameraSession, Coordinator, create_app
from object_detection.locate_anything.detector import LocateAnythingDetector, parse_boxes
from tracking.csrt import CSRTTracker
from settings import Settings


def options(**kwargs):
    sid = uuid.uuid4().hex
    return WorkerStart(session_id=sid, room='camera-'+sid, publisher='camera-'+sid,
        question='Watch the bottle', tracking=True, **kwargs)


def box():
    return TrackedBox(track_id=1, label='bottle', box=[.1, .2, .4, .8])


class ProtocolTests(unittest.TestCase):
    def test_nvidia_output_labels_and_invalid_coordinates(self):
        answer = '<ref>red bottle</ref><box><100><200><400><800></box>'
        answer += '<box><900><0><100><500></box><box><0><0><1001><400></box>'
        result = parse_boxes(answer, 'fallback')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, 'red bottle')
        self.assertEqual(result[0].box, [.1, .2, .4, .8])
        self.assertEqual(parse_boxes('<box>none</box>', 'object'), [])

    def test_nonfinite_box_and_oversized_updates_rejected(self):
        for coords in ([0, 0, float('nan'), 1], [0, 0, 2, 1], [.5, .5, .2, .2]):
            with self.assertRaises(ValueError):
                TrackedBox(track_id=1, label='object', box=coords)
        with self.assertRaises(ValueError):
            TrackingUpdate(session_id=uuid.uuid4().hex, sequence=1,
                frame_time=1, status='tracking', boxes=[box()]*9)

class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_tracking_api_auth_and_stale_clear(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            c = Coordinator(workspace=directory)
            c.settings.url = c.settings.api_key = c.settings.api_secret = ''
            c.session = CameraSession(options())
            c.session.tracking = {'boxes': [box().model_dump()]}
            c.session.tracking_received = time.monotonic()
            body = {'session_id': c.session.options.session_id}
            with TestClient(create_app(c)) as client:
                headers = {'X-Access-Key': c.access_key}
                self.assertEqual(client.post('/api/tracks', json=body).status_code, 401)
                self.assertTrue(client.post('/api/tracks', json=body, headers=headers).json()['tracking']['boxes'])
                c.session.tracking_received -= 2
                self.assertIsNone(client.post('/api/tracks', json=body, headers=headers).json()['tracking'])
                c.session.active = False


    async def test_persistent_locator_bridge_and_diagnostics_isolation(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'locateanything_worker.py').write_text(
                "class LocateAnythingWorker:\n"
                "    def __init__(self, *args, **kwargs): print('model diagnostic')\n"
                "    def ground_multi(self, image, query, **kwargs):\n"
                "        print('inference diagnostic')\n"
                "        return {'answer': '<ref>bottle</ref><box><100><200><400><800></box>'}\n")
            with patch.dict(os.environ, {'LOCATE_ANYTHING_SOURCE': directory,
                    'LOCATE_ANYTHING_PYTHON': sys.executable, 'LOCATE_ANYTHING_BATCH_RUNTIME': '0'}):
                locator = LocateAnythingDetector()
                try:
                    await locator.start()
                    process_id = locator.process.pid
                    for _ in range(2):
                        boxes = await locator.detect(Image.new('RGB', (80, 80)), 'bottle')
                        self.assertEqual(boxes[0].label, 'bottle')
                        self.assertEqual(locator.process.pid, process_id)
                finally:
                    await locator.close()
                self.assertIsNone(locator.process)

    async def test_crop_coordinates_mapped_to_full_frame(self):
        class Detector:
            async def detect(self, image, query):
                self.query = query
                return [box()]
        detector = Detector()
        session = VideoSession(SimpleNamespace(detector=detector), options(
            crop=Crop(x=.25, y=.25, width=.5, height=.5)))
        await session.localize(Image.new('RGB', (80, 80)), 2, 'bottle')
        timestamp, boxes, generation = session.pending_detection
        self.assertEqual(timestamp, 2)
        self.assertEqual(boxes[0].box, [.3, .35, .45, .65])

    async def test_toggle_discards_inflight_detection_and_keeps_models_loaded(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class Detector:
            async def detect(self, image, query):
                entered.set()
                await release.wait()
                return [box()]
        detector = Detector()
        tracker = SimpleNamespace(reset=lambda: None)
        async def cpu(fn, *args):
            return fn(*args)
        worker = SimpleNamespace(detector=detector, cpu=cpu,
            settings=SimpleNamespace(localization_retry_seconds=.001))
        session = VideoSession(worker, options())
        session.tracker = tracker
        task = asyncio.create_task(session.localize(None, 1, 'bottle'))
        await entered.wait()
        await session.set_tracking(False)
        release.set()
        await task
        self.assertIsNone(session.pending_detection)
        self.assertIs(worker.detector, detector)
        session.trigger_localization(SimpleNamespace(response='bottle'), None, 2)
        self.assertIsNone(session.detection_request)
        await session.set_tracking(True)
        self.assertEqual(session.generation, 2)
        session.trigger_localization(SimpleNamespace(response='bottle'), None, 3)
        self.assertEqual(session.detection_request[1:], (3, 'bottle', 2))

    async def test_visual_tracking_runs_without_repeated_localization(self):
        image = Image.new('RGB', (80, 80))
        detector = SimpleNamespace(detect=AsyncMock(return_value=[box()]))
        updates = []
        tracker = SimpleNamespace(initialize=lambda im, boxes: boxes,
            update=lambda im: updates.append(im) or [box()])
        async def cpu(fn, *args):
            return fn(*args)
        worker = SimpleNamespace(detector=detector, cpu=cpu,
            settings=SimpleNamespace(localization_retry_seconds=.001))
        session = VideoSession(worker, options())
        session.tracker = tracker
        session.epoch = time.monotonic()-1.2
        session.latest_video_time = 1.2
        session.frames.append((1., image))
        session.tracking_frames.extend([(1., image), (1.1, image), (1.2, image)])
        publications = []
        async def publish(timestamp, boxes):
            publications.append((session.track_status, boxes))
            if session.track_status == 'tracking':
                if len(publications) == 1:
                    session.tracking_frames.append((1.3, image))
                    session.latest_video_time = 1.3
                    session.epoch = time.monotonic()-1.3
                    # New OneStreamer responses cannot replace a healthy visual target.
                    session.trigger_localization(SimpleNamespace(response='another object'), image, 1.3)
                else:
                    session.stopped.set()
        session.publish_tracks = publish
        session.trigger_localization(SimpleNamespace(response='bottle'), image, 1.)
        await asyncio.wait_for(asyncio.gather(session.detect_loop(), session.track_loop()), 2)
        self.assertEqual(detector.detect.await_count, 1)
        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(session.target_query, 'bottle')
        self.assertTrue(publications[-1][1])

    async def test_csrt_loss_relocalizes_without_another_observer_response(self):
        image = Image.new('RGB', (80, 80))
        detector = SimpleNamespace(detect=AsyncMock(return_value=[box()]))
        initialized = []
        def initialize(im, detections):
            initialized.append(im)
            return [box().model_copy(update={'track_id': len(initialized)})]
        tracker = SimpleNamespace(initialize=initialize, update=lambda im: [])
        async def cpu(fn, *args):
            return fn(*args)
        session = VideoSession(SimpleNamespace(detector=detector, cpu=cpu,
            settings=SimpleNamespace(localization_retry_seconds=.001)), options())
        session.tracker = tracker
        session.epoch = time.monotonic()-1.1
        session.latest_video_time = 1.1
        session.frames.append((1., image))
        session.tracking_frames.extend([(1., image), (1.1, image)])
        publications = []
        async def publish(timestamp, boxes):
            publications.append((session.track_status, boxes))
            if session.track_status == 'lost':
                session.frames.append((2., image))
                session.tracking_frames.append((2., image))
                session.latest_video_time = 2.
                session.epoch = time.monotonic()-2.
            elif session.track_status == 'tracking':
                session.stopped.set()
        session.publish_tracks = publish
        session.trigger_localization(SimpleNamespace(response='bottle'), image, 1.)
        await asyncio.wait_for(asyncio.gather(session.detect_loop(), session.track_loop()), 2)
        self.assertEqual(detector.detect.await_count, 2)
        self.assertEqual(publications[-1][1][0].track_id, 2)

    async def test_tracking_ready_is_withheld_until_frame_catchup(self):
        image = Image.new('RGB', (80, 80))
        async def cpu(fn, *args):
            return fn(*args)
        session = VideoSession(SimpleNamespace(cpu=cpu, settings=SimpleNamespace()), options())
        session.tracker = SimpleNamespace(initialize=lambda im, detections: detections,
            update=lambda im: [box()])
        session.epoch = time.monotonic()-3
        session.latest_video_time = 3
        session.tracking_frames.extend([(1., image), (2., image)])
        session.pending_detection = (1., [box()], 0)
        session.mapping_pending = True
        publications = []
        async def publish(timestamp, boxes):
            publications.append((session.track_status, boxes))
            if len(publications) == 1:
                self.assertEqual(session.track_status, 'localizing')
                self.assertEqual(boxes, [])
                session.tracking_frames.append((3., image))
            else:
                session.stopped.set()
        session.publish_tracks = publish
        await asyncio.wait_for(session.track_loop(), 1)
        self.assertEqual(publications[-1][0], 'tracking')
        self.assertTrue(publications[-1][1])

    async def test_expired_mapping_frame_is_not_used_on_current_video(self):
        image = Image.new('RGB', (80, 80))
        session = VideoSession(SimpleNamespace(settings=SimpleNamespace()), options())
        session.tracker = SimpleNamespace(initialize=lambda *args: self.fail('Expired box must not initialize CSRT'))
        session.tracking_frames.append((2., image))
        session.pending_detection = (1., [box()], 0)
        session.mapping_pending = True
        async def publish(timestamp, boxes):
            self.assertEqual(boxes, [])
            self.assertIn('expired', session.track_message)
            session.stopped.set()
        session.publish_tracks = publish
        await session.track_loop()
        self.assertFalse(session.mapping_pending)

    async def test_models_preload_once_and_disabled_flag_skips_plugins(self):
        for enabled in (True, False):
            with patch.dict(os.environ, {'TRACKING_MODELS_ENABLED': str(enabled).lower()}):
                settings = Settings()
            detector = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
            worker = GPUWorker(settings, engine_factory=lambda: object())
            with patch('gpu_worker.create_detector', return_value=detector) as factory, \
                    patch('gpu_worker.create_tracker', return_value=object()):
                try:
                    await worker.prepare_models()
                    await worker.prepare_models()
                    self.assertEqual(factory.call_count, int(enabled))
                    self.assertEqual(detector.start.await_count, int(enabled))
                finally:
                    await worker.close()

    async def test_runtime_api_and_old_generation_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            c = Coordinator(workspace=directory)
            c.settings.url = c.settings.api_key = c.settings.api_secret = ''
            session = CameraSession(options())
            c.session = session
            c.attach_handlers(session)
            sid = session.options.session_id
            with TestClient(create_app(c)) as client:
                headers = {'X-Access-Key': c.access_key}
                self.assertEqual(client.post('/api/tracking', json={'session_id': sid, 'enabled': False}).status_code, 401)
                with patch.object(c, 'command', new=AsyncMock(return_value={'enabled': False, 'generation': 1})):
                    result = client.post('/api/tracking', json={'session_id': sid, 'enabled': False}, headers=headers)
                self.assertEqual(result.status_code, 200)
                self.assertFalse(session.options.tracking)
                self.assertEqual(session.tracking_generation, 1)
                session.options.tracking = True
                def send(generation, identity=None):
                    update = TrackingUpdate(session_id=sid, generation=generation, sequence=generation+1,
                        frame_time=1, status='tracking', boxes=[box()])
                    session.room.emit('data_received', SimpleNamespace(topic='onestreamer.tracks.v1',
                        participant=SimpleNamespace(identity=identity or c.settings.worker_identity),
                        data=update.model_dump_json().encode()))
                send(0)
                self.assertIsNone(session.tracking)
                send(1, 'intruder')
                self.assertIsNone(session.tracking)
                send(1)
                self.assertEqual(session.tracking['generation'], 1)
                session.active = False


class CSRTTests(unittest.TestCase):
    @staticmethod
    def scene(x=20, y=30, width=160):
        import numpy as np
        pixels = np.full((120, width, 3), 25, dtype=np.uint8)
        texture = np.random.default_rng(7).integers(0, 256, (32, 40, 3), dtype=np.uint8)
        pixels[y:y+32, x:x+40] = texture
        return Image.fromarray(pixels)

    def test_real_csrt_follows_texture_and_keeps_id(self):
        tracker = CSRTTracker()
        target = box().model_copy(update={'box': [20/160, 30/120, 60/160, 62/120]})
        initial = tracker.initialize(self.scene(), [target])
        self.assertEqual(len(initial), 1)
        for step in range(1, 6):
            result = tracker.update(self.scene(x=20+step*3, y=30+step))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].track_id, initial[0].track_id)
            self.assertAlmostEqual(result[0].box[0], (20+step*3)/160, delta=.04)
            self.assertEqual(result[0].label, 'bottle')

    def test_reacquisition_uses_new_ids_and_reset_clears(self):
        tracker = CSRTTracker()
        first = tracker.initialize(self.scene(), [box()])
        second = tracker.initialize(self.scene(), [box()])
        self.assertGreater(second[0].track_id, first[0].track_id)
        tracker.reset()
        self.assertEqual(tracker.update(self.scene()), [])

    def test_frame_size_change_and_failed_update_clear_boxes(self):
        tracker = CSRTTracker()
        tracker.initialize(self.scene(), [box()])
        self.assertEqual(tracker.update(self.scene(width=180)), [])
        failed = SimpleNamespace(update=lambda image: (False, (0, 0, 0, 0)))
        tracker.size = (160, 120)
        tracker.tracks = [(1, 'bottle', failed)]
        self.assertEqual(tracker.update(self.scene()), [])
        self.assertEqual(tracker.tracks, [])

    def test_backend_defaults_load_models_and_select_csrt(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            self.assertTrue(settings.tracking_models_enabled)
            self.assertEqual(settings.tracking_plugin, 'csrt')
