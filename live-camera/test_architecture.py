"""Protocol/security tests; no GPU, LiveKit service or paid API calls required."""
import asyncio
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
import jwt
from PIL import Image

from gpu_worker import GPUWorker
from protocol import (Crop, MAX_EVIDENCE_BYTES, Observation, Start, WorkerStart,
                      decode_evidence, encode_evidence)
from server import CameraSession, Coordinator, create_app
from settings import Settings
from verification import Reviewer
from test_verification import decision_result, mock_client


def observation(sid=None):
    return Observation(session_id=sid or uuid.uuid4().hex, event_id=uuid.uuid4().hex,
                       state='response', start=1, end=2, response='A blue object')


def options(sid=None):
    sid = sid or uuid.uuid4().hex
    return WorkerStart(session_id=sid, room='camera-'+sid, publisher='camera-'+sid,
                       question='Describe objects', verify=True)


class ProtocolTests(unittest.TestCase):
    def test_example_credentials_do_not_enable_livekit_connection(self):
        with patch.dict('os.environ', {'LIVEKIT_URL': 'wss://YOUR_PROJECT.livekit.cloud',
                'LIVEKIT_API_KEY': 'YOUR_LIVEKIT_API_KEY',
                'LIVEKIT_API_SECRET': 'YOUR_LIVEKIT_API_SECRET'}):
            settings = Settings()
        self.assertFalse(settings.configured)
        self.assertIn('Replace the example', settings.configuration_error)

    def test_evidence_roundtrip_preserves_event_and_selected_frames(self):
        event = observation()
        packet = encode_evidence(event, [Image.new('RGB', (100, 80), c)
            for c in ['red', 'green', 'blue', 'yellow']], [1, 1.25, 1.5, 1.75])
        result, images, times = decode_evidence(packet)
        self.assertEqual(event, result)
        self.assertEqual(times, [1, 1.75])
        self.assertEqual(len(images), 2)
        self.assertGreater(images[0].getpixel((0, 0))[0], 240)
        self.assertGreater(images[1].getpixel((0, 0))[1], 240)

    def test_evidence_rejects_truncation_and_oversize(self):
        packet = encode_evidence(observation(), [Image.new('RGB', (80, 80))]*4,
                                 [1, 1.25, 1.5, 1.75])
        for invalid in [b'', packet[:-1], b'x'*(MAX_EVIDENCE_BYTES+1), struct.pack('!I', 30000)]:
            with self.assertRaises(ValueError):
                decode_evidence(invalid)

    def test_nonfinite_time_and_mismatched_room_rejected(self):
        packet = encode_evidence(observation(), [Image.new('RGB', (80, 80))]*4,
                                 [1, 1.25, 1.5, 1.75])
        size, = struct.unpack('!I', packet[:4])
        manifest = json.loads(packet[4:4+size])
        manifest['timestamps'][1] = float('nan')
        header = json.dumps(manifest).encode()
        with self.assertRaises(ValueError):
            decode_evidence(struct.pack('!I', len(header))+header+packet[4+size:])
        with self.assertRaises(ValueError):
            WorkerStart(**{**options().model_dump(), 'room': 'camera-'+uuid.uuid4().hex})
        with self.assertRaises(ValueError):
            Crop(x=.8, y=0, width=.5, height=1)
        with self.assertRaises(ValueError):
            Start(question='   ')

    def test_camera_token_cannot_subscribe_or_send_data(self):
        with patch.dict('os.environ', {'LIVEKIT_API_KEY': 'test', 'LIVEKIT_API_SECRET': 's'*32}):
            settings = Settings()
            claims = jwt.decode(settings.token('camera-a', 'camera-b', camera=True),
                                settings.api_secret, algorithms=['HS256'])
        grants = claims['video']
        self.assertEqual(grants['room'], 'camera-b')
        self.assertEqual(grants['canPublishSources'], ['camera'])
        self.assertFalse(grants['canSubscribe'])
        self.assertFalse(grants['canPublishData'])


class BackendTests(unittest.TestCase):
    def test_local_api_auth_and_missing_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            c = Coordinator(workspace=directory)
            c.settings.url = c.settings.api_key = c.settings.api_secret = ''
            with TestClient(create_app(c)) as client:
                self.assertEqual(client.get('/health').json()['ready'], False)
                self.assertEqual(client.get('/').status_code, 200)
                self.assertEqual(client.get('/app.js').status_code, 200)
                self.assertEqual(client.post('/api/start', json={'question': 'test'}).status_code, 401)
                response = client.post('/api/start', json={'question': 'test'},
                                       headers={'X-Access-Key': c.access_key})
                self.assertEqual(response.status_code, 503)
                self.assertFalse((Path(directory)/'local-access-key').stat().st_mode & 0o077)


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_correlated_deduplicated_and_old_session_rejected(self):
        calls = []
        def handle(request):
            calls.append(request)
            import httpx
            return httpx.Response(200, json=decision_result())
        class Reader:
            def __init__(self, packet, event_id=None):
                self.packet, self.closed = packet, False
                if event_id:
                    self.info = SimpleNamespace(name=event_id)
            def __aiter__(self): return self.chunks()
            async def chunks(self):
                yield self.packet[:20]
                yield self.packet[20:]
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as directory:
            reviewer = Reviewer(directory, client=mock_client(handle), cooldown=0)
            c = Coordinator(workspace=directory, reviewer=reviewer)
            s = CameraSession(options())
            c.session = s
            event = observation(s.options.session_id)
            packet = encode_evidence(event, [Image.new('RGB', (80, 80), 'blue')]*4, [1, 1.25, 1.5, 1.75])
            reader = Reader(packet, event.event_id)
            await c.receive_evidence(s, reader)
            await asyncio.to_thread(reviewer.future.result, 3)
            self.assertTrue(reader.closed)
            result = reviewer.get_results(s.options.session_id)[0]
            self.assertEqual(result['event_id'], event.event_id)
            self.assertEqual(s.events[event.event_id]['review']['id'], result['id'])
            await c.receive_evidence(s, Reader(packet))
            self.assertEqual(len(calls), 1)
            # A failed stream can arrive before its observation; failure must not
            # leave the later UI event marked pending forever.
            failed = observation(s.options.session_id)
            failed.evidence_status = 'pending'
            await c.receive_evidence(s, Reader(b'bad', failed.event_id))
            s.record(failed)
            self.assertEqual(s.events[failed.event_id]['evidence_status'], 'transfer_failed')
            s.active = False
            await c.receive_evidence(s, Reader(packet))
            self.assertEqual(len(calls), 1)
            reviewer.close()

    async def test_worker_rejects_foreign_coordinator_before_gpu_work(self):
        worker = GPUWorker(Settings(), engine_factory=lambda: None)
        response = json.loads(await worker.rpc(SimpleNamespace(caller_identity='intruder',
                                                               payload='{"action":"start"}')))
        self.assertIn('Unauthorized', response['error'])
        self.assertIsNone(worker.active)
        worker.executor.shutdown()


if __name__ == '__main__':
    unittest.main()
