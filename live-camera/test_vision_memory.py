"""Persistent memory, coverage and GPT-Live protocol tests; no paid calls or GPU."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
import uuid

from PIL import Image

from onestreamer.adapter import parse_output
from protocol import Observation, MemoryQuery, Start, WorkerStart, encode_evidence
from server import CameraSession, Coordinator, create_app
from settings import Settings
from vision_memory import VisionMemory
from voice import VoiceAgent, compact_context
from fastapi.testclient import TestClient
import jwt


def event(sid, start=1, end=2, **kwargs):
    return Observation(session_id=sid, event_id=uuid.uuid4().hex,
                       start=start, end=end, state='response', **kwargs)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/'memory.sqlite3'
        self.memory = VisionMemory(self.path, evidence_limit=1)
        self.sid = uuid.uuid4().hex

    def tearDown(self):
        self.memory.close()
        self.directory.cleanup()

    def test_persistence_deduplication_and_session_isolation(self):
        observation = event(self.sid, description='cup beside laptop')
        self.assertTrue(self.memory.record(observation))
        self.assertFalse(self.memory.record(observation))
        self.memory.record(event(uuid.uuid4().hex, description='a secret cup'))
        self.memory.close()
        self.memory = VisionMemory(self.path)
        result = self.memory.search(MemoryQuery(session_id=self.sid, query='cup'))
        self.assertEqual([r['description'] for r in result['entries']], ['cup beside laptop'])
        self.assertEqual(len(self.memory.search(MemoryQuery(session_id=self.sid)).get('entries')), 1)

    def test_sampling_gaps_silence_and_evidence_retention(self):
        a = event(self.sid, 1, 3, response='cup', sample_times=[1, 1.25, 2.5, 2.75])
        b = event(self.sid, 5, 6, response='cup moved')
        self.memory.record(a); self.memory.record(b)
        images = [Image.new('RGB', (40, 40))]*2
        self.memory.save_evidence(a, images, [1, 2.75])
        self.memory.save_evidence(b, images, [5, 5.75])
        result = self.memory.search(MemoryQuery(session_id=self.sid, start=0, end=7))
        self.assertEqual(result['gaps'], [[0, 1], [1.5, 2.5], [3, 5], [6, 7]])
        self.assertFalse(result['entries'][0]['evidence_available'])
        self.assertTrue(result['entries'][1]['evidence_available'])
        inspected = self.memory.inspect(MemoryQuery(session_id=self.sid, start=5, end=6))
        self.assertEqual(sum(c['type'] == 'input_image' for c in inspected), 2)
        self.assertIn('no description, not no activity', result['limitations'])

    def test_auth_and_query_bounds(self):
        with self.assertRaises(ValueError):
            MemoryQuery(session_id=self.sid, start=3, end=1)
        with self.assertRaises(ValueError):
            Start(question='test', voice=True, vision_memory=False)
        with self.assertRaises(ValueError):
            event(self.sid, sample_times=[float('nan')])
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.addCleanup(loop.close)
        c = Coordinator(workspace=self.directory.name)
        c.settings.url = ''
        with TestClient(create_app(c)) as client:
            self.assertEqual(client.post('/api/memory', json={'session_id': self.sid}).status_code, 401)
            self.assertEqual(client.post('/api/memory', json={'session_id': self.sid},
                headers={'X-Access-Key': c.access_key}).status_code, 200)

    def test_continuous_sections_and_legacy_output(self):
        self.assertEqual(parse_output('</Observe>cup near desk</Summary>cup placed</Response>There it is'),
                         ('response', 'There it is', 'cup near desk', 'cup placed'))
        self.assertEqual(parse_output('</Observe>person standing'), ('silence', None, 'person standing', None))
        self.assertEqual(parse_output('</Standby>'), ('standby', None, None, None))
        self.assertEqual(parse_output('</Silence>'), ('silence', None, None, None))

    def test_voice_tokens_only_enable_expected_sources(self):
        settings = Settings(); settings.api_key = 'test'; settings.api_secret = 'x'*32
        claims = jwt.decode(settings.token('publisher', 'room', camera=True, voice=True),
                            settings.api_secret, algorithms=['HS256'])['video']
        self.assertEqual(claims['canPublishSources'], ['camera', 'microphone'])
        self.assertTrue(claims['canSubscribe']); self.assertFalse(claims['canPublishData'])
        agent = jwt.decode(settings.token('voice', 'room', agent=True),
                           settings.api_secret, algorithms=['HS256'])['video']
        self.assertEqual(agent['canPublishSources'], ['microphone'])
        self.assertTrue(agent['canPublish'])


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.memory = VisionMemory(Path(self.directory.name)/'memory.sqlite3')
        sid = uuid.uuid4().hex
        options = WorkerStart(session_id=sid, room='camera-'+sid, publisher='camera-'+sid,
                              question='Observe', voice=True)
        self.session = CameraSession(options, memory=self.memory)
        self.agent = VoiceAgent(Settings(), self.session, self.memory, key='fake')
        self.agent.ws = SimpleNamespace(send_json=AsyncMock(), close=AsyncMock())

    async def asyncTearDown(self):
        await self.agent.close()
        self.memory.close(); self.directory.cleanup()

    async def test_background_updates_bounded_and_do_not_force_speech(self):
        self.agent.state = 'listening'
        self.session.voice = self.agent
        observation = event(self.session.options.session_id, description='cup near laptop')
        self.session.record(observation); self.session.record(observation)
        self.assertEqual(self.agent.updates.qsize(), 1)
        task = asyncio.create_task(self.agent.send_context())
        for _ in range(10):
            await asyncio.sleep(.001)
            if self.agent.ws.send_json.await_count:
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        payload = self.agent.ws.send_json.call_args.args[0]
        self.assertEqual(payload['type'], 'session.thinking.append')
        for i in range(20):
            self.agent.observe(event(self.session.options.session_id, description=f'cup {i}'))
        self.assertEqual(self.agent.updates.qsize(), 8)
        self.assertEqual(self.agent.context_dropped, 12)
        self.assertLessEqual(len(compact_context('🦉'*500).encode()), 450)

    async def test_memory_tool_cannot_select_another_session(self):
        self.memory.record(event(self.session.options.session_id, description='own cup'))
        self.memory.record(event(uuid.uuid4().hex, description='secret cup'))
        output = await self.agent.tool_output({'name': 'search_vision_memory', 'arguments': '{}'})
        self.assertIn('own cup', output); self.assertNotIn('secret cup', output)
        output = await self.agent.tool_output({'name': 'search_vision_memory',
            'arguments': json.dumps({'session_id': uuid.uuid4().hex})})
        self.assertIn('Invalid', output)

    async def test_nested_tool_calls_deduplicated_and_continued_as_batch(self):
        self.memory.record(event(self.session.options.session_id, description='a cup'))
        item = {'type': 'function_call', 'call_id': 'call_1', 'name': 'search_vision_memory', 'arguments': '{}'}
        envelope = {'type': 'response.event', 'delegation_id': 'del_1',
                    'event': {'type': 'response.output_item.done', 'item': item}}
        await self.agent.handle_event(envelope); await self.agent.handle_event(envelope)
        self.assertEqual(len(self.agent.pending_calls['del_1']), 1)
        await self.agent.handle_event({'type': 'response.event', 'delegation_id': 'del_1',
                                      'event': {'type': 'response.completed'}})
        await asyncio.gather(*self.agent.tasks)
        sent = [c.args[0] for c in self.agent.ws.send_json.call_args_list]
        self.assertEqual([e['type'] for e in sent], ['response.item.create', 'response.create'])
        self.assertEqual(sent[0]['item']['call_id'], 'call_1')
        self.assertIn('a cup', sent[0]['item']['output'])

    async def test_audio_out_and_input_are_independent(self):
        self.agent.source = SimpleNamespace(capture_frame=AsyncMock(), aclose=AsyncMock())
        await self.agent.handle_event({'type': 'session.output_audio.delta',
            'delta': base64.b64encode(bytes(1920)).decode()})
        self.assertEqual(self.agent.source.capture_frame.await_count, 2)
        self.assertEqual(self.agent.ws.send_json.await_count, 0)

    async def test_api_error_is_visible_and_closes_transport(self):
        await self.agent.run_guarded(self.agent.handle_event({'type': 'error', 'error': {'message': 'private'}}))
        self.assertEqual(self.agent.state, 'error')
        self.assertNotIn('private', self.agent.error)
        self.agent.ws.close.assert_awaited_once()


class MemoryDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_saved_without_openai_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator(workspace=directory)
            sid = uuid.uuid4().hex
            options = WorkerStart(session_id=sid, room='camera-'+sid, publisher='camera-'+sid,
                                  question='Observe', verify=False, vision_memory=True)
            session = CameraSession(options, memory=coordinator.memory)
            coordinator.session = session
            observation = event(sid, description='A blue cup')
            packet = encode_evidence(observation, [Image.new('RGB', (80, 80), 'blue')]*2, [1, 1.75])
            class Reader:
                info = SimpleNamespace(name=observation.event_id)
                closed = False
                def __aiter__(self): return self.chunks()
                async def chunks(self): yield packet
                def close(self): self.closed = True
            reader = Reader()
            await coordinator.receive_evidence(session, reader)
            result = coordinator.memory.search(MemoryQuery(session_id=sid))
            self.assertTrue(reader.closed)
            self.assertTrue(result['entries'][0]['evidence_available'])
            self.assertEqual(coordinator.reviewer.get_results(sid), [])
            # The later text packet cannot erase the received-evidence UI status.
            session.record(observation)
            self.assertEqual(session.events[observation.event_id]['evidence_status'], 'received')
            await asyncio.to_thread(coordinator.reviewer.close)
            coordinator.memory.close()
