"""Local LiveKit audio roundtrip with a fake GPT-Live socket; no API key or GPU.
Run livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1 first.
"""
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import uuid

import aiohttp
from livekit import rtc
from PIL import Image

from protocol import Observation, WorkerStart
from server import CameraSession
from settings import Settings
from vision_memory import VisionMemory
from voice import VoiceAgent
from validate_api import until


class FakeLiveSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.audio_count = 0
        self.injected = False

    async def send_json(self, event):
        self.sent.append(event)
        if event['type'] == 'session.start':
            await self.incoming.put({'type': 'session.started', 'session': {'model': 'gpt-live-1'}})
        elif event['type'] == 'session.input_audio.append':
            self.audio_count += 1
            # Echo the actual received PCM through the model-output path.
            await self.incoming.put({'type': 'session.output_audio.delta', 'delta': event['audio']})
            if not self.injected:
                self.injected = True
                await self.incoming.put({'type': 'response.event', 'delegation_id': 'del_memory',
                    'event': {'type': 'response.output_item.done', 'item': {'type': 'function_call',
                        'name': 'inspect_interval', 'arguments': '{"start":1,"end":2,"query":""}', 'call_id': 'call_memory'}}})
                await self.incoming.put({'type': 'response.event', 'delegation_id': 'del_memory',
                    'event': {'type': 'response.completed'}})

    async def receive_json(self):
        return await self.incoming.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.incoming.get()
        if event is None:
            raise StopAsyncIteration
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))

    async def close(self):
        if not self.closed:
            self.closed = True
            await self.incoming.put(None)


async def validate():
    settings = Settings()
    settings.url, settings.api_key, settings.api_secret = 'ws://127.0.0.1:7880', 'devkey', 'secret'
    sid = uuid.uuid4().hex
    options = WorkerStart(session_id=sid, room='camera-'+sid, publisher='camera-'+sid,
                          question='Describe the scene', voice=True)
    publisher = rtc.Room()
    tasks, heard = [], []
    socket = FakeLiveSocket()
    async def connector():
        return socket
    with tempfile.TemporaryDirectory() as directory:
        memory = VisionMemory(Path(directory)/'memory.sqlite3')
        session = CameraSession(options, memory=memory)
        agent = VoiceAgent(settings, session, memory, key='fake', connector=connector)
        session.voice = agent
        mic_source = None
        async def listen(track):
            stream = rtc.AudioStream(track, sample_rate=24000, num_channels=1, capacity=5)
            try:
                async for frame in stream:
                    heard.append(bytes(frame.frame.data))
            finally:
                await stream.aclose()
        @publisher.on('track_subscribed')
        def subscribed(track, publication, participant):
            if participant.identity == agent.identity and track.kind == rtc.TrackKind.KIND_AUDIO:
                tasks.append(asyncio.create_task(listen(track)))
        @publisher.on('track_published')
        def published(publication, participant):
            if participant.identity == agent.identity:
                publication.set_subscribed(True)
        try:
            await agent.open()
            await publisher.connect(settings.url,
                settings.token(options.publisher, options.room, camera=True, voice=True),
                rtc.RoomOptions(auto_subscribe=False))
            for participant in publisher.remote_participants.values():
                if participant.identity == agent.identity:
                    for publication in participant.track_publications.values():
                        publication.set_subscribed(True)
            observation = Observation(session_id=sid, event_id=uuid.uuid4().hex,
                state='silence', start=1, end=2, description='A cup beside the laptop')
            session.record(observation)
            memory.save_evidence(observation, [Image.new('RGB', (80, 80), 'blue')]*2, [1, 1.75])
            mic_source = rtc.AudioSource(24000, 1, queue_size_ms=100)
            mic = rtc.LocalAudioTrack.create_audio_track('test-mic', mic_source)
            await publisher.local_participant.publish_track(mic,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            async def pump():
                # Alternating nonzero samples survive Opus encoding; no sound hardware used.
                pcm = (int(6000).to_bytes(2, 'little', signed=True)*24 +
                       int(-6000).to_bytes(2, 'little', signed=True)*24)*10
                while True:
                    await mic_source.capture_frame(rtc.AudioFrame(pcm, 24000, 1, 480))
                    await asyncio.sleep(.02)
            tasks.append(asyncio.create_task(pump()))
            await until(lambda: socket.audio_count >= 10 and len(heard) >= 5 and any(
                e['type'] == 'response.create' for e in socket.sent), 20)
            outputs = [e['item']['output'] for e in socket.sent if e['type'] == 'response.item.create']
            assert outputs and sum(p['type'] == 'input_image' for p in outputs[0]) == 2
            assert any(e['type'] == 'session.thinking.append' for e in socket.sent)
            assert any(any(blob) for blob in heard)
            await agent.close()
            assert socket.closed and agent.state == 'off'
            print(json.dumps({'result': 'PASS', 'microphone_frames': socket.audio_count,
                'speaker_frames': len(heard), 'checks': ['real LiveKit microphone subscription',
                '24 kHz PCM input', 'audio output over LiveKit', 'background visual context',
                'memory tool with image evidence', 'Responses continuation', 'voice cleanup'],
                'limitation': 'GPT-Live socket is simulated; provider/model access is not tested'}, indent=2))
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            await agent.close()
            await publisher.disconnect()
            if mic_source:
                await mic_source.aclose()
            memory.close()


if __name__ == '__main__':
    asyncio.run(validate())
