"""Outbound-only LiveKit worker. No HTTP listener and no OpenAI dependency."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import json
import logging
import signal
import time
import uuid

from livekit import rtc
from PIL import Image

from onestreamer import OneStreamer
from protocol import (EVENT_TOPIC, EVIDENCE_TOPIC, Observation, SessionID,
                      WorkerStart, crop_frame, encode_evidence, TRACK_TOPIC, TrackingUpdate, TrackingControl)
from object_detection import create_detector
from tracking import create_tracker
from tracking.verification import tracks_match
from settings import Settings

log = logging.getLogger('gpu-worker')


class GPUWorker:
    def __init__(self, settings, engine_factory=OneStreamer):
        self.settings, self.engine_factory = settings, engine_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='onestreamer')
        self.engine = None
        self.detector = None
        self.tracker = None
        self.tracking_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-tracker')
        self.control = rtc.Room()
        self.active = None
        self.guard = asyncio.Lock()

    async def gpu(self, fn, *args):
        # Shield prevents cancellation from releasing ownership of running CUDA work.
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(self.executor, fn, *args))

    async def cpu(self, fn, *args):
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(self.tracking_executor, fn, *args))

    async def prepare_models(self):
        if self.engine is None:
            self.engine = await self.gpu(self.engine_factory)
        if self.settings.tracking_models_enabled and self.detector is None:
            tracker = await self.cpu(create_tracker, self.settings.tracking_plugin)
            detector = create_detector(self.settings.detection_plugin)
            await detector.start()
            self.detector, self.tracker = detector, tracker

    async def connect(self):
        await self.prepare_models()
        await self.control.connect(self.settings.url,
            self.settings.token(self.settings.worker_identity, self.settings.control_room),
            rtc.RoomOptions(auto_subscribe=False))
        self.control.local_participant.register_rpc_method('camera.control', self.rpc)
        log.info('Model ready; waiting in control room %s', self.settings.control_room)

    async def rpc(self, invocation):
        if invocation.caller_identity != self.settings.backend_identity:
            return json.dumps({'error': 'Unauthorized coordinator'})
        try:
            body = json.loads(invocation.payload)
            action = body.pop('action')
            async with self.guard:
                if action == 'start':
                    options = WorkerStart.model_validate(body)
                    if self.active:
                        raise ValueError('GPU is busy or finishing the previous session')
                    if options.tracking and self.detector is None:
                        raise ValueError('Tracking models are disabled; set TRACKING_MODELS_ENABLED=true and restart the GPU worker')
                    session = VideoSession(self, options)
                    self.active = session
                    try:
                        await session.open()
                    except BaseException:
                        await session.close()
                        self.active = None
                        raise
                    session.task = asyncio.create_task(self.run_session(session))
                    return json.dumps({'ready': True})
                sid = SessionID.model_validate(body).session_id
                if action == 'stop' and self.active is None:
                    return json.dumps({'ok': True})
                if not self.active or self.active.options.session_id != sid:
                    raise ValueError('Session ended')
                if action == 'tracking':
                    control = TrackingControl.model_validate(body)
                    await self.active.set_tracking(control.enabled)
                    return json.dumps({'enabled': control.enabled, 'generation': self.active.generation})
                if action == 'heartbeat':
                    self.active.touched = time.monotonic()
                elif action == 'stop':
                    self.active.stopped.set()
                elif action == 'verify':
                    return json.dumps(await self.active.manual_review())
                else:
                    raise ValueError('Unknown action')
                return json.dumps({'ok': True})
        except Exception as exc:
            log.warning('Control request failed: %s', type(exc).__name__)
            return json.dumps({'error': str(exc) if isinstance(exc, ValueError) else 'GPU command failed'})

    async def run_session(self, session):
        try:
            await session.run()
        except Exception:
            log.exception('Session failed')
            with suppress(Exception):
                await session.publish(Observation(
                    session_id=session.options.session_id, event_id=uuid.uuid4().hex,
                    state='error', start=0, end=0,
                    response='GPU session failed. Stop and start a new session.'))
        finally:
            try:
                await session.close()
            finally:
                if self.active is session:
                    self.active = None

    async def close(self):
        if self.active:
            self.active.stopped.set()
            if self.active.task:
                await self.active.task
        await self.control.disconnect()
        if self.detector:
            await self.detector.close()
        self.tracking_executor.shutdown(wait=True, cancel_futures=True)
        self.executor.shutdown(wait=True, cancel_futures=True)


class VideoSession:
    def __init__(self, worker, options):
        self.worker, self.options = worker, options
        self.room = rtc.Room()
        self.frames = deque(maxlen=4)
        self.tracker = None
        self.tracking_task = self.localization_task = None
        self.pending_detection = None
        self.mapping_pending = False
        self.tracker_active = False
        self.tracking_frames = deque(maxlen=180)
        self.tracker_time = -1
        self.last_localization = float('-inf')
        self.track_history = deque(maxlen=180)
        self.tracker_revision = 0
        self.pending_verification = None
        self.verifying = False
        self.verification_error = ''
        self.last_verification_attempt = float('-inf')
        self.last_verified = None
        self.track_sequence = 0
        self.generation = 0
        self.track_status = 'waiting'
        self.track_message = ''
        self.latest_video_time = 0
        self.target_query = ''
        self.detection_request = None
        self.stopped = asyncio.Event()
        self.epoch = self.touched = time.monotonic()
        self.last_sample = float('-inf')
        self.last_consumed = -1
        self.receiver = self.sender = self.task = None
        self.evidence = asyncio.Queue(maxsize=1)
        self.sending = False

    async def open(self):
        @self.room.on('track_published')
        def published(publication, participant):
            if (participant.identity == self.options.publisher
                    and publication.source == rtc.TrackSource.SOURCE_CAMERA):
                publication.set_subscribed(True)

        @self.room.on('track_subscribed')
        def subscribed(track, publication, participant):
            if (participant.identity == self.options.publisher
                    and publication.source == rtc.TrackSource.SOURCE_CAMERA
                    and track.kind == rtc.TrackKind.KIND_VIDEO and self.receiver is None):
                self.receiver = asyncio.create_task(self.receive(track))

        @self.room.on('participant_disconnected')
        def departed(participant):
            if participant.identity in (self.options.publisher, self.worker.settings.backend_identity):
                self.stopped.set()

        @self.room.on('track_unsubscribed')
        def unsubscribed(track, publication, participant):
            if participant.identity == self.options.publisher:
                self.stopped.set()

        @self.room.on('disconnected')
        def disconnected(*_):
            self.stopped.set()

        self.tracker = self.worker.tracker
        if self.tracker:
            await self.worker.cpu(self.tracker.reset)
        await self.worker.gpu(self.worker.engine.start, self.options)
        await self.room.connect(self.worker.settings.url,
            self.worker.settings.token(self.worker.settings.worker_identity, self.options.room),
            rtc.RoomOptions(auto_subscribe=False))
        self.sender = asyncio.create_task(self.send_evidence())
        if self.tracker:
            self.tracking_task = asyncio.create_task(self.track_loop())
            self.localization_task = asyncio.create_task(self.detect_loop())

    async def receive(self, track):
        stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGB24)
        try:
            async for frame_event in stream:
                now = time.monotonic()
                if self.stopped.is_set():
                    break
                self.latest_video_time = now-self.epoch
                sample = now-self.last_sample >= .25
                if not sample and not self.options.tracking:
                    continue
                frame = frame_event.frame
                if min(frame.width, frame.height) < 32 or max(frame.width, frame.height) > 4096:
                    raise ValueError('Unexpected video dimensions')
                image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.data))
                rotation = {rtc.VideoRotation.VIDEO_ROTATION_90: 90,
                            rtc.VideoRotation.VIDEO_ROTATION_180: 180,
                            rtc.VideoRotation.VIDEO_ROTATION_270: 270}.get(frame_event.rotation, 0)
                if rotation:
                    image = image.rotate(-rotation, expand=True)
                image.thumbnail((1280, 1280))
                # Receiver-relative time: do not misrepresent arrival time as capture time.
                if sample:
                    self.last_sample = now
                    self.frames.append((now-self.epoch, image))
                if self.options.tracking:
                    small = image.copy()
                    small.thumbnail((640, 640))
                    self.tracking_frames.append((now-self.epoch, small))

        except Exception:
            log.exception('Video receiver failed')
        finally:
            await stream.aclose()
            self.stopped.set()

    async def publish(self, observation):
        await asyncio.wait_for(self.room.local_participant.publish_data(
            observation.model_dump_json().encode(), reliable=True, topic=EVENT_TOPIC,
            destination_identities=[self.worker.settings.backend_identity]), 5)

    def queue_evidence(self, observation, images, times):
        if self.sending or not self.evidence.empty():
            observation.evidence_status = 'busy'
            return False
        observation.evidence_status = 'pending'
        self.evidence.put_nowait((observation, images, times))
        return True

    async def send_evidence(self):
        while True:
            observation, images, times = await self.evidence.get()
            self.sending = True
            try:
                packet = await asyncio.to_thread(encode_evidence, observation, images, times)
                await asyncio.wait_for(self.room.local_participant.send_bytes(
                    packet, topic=EVIDENCE_TOPIC, name=observation.event_id,
                    mime_type='application/octet-stream',
                    destination_identities=[self.worker.settings.backend_identity]), 15)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception('Evidence transfer failed')
                observation.evidence_status = 'transfer_failed'
                with suppress(Exception):
                    await self.publish(observation)
            finally:
                self.sending = False
                self.evidence.task_done()

    async def manual_review(self):
        if not self.options.verify:
            raise ValueError('Enable OpenAI verification when starting the session')
        if len(self.frames) < 4 or time.monotonic()-self.epoch-self.frames[-1][0] > 2:
            raise ValueError('Waiting for fresh camera frames')
        batch = list(self.frames)
        times = [t for t, _ in batch]
        observation = Observation(session_id=self.options.session_id, event_id=uuid.uuid4().hex,
            state='snapshot', start=times[0], end=times[-1]+.25)
        images = [crop_frame(im, self.options.crop) for _, im in batch]
        queued = self.queue_evidence(observation, images, times)
        await self.publish(observation)
        return {'event_id': observation.event_id, 'status': 'pending' if queued else 'busy'}

    async def run(self):
        while not self.stopped.is_set():
            if time.monotonic()-self.touched > 45:
                break
            if len(self.frames) == 4 and self.frames[0][0] > self.last_consumed:
                batch = list(self.frames)
                self.last_consumed = batch[-1][0]
                result, images, times = await self.worker.gpu(self.worker.engine.step, batch)
                if self.stopped.is_set():
                    break
                if self.options.tracking and result.state == 'response':
                    self.trigger_localization(result, images[-1], times[-1])
                if self.options.verify and result.state == 'response':
                    self.queue_evidence(result, images, times)
                await self.publish(result)
            await asyncio.sleep(.05)

    async def set_tracking(self, enabled):
        if enabled and self.worker.detector is None:
            raise ValueError('Tracking models are disabled on the GPU worker')
        if self.options.tracking == enabled:
            return
        self.generation += 1
        self.options.tracking = enabled
        self.target_query = ''
        self.detection_request = self.pending_detection = None
        self.mapping_pending = self.tracker_active = False
        self.tracking_frames.clear()
        self.tracker_time = -1
        self.last_localization = float('-inf')
        self.track_history.clear()
        self.tracker_revision += 1
        self.pending_verification = None
        self.last_verified = None
        self.last_verification_attempt = float('-inf')
        self.verification_error = ''
        self.verifying = False
        self.track_status, self.track_message = 'waiting', ''
        if self.tracker:
            await self.worker.cpu(self.tracker.reset)
        # Any already-running detection is allowed to finish, then discarded by generation.
        # The detector process and CSRT dependencies remain loaded.

    def trigger_localization(self, result, image, timestamp):
        if not self.options.tracking or self.tracker_active or self.mapping_pending:
            return
        query = self.options.tracking_target.strip() or (result.response or '')[:500]
        if query:
            # Keep the chosen target description through loss/reacquisition.
            self.target_query = self.target_query or query
            if time.monotonic()-self.last_localization >= self.worker.settings.localization_retry_seconds:
                self.mapping_pending = True
                self.detection_request = (image, timestamp, self.target_query, self.generation)

    async def localize(self, image, timestamp, query, generation=None):
        generation = self.generation if generation is None else generation
        self.mapping_pending = True
        self.track_status, self.track_message = 'localizing', 'Target noticed—please hold position until mapped'
        try:
            boxes = await self.worker.detector.detect(image, query)
            if self.stopped.is_set() or not self.options.tracking or generation != self.generation:
                return
            boxes = self.full_view_boxes(boxes)
            if boxes:
                self.pending_detection = (timestamp, boxes, generation)
            else:
                self.mapping_pending = False
                self.track_status, self.track_message = 'lost', 'Target not visible—hold it in view; searching automatically'
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Detection failed')
            if self.options.tracking and generation == self.generation:
                self.mapping_pending = False
                self.track_status, self.track_message = 'error', 'Localization failed; check GPU worker logs'

    def full_view_boxes(self, boxes):
        crop = self.options.crop
        if crop:
            for box in boxes:
                x1, y1, x2, y2 = box.box
                box.box = [crop.x+x1*crop.width, crop.y+y1*crop.height,
                           crop.x+x2*crop.width, crop.y+y2*crop.height]
        return boxes

    async def verify_target(self, image, timestamp, tracked, generation, revision):
        if (self.stopped.is_set() or not self.options.tracking or not self.tracker_active
                or generation != self.generation or revision != self.tracker_revision):
            return
        self.verifying = True
        self.last_verification_attempt = time.monotonic()
        try:
            detections = self.full_view_boxes(await self.worker.detector.detect(
                crop_frame(image, self.options.crop), self.target_query))
            if (self.stopped.is_set() or not self.options.tracking or generation != self.generation
                    or revision != self.tracker_revision or not self.tracker_active):
                return
            matched = tracks_match(tracked, detections,
                getattr(self.worker.settings, 'track_verification_min_iou', .2))
            self.pending_verification = (timestamp, detections, matched, generation, revision)
            self.verification_error = ''
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Target verification unavailable')
            if generation == self.generation and revision == self.tracker_revision:
                self.verification_error = 'Target check unavailable; retrying'
        finally:
            if generation == self.generation and revision == self.tracker_revision:
                self.verifying = False

    async def maybe_verify(self):
        if self.mapping_pending or self.pending_verification or not self.track_history:
            return
        interval = getattr(self.worker.settings, 'track_verification_interval', 3)
        if time.monotonic()-self.last_verification_attempt < interval:
            return
        timestamp, tracked = self.track_history[-1]
        # Both CSRT and the detector must inspect exactly the same image/time.
        image = next((im for t, im in self.tracking_frames if t == timestamp), None)
        if image is None or not tracked or time.monotonic()-self.epoch-timestamp > 1:
            return
        await self.verify_target(image, timestamp, tracked, self.generation, self.tracker_revision)

    async def detect_loop(self):
        last_timestamp = -1
        while not self.stopped.is_set():
            if not self.options.tracking or not self.target_query:
                await asyncio.sleep(.05)
                continue
            if self.tracker_active:
                await self.maybe_verify()
                await asyncio.sleep(.05)
                continue
            request = self.detection_request
            self.detection_request = None
            if request is None:
                if self.mapping_pending or time.monotonic()-self.last_localization < self.worker.settings.localization_retry_seconds:
                    await asyncio.sleep(.05)
                    continue
                if self.frames:
                    timestamp, image = self.frames[-1]
                    request = (crop_frame(image, self.options.crop), timestamp, self.target_query, self.generation)
            if request and request[1] > last_timestamp:
                last_timestamp = request[1]
                self.last_localization = time.monotonic()
                await self.localize(*request)
            else:
                # No fresh camera frame: don't repeatedly localize an old snapshot.
                if self.pending_detection is None:
                    self.mapping_pending = False
                await asyncio.sleep(.05)

    async def publish_tracks(self, timestamp, boxes):
        self.track_sequence += 1
        update = TrackingUpdate(session_id=self.options.session_id, generation=self.generation,
            sequence=self.track_sequence, frame_time=timestamp, status=self.track_status,
            boxes=boxes, message=self.track_message)
        await asyncio.wait_for(self.room.local_participant.publish_data(
            update.model_dump_json().encode(), reliable=True, topic=TRACK_TOPIC,
            destination_identities=[self.worker.settings.backend_identity]), 2)

    def lose_target(self, message):
        self.tracker_revision += 1
        self.pending_verification = None
        self.verifying = False
        self.track_history.clear()
        self.tracker_active = self.mapping_pending = False
        self.track_status, self.track_message = 'lost', message

    async def track_loop(self):
        boxes = []
        last_publish = float('-inf')
        try:
            while not self.stopped.is_set():
                if not self.options.tracking:
                    boxes = []
                    await asyncio.sleep(.05)
                    continue
                generation = self.generation
                if self.pending_verification:
                    timestamp, detections, matched, checked_generation, revision = self.pending_verification
                    self.pending_verification = None
                    if checked_generation == generation and revision == self.tracker_revision and self.tracker_active:
                        max_age = getattr(self.worker.settings, 'track_verification_max_age', 15)
                        if time.monotonic()-self.epoch-timestamp <= max_age:
                            if matched:
                                self.last_verified = self.epoch+timestamp
                            else:
                                boxes = []
                                self.lose_target('Target not visible—searching automatically')
                                # A matching target elsewhere is a fresh candidate, never
                                # silently treated as the same physical object.
                                if detections:
                                    self.mapping_pending = True
                                    self.pending_detection = (timestamp, detections, generation)
                                    self.track_status, self.track_message = 'localizing', 'Track drift detected—hold position while we map the target again'
                                await self.publish_tracks(max(0, self.tracker_time), [])
                if generation != self.generation or not self.options.tracking:
                    boxes = []
                    continue
                max_age = getattr(self.worker.settings, 'track_verification_max_age', 15)
                if self.tracker_active and self.last_verified is not None and time.monotonic()-self.last_verified > max_age:
                    boxes = []
                    self.lose_target('Target verification expired—hold position while we locate again')
                if self.pending_detection:
                    timestamp, detections, detected_generation = self.pending_detection
                    self.pending_detection = None
                    if detected_generation == generation:
                        original = next((im for t, im in self.tracking_frames if t == timestamp), None)
                        if original is None:
                            boxes = []
                            self.lose_target('Mapping frame expired—hold position while we locate again')
                        else:
                            boxes = await self.worker.cpu(self.tracker.initialize, original, detections)
                            if generation != self.generation:
                                boxes = []
                                continue
                            self.tracker_time = timestamp
                            self.tracker_revision += 1
                            self.track_history.clear()
                            self.track_history.append((timestamp, list(boxes)))
                            self.last_verified = self.epoch+timestamp
                            self.last_verification_attempt = time.monotonic()
                            self.tracker_active = bool(boxes)
                            if not boxes:
                                self.lose_target('Target too small to track—hold it closer while we locate again')
                if self.tracker_active:
                    frames = list(self.tracking_frames)
                    if frames and self.tracker_time < frames[0][0]:
                        boxes = []
                        self.lose_target('Tracking fell behind—hold position while we locate again')
                    else:
                        # Small replay batches keep status publication responsive while
                        # catching up from the exact localization frame.
                        for timestamp, image in [(t, im) for t, im in frames if t > self.tracker_time][:8]:
                            boxes = await self.worker.cpu(self.tracker.update, image)
                            if generation != self.generation or self.stopped.is_set():
                                boxes = []
                                break
                            self.tracker_time = timestamp
                            self.track_history.append((timestamp, list(boxes)))
                            if not boxes:
                                self.lose_target('Target lost—hold it in view while we locate again')
                                break
                if generation != self.generation or not self.options.tracking:
                    boxes = []
                    continue
                lag = max(0, time.monotonic()-self.epoch-self.tracker_time)
                visible = boxes
                if self.tracker_active:
                    if lag <= .5:
                        self.mapping_pending = False
                        self.track_status, self.track_message = 'tracking', (
                            'Tracking active—checking target' if self.verifying else self.verification_error
                            or 'Tracking active—you can move the object')
                    else:
                        self.mapping_pending = True
                        self.track_status, self.track_message = 'localizing', 'Mapped; aligning with live video—please keep holding position'
                        visible = []
                if time.monotonic()-self.epoch-self.latest_video_time > 1:
                    visible = []
                    if self.tracker_active:
                        self.track_status, self.track_message = 'waiting', 'Waiting for fresh camera frames'
                now = time.monotonic()
                if now-last_publish >= .1:
                    await self.publish_tracks(max(0, self.tracker_time), visible)
                    last_publish = now
                await asyncio.sleep(.01)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Tracker failed')
            self.track_status, self.track_message = 'error', 'Tracker failed; restart the session'
            with suppress(Exception):
                await self.publish_tracks(self.latest_video_time, [])

    async def close(self):
        self.stopped.set()
        # Let the in-flight detector reply drain before the next session uses it.
        if self.localization_task:
            with suppress(Exception):
                await self.localization_task
        for task in (self.receiver, self.sender, self.tracking_task):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        try:
            await self.room.disconnect()
        finally:
            await self.worker.gpu(self.worker.engine.stop)
            self.frames.clear()
            self.tracking_frames.clear()
            self.track_history.clear()
            if self.tracker:
                await self.worker.cpu(self.tracker.reset)


async def main():
    settings = Settings()
    if not settings.configured:
        raise RuntimeError(settings.configuration_error)
    worker, done = GPUWorker(settings), asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, done.set)
    try:
        # Load failures should fail startup; transient LiveKit outages should retry
        # without repeatedly loading another model into GPU memory.
        await worker.prepare_models()
        delay = 1
        while not done.is_set():
            if not worker.control.isconnected():
                if worker.active:
                    worker.active.stopped.set()
                try:
                    await worker.connect()
                    delay = 1
                except Exception as exc:
                    log.warning('Control reconnect unavailable (%s)', type(exc).__name__)
                    delay = min(delay*2, 30)
            try:
                await asyncio.wait_for(done.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
    finally:
        await worker.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
