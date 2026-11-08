"""Silent drift, same-frame verification, absence, and reappearance regression tests."""
import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from PIL import Image
from gpu_worker import VideoSession
from tracking.verification import tracks_match
from test_tracking import options, box


class MatchingTests(unittest.TestCase):
    def test_spatial_match_not_just_same_color_or_label(self):
        hair = box().model_copy(update={'box': [.65, .1, .95, .4]})
        self.assertFalse(tracks_match([hair], [box()], .2))
        self.assertTrue(tracks_match([box()], [box()], .2))
        self.assertFalse(tracks_match([box()], [], .2))

    def test_multiple_tracks_need_distinct_detections(self):
        self.assertFalse(tracks_match([box(), box()], [box()], .2))
        self.assertTrue(tracks_match([box(), box()], [box(), box()], .2))


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def session(self, detections):
        async def cpu(fn, *args):
            return fn(*args)
        worker = SimpleNamespace(detector=SimpleNamespace(detect=AsyncMock(return_value=detections)),
            cpu=cpu, settings=SimpleNamespace(localization_retry_seconds=.001,
                track_verification_interval=.001, track_verification_max_age=15,
                track_verification_min_iou=.2))
        session = VideoSession(worker, options(tracking_target='black device'))
        session.epoch = time.monotonic()-1.2
        session.latest_video_time = 1.2
        session.tracker_active = True
        session.target_query = 'black device'
        session.tracker_revision = 7
        session.tracker_time = 1.2
        session.last_verified = session.epoch+1
        session.image = Image.new('RGB', (80, 80))
        session.tracking_frames.extend([(1., session.image), (1.2, session.image)])
        session.frames.append((1.2, session.image))
        return session

    async def test_drift_on_hair_reinitializes_from_independent_device_box(self):
        session = self.session([box()])
        hair = box().model_copy(update={'box': [.65, .1, .95, .4]})
        session.track_history.append((1., [hair]))
        await session.maybe_verify()
        self.assertFalse(session.pending_verification[2])
        self.assertEqual(session.worker.detector.detect.call_args.args[1], 'black device')
        initialized = []
        recovered = box().model_copy(update={'track_id': 2})
        session.tracker = SimpleNamespace(
            initialize=lambda image, detections: initialized.append(detections) or [recovered],
            update=lambda image: [recovered])
        async def publish(timestamp, boxes):
            if not boxes:
                self.assertIn('drift', session.track_message)
                return
            self.assertEqual(boxes[0].track_id, 2)
            self.assertEqual(boxes[0].box, box().box)
            session.stopped.set()
        session.publish_tracks = publish
        await session.track_loop()
        self.assertEqual(len(initialized), 1)

    async def test_absent_target_clears_box_then_reappearing_target_relocalizes(self):
        session = self.session([])
        session.track_history.append((1., [box()]))
        await session.maybe_verify()
        session.tracker = SimpleNamespace(initialize=lambda image, detections: [box()], update=lambda image: [box()])
        published = []
        async def publish(timestamp, boxes):
            published.append(boxes)
            self.assertEqual(boxes, [])
            self.assertIn('not visible', session.track_message)
            session.stopped.set()
        session.publish_tracks = publish
        await session.track_loop()
        self.assertFalse(session.tracker_active)
        # The model can now find the returned device; no new OneStreamer event is needed.
        session.stopped.clear()
        session.worker.detector.detect.return_value = [box()]
        original_detect = session.worker.detector.detect
        async def detect(image, query):
            result = await original_detect(image, query)
            asyncio.get_running_loop().call_soon(session.stopped.set)
            return result
        session.worker.detector = SimpleNamespace(detect=detect)
        await asyncio.wait_for(session.detect_loop(), 1)
        self.assertIsNotNone(session.pending_detection)
        self.assertEqual(session.pending_detection[1][0].box, box().box)

    async def test_verification_compares_historical_boxes_not_current_motion(self):
        session = self.session([box()])
        # Current video differs from the exact frame for which we have a tracked box.
        session.tracking_frames[-1] = (1.2, Image.new('RGB', (80, 80), 'blue'))
        session.track_history.append((1., [box()]))
        await session.maybe_verify()
        self.assertTrue(session.pending_verification[2])
        image = session.worker.detector.detect.call_args.args[0]
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    async def test_old_revision_and_disabled_results_are_discarded(self):
        session = self.session([box()])
        await session.verify_target(session.image, 1., [box()], 0, 6)
        self.assertIsNone(session.pending_verification)
        session.options.tracking = False
        await session.verify_target(session.image, 1., [box()], 0, 7)
        self.assertIsNone(session.pending_verification)
        self.assertEqual(session.worker.detector.detect.await_count, 0)

    async def test_inflight_verification_cannot_revive_replaced_track(self):
        session = self.session([box()])
        entered, release = asyncio.Event(), asyncio.Event()
        async def detect(image, query):
            entered.set()
            await release.wait()
            return [box()]
        session.worker.detector.detect = detect
        task = asyncio.create_task(session.verify_target(session.image, 1., [box()], 0, 7))
        await entered.wait()
        session.lose_target('Lost')
        session.tracker_active = True
        session.tracker_revision += 1
        last_verified = session.last_verified
        release.set()
        await task
        self.assertIsNone(session.pending_verification)
        self.assertEqual(session.last_verified, last_verified)
        self.assertFalse(session.verifying)

    async def test_successful_check_keeps_id_and_renews_source_time(self):
        session = self.session([box()])
        session.track_history.append((1.2, [box()]))
        await session.maybe_verify()
        session.tracker = SimpleNamespace(update=lambda im: self.fail('No new frame to update'))
        async def publish(timestamp, boxes):
            self.assertTrue(session.tracker_active)
            self.assertEqual(session.tracker_revision, 7)
            self.assertEqual(session.last_verified, session.epoch+1.2)
            session.stopped.set()
        session.publish_tracks = publish
        await session.track_loop()

    async def test_verifier_errors_do_not_refresh_trust_and_expired_track_hides(self):
        session = self.session([box()])
        session.worker.detector.detect.side_effect = RuntimeError('offline')
        old_verified = session.last_verified
        with self.assertLogs('gpu-worker', level='ERROR'):
            await session.verify_target(session.image, 1., [box()], 0, 7)
        self.assertEqual(session.last_verified, old_verified)
        self.assertTrue(session.verification_error)
        session.last_verified = time.monotonic()-20
        async def publish(timestamp, boxes):
            self.assertEqual(boxes, [])
            self.assertIn('expired', session.track_message)
            session.stopped.set()
        session.publish_tracks = publish
        await session.track_loop()
        self.assertFalse(session.tracker_active)
