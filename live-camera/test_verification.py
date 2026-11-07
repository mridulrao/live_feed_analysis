"""Verifier tests use a fake SDK client and never make paid API requests."""
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from verification import Assessment, Reviewer


class VerificationTests(unittest.TestCase):
    def test_unconfigured_never_submits(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            r = Reviewer(directory)
            self.assertEqual(r.enqueue('a', 'q', 'claim', [], [])['status'], 'not_configured')
            r.close()

    def test_background_single_call_and_schema(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def parse(**kwargs):
            calls.append(kwargs)
            entered.set()
            release.wait(3)
            return SimpleNamespace(output_parsed=Assessment(
                verdict='disagree', object_label='laptop charger', certainty='medium',
                evidence=['Attached cable'], explanation='Visible cable and power brick.'))

        with tempfile.TemporaryDirectory() as directory:
            r = Reviewer(directory, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
            images = [Image.new('RGB', (200, 150)) for _ in range(4)]
            first = r.enqueue('session', 'Identify object', 'phone', images, [0, .25, .5, .75])
            self.assertEqual(first['status'], 'pending')
            self.assertTrue(entered.wait(2))
            self.assertEqual(r.enqueue('session', 'q', 'c', images, [0, .25, .5, .75])['status'], 'busy')
            try:
                self.assertFalse(calls[0]['store'])
                self.assertEqual(calls[0]['text_format'], Assessment)
                self.assertEqual(sum(x['type'] == 'input_image' for x in calls[0]['input'][0]['content']), 2)
            finally:
                release.set()
                r.future.result(timeout=3)
            result = r.get_results('session')[0]
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['assessment']['verdict'], 'disagree')
            self.assertEqual(r.get_results('other-session'), [])
            self.assertEqual(r.enqueue('session', 'q', 'c', images, [1, 1.25, 1.5, 1.75])['status'], 'cooldown')
            r.close()

    def test_failure_is_not_reported_as_agreement(self):
        def parse(**kwargs):
            raise RuntimeError('SENSITIVE EXCEPTION CONTENT')
        with tempfile.TemporaryDirectory() as directory:
            r = Reviewer(directory, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
            r.enqueue('s', 'q', 'claim', [Image.new('RGB', (100, 100))] * 4, [0, .25, .5, .75])
            r.future.result(timeout=3)
            result = r.get_results('s')[0]
            self.assertEqual(result['status'], 'error')
            self.assertNotIn('SENSITIVE', result['error'])
            self.assertNotIn('assessment', result)
            r.close()


if __name__ == '__main__':
    unittest.main()
