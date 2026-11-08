"""Decisions HTTP contract tests; never make paid API requests."""
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from PIL import Image
from verification import (CLAIM_CHOICES, DECISIONS_MODEL, DECISIONS_URL,
                          Reviewer, parse_assessment)


def decision_result(choice='agree', choices=None):
    values = list(choices or CLAIM_CHOICES)
    return {'model': DECISIONS_MODEL, 'answers': [{
        'type': 'choice', 'name': 'visual_assessment', 'choice': choice,
        'confidence': .82,
        'probabilities': [{'value': v, 'probability': .9 if v == choice else .05} for v in values]}]}


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=30)


class VerificationTests(unittest.TestCase):
    def test_unconfigured_never_submits(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            r = Reviewer(directory)
            self.assertEqual(r.enqueue('a', 'q', 'claim', [], [])['status'], 'not_configured')
            r.close()

    def test_background_single_decision_call_and_contract(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def handle(request):
            calls.append(request)
            entered.set()
            release.wait(3)
            return httpx.Response(200, json=decision_result('disagree'))
        with tempfile.TemporaryDirectory() as directory:
            r = Reviewer(directory, client=mock_client(handle))
            images = [Image.new('RGB', (200, 150)) for _ in range(4)]
            first = r.enqueue('session', 'Identify object', 'phone', images, [0, .25, .5, .75], event_id='event')
            self.assertEqual(first['status'], 'pending')
            self.assertTrue(entered.wait(2))
            self.assertEqual(r.enqueue('session', 'q', 'c', images, [0, .25, .5, .75])['status'], 'busy')
            try:
                self.assertEqual(str(calls[0].url), DECISIONS_URL)
                self.assertEqual(calls[0].method, 'POST')
                body = json.loads(calls[0].content)
                self.assertEqual(set(body), {'model', 'input', 'questions'})
                self.assertEqual(body['model'], 'gpt-6-luna')
                self.assertEqual(body['questions'][0]['type'], 'choice')
                self.assertEqual({c['value'] for c in body['questions'][0]['choices']}, set(CLAIM_CHOICES))
                self.assertEqual(sum(x['type'] == 'input_image' for x in body['input'][0]['content']), 2)
            finally:
                release.set()
                r.future.result(timeout=3)
            result = r.get_results('session')[0]
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['event_id'], 'event')
            self.assertEqual(result['assessment']['verdict'], 'disagree')
            self.assertEqual(result['assessment']['confidence'], .82)
            self.assertEqual(r.get_results('other-session'), [])
            self.assertEqual(r.enqueue('session', 'q', 'c', images, [1, 1.25, 1.5, 1.75])['status'], 'cooldown')
            r.close()

    def test_manual_check_uses_task_choices_not_object_identification(self):
        def handle(request):
            body = json.loads(request.content)
            choices = [c['value'] for c in body['questions'][0]['choices']]
            self.assertEqual(choices, ['matches_task', 'does_not_match_task', 'unclear'])
            return httpx.Response(200, json=decision_result('matches_task', choices))
        with tempfile.TemporaryDirectory() as directory:
            r = Reviewer(directory, client=mock_client(handle))
            r.enqueue('s', 'A person is holding a cup', '', [Image.new('RGB', (100, 100))]*2, [0, .25])
            r.future.result(timeout=3)
            assessment = r.get_results('s')[0]['assessment']
            self.assertEqual(assessment['kind'], 'task')
            self.assertEqual(assessment['verdict'], 'matches_task')
            self.assertNotIn('object_label', assessment)
            r.close()

    def test_failure_and_refusal_are_not_reported_as_agreement(self):
        scenarios = [
            (httpx.Response(401, text='SENSITIVE EXCEPTION CONTENT'), 'error'),
            (httpx.Response(200, json={'answers': [{'type': 'refusal', 'name': 'visual_assessment'}]}), 'refused'),
            (httpx.Response(200, json={'answers': []}), 'error'),
            (httpx.Response(200, text='invalid json'), 'error'),
        ]
        for response, expected in scenarios:
            with self.subTest(expected=expected, body=response.content), tempfile.TemporaryDirectory() as directory:
                r = Reviewer(directory, client=mock_client(lambda request: response))
                r.enqueue('s', 'q', 'claim', [Image.new('RGB', (100, 100))]*2, [0, .25])
                r.future.result(timeout=3)
                result = r.get_results('s')[0]
                self.assertEqual(result['status'], expected)
                self.assertNotIn('SENSITIVE', json.dumps(result))
                self.assertNotIn('assessment', result)
                r.close()

    def test_invalid_distributions_and_names_rejected(self):
        for mutation in ['choice', 'name', 'confidence', 'probabilities']:
            result = decision_result()
            answer = result['answers'][0]
            answer[mutation] = {'choice': 'made_up', 'name': 'wrong', 'confidence': float('nan'),
                'probabilities': [{'value': 'agree', 'probability': 1}]*3}[mutation]
            with self.subTest(field=mutation), self.assertRaises(ValueError):
                parse_assessment(result, True)

    def test_unsupported_model_rejected_without_request(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'requires gpt-6-luna'):
                Reviewer(directory, model='gpt-4.1-mini')

    def test_timeout_remains_an_unavailable_review(self):
        def handle(request):
            raise httpx.ReadTimeout('SENSITIVE', request=request)
        with tempfile.TemporaryDirectory() as directory:
            r = Reviewer(directory, client=mock_client(handle))
            r.enqueue('s', 'q', 'claim', [Image.new('RGB', (100, 100))]*2, [0, .25])
            r.future.result(timeout=3)
            result = r.get_results('s')[0]
            self.assertEqual(result['status'], 'error')
            self.assertNotIn('SENSITIVE', result['error'])
            r.close()


if __name__ == '__main__':
    unittest.main()
