"""Persistent JSON-lines bridge; run with LocateAnything's isolated Python environment."""
import base64
from contextlib import redirect_stdout
import io
import json
import os
import sys

from PIL import Image


def main():
    wire = sys.stdout
    # Model diagnostics must never enter the JSON protocol on stdout.
    with redirect_stdout(sys.stderr):
        sys.path.insert(0, os.environ['LOCATE_ANYTHING_SOURCE'])
        model_path = os.environ.get('LOCATE_ANYTHING_MODEL', 'nvidia/LocateAnything-3B')
        sys.path.insert(0, model_path)
        from locateanything_worker import LocateAnythingWorker
        worker = LocateAnythingWorker(model_path,
            use_batch_runtime=os.environ.get('LOCATE_ANYTHING_BATCH_RUNTIME', '0') == '1')
    print(json.dumps({'ready': True}), file=wire, flush=True)
    for line in sys.stdin:
        try:
            body = json.loads(line)
            image = Image.open(io.BytesIO(base64.b64decode(body['image']))).convert('RGB')
            with redirect_stdout(sys.stderr):
                result = worker.ground_multi(image, body['query'],
                    max_new_tokens=512, verbose=False)
            reply = {'answer': result['answer']}
        except Exception as exc:
            print(f'Localization failed: {exc}', file=sys.stderr, flush=True)
            reply = {'error': 'LocateAnything inference failed; check GPU logs'}
        print(json.dumps(reply), file=wire, flush=True)


if __name__ == '__main__':
    main()
