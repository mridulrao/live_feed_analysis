#!/usr/bin/env bash
# Run locally; hidden input is handled by Python, never a CLI argument.
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LIVE_CAMERA_CONFIG_DIR="$APP_DIR"
"$APP_DIR/../.venv/bin/python" - <<'PY'
import getpass
import os
from pathlib import Path
from dotenv import set_key
path = Path(os.environ['LIVE_CAMERA_CONFIG_DIR']) / '.env'
key = getpass.getpass('OpenAI API key (hidden): ')
if not key or any(c.isspace() for c in key):
    raise SystemExit('Expected a nonempty API key without whitespace.')
if not path.exists():
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
path.chmod(0o600)
set_key(str(path), 'OPENAI_API_KEY', key)
path.chmod(0o600)
print('Key saved in local live-camera/.env. Restart the local backend to enable reviews.')
PY
