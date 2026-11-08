"""Server-side LiveKit configuration shared by the coordinator and GPU worker."""
import os
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from livekit import api

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / '.env')
load_dotenv(os.environ.get('LIVE_CAMERA_ENV', str(ROOT / '.env')))


class Settings:
    def __init__(self):
        # Worker capability/lifetime flag; independent of the per-session UI switch.
        self.tracking_models_enabled = os.environ.get('TRACKING_MODELS_ENABLED', 'true').strip().lower() in ('true', '1', 'yes', 'on')
        self.detection_plugin = os.environ.get('DETECTION_PLUGIN', 'locate_anything')
        self.tracking_plugin = os.environ.get('TRACKING_PLUGIN', 'csrt')
        self.localization_retry_seconds = float(os.environ.get('LOCALIZATION_RETRY_SECONDS', '1'))
        if not .1 <= self.localization_retry_seconds <= 30:
            raise ValueError('LOCALIZATION_RETRY_SECONDS must be between .1 and 30')
        self.url = os.environ.get('LIVEKIT_URL', '')
        self.api_key = os.environ.get('LIVEKIT_API_KEY', '')
        self.api_secret = os.environ.get('LIVEKIT_API_SECRET', '')
        self.control_room = os.environ.get('LIVEKIT_CONTROL_ROOM', 'onestreamer-control')
        self.backend_identity = os.environ.get('LIVEKIT_BACKEND_IDENTITY', 'camera-backend')
        self.worker_identity = os.environ.get('LIVEKIT_WORKER_IDENTITY', 'camera-gpu')
        if self.backend_identity == self.worker_identity:
            raise ValueError('Backend and worker identities must differ')

    @property
    def configuration_error(self):
        if not (self.url and self.api_key and self.api_secret):
            return 'Set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the backend environment.'
        parsed = urlparse(self.url)
        if parsed.scheme not in ('ws', 'wss') or not parsed.hostname:
            return 'LIVEKIT_URL must be your LiveKit WebSocket URL (ws:// or wss://).'
        if ('your_project' in parsed.hostname.lower()
                or self.api_key == 'YOUR_LIVEKIT_API_KEY'
                or self.api_secret == 'YOUR_LIVEKIT_API_SECRET'):
            return 'Replace the example LiveKit URL, API key and API secret with values from your LiveKit project.'
        return None

    @property
    def configured(self):
        return self.configuration_error is None

    def token(self, identity, room, *, camera=False):
        return (api.AccessToken(self.api_key, self.api_secret)
                .with_identity(identity).with_ttl(timedelta(minutes=15 if camera else 360))
                .with_grants(api.VideoGrants(
                    room_join=True, room=room, can_publish=camera,
                    can_subscribe=not camera, can_publish_data=not camera,
                    can_publish_sources=['camera'] if camera else [],
                    can_update_own_metadata=False)).to_jwt())
