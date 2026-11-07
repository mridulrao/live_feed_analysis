"""Server-side LiveKit configuration shared by the coordinator and GPU worker."""
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from livekit import api

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / '.env')
load_dotenv(os.environ.get('LIVE_CAMERA_ENV', str(ROOT / '.env')))


class Settings:
    def __init__(self):
        self.url = os.environ.get('LIVEKIT_URL', '')
        self.api_key = os.environ.get('LIVEKIT_API_KEY', '')
        self.api_secret = os.environ.get('LIVEKIT_API_SECRET', '')
        self.control_room = os.environ.get('LIVEKIT_CONTROL_ROOM', 'onestreamer-control')
        self.backend_identity = os.environ.get('LIVEKIT_BACKEND_IDENTITY', 'camera-backend')
        self.worker_identity = os.environ.get('LIVEKIT_WORKER_IDENTITY', 'camera-gpu')
        if self.backend_identity == self.worker_identity:
            raise ValueError('Backend and worker identities must differ')

    @property
    def configured(self):
        return bool(self.url and self.api_key and self.api_secret)

    def token(self, identity, room, *, camera=False):
        return (api.AccessToken(self.api_key, self.api_secret)
                .with_identity(identity).with_ttl(timedelta(minutes=15 if camera else 360))
                .with_grants(api.VideoGrants(
                    room_join=True, room=room, can_publish=camera,
                    can_subscribe=not camera, can_publish_data=not camera,
                    can_publish_sources=['camera'] if camera else [],
                    can_update_own_metadata=False)).to_jwt())
