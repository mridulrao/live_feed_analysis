# Runtime components

Start with the [repository setup guide](../README.md).

| File | Runs on | Responsibility |
|---|---|---|
| `server.py` | Local computer | FastAPI UI/API, room coordination, evidence validation and OpenAI review |
| `static/index.html`, `static/app.js` | Browser | Camera publishing through LiveKit; local control/results HTTP calls |
| `gpu_worker.py` | Cloud GPU | LiveKit control RPC, video subscription/sampling, inference scheduling and evidence delivery |
| `inference.py` | Cloud GPU | OneStreamer loading, per-session context, crop/resize and inference |
| `protocol.py` | Both | Validated session/events and bounded binary evidence envelopes |
| `settings.py` | Both | Environment loading and room-scoped token generation |
| `verification.py` | Local computer | Bounded background OpenAI Decisions API visual classification |

## Connections

Both server processes join `LIVEKIT_CONTROL_ROOM` using distinct identities. Only the configured backend identity can call GPU control RPC. A camera session has a random ID and its own `camera-<id>` room. The backend and worker join this room; the browser gets a 15-minute camera-only publish token, with no data publishing or media subscription grant.

GPU observations are reliable room data messages addressed only to the backend. Evidence is a LiveKit byte stream containing a size-prefixed JSON manifest and two JPEGs. The backend enforces a 2 MB envelope limit, a stream read timeout, JPEG dimensions/format, session IDs, worker identity and event deduplication. It does not subscribe to the video track. Events and reviews reach the browser through authenticated local polling, which also maintains the session lease.

Only one evidence transfer is in flight at a time. Transfers can be skipped as busy; OpenAI calls can separately be skipped as busy/cooldown. Evidence transport errors do not stop GPU inference. Stop/camera departure ends the session, disconnects its room, waits for current GPU work and resets its context. The coordinator removes the session room to disconnect stalled publishers.

OpenAI remains optional. A configured key alone does not send snapshots: verification must be enabled for the session. API calls receive selected snapshots, the user prompt and the unverified claim; the GPU never imports the OpenAI reviewer. The default response is asynchronous feedback, not a gate on local observations. No automated speaking/actions are implemented.

## Troubleshooting

- **Missing `hf_transfer` during model download:** a GPU template may set `HF_HUB_ENABLE_HF_TRANSFER=1` even though the isolated model environment does not contain that optional package. Updated setup/start scripts disable that optional downloader. After copying the updated source into `/workspace/live-camera`, run `bash /workspace/live-camera/finish-setup.sh` to retry the model download and finish setup without repeating dependency installation. Preserve existing model/cache files.
- **LiveKit unconfigured:** edit the local `live-camera/.env` and restart the local backend.
- **Waiting for GPU worker:** check the worker's model load logs, LiveKit project/room/identities on both machines, and outbound network access.
- **Unauthorized coordinator:** the `LIVEKIT_BACKEND_IDENTITY` values differ between machines.
- **GPU busy/finishing:** a previous inference must finish before context can be reset. Retry after shutdown completes.
- **No video observations:** check browser camera permission, LiveKit connection status and GPU logs. A worker only subscribes to the configured session publisher's camera track.
- **OpenAI unavailable:** configure the key locally, restart the backend and enable the checkbox before starting. Busy/cooldown states are not successful reviews.
- **Browser SDK missing:** run `npm ci --prefix live-camera/web` from the repository root.
- **Webcam blocked:** use `http://localhost:8000` or HTTPS. Opening the HTML file directly is unsupported.

Keep credentials out of Git. Example environment files are committed; actual `.env` files, the local access key, outputs, models and downloaded dependencies are ignored. GPU scripts use `/workspace/.onestreamer-livekit.env` by default; override `LIVE_CAMERA_ENV` if needed. The two server processes currently each use a server-side LiveKit signing key; the browser never receives it.
