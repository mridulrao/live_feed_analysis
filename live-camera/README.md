# Runtime components

Start with the [repository setup guide](../README.md).

| File | Runs on | Responsibility |
|---|---|---|
| `server.py` | Local computer | FastAPI UI/API, room coordination, evidence validation and OpenAI review |
| `static/index.html`, `static/app.js` | Browser | Camera publishing through LiveKit; local control/results HTTP calls |
| `gpu_worker.py` | Cloud GPU | LiveKit control RPC, video subscription/sampling, inference scheduling and evidence delivery |
| `onestreamer/adapter.py` | Cloud GPU | OneStreamer loading, per-session context, crop/resize and inference |
| `object_detection/locate_anything/` | Cloud GPU | LocateAnything detection plugin and isolated persistent process |
| `tracking/csrt.py` | Cloud GPU host CPU | Box-initialized CSRT visual tracking |
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

## Event-triggered localization and tracking

The components are separated by responsibility:

```text
live-camera/
  onestreamer/
    adapter.py                   # continuous observation
  object_detection/
    __init__.py                  # detector interface and factory
    locate_anything/
      detector.py                # detection plugin / persistent-process client
      process.py                 # isolated LocateAnything model runtime
  tracking/
    __init__.py                  # tracker factory
    csrt.py                      # OpenCV CSRT visual tracker
  gpu_worker.py                  # session lifecycle and pipeline scheduling
```

OneStreamer response → LocateAnything initial bounding boxes → CSRT visual tracking
→ labeled browser boxes. LocateAnything runs in its own persistent Python process on the same GPU;
CSRT runs on the CPU and does not load neural weights or another detector.
The factories select plugins using `DETECTION_PLUGIN=locate_anything` and
`TRACKING_PLUGIN=csrt`. A detector implements `start`, `detect(image, query)`
and `close`, returning normalized `DetectionBox` objects. A tracker implements
`reset`, `initialize(image, detections)` and `update(image)`.

### Model lifetime versus UI processing

`TRACKING_MODELS_ENABLED=true` is the backend default. The GPU worker loads
OneStreamer, the detector and CSRT before joining LiveKit, and retains them
across UI toggles and camera sessions. Missing detection dependencies/configuration
cause startup to fail with a setup error. Set `TRACKING_MODELS_ENABLED=false` for
an observation-only worker: detector/tracker plugins are not loaded, and attempts
to enable tracking are rejected. Changing this backend flag requires a worker restart.

The single **Enable tracking** checkbox controls whether frames are sent through
detection and tracking. It works before starting and during an active camera
session, using authenticated `/api/tracking` and the coordinator-only LiveKit RPC.
Turning it off clears track state and boxes; it does not unload the detector or
CSRT. An already-running detection can finish, but its result is discarded.
Generation checks in the worker, backend and browser reject results from before a
toggle. Turning it on starts waiting for the next relevant OneStreamer response;
no model-loading step is needed. Track IDs reset when toggled or a new session starts.

### GPU installation

After the usual OneStreamer installation, copy the updated `live-camera` directory
to the Linux CUDA host and run:

```bash
bash /workspace/live-camera/setup-tracking.sh
```

This installs NVIDIA's Eagle/Embodied project and downloads LocateAnything in a
separate Python environment, preserving OneStreamer's Transformers dependencies.
CSRT's pinned `opencv-contrib-python-headless==4.11.0.86` dependency is installed
in the main worker environment. Competing OpenCV wheels are removed there because
they share the `cv2` namespace. The isolated LocateAnything environment is unaffected. Add the printed settings to `/workspace/.onestreamer-livekit.env`
(or your `LIVE_CAMERA_ENV` file), then run:

```bash
bash /workspace/live-camera/restart.sh
```

Defaults are `/workspace/locate-anything-env/bin/python`,
`/workspace/Eagle/Embodied`, and `/workspace/models/LocateAnything-3B`; environment
overrides are listed in `.env.worker.example`. The setup requires network access,
model disk space and the existing CUDA image. Existing Eagle checkouts are preserved.
Installed versions are recorded in `/workspace/locate-anything-installed.txt`.
NVIDIA's model uses its upstream `trust_remote_code` implementation.

The default LocateAnything runtime uses ordinary attention. To use
`LOCATE_ANYTHING_BATCH_RUNTIME=1`, install compatible FlashAttention in its separate
environment and retain the model's `batch_utils/` and `kernel_utils/` directories.
MagiAttention is not required for the default runtime.

### Initial mapping, visual tracking and reacquisition

An optional short target description, such as “the wallet,” overrides the
OneStreamer response as the grounding query. Otherwise the response text is used.
A relevant OneStreamer response triggers localization on the last frame from its
exact inference batch. The UI displays **Target noticed—please hold position until
mapped** beside the camera, even when Settings is collapsed. Crop-relative boxes
are mapped back to the full view.

The worker retains up to 180 full-view frames resized to 640 pixels on the longest
edge, roughly 12 seconds at the browser's 15 FPS. CSRT initializes on the exact
source frame and processes subsequent buffered images in small batches to catch
up. Until the tracked frame is within 500 ms of the current worker time, the UI
continues asking the user to hold still and hides stale boxes. Only then does it
show **Tracking active—you can move the object**. This threshold is worker-side;
network transit and browser polling introduce additional latency.

While CSRT is active, LocateAnything periodically verifies that it is still following the target. OneStreamer continues
observing, and additional responses do not replace a healthy target. CSRT follows
the initial target set frame by frame for as long as it reports success, without a
fixed tracking timeout. Up to eight initial objects can be followed. The visual
tracker depends on CPU throughput and does not guarantee 15 FPS.

If all targets are lost, frame geometry changes, or processing falls behind the
buffer, boxes clear and the UI asks the user to hold the target in view. The
existing target description is used to localize a fresh frame; another OneStreamer
response is not required. `LOCALIZATION_RETRY_SECONDS=1` limits retry frequency.
There is one in-flight localization and one pending initialization. Old source
frames that have expired from the buffer are rejected instead of applying an old
box to current video. Reacquired objects receive new IDs: this does not establish
that the same physical object returned.

CSRT uses appearance and can drift or report success on an incorrect region after
occlusion, fast motion or appearance changes. The worker now uses LocateAnything
as an independent check of the original target description, even when CSRT reports
success. It compares the detector's boxes against the CSRT boxes recorded for the
exact same frame, avoiding false drift decisions caused by motion during inference.
Every tracked object must overlap a distinct detected target.

Worker defaults (also listed in `.env.worker.example`):

```env
TRACK_VERIFICATION_INTERVAL_SECONDS=3
TRACK_VERIFICATION_MAX_AGE_SECONDS=15
TRACK_VERIFICATION_MIN_IOU=0.2
```

Only one detection/verification runs at a time. Checks begin at most once per
configured interval; inference time can lengthen that cadence. CSRT continues
following frames while a check is in flight. On disagreement, the old box is
immediately cleared. If the detector finds the target elsewhere, CSRT initializes
on that frame, catches up, and assigns fresh IDs. If no target is found, the UI
shows **Target not visible—searching automatically**, and fresh frames are searched
until the target returns without another OneStreamer response being required.

Successful checks renew trust based on their source-frame time. Errors and replies
from before a toggle or tracker replacement cannot renew it. Tracks without a
successful localization/verification within the maximum age are cleared and searched
again. The verification process stays within the existing detector lifetime and
the Enable tracking toggle; disabling tracking discards in-flight results.

This checks agreement with the target description, not physical-object identity.
LocateAnything can also misidentify a target; overlapping device/hair regions may
still pass the IoU threshold. There is no reference-image identity embedding or
claim of guaranteed recovery. Periodic checks add GPU work, so measure their latency
alongside OneStreamer on the actual deployment.

Tracking updates normally publish at most 10 times per second, with immediate
clear messages on verification failure, and use a separate
authenticated `/api/tracks` poll at 100 ms. The backend retains only the latest
result. The browser accounts for letterboxing and mirrors box positions to match
the preview while keeping labels readable. Stale boxes clear on frame/update
stalls, polling failures, toggle-off and session stop. Tracking polls do not renew
the main session lease. Timestamps describe frame reception rather than capture.

Run the Python tests and UI behavior tests with:

```bash
python -m unittest discover -s live-camera -p 'test_*.py'
node --test live-camera/test_ui.cjs
```

CSRT tests require the worker dependencies. They exercise real OpenCV tracking
on textured moving synthetic objects, ID continuity, fresh IDs on reacquisition,
reset and frame-size changes. Pipeline tests cover one-shot localization, visual
frame catch-up, expired source frames, loss-triggered reacquisition, plugin
preloading, runtime toggles, stale results, crop mapping, the detector process
bridge and API authentication.
Model startup, combined VRAM and end-to-end latency still require L40S hardware
validation. Gemini 2 depth ingestion is not included.
