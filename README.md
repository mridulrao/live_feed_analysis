# OneStreamer live camera over LiveKit

Run the frontend and review backend on your computer. Run OneStreamer on a cloud NVIDIA GPU. Both connect outbound to the same LiveKit project; the GPU needs no public HTTP endpoint.

```text
Browser webcam ── WebRTC ──► LiveKit session room ──► GPU worker
                                  ▲                     │
                                  │        observations + exact cropped snapshots
                                  │                     │
Local backend ◄───────────────────┴─────────────────────┘
     │
     ├── optional OpenAI visual review
     └── local HTTP controls/results ──► browser
```

The backend and GPU worker wait in a shared control room. The backend creates a fresh media room for every camera session and configures the worker through LiveKit RPC. The browser receives a short-lived, camera-only publishing token; LiveKit signing credentials and OpenAI credentials remain server-side.

## 1. Local computer

Install Python 3.11+, uv and Node.js/npm. From this repository:

```bash
uv sync --frozen
npm ci --prefix live-camera/web
cp live-camera/.env.example live-camera/.env
chmod 600 live-camera/.env
```

Edit `live-camera/.env` with your LiveKit project's URL, API key and API secret. Optional OpenAI reviews use `OPENAI_API_KEY` and `OPENAI_VERIFY_MODEL` in this **local** file. Reviews use `POST /v1/decisions` with `gpt-6-luna`, the endpoint’s supported model. Other `OPENAI_VERIFY_MODEL` values are rejected at startup. You can also set the key interactively with `bash live-camera/configure-openai.sh` after installing dependencies.

Start the local backend and frontend:

```bash
uv run --frozen uvicorn server:app --app-dir live-camera --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://localhost:8000**, which browsers treat as a secure context for camera access. In another terminal:

```bash
cat .runtime/local-access-key
```

Paste that generated local access key into the page. It protects local control endpoints and is unrelated to your LiveKit or OpenAI keys. Wait for **GPU worker connected**, then start the camera. The UI retains crop/detail controls, observation states and manual OpenAI checks. No microphone is requested yet.

Do not use `--reload` or multiple backend workers: the coordinator owns persistent room connections and one active session. Use the same hostname consistently; an HTTP LAN address will need HTTPS for camera access.

## 2. Cloud GPU

The existing setup remains Linux x86_64 + NVIDIA GPU, CUDA development toolkit (`nvcc`), Python 3, Git and at least 50 GB disk. The previous model deployment was tested on a RunPod L40S (48 GB). This transport migration has not yet been validated against the actual GPU model.

For a fresh pod, configure persistent storage before installation, clone this repository and run:

```bash
bash live-camera/setup-runpod.sh
cp /workspace/live-camera/.env.worker.example /workspace/.onestreamer-livekit.env
chmod 600 /workspace/.onestreamer-livekit.env
```

Edit `/workspace/.onestreamer-livekit.env` with the **same LiveKit project, control room and identities** as the local backend. No OpenAI key is needed on the GPU. Then:

```bash
bash /workspace/live-camera/restart.sh
tail -f /workspace/logs/camera-server.log
```

Wait for `Model ready; waiting in control room`. The worker loads the model once and joins camera rooms as requested by the backend.

For an existing deployment, copy the updated `live-camera/` source into `/workspace/live-camera` (preserve credentials outside that directory), install the new transport dependencies and restart:

```bash
/workspace/onestreamer-env/bin/python -m pip install -r /workspace/live-camera/requirements-worker.txt
bash /workspace/live-camera/restart.sh
```

The restart script can stop the previous HTTP server as well as the new worker. The old Pod camera access key and OpenAI environment file are unused by this worker. The previous port-8000 RunPod proxy URL is replaced by your local UI; no inbound port on the GPU is required. Both machines need outbound WebRTC connectivity to LiveKit, including TURN fallback where necessary.

## Processing and reviews

- Browser publishes up to 1280×720 at 15 FPS. The GPU host samples at 4 FPS.
- Video reception runs separately from inference. Only the latest four fresh samples are eligible for the next update; buffers stay bounded when inference is slow.
- OneStreamer retains its four-frame updates, 32-frame rolling context, image budgets (224²/336²/448² pixel area), and centered crops.
- Times are seconds since session creation on the GPU, measured at **frame reception**. They are not exact browser capture timestamps. Network delay is therefore not included in the reported inference time.
- For each reviewable response, the GPU sends a binary evidence stream containing the first and last cropped snapshots from that exact inference batch, along with the session/event IDs and sample times. Byte streams are addressed only to the backend. They are not ordinary data-message attachments.
- The backend calls OpenAI asynchronously. At most one review is in flight with a five-second minimum interval. Busy, cooldown, unclear and failure states remain explicit. The camera/inference pipeline does not wait for OpenAI.
- Manual **Check task against current view** asks the GPU for its latest fresh sampled frames and classifies whether the visible scene matches your instruction; it does not identify arbitrary objects or reuse an old model claim.
- Selected snapshots are held in memory, not saved as image files. JSONL observations and reviews are written under `.runtime/outputs/` on the local computer. OpenAI receives snapshots only when verification is enabled; normal video transits LiveKit to the GPU.
- Two snapshots can support object identification, but cannot establish historical counts or arbitrary action sequences. Decisions returns agree/disagree/unclear, or matches-task/does-not-match-task/unclear for manual checks, with choice probabilities and confidence. These are model estimates, not measured accuracy. The endpoint does not generate object labels or written explanations.

The current deployment deliberately supports **one camera session per GPU worker/coordinator pair**. Give separate pairs different control rooms and identities. A lost browser stops publication; a local polling lease expires after about 35–45 seconds, and the GPU independently stops after 45 seconds without backend heartbeats. Stopping waits for running inference before resetting model state; a new session may briefly report that the GPU is finishing.

The worker uses decoded RGB frames and PIL preprocessing on the GPU host; this is not an NVDEC/zero-copy GPU decode implementation. LiveKit supplies the future media foundation for a voice agent; a voice agent and speech-based decision policy are not implemented yet.

## Validation

Unit tests make no paid calls and need no GPU or LiveKit server:

```bash
uv run --frozen python -m unittest discover -s live-camera -p 'test_*.py' -v
node --check live-camera/static/app.js
```

For a real local WebRTC transport test, install/run a local LiveKit server in a separate terminal:

```bash
livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1
```

Then:

```bash
uv run --frozen python live-camera/validate_api.py
```

This test uses a synthetic video publisher, a deliberately slow fake OneStreamer adapter and a fake OpenAI client. It checks real video subscription, concurrent sampling, backpressure, cropped evidence byte streams, manual review, correlation, stop/room cleanup and session isolation. It is restricted to localhost with LiveKit's development credentials and never calls OpenAI.

See [runtime details](live-camera/README.md) for component boundaries and troubleshooting. The ignored local `OneStreamer/` directory is upstream code; setup downloads [MCG-NJU/OneStreamer](https://github.com/MCG-NJU/OneStreamer) at revision `338e5fe77383fc8d691bd6226c0358bcb2c83704`. Upstream code and model weights retain their respective licenses.
