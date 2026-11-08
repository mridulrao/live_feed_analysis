/* Camera media travels through LiveKit; HTTP calls only control the local backend. */
const $ = id => document.getElementById(id);
$('start').parentElement.before($('extraControls').content.cloneNode(true));
const guide = document.createElement('div');
guide.className = 'crop-guide';
guide.innerHTML = '<span>OBJECT CROP</span>';
$('video').parentElement.append(guide);
const reviewButton = document.createElement('button');
reviewButton.id = 'verifyNow';
reviewButton.className = 'secondary';
reviewButton.disabled = true;
reviewButton.textContent = 'Check task against current view';
const reviewStatus = document.createElement('div');
reviewStatus.id = 'reviewStatus';
reviewStatus.setAttribute('role', 'status');
$('status').after(reviewButton, reviewStatus);
let room = null, cameraTrack = null, session = null, opening = false, stopping = false;
let polling = null, pollBusy = false, verifierReady = false, epoch = 0;
let history = [], reviewResults = new Map();
const status = (message, error = false) => {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
};
async function api(path, body) {
  const response = await fetch('/api/' + path, {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Access-Key': $('key').value.trim()},
    body: JSON.stringify(body), signal: AbortSignal.timeout(40000)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid session settings.');
  return data;
}
function controls(active) {
  $('start').disabled = active;
  $('stop').disabled = !active;
  for (const id of ['prompt', 'key', 'camera', 'budget', 'crop']) $(id).disabled = active;
  $('verify').disabled = active || !verifierReady;
  reviewButton.disabled = !active || !$('verify').checked;
}
function cropSettings() {
  const z = Number($('crop').value);
  return z === 1 ? null : {x: (1-1/z)/2, y: (1-1/z)/2, width: 1/z, height: 1/z};
}
function updateGuide() {
  const c = cropSettings();
  guide.style.display = c && cameraTrack ? 'block' : 'none';
  if (c) {
    guide.style.left = c.x*100+'%'; guide.style.top = c.y*100+'%';
    guide.style.width = c.width*100+'%'; guide.style.height = c.height*100+'%';
  }
}
$('crop').onchange = updateGuide;
function appendReview(row, result) {
  const text = document.createElement('p'); text.className = 'review';
  if (result.status === 'complete') {
    const a = result.assessment;
    text.textContent = 'OpenAI '+(a.kind === 'claim' ? 'claim review: ' : 'task check: ')
      +a.verdict.replaceAll('_', ' ').toUpperCase()+' · confidence '+(a.confidence*100).toFixed(1)+'%';
    if (a.verdict !== 'agree' && a.verdict !== 'matches_task') text.classList.add('warn');
    const note = document.createElement('div'); note.className = 'review-note';
    note.textContent = a.probabilities.map(p => p.value.replaceAll('_', ' ')+': '
      +(p.probability*100).toFixed(1)+'%').join(' · ')
      +'\nModel estimates; not measured accuracy.';
    row.append(text, note);
  } else {
    text.textContent = result.status === 'pending' ? 'OpenAI review pending…'
      : result.status === 'error' ? result.error
      : result.status === 'refused' ? 'OpenAI declined this decision.' : 'OpenAI review skipped: '+result.status;
    row.append(text);
  }
}
function render() {
  const shown = history.filter(r => r.state === 'response' || r.state === 'error' ||
    ($('showAll').checked && r.state !== 'snapshot'));
  $('feed').replaceChildren();
  if (!shown.length) {
    const empty = document.createElement('div'); empty.className = 'empty';
    empty.textContent = 'Watching for an event that matches your prompt…'; $('feed').append(empty);
  }
  for (const r of shown.slice(-100).reverse()) {
    const row = document.createElement('div'); row.className = 'entry';
    const at = document.createElement('time'); at.textContent = r.end.toFixed(1)+'s · '+r.state.toUpperCase();
    const text = document.createElement('p');
    text.textContent = r.response || (r.state === 'standby' ? 'Relevant activity in progress.' : 'Observing.');
    row.append(at, text);
    if (r.review) appendReview(row, reviewResults.get(r.review.id) || r.review);
    else if ($('verify').checked && r.state === 'response') {
      const note = document.createElement('p'); note.className = 'review';
      note.textContent = r.evidence_status === 'pending' ? 'Receiving evidence from GPU…'
        : 'Review unavailable: '+r.evidence_status;
      row.append(note);
    }
    $('feed').append(row);
  }
  const manual = history.filter(r => r.state === 'snapshot').at(-1);
  if (manual) {
    reviewStatus.replaceChildren();
    if (manual.review) appendReview(reviewStatus, reviewResults.get(manual.review.id) || manual.review);
    else reviewStatus.textContent = 'Evidence: '+manual.evidence_status;
  }
}
async function poll() {
  if (!session || pollBusy || stopping) return;
  pollBusy = true;
  const requested = session;
  try {
    const data = await api('events', {session_id: requested});
    if (requested !== session) return;
    history = data.events;
    reviewResults = new Map(data.reviews.map(r => [r.id, r]));
    render();
    const last = history.at(-1);
    if (last) {
      $('gpuTime').textContent = last.inference_seconds.toFixed(2)+'s';
      $('roundTrip').textContent = 'LiveKit';
    }
    $('elapsed').textContent = ((performance.now()-epoch)/1000).toFixed(1)+'s';
    const failure = history.find(r => r.state === 'error');
    if (failure || !data.active) {
      status(failure?.response || 'Session ended. Start a new session.', true);
      await stop(true);
    } else status(history.length ? 'Watching for your requested event…' : 'Camera published. Waiting for GPU observations…');
  } catch (error) {
    status(error.message, true);
    // Keep polling after transient local API failures; the lease handles a closed tab.
  } finally { pollBusy = false; }
}
async function closeMedia() {
  clearInterval(polling); polling = null;
  if (cameraTrack) { cameraTrack.stop(); cameraTrack.detach(); cameraTrack = null; }
  const oldRoom = room; room = null;
  if (oldRoom) await oldRoom.disconnect();
  $('video').srcObject = null; $('video').classList.remove('active');
  $('cameraTag').textContent = 'CAMERA OFF'; guide.style.display = 'none';
}
async function stop(preserveError = false) {
  if (stopping) return;
  stopping = true;
  const sid = session; session = null;
  controls(true); $('stop').disabled = true;
  try {
    await closeMedia();
    if (sid) await api('stop', {session_id: sid});
    if (!preserveError) status('Stopped. Camera is off.');
  } catch (error) { status(error.message, true); }
  finally { stopping = false; controls(false); }
}
$('stop').onclick = () => stop();
reviewButton.onclick = async () => {
  if (!session) return;
  reviewButton.disabled = true;
  try {
    const result = await api('verify', {session_id: session});
    reviewStatus.textContent = result.status === 'pending' ? 'Requesting GPU evidence…' : 'Review '+result.status;
  } catch (error) { reviewStatus.textContent = error.message; }
  finally { reviewButton.disabled = !session || !$('verify').checked; }
};
$('showAll').onchange = render;
$('start').onclick = async () => {
  if (opening || session || stopping) return;
  if (!$('key').value.trim() || !$('prompt').value.trim()) {
    status('Enter your local access key and an instruction.', true); return;
  }
  if (!window.LivekitClient) { status('Install the browser dependencies with npm ci --prefix live-camera/web.', true); return; }
  opening = true; controls(true); $('stop').disabled = true;
  try {
    status('Requesting camera access…');
    cameraTrack = await LivekitClient.createLocalVideoTrack({
      deviceId: $('camera').value ? {exact: $('camera').value} : undefined,
      resolution: {width: 1280, height: 720, frameRate: 15}
    });
    cameraTrack.attach($('video')); await $('video').play(); $('video').classList.add('active');
    $('video').parentElement.style.aspectRatio = $('video').videoWidth+'/'+$('video').videoHeight;
    updateGuide();
    const devices = await navigator.mediaDevices.enumerateDevices();
    const selected = cameraTrack.mediaStreamTrack.getSettings().deviceId;
    $('camera').replaceChildren(...devices.filter(d => d.kind === 'videoinput')
      .map((d, i) => new Option(d.label || 'Camera '+(i+1), d.deviceId)));
    if (selected) $('camera').value = selected;
    status('Starting GPU session…');
    const result = await api('start', {question: $('prompt').value.trim(),
      image_budget: Number($('budget').value), crop: cropSettings(), verify: $('verify').checked});
    session = result.session_id;
    const nextRoom = new LivekitClient.Room({adaptiveStream: false, dynacast: false});
    room = nextRoom;
    nextRoom.on(LivekitClient.RoomEvent.Disconnected, () => {
      if (session && !stopping && room === nextRoom) { status('LiveKit disconnected. Restart the session.', true); void stop(true); }
    });
    await nextRoom.connect(result.url, result.token, {autoSubscribe: false});
    await nextRoom.localParticipant.publishTrack(cameraTrack, {
      source: LivekitClient.Track.Source.Camera, simulcast: false,
      videoEncoding: {maxBitrate: 1800000, maxFramerate: 15}
    });
    history = []; reviewResults.clear(); reviewStatus.textContent = ''; epoch = performance.now(); render();
    $('sessionInfo').textContent = 'LiveKit · GPU samples at 4 FPS · 32-frame context · '+result.image_budget+' pixel budget';
    $('cameraTag').textContent = 'LIVE · WEBRTC';
    cameraTrack.mediaStreamTrack.addEventListener('ended', () => void stop());
    controls(true); polling = setInterval(poll, 1000); await poll();
  } catch (error) {
    status(error.message || 'Unable to start camera', true); await stop(true);
  } finally { opening = false; }
};
window.addEventListener('pagehide', () => {
  cameraTrack?.stop();
  if (session) fetch('/api/stop', {method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Access-Key': $('key').value.trim()},
    body: JSON.stringify({session_id: session}), keepalive: true}).catch(() => {});
});
async function health() {
  try {
    const data = await fetch('/health').then(r => r.json());
    verifierReady = !!data.verification_configured;
    if (!session && !opening) {
      controls(false);
      status(!data.livekit_configured ? (data.configuration_error || 'Configure LiveKit in your local .env and restart the backend.')
        : data.ready ? 'GPU worker connected. Ready to start.' : 'Waiting for the GPU worker to connect to LiveKit.');
    }
    $('verifyConfig').textContent = verifierReady ? 'Available · '+data.verification_model
      : 'Set OPENAI_API_KEY in the local backend .env and restart.';
  } catch { if (!session) status('Cannot reach the local backend.', true); }
}
void health(); setInterval(health, 5000);
