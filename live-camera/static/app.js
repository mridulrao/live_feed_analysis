/* Camera media travels through LiveKit; HTTP calls only control the local backend. */
const $ = id => document.getElementById(id);
const guide = document.createElement('div');
guide.className = 'crop-guide';
guide.innerHTML = '<span>OBJECT CROP</span>';
$('video').parentElement.append(guide);
const reviewButton = $('verifyNow');
const reviewStatus = $('reviewStatus');
let microphoneTrack = null, voiceIdentity = null, voiceReady = false;
const remoteAudio = new Set();
let room = null, cameraTrack = null, session = null, opening = false, stopping = false;
let polling = null, pollBusy = false, verifierReady = false, epoch = 0;
let history = [], reviewResults = new Map();
let trackPolling = null, trackBusy = false, latestTracks = null, tracksReceived = 0;
let trackingGeneration = 0, trackingToggleBusy = false;
function showTrackingStatus(message) {
  $('trackingStatus').textContent = message;
  const notice = $('trackingNotice');
  notice.textContent = message;
  notice.hidden = !session || !$('tracking').checked;
}
$('tracking').onchange = async () => {
  const enabled = $('tracking').checked;
  latestTracks = null;
  showTrackingStatus(enabled ? 'Enabling tracking…' : 'Tracking off');
  clearInterval(trackPolling); trackPolling = null;
  if (!session) return;
  const requested = session;
  trackingToggleBusy = true;
  $('tracking').disabled = true;
  try {
    const result = await api('tracking', {session_id: requested, enabled});
    if (requested !== session) return;
    trackingGeneration = result.generation;
    showTrackingStatus(enabled ? 'Watching for your target…' : 'Tracking off');
  } catch (error) {
    if (requested !== session) return;
    $('tracking').checked = !enabled;
    showTrackingStatus(error.message);
  } finally {
    trackingToggleBusy = false;
    $('tracking').disabled = opening || stopping || trackingToggleBusy;
    if (requested === session && $('tracking').checked) {
      trackPolling = setInterval(pollTracks, 100); void pollTracks();
    }
  }
};
async function pollTracks() {
  if (!session || trackBusy || stopping || !$('tracking').checked) return;
  trackBusy = true;
  const requested = session;
  try {
    const data = await api('tracks', {session_id: requested});
    if (requested !== session || !$('tracking').checked) return;
    latestTracks = data.tracking?.generation === trackingGeneration ? data.tracking : null;
    tracksReceived = performance.now();
    showTrackingStatus(latestTracks
      ? latestTracks.message || 'Watching for your target…'
      : 'Waiting for tracking updates…');
  } catch {
    if (requested !== session) return;
    latestTracks = null;
    showTrackingStatus('Tracking connection interrupted');
  } finally { trackBusy = false; }
}
function drawTracks() {
  const canvas = $('tracks'), video = $('video'), stage = video.parentElement;
  const width = stage.clientWidth, height = stage.clientHeight;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  if (session && $('tracking').checked && latestTracks && performance.now()-tracksReceived < 750 && video.videoWidth) {
    const scale = Math.min(width/video.videoWidth, height/video.videoHeight);
    const vw = video.videoWidth*scale, vh = video.videoHeight*scale;
    const left = (width-vw)/2, top = (height-vh)/2;
    ctx.strokeStyle = '#cef894'; ctx.fillStyle = '#cef894'; ctx.lineWidth = 2; ctx.font = '13px system-ui';
    for (const track of latestTracks.boxes) {
      const [x1, y1, x2, y2] = track.box;
      // Mirror coordinates to match the local preview, while keeping text readable.
      const x = left+(1-x2)*vw, y = top+y1*vh;
      ctx.strokeRect(x, y, (x2-x1)*vw, (y2-y1)*vh);
      const label = track.label+' #'+track.track_id;
      const textWidth = Math.min(width-8, ctx.measureText(label).width+10);
      const tx = Math.max(0, Math.min(x, width-textWidth)), ty = Math.max(20, y);
      ctx.fillRect(tx, ty-20, textWidth, 20);
      ctx.fillStyle = '#101211'; ctx.fillText(label, tx+5, ty-5, textWidth-10); ctx.fillStyle = '#cef894';
    }
  }
  requestAnimationFrame(drawTracks);
}
requestAnimationFrame(drawTracks);
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
  for (const id of ['prompt', 'key', 'camera', 'budget', 'crop', 'trackingTarget', 'visionMemory']) $(id).disabled = active;
  $('tracking').disabled = opening || stopping || trackingToggleBusy;
  $('verify').disabled = active || !verifierReady;
  $('voice').disabled = active || !voiceReady;
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
    const video = $('video'), stage = video.parentElement;
    if (!video.videoWidth || !video.videoHeight) return;
    const scale = Math.min(stage.clientWidth/video.videoWidth, stage.clientHeight/video.videoHeight);
    const width = video.videoWidth*scale, height = video.videoHeight*scale;
    guide.style.left = ((stage.clientWidth-width)/2+c.x*width)+'px';
    guide.style.top = ((stage.clientHeight-height)/2+c.y*height)+'px';
    guide.style.width = c.width*width+'px'; guide.style.height = c.height*height+'px';
  }
}
$('crop').onchange = updateGuide;
window.addEventListener('resize', updateGuide);
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
  const shown = history.filter(r => r.description || r.summary || r.state === 'response' || r.state === 'error' ||
    ($('showAll').checked && r.state !== 'snapshot'));
  $('feed').replaceChildren();
  if (!shown.length) {
    const empty = document.createElement('div'); empty.className = 'empty';
    empty.textContent = $('visionMemory').checked ? 'Building continuous visual context…' : 'Watching for an event that matches your prompt…'; $('feed').append(empty);
  }
  for (const r of shown.slice(-100).reverse()) {
    const row = document.createElement('div'); row.className = 'entry';
    const at = document.createElement('time'); at.textContent = r.start.toFixed(1)+'–'+r.end.toFixed(1)+'s · '+(r.description && r.state === 'silence' ? 'OBSERVATION' : r.state.toUpperCase());
    const text = document.createElement('p');
    text.textContent = r.description || r.response || r.summary || (r.state === 'standby' ? 'Relevant activity in progress.' : 'Observing.');
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
    $('voiceStatus').textContent = data.voice?.error || ('Voice '+(data.voice?.state || 'off'));
    $('voiceTranscript').textContent = (data.voice?.transcripts || []).slice(-4).map(t => (t.role === 'user' ? 'You: ' : 'Assistant: ')+t.text).join('\n');
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
    } else status(history.length ? ($('visionMemory').checked ? 'Observing continuously · vision memory active' : 'Watching for your requested event…') : 'Camera published. Waiting for GPU observations…');
  } catch (error) {
    status(error.message, true);
    // Keep polling after transient local API failures; the lease handles a closed tab.
  } finally { pollBusy = false; }
}
async function closeMedia() {
  clearInterval(polling); polling = null;
  clearInterval(trackPolling); trackPolling = null; latestTracks = null; showTrackingStatus('Tracking off');
  if (microphoneTrack) { microphoneTrack.stop(); microphoneTrack = null; }
  for (const audio of remoteAudio) { audio.pause(); audio.srcObject = null; audio.remove(); }
  remoteAudio.clear(); voiceIdentity = null;
  $('muteMic').hidden = true; $('resumeAudio').hidden = true;
  $('voiceStatus').textContent = 'Voice off';
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
    $('settings').open = true;
    (!$('key').value.trim() ? $('key') : $('prompt')).focus();
    status('Enter your local access key and an instruction in Settings.', true); return;
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
    updateGuide();
    const devices = await navigator.mediaDevices.enumerateDevices();
    const selected = cameraTrack.mediaStreamTrack.getSettings().deviceId;
    $('camera').replaceChildren(...devices.filter(d => d.kind === 'videoinput')
      .map((d, i) => new Option(d.label || 'Camera '+(i+1), d.deviceId)));
    if (selected) $('camera').value = selected;
    if ($('voice').checked) {
      $('visionMemory').checked = true;
      status('Requesting microphone access…');
      microphoneTrack = await LivekitClient.createLocalAudioTrack({echoCancellation: true, noiseSuppression: true, autoGainControl: true});
    }
    status('Starting GPU session…');
    const result = await api('start', {question: $('prompt').value.trim(),
      image_budget: Number($('budget').value), crop: cropSettings(), verify: $('verify').checked,
      vision_memory: $('visionMemory').checked, voice: $('voice').checked,
      tracking: $('tracking').checked, tracking_target: $('trackingTarget').value.trim()});
    session = result.session_id; trackingGeneration = 0;
    $('memorySession').value = session;
    voiceIdentity = result.voice_identity;
    showTrackingStatus($('tracking').checked ? 'Watching for your target…' : 'Tracking off');
    const nextRoom = new LivekitClient.Room({adaptiveStream: false, dynacast: false});
    room = nextRoom;
    nextRoom.on(LivekitClient.RoomEvent.Disconnected, () => {
      if (session && !stopping && room === nextRoom) { status('LiveKit disconnected. Restart the session.', true); void stop(true); }
    });
    nextRoom.on(LivekitClient.RoomEvent.TrackPublished, (publication, participant) => {
      if (participant.identity === voiceIdentity && publication.kind === LivekitClient.Track.Kind.Audio) publication.setSubscribed(true);
    });
    nextRoom.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (participant.identity !== voiceIdentity || track.kind !== LivekitClient.Track.Kind.Audio) return;
      const audio = track.attach(); audio.autoplay = true; document.body.append(audio); remoteAudio.add(audio);
      audio.play().catch(() => { $('resumeAudio').hidden = false; });
    });
    nextRoom.on(LivekitClient.RoomEvent.TrackUnsubscribed, track => {
      for (const audio of track.detach()) { remoteAudio.delete(audio); audio.remove(); }
    });
    await nextRoom.connect(result.url, result.token, {autoSubscribe: false});
    for (const participant of nextRoom.remoteParticipants.values()) {
      if (participant.identity === voiceIdentity) for (const publication of participant.audioTrackPublications.values()) publication.setSubscribed(true);
    }
    if (microphoneTrack) {
      await nextRoom.localParticipant.publishTrack(microphoneTrack, {source: LivekitClient.Track.Source.Microphone});
      $('muteMic').hidden = false; $('muteMic').textContent = 'Mute microphone';
      await nextRoom.startAudio().catch(() => { $('resumeAudio').hidden = false; });
    }
    await nextRoom.localParticipant.publishTrack(cameraTrack, {
      source: LivekitClient.Track.Source.Camera, simulcast: false,
      videoEncoding: {maxBitrate: 1800000, maxFramerate: 15}
    });
    history = []; reviewResults.clear(); reviewStatus.textContent = ''; epoch = performance.now(); render();
    $('sessionInfo').textContent = 'LiveKit · GPU samples at 4 FPS · 32-frame context · '+result.image_budget+' pixel budget';
    $('cameraTag').textContent = 'LIVE · WEBRTC';
    cameraTrack.mediaStreamTrack.addEventListener('ended', () => void stop());
    controls(true); polling = setInterval(poll, 1000);
    if ($('tracking').checked) { trackPolling = setInterval(pollTracks, 100); void pollTracks(); }
    await poll();
  } catch (error) {
    status(error.message || 'Unable to start camera', true); await stop(true);
  } finally { opening = false; controls(!!session); }
};
window.addEventListener('pagehide', () => {
  cameraTrack?.stop(); microphoneTrack?.stop();
  if (session) fetch('/api/stop', {method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Access-Key': $('key').value.trim()},
    body: JSON.stringify({session_id: session}), keepalive: true}).catch(() => {});
});
async function health() {
  try {
    const data = await fetch('/health').then(r => r.json());
    verifierReady = !!data.verification_configured;
    voiceReady = !!data.voice_configured;
    $('voiceConfig').textContent = voiceReady ? 'Available · '+data.voice_model : 'Set OPENAI_API_KEY in the backend environment and restart.';
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

$('voice').onchange = () => { if ($('voice').checked) $('visionMemory').checked = true; };
$('visionMemory').onchange = () => { if (!$('visionMemory').checked) $('voice').checked = false; };
$('muteMic').onclick = async () => {
  if (!microphoneTrack) return;
  try {
    if (microphoneTrack.isMuted) await microphoneTrack.unmute(); else await microphoneTrack.mute();
    $('muteMic').textContent = microphoneTrack.isMuted ? 'Unmute microphone' : 'Mute microphone';
  } catch (error) { $('voiceStatus').textContent = error.message; }
};
$('resumeAudio').onclick = async () => {
  try { await room?.startAudio(); for (const audio of remoteAudio) await audio.play(); $('resumeAudio').hidden = true; }
  catch { $('voiceStatus').textContent = 'Speaker playback blocked. Try enabling audio again.'; }
};
$('searchMemory').onclick = async () => {
  $('searchMemory').disabled = true;
  try {
    const result = await api('memory', {session_id: $('memorySession').value.trim(), query: $('memoryQuery').value.trim(),
      start: $('memoryStart').value === '' ? null : Number($('memoryStart').value),
      end: $('memoryEnd').value === '' ? null : Number($('memoryEnd').value)});
    $('memoryResults').textContent = result.entries.map(e => `${e.start.toFixed(2)}–${e.end.toFixed(2)}s · ${e.state} · ${e.evidence_available ? 'snapshots retained' : 'no retained snapshots'}\n${e.description || e.response || e.summary || 'Processed without a description; activity unknown.'}`).join('\n\n')
      + '\n\nCoverage gaps: '+(result.gaps.map(g => g.map(t => t.toFixed(2)).join('–')+'s').join(', ') || 'None within the recorded range.')
      + '\n'+result.limitations;
  } catch (error) { $('memoryResults').textContent = error.message; }
  finally { $('searchMemory').disabled = false; }
};
