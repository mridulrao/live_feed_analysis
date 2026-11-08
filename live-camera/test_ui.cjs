/* Run with node --test live-camera/test_ui.cjs. No browser or network required. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

async function page() {
  const elements = new Map();
  function element() {
    return {checked: false, disabled: false, value: '', textContent: '', style: {},
      append() {}, replaceChildren() {}, classList: {toggle() {}, add() {}, remove() {}},
      parentElement: {append() {}}, addEventListener() {}};
  }
  const context = vm.createContext({
    document: {getElementById(id) {if (!elements.has(id)) elements.set(id, element()); return elements.get(id);},
      createElement: element},
    window: {addEventListener() {}}, performance: {now: () => 1000},
    requestAnimationFrame() {}, setInterval: () => 1, clearInterval() {}, AbortSignal,
    fetch: async () => ({json: async () => ({ready: true, livekit_configured: true})})
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'static/app.js'), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  vm.runInContext("session = 'camera-session';", context);
  return {context, elements, run: code => vm.runInContext(code, context)};
}

test('Enable tracking stays available during a session and controls the live API', async () => {
  const {elements, run} = await page();
  run('controls(true)');
  assert.equal(elements.get('tracking').disabled, false);
  // Replace the HTTP boundary, keeping the real change handler and polling logic.
  run('globalThis.requests = []; api = async (path, body) => {requests.push({path, body}); return {enabled: body.enabled, generation: 1};};');
  elements.get('tracking').checked = true;
  await elements.get('tracking').onchange();
  assert.equal(run('requests[0].path'), 'tracking');
  assert.equal(run('requests[0].body.enabled'), true);
  assert.equal(run('trackingGeneration'), 1);
  elements.get('tracking').checked = false;
  await elements.get('tracking').onchange();
  assert.equal(run('requests.filter(r => r.path === "tracking").at(-1).body.enabled'), false);
  assert.equal(run('latestTracks'), null);
  assert.equal(run('trackPolling'), null);
});

test('failed toggle restores the previous UI state', async () => {
  const {elements, run} = await page();
  run('api = async () => {throw new Error("Tracking disabled by worker");};');
  elements.get('tracking').checked = true;
  await elements.get('tracking').onchange();
  assert.equal(elements.get('tracking').checked, false);
  assert.match(elements.get('trackingStatus').textContent, /disabled by worker/);
  assert.equal(elements.get('tracking').disabled, false);
});

test('polling rejects an update from before the last toggle', async () => {
  const {elements, run} = await page();
  elements.get('tracking').checked = true;
  run('trackingGeneration = 2; api = async () => ({tracking: {generation: 1, boxes: [{label: "stale"}]}});');
  await run('pollTracks()');
  assert.equal(run('latestTracks'), null);
});

test('hold-position instructions appear beside the camera and hide when disabled', async () => {
  const {elements, run} = await page();
  elements.get('tracking').checked = true;
  run('api = async () => ({tracking: {generation: 0, status: "localizing", message: "Target noticed—please hold position until mapped", boxes: []}});');
  await run('pollTracks()');
  assert.equal(elements.get('trackingNotice').hidden, false);
  assert.match(elements.get('trackingNotice').textContent, /hold position/);
  run('api = async (path, body) => ({enabled: body.enabled, generation: 1});');
  elements.get('tracking').checked = false;
  await elements.get('tracking').onchange();
  assert.equal(elements.get('trackingNotice').hidden, true);
});

test('verification failure replaces old boxes with target-absent status', async () => {
  const {elements, run} = await page();
  elements.get('tracking').checked = true;
  run('latestTracks = {boxes: [{label: "device"}]}; api = async () => ({tracking: {generation: 0, status: "lost", message: "Target not visible—searching automatically", boxes: []}});');
  await run('pollTracks()');
  assert.equal(run('latestTracks.boxes.length'), 0);
  assert.match(elements.get('trackingNotice').textContent, /not visible/);
});
