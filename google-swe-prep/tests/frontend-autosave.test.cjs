/* Run with: node --test tests/frontend-autosave.test.cjs
   Exercise the shipped autosave functions without launching a browser. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const app = fs.readFileSync(path.join(__dirname, '../google_prep/gui/static/app.js'), 'utf8');
const start = app.indexOf('  async function save(withText, keepalive = false) {');
const end = app.indexOf('\n  function togglePause()', start);
assert.ok(start > 0 && end > start, 'The shipped autosave functions must be present');

function setup(api) {
  const drafts = new Map();
  const env = {
    api, code: { value: 'first version' }, trace: { value: 'trace' }, savedTag: { textContent: '' },
    performance: { now: () => 100 }, localStorage: { removeItem: key => drafts.delete(key) },
    toast: () => {}, setTimeout: () => 1, clearTimeout: () => {},
    drafts,
  };
  vm.createContext(env);
  vm.runInContext(`
    let dirty = true, editRevision = 1, saveTail = Promise.resolve(), lastSaved = 0, saveTimer;
    const draftKey = 'current-session';
    const B = {problem_id:'G-1', practice:true, minutes:60, elapsed:30, ticks:[false,false,false,false], shown:-1, hints:0};
    function localDraft() { drafts.set(draftKey, {code:code.value, trace:trace.value}); }
    ${app.slice(start, end)}
    globalThis.subject = {save, onInput, get dirty() {return dirty;}};
  `, env);
  return { ...env, subject: env.subject, drafts };
}

test('failed text save is retried without another edit', async () => {
  const calls = [];
  const env = setup(async (_, payload) => {
    calls.push(payload);
    if (calls.length === 1) throw new Error('connection interrupted');
  });
  assert.equal(await env.subject.save(true), false);
  assert.equal(env.subject.dirty, true);
  assert.equal(env.drafts.get('current-session').code, 'first version');
  assert.equal(await env.subject.save(false), true);
  assert.equal(calls[1].code, 'first version');
  assert.equal(calls[1].trace, 'trace');
  assert.equal(env.subject.dirty, false);
  assert.equal(env.drafts.has('current-session'), false);
});

test('an older acknowledgement does not clear a newer pending edit', async () => {
  let releaseFirst, firstStarted;
  const started = new Promise(resolve => {firstStarted = resolve;});
  const env = setup(() => new Promise(resolve => {releaseFirst = resolve; firstStarted();}));
  const first = env.subject.save(true);
  await started;
  env.code.value = 'newer version';
  env.subject.onInput();
  releaseFirst();
  assert.equal(await first, true);
  assert.equal(env.subject.dirty, true);
  assert.equal(env.drafts.get('current-session').code, 'newer version');
  assert.notEqual(env.savedTag.textContent.slice(0, 6), 'Saved ');
});

test('save requests are serialized and the latest snapshot wins', async () => {
  let releaseFirst, notifyFirst;
  const started = new Promise(resolve => {notifyFirst = resolve;});
  const calls = [];
  let concurrent = 0, maximum = 0;
  const env = setup(async (_, payload) => {
    calls.push(payload); concurrent++; maximum = Math.max(maximum, concurrent);
    if (calls.length === 1) await new Promise(resolve => {releaseFirst = resolve; notifyFirst();});
    concurrent--;
  });
  const first = env.subject.save(true);
  await started;
  env.code.value = 'latest version'; env.subject.onInput();
  const second = env.subject.save(true);
  releaseFirst();
  await Promise.all([first, second]);
  assert.equal(maximum, 1);
  assert.equal(calls.at(-1).code, 'latest version');
  assert.equal(env.subject.dirty, false);
  assert.equal(env.drafts.size, 0);
});
