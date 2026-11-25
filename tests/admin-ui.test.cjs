// Run with node --test tests/admin-ui.test.cjs. No browser, network or npm dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../src/localive/static/admin.js"), "utf8");
const settled = () => new Promise((resolve) => setImmediate(resolve));

async function page(hash, initialStatus = {}) {
  const status = { sessions: [], brains: ["qwen"], voices: {}, connection: { urls: [] }, ...initialStatus };
  const model = { id: "m", path: "models/m.gguf", state: "downloading", family: "M" };
  const calls = [], timers = new Map(), nodes = new Map(), listeners = new Map();
  let timerId = 0;
  function schedule(callback, delay, repeat) {
    timers.set(++timerId, { callback, delay, repeat });
    return timerId;
  }
  function node(selector) {
    if (["#hf-list", "#hf-repo", "#show-advanced"].includes(selector)) return null;
    if (!nodes.has(selector)) nodes.set(selector, {
      querySelectorAll: () => [], querySelector: () => null, innerHTML: "",
    });
    return nodes.get(selector);
  }
  const browser = {
    URL, AbortSignal,
    location: { hash }, document: { querySelector: node, querySelectorAll: () => [] },
    window: { addEventListener: (type, fn) => listeners.set(type, fn), getSelection: () => null },
    sessionStorage: { getItem() {} }, localStorage: { getItem() {} },
    setTimeout: (fn, delay) => schedule(fn, delay, false),
    setInterval: (fn, delay) => schedule(fn, delay, true),
    clearTimeout: (id) => timers.delete(id), clearInterval: (id) => timers.delete(id),
    async fetch(endpoint) {
      calls.push(endpoint);
      const choices = model.state === "done" ? [model.path] : [];
      const setting = { key: "brain.model", choices, value: "", default: "", label: "Modèle" };
      const body = {
        "/api/status": status, "/api/models": { models: [model] },
        "/api/settings": { groups: [{ id: "brain", settings: [setting] }] },
      }[endpoint];
      assert.ok(body, endpoint);
      return { ok: true, status: 200, json: async () => body };
    },
  };
  const context = vm.createContext(browser);
  await vm.runInContext(source, context);
  await settled();
  async function tick(delay) {
    const due = [...timers].filter(([, timer]) => timer.delay === delay);
    for (const [id, timer] of due) {
      if (!timer.repeat) timers.delete(id);
      await timer.callback();
    }
    await settled();
    return due.length;
  }
  return { calls, tick, model, status, view: node("#view"), context, listeners };
}

test("home refresh preserves a manual copy selection until it is released", async () => {
  const { context, tick, view, status } = await page("#home", { ready: true, brain: "first" });
  const address = {};
  view.contains = (node) => node === address;
  context.window.getSelection = () => ({ isCollapsed: false, anchorNode: address });
  status.brain = "next";
  await tick(5000);
  assert.match(view.innerHTML, /<strong>first<\/strong>/);
  context.window.getSelection = () => null;
  await tick(5000);
  assert.match(view.innerHTML, /<strong>next<\/strong>/);
});

test("the hangup method is shown when enabled and labels the local model", async () => {
  const { context, view } = await page("#home");
  context.turnSettings = { id: "turn", title: "Fin de tour", settings: [
    { key: "turn.end_on_goodbye", type: "bool", label: "Raccrocher", value: false, choices: [] },
    { key: "turn.goodbye_mode", type: "str", label: "Détection du raccrochage", value: "phrases",
      choices: ["phrases", "d3"], only: "turn.end_on_goodbye=true" },
  ] };
  vm.runInContext("renderSection(turnSettings)", context);
  assert.doesNotMatch(view.innerHTML, /data-key="turn.goodbye_mode"/);
  vm.runInContext('state.dirty.set("turn.end_on_goodbye", true); renderSection(turnSettings)', context);
  assert.match(view.innerHTML, /data-key="turn.goodbye_mode" aria-label="Détection du raccrochage"/);
  assert.match(view.innerHTML, /Formules prédéfinies/);
  assert.match(view.innerHTML, /Décideur local \(d3-mini\)/);
});

test("mobile menu closes on Escape and a cancelled navigation keeps the draft URL", async () => {
  const { context, listeners } = await page("#brain");
  const node = context.document.querySelector;
  const toggle = node("#nav-toggle");
  let expanded = "false", focused;
  toggle.getAttribute = () => expanded;
  toggle.setAttribute = (_, value) => { expanded = value; };
  toggle.focus = () => { focused = toggle; };
  toggle.onclick();
  assert.equal(expanded, "true");
  node(".sidebar").onkeydown({ key: "Escape" });
  assert.equal(expanded, "false");
  assert.equal(focused, toggle);

  const oldURL = "http://localhost/#brain";
  context.location.hash = "#connect";
  context.confirm = () => false;
  context.history = { replaceState: (_, __, url) => { context.location.hash = new URL(url).hash; } };
  vm.runInContext('state.dirty.set("llm.temperature", 0.4)', context);
  listeners.get("hashchange")({ oldURL });
  assert.equal(context.location.hash, "#brain");
  assert.equal(vm.runInContext('state.dirty.get("llm.temperature")', context), 0.4);
});

test("one status poll per interval, including after repeated home redraws", async () => {
  const { tick, calls, context } = await page("#home");
  const initial = calls.filter((p) => p === "/api/status").length;
  for (let i = 0; i < 5; i++) assert.equal(await tick(5000), 1);
  assert.equal(calls.filter((p) => p === "/api/status").length, initial + 5);
  context.location.hash = "#connect";
  await vm.runInContext("route()", context);
  assert.equal(await tick(5000), 0);
});

test("the permanent restart button sends one POST and reconnects to the new server", async () => {
  const { context, tick, status } = await page("#home", { ready: true });
  const find = context.document.querySelector;
  const button = find("#restart");
  assert.equal(find("#banner").hidden, true);
  status.restart_pending = ["voice.threads"];
  await tick(5000);
  assert.equal(find("#banner").hidden, false);
  assert.equal(find("#restart"), button);
  assert.doesNotMatch(find("#banner").innerHTML, /<button/);
  let posts = 0, polls = 0, reloads = 0;
  context.confirm = () => true;
  context.location.reload = () => { reloads++; };
  context.fetch = async (path, options) => {
    if (path === "/api/restart") {
      assert.equal(options.method, "POST");
      posts++;
      return { ok: true, json: async () => ({ restarting: true }) };
    }
    assert.equal(path, "/health");
    assert.equal(options.cache, "no-store");
    polls++;
    if (polls === 2) throw new TypeError("server offline");
    return { ok: true, json: async () => ({ restarting: polls === 1, ready: false }) };
  };
  const restarting = button.onclick();
  await settled();
  assert.equal(button.disabled, true);
  await button.onclick();
  assert.equal(posts, 1);
  await tick(2000);
  await tick(2000);
  assert.equal(reloads, 0);
  await tick(2000);
  await restarting;
  assert.equal(reloads, 1);
});

test("cancelling or refusing a restart preserves the unsaved settings", async () => {
  const { context, tick } = await page("#brain", { ready: true });
  const find = context.document.querySelector;
  const button = find("#restart");
  vm.runInContext('state.dirty.set("llm.temperature", 0.4)', context);
  let posts = 0;
  context.fetch = async () => {
    posts++;
    return { ok: false, status: 409, json: async () => ({ error: "Une conversation est en cours." }) };
  };
  context.confirm = (message) => {
    assert.match(message, /modifications non enregistrées/);
    return false;
  };
  await button.onclick();
  assert.equal(posts, 0);
  context.confirm = () => true;
  await button.onclick();
  assert.equal(posts, 1);
  assert.equal(button.disabled, false);
  assert.equal(find("#toast").textContent, "Une conversation est en cours.");
  assert.equal(vm.runInContext('state.dirty.get("llm.temperature")', context), 0.4);
  assert.equal(await tick(2000), 0);
});

test("the first completed download appears in the model selector", async () => {
  const { tick, model, calls, view } = await page("#brain");
  model.state = "done";
  model.installed = true;
  await tick(1500);
  assert.equal(calls.filter((p) => p === "/api/settings").length, 2);
  assert.match(view.innerHTML, /<option value="models\/m\.gguf"/);
});

test("connection page gives WebSocket endpoints and preserves the HTTP bases for Cornelius", async () => {
  const connection = {
    urls: ["http://127.0.0.1:8787/v1", "https://[2001:db8::1]:8080/voice/v1/"],
    api_key_required: true,
  };
  const { view } = await page("#connect", { connection });
  for (const url of ["ws://127.0.0.1:8787/live", "wss://[2001:db8::1]:8080/voice/live", ...connection.urls]) {
    assert.ok(view.innerHTML.includes(`<code tabindex="-1">${url}</code>`));
    assert.ok(view.innerHTML.includes(`data-copy="${url}"`));
    assert.ok(view.innerHTML.includes(`aria-label="Copier ${url}"`));
  }
  assert.match(view.innerHTML, /session\.start/);
  assert.match(view.innerHTML, /Authorization: Bearer/);
  assert.match(view.innerHTML, /Voix de Cornelius/);
});

test("copying an endpoint works over LAN HTTP and leaves a manual selection when denied", async () => {
  const url = "ws://192.168.1.10:8787/live";
  const { view, context } = await page("#connect");
  const toast = context.document.querySelector("#toast");
  let focused, selected, copied;
  const address = { focus() { focused = address; } };
  const button = { dataset: { copy: url }, previousElementSibling: address, focus() { focused = button; } };
  view.querySelectorAll = () => [button];
  context.navigator = {};
  context.document.createRange = () => ({ selectNodeContents(node) { this.node = node; } });
  context.window.getSelection = () => ({
    removeAllRanges() { selected = null; },
    addRange(range) { selected = range.node; },
  });
  context.document.execCommand = (command) => {
    assert.equal(command, "copy");
    assert.equal(selected, address);
    return true;
  };
  vm.runInContext("bindCopy()", context);
  await button.onclick();
  assert.equal(focused, button);
  assert.equal(toast.textContent, "Adresse copiée.");

  context.navigator.clipboard = { async writeText() { throw new Error("permission denied"); } };
  context.document.execCommand = () => false;
  await button.onclick();
  assert.equal(selected, address);
  assert.equal(focused, address);
  assert.match(toast.textContent, /Adresse sélectionnée/);

  context.navigator.clipboard.writeText = async (text) => { copied = text; };
  context.document.execCommand = () => assert.fail("secure clipboard already succeeded");
  await button.onclick();
  assert.equal(copied, url);
  assert.equal(toast.textContent, "Adresse copiée.");
});

test("home displays the live endpoint and the current router state", async () => {
  const connection = { urls: ["http://192.168.1.10:9000/v1"] };
  const { view, tick, status } = await page("#home", { connection, tool_router_state: "idle" });
  assert.match(view.innerHTML, /data-copy="ws:\/\/192\.168\.1\.10:9000\/live"/);
  assert.match(view.innerHTML, /routeur en attente d’un grand catalogue/);
  status.tool_router_state = "missing";
  await tick(5000);
  assert.match(view.innerHTML, /fichier du routeur absent/);
  status.decision_model_state = "downloading";
  status.decision_model_download_pct = 42;
  await tick(5000);
  assert.match(view.innerHTML, /téléchargement de d3-mini · 42 %/);
  status.decision_model_download_pct = 100;
  await tick(5000);
  assert.match(view.innerHTML, /vérification des poids de d3-mini/);
  status.decision_model_state = "download_failed";
  await tick(5000);
  assert.match(view.innerHTML, /téléchargement de d3-mini interrompu/);
  status.decision_model_state = "idle";
  status.tool_router_state = "off";
  await tick(5000);
  assert.match(view.innerHTML, /tri des outils désactivé/);
  status.tool_router_state = "loading";
  await tick(5000);
  assert.match(view.innerHTML, /chargement du routeur/);
  status.tool_router = true;
  status.tool_router_file = "routeur A&B.gguf";
  await tick(5000);
  assert.match(view.innerHTML, /routeur prêt \(routeur A&amp;B\.gguf\)/);
});
