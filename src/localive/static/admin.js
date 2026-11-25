// Localive administration: status, client connections, explained settings, live log.
// Vanilla JS, no build step. Talks to /api/status, /api/settings, /api/logs.

const $ = (selector, root = document) => root.querySelector(selector);
const view = $("#view");
const state = {
  token: sessionStorage.getItem("localive-token") || "",
  schema: null,          // /api/settings
  status: null,          // /api/status
  dirty: new Map(),      // key -> new value
  advanced: localStorage.getItem("localive-advanced") === "1",
  logIndex: 0,
  timers: [],
  pageVersion: 0,
  homeTimer: null,
  homeMarkup: "",
};

// ------------------------------------------------------------------ icons
const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
  plug: '<path d="M9 2v6M15 2v6M6 8h12v3a6 6 0 0 1-12 0zM12 17v5"/>',
  voice: '<path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  ear: '<path d="M6 8.5a6 6 0 1 1 12 0c0 3.5-3 4.5-3 7.5a3.5 3.5 0 0 1-6.5 1.8"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-1.5 2-1.5 3.5"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M9 2h6"/>',
  split: '<path d="M16 3h5v5M8 3H3v5M21 3l-7 7M3 3l7 7M12 14v7"/>',
  brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h0a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h0a3 3 0 0 1-3-3"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  waves: '<path d="M2 12c2-3 4-3 6 0s4 3 6 0 4-3 6 0M2 7c2-3 4-3 6 0s4 3 6 0 4-3 6 0M2 17c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/>',
  shield: '<path d="M12 3 4 6v6c0 4.5 3.4 8 8 9 4.6-1 8-4.5 8-9V6z"/>',
  log: '<path d="M4 5h16M4 10h16M4 15h10M4 20h7"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.layers}</svg>`;

const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtNumber = (value) => (typeof value === "number" ? value.toLocaleString("fr-FR", { maximumFractionDigits: 3 }) : value);

// -------------------------------------------------------------------- api
async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    await askToken();
    return api(path, options);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error?.message || body.error || `HTTP ${response.status}`);
  return body;
}

function askToken() {
  return new Promise((resolve) => {
    const dialog = $("#login");
    $("#login-form").onsubmit = () => {
      state.token = $("#login-token").value.trim();
      sessionStorage.setItem("localive-token", state.token);
      resolve();
    };
    if (!dialog.open) dialog.showModal();
  });
}

function toast(text, error = false) {
  const element = $("#toast");
  element.textContent = text;
  element.className = `toast${error ? " err" : ""}`;
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (element.hidden = true), error ? 7000 : 3500);
}

// ------------------------------------------------------------- navigation
const PAGES = [
  { id: "home", title: "Accueil", icon: "home" },
  { id: "connect", title: "Connecter un client", icon: "plug" },
];

function renderNav() {
  const groups = state.schema?.groups || [];
  const pending = new Set(state.status?.restart_pending || []);
  const pageLinks = PAGES.map((p) => link(p.id, p.title, p.icon)).join("");
  const settingLinks = groups
    .map((g) => link(g.id, g.title, g.icon, g.settings.some((s) => pending.has(s.key))))
    .join("");
  $("#nav").innerHTML = `${pageLinks}<span class="eyebrow nav-label">Réglages</span>${settingLinks}
    <span class="eyebrow nav-label">Suivi</span>${link("log", "Journal", "log")}`;
  markCurrent();
}

function link(id, title, iconName, dot = false) {
  return `<a class="nav-item" href="#${id}" data-page="${id}">${icon(iconName)}<span>${esc(title)}</span>${dot ? '<span class="dot" title="Redémarrage nécessaire"></span>' : ""}</a>`;
}

function markCurrent() {
  const current = location.hash.slice(1) || "home";
  document.querySelectorAll(".nav-item").forEach((a) => {
    a.toggleAttribute("aria-current", a.dataset.page === current);
    if (a.dataset.page === current) a.setAttribute("aria-current", "page");
  });
}

const navToggle = $("#nav-toggle");
navToggle.onclick = () => {
  const open = navToggle.getAttribute("aria-expanded") === "true";
  navToggle.setAttribute("aria-expanded", String(!open));
};

function closeNavigation() {
  if (navToggle.getAttribute("aria-expanded") === "true") {
    navToggle.setAttribute("aria-expanded", "false");
    $("#main").focus();
  }
}

$("#nav").onclick = (event) => {
  if (event.target.closest('a[aria-current="page"]')) closeNavigation();
};
$(".sidebar").onkeydown = (event) => {
  if (event.key === "Escape" && navToggle.getAttribute("aria-expanded") === "true") {
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.focus();
  }
};

async function route() {
  const version = ++state.pageVersion;
  state.homeMarkup = "";
  clearTimeout(state.homeTimer);
  state.timers.forEach(clearInterval);
  state.timers = [];
  const page = location.hash.slice(1) || "home";
  markCurrent();
  if (page === "home") return renderHome(version);
  if (page === "connect") return renderConnect();
  if (page === "log") return renderLog();
  const group = state.schema?.groups.find((g) => g.id === page);
  if (group) return renderSection(group);
  location.hash = "home";
}

// ----------------------------------------------------------------- status
async function refreshStatus() {
  try {
    state.status = await api("/api/status");
  } catch (error) {
    state.status = null;
  }
  const s = state.status;
  const foot = $("#foot-state");
  if (restartButton.disabled || s?.restarting) {
    foot.className = "pill warn";
    foot.textContent = "redémarrage";
  } else if (!s) {
    foot.className = "pill err";
    foot.textContent = "injoignable";
  } else {
    foot.className = `pill ${s.ready ? "ok" : s.error ? "err" : "warn"}`;
    foot.textContent = s.ready ? "prêt" : s.error ? "échec" : "démarrage";
    $("#foot-version").textContent = `v${s.version}`;
  }
  renderBanner();
  return s;
}

function renderBanner() {
  const pending = state.status?.restart_pending || [];
  const banner = $("#banner");
  if (!pending.length) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const change = pending.length === 1 ? "réglage enregistré prendra" : "réglages enregistrés prendront";
  banner.textContent = `${pending.length} ${change} effet au redémarrage de Localive.`;
}

const restartButton = $("#restart");
restartButton.onclick = async () => {
  if (restartButton.disabled) return;
  const discardDrafts = state.dirty.size > 0;
  const question = discardDrafts
    ? "Redémarrer Localive et abandonner les modifications non enregistrées ?"
    : "Redémarrer Localive ? Les modèles vont être rechargés.";
  if (!confirm(question)) return;
  try {
    await restartServer(discardDrafts);
  } catch (error) {
    toast(error.message, true);
  }
};

async function restartServer(discardDrafts = false) {
  if (restartButton.disabled) return;
  const label = restartButton.textContent;
  restartButton.disabled = true;
  restartButton.textContent = "Redémarrage…";
  try {
    await api("/api/restart", { method: "POST" });
    if (discardDrafts) {
      state.dirty.clear();
      renderSavebar();
    }
    toast("Redémarrage demandé. La page se reconnectera automatiquement.");
    const deadline = Date.now() + 300000;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      try {
        const signal = globalThis.AbortSignal?.timeout?.(5000);
        const response = await fetch("/health", { cache: "no-store", signal });
        if (response.ok && (await response.json()).restarting === false) {
          location.reload();
          return;
        }
      } catch { /* The launcher is replacing the process. */ }
    }
    throw new Error("Localive ne répond pas à cette adresse. Vérifiez le lanceur et le port du serveur.");
  } finally {
    restartButton.disabled = false;
    restartButton.textContent = label;
  }
}

function statePill(ok, warn, text) {
  return `<span class="pill ${ok ? "ok" : warn ? "warn" : "err"}">${esc(text)}</span>`;
}

const SEARCH_TEXT = {
  ready: "prête", starting: "démarrage", idle: "à la demande", off: "désactivée", docker_missing: "Docker absent",
  docker_stopped: "Docker arrêté", unreachable: "injoignable", error: "erreur",
};

const ROUTER_TEXT = {
  off: "tri des outils désactivé · catalogue complet",
  missing: "fichier du routeur absent · catalogue complet",
  download_failed: "téléchargement de d3-mini interrompu · reprise au prochain besoin",
  idle: "routeur en attente d’un grand catalogue",
  loading: "chargement du routeur",
  unavailable: "routeur indisponible · catalogue complet",
};
const SESSION_TEXT = { listening: "À l’écoute", thinking: "Réflexion", speaking: "Prise de parole" };

function routerStatus(s) {
  if (s.decision_model_state === "downloading") {
    return s.decision_model_download_pct === 100
      ? "vérification des poids de d3-mini"
      : `téléchargement de d3-mini · ${s.decision_model_download_pct} %`;
  }
  if (s.decision_model_state === "download_failed") return ROUTER_TEXT.download_failed;
  if (s.tool_router) return s.tool_router_file ? `routeur prêt (${s.tool_router_file})` : "routeur prêt";
  return ROUTER_TEXT[s.tool_router_state] || "catalogue complet";
}

async function renderHome(version) {
  const s = await refreshStatus();
  if (version !== state.pageVersion) return;
  // Schedule after the request finishes; leave the token form alone while it is being edited.
  const hfAccess = (s?.error_kind || "").startsWith("hf_");
  if (!hfAccess) state.homeTimer = setTimeout(() => renderHome(version), 5000);
  if (!s) {
    state.homeMarkup = "";
    view.innerHTML = `<div class="page-head"><h1>Localive</h1><p class="muted">Le serveur ne répond pas.</p></div>`;
    return;
  }
  const search = s.search || {};
  const firstUrl = s.connection.urls[0];
  const uptime = s.uptime_s > 3600 ? `${Math.floor(s.uptime_s / 3600)} h ${Math.floor((s.uptime_s % 3600) / 60)} min` : `${Math.floor(s.uptime_s / 60)} min`;
  const sessions = s.sessions.length
    ? `<div class="session-table" role="region" aria-label="Conversations en cours" tabindex="0"><table><thead><tr><th>Session</th><th>Transport</th><th>État</th><th>Cerveau</th><th>Voix</th></tr></thead><tbody>${s.sessions
        .map((x) => `<tr><td class="mono">${esc(x.id.slice(-8))}</td><td>${x.transport === "webrtc" ? "WebRTC" : "WebSocket"}</td><td>${esc(SESSION_TEXT[x.state] || x.state)}</td><td>${esc(x.brain)}</td><td>${esc(x.voice)}</td></tr>`)
        .join("")}</tbody></table></div>`
    : `<p class="muted">Aucune conversation en cours.</p>`;
  const markup = `
    <div class="page-head">
      <h1>${s.ready ? "Localive est prêt" : hfAccess ? "Une dernière étape : la voix" : s.error ? "Localive n'a pas démarré" : "Localive démarre…"}</h1>
      ${hfAccess ? '<p class="muted">La voix Phonon demande un compte Hugging Face, l’acceptation de ses conditions et un jeton d’accès.</p>' : s.error ? `<p class="muted">${esc(s.error)}</p>` : ""}
    </div>
    ${hfAccess ? hfCard(s.error_kind) : ""}
    <div class="card connection-card">
      <div class="card-head"><h2>Adresse WebSocket Live</h2><a class="btn small" href="#connect">Connecter un client</a></div>
      ${firstUrl ? copyAddress(liveSocketUrl(firstUrl)) : ""}
    </div>
    <div class="grid">
      <div class="card stat"><span class="eyebrow">Écoute</span><strong>${esc(s.asr || "—")}</strong><small>transcription</small></div>
      <div class="card stat"><span class="eyebrow">Cerveau</span><strong>${esc(s.brain_file || s.brain)}</strong><small>${esc(routerStatus(s))}</small></div>
      <div class="card stat"><span class="eyebrow">Voix</span><strong>${esc(s.voices.female)} · ${esc(s.voices.male)}</strong><small>${esc(s.tts || "Phonon")}</small></div>
      <div class="card stat"><span class="eyebrow">Recherche web</span>
        <div class="row">${statePill(search.state === "ready", ["idle", "starting", "off"].includes(search.state), SEARCH_TEXT[search.state] || search.state)}</div>
        <small>${esc(search.message || "")}</small>
        ${["ready", "off", "starting"].includes(search.state) ? "" : `<button class="btn small" id="search-retry">${search.state === "idle" ? "Préparer maintenant" : "Réessayer"}</button>`}
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <div class="card-head"><h2>Conversations</h2><span class="muted mono">en service depuis ${uptime}</span></div>
      ${sessions}
    </div>`;
  if (markup === state.homeMarkup) return;
  const selection = window.getSelection();
  if (s.ready && selection && !selection.isCollapsed && view.contains(selection.anchorNode)) return;
  view.innerHTML = markup;
  state.homeMarkup = markup;
  bindCopy();
  if (hfAccess) bindHfCard();
  const retry = $("#search-retry");
  if (retry) retry.onclick = async () => {
    await api("/api/search/retry", { method: "POST" });
    toast("Préparation de SearXNG…");
  };
}

const PHONON = "https://huggingface.co/gradium/phonon-40m-fr";

function hfCard(kind) {
  return `<div class="card" id="hf-card" style="margin-bottom:14px">
    <ol class="steps">
      <li><strong>Créez un compte gratuit</strong> sur <a href="https://huggingface.co/join" target="_blank" rel="noopener">huggingface.co</a> (ou connectez-vous).</li>
      <li><strong>Acceptez les conditions de la voix</strong> sur <a href="${PHONON}" target="_blank" rel="noopener">huggingface.co/gradium/phonon-40m-fr</a>${kind === "hf_terms" ? " : c'est l'étape qui manque pour votre jeton actuel." : "."}</li>
      <li><strong>Créez un jeton</strong> de type <em>Read</em> dans <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Settings → Access Tokens</a>, puis collez-le ici :
        <div class="copy">
          <input type="password" id="hf-token" placeholder="hf_…" autocomplete="off" spellcheck="false" aria-label="Jeton Hugging Face">
          <button class="btn primary" id="hf-save">Vérifier et démarrer</button>
        </div>
        <p class="muted" id="hf-result" role="status" style="margin-top:8px"></p></li>
    </ol>
  </div>`;
}

function bindHfCard() {
  const result = $("#hf-result");
  const button = $("#hf-save");
  const input = $("#hf-token");
  input.addEventListener("keydown", (event) => event.key === "Enter" && button.click());
  button.onclick = async () => {
    const token = input.value.trim();
    if (!token) return input.focus();
    button.disabled = true;
    result.textContent = "Vérification auprès de Hugging Face…";
    try {
      const check = await api("/api/hf/check", { method: "POST", body: JSON.stringify({ token }) });
      if (check.state !== "ok") {
        result.innerHTML = check.state === "terms"
          ? `Jeton valide, mais les conditions ne sont pas encore acceptées : <a href="${PHONON}" target="_blank" rel="noopener">acceptez-les</a>, puis réessayez.`
          : esc(check.message);
        button.disabled = false;
        return;
      }
      await api("/api/settings", { method: "PUT", body: JSON.stringify({ "huggingface.token": token }) });
      result.textContent = "Jeton enregistré. Redémarrage de Localive : la voix se télécharge (quelques minutes la première fois)…";
      await restartServer();
    } catch (error) {
      result.textContent = error.message;
      button.disabled = false;
    }
  };
}

function bindCopy() {
  view.querySelectorAll("[data-copy]").forEach((button) => {
    button.onclick = async () => {
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(button.dataset.copy);
          return toast("Adresse copiée.");
        } catch { /* A browser can also deny a secure clipboard request. */ }
      }
      // The Clipboard API is usually unavailable over plain HTTP on the LAN.
      const address = button.previousElementSibling;
      address.focus();
      const selection = window.getSelection();
      if (!selection) return toast("Sélectionnez l’adresse puis choisissez Copier.", true);
      const range = document.createRange();
      range.selectNodeContents(address);
      selection.removeAllRanges();
      selection.addRange(range);
      try {
        if (document.execCommand?.("copy")) {
          selection.removeAllRanges();
          button.focus();
          return toast("Adresse copiée.");
        }
      } catch { /* Keep the address selected for manual copying. */ }
      toast("Adresse sélectionnée. Utilisez la commande Copier de votre appareil.");
    };
  });
}

function liveSocketUrl(baseUrl) {
  const url = new URL(baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = url.pathname.replace(/\/v1\/?$/, "/live");
  return url.href;
}

function copyAddress(url) {
  const address = esc(url);
  return `<div class="copy"><code tabindex="-1">${address}</code><button class="btn small" data-copy="${address}" aria-label="Copier ${address}">Copier</button></div>`;
}

async function renderConnect() {
  const s = state.status || (await refreshStatus());
  if (!s) return;
  const c = s.connection;
  view.innerHTML = `
    <div class="page-head">
      <h1>Connecter un client</h1>
    </div>
    <div class="card">
      <h2>Application WebSocket</h2>
      <p class="muted">Ouvrez la connexion et envoyez <code>{"type":"session.start","session":{}}</code>. Après <code>session.started</code>, envoyez le son en PCM16 mono à 16 kHz.</p>
      ${c.urls.map(liveSocketUrl).map(copyAddress).join("")}
      ${c.api_key_required ? '<p class="muted">Envoyez le jeton Localive dans l’en-tête <code>Authorization: Bearer &lt;jeton&gt;</code>.</p>' : ""}
    </div>
    <div class="card">
      <h2>Cornelius</h2>
      <ol class="steps">
        <li><strong>Dans Cornelius, ouvrez Administration → Mise en route → Voix de Cornelius.</strong>
          <p class="muted">Fournisseur de la conversation : <strong>OpenAI</strong>.</p></li>
        <li><strong>Dans « Serveur Live », collez l'adresse de Localive.</strong>
          <p class="muted">Utilisez l’adresse du réseau si Cornelius tourne sur une autre machine. L’adresse de boucle locale désigne la machine du client.</p>
          ${c.urls.map(copyAddress).join("")}</li>
        <li><strong>Clé OpenAI de Cornelius (OPENAI_API_KEY) :</strong>
          <p class="muted">${c.api_key_required
            ? "Cornelius envoie sa clé à Localive comme jeton : elle doit être <strong>identique au jeton d'accès de Localive</strong> (Serveur et sécurité)."
            : "Cornelius en exige une : gardez la vôtre si vous en avez une (ses autres fonctions s'en servent), sinon mettez n'importe quelle valeur. Localive l'ignore, et la conversation ne passe plus par OpenAI."}</p></li>
        <li><strong>Cornelius sur une autre machine ?</strong>
          <p class="muted">Dans <a href="#server">Serveur et sécurité</a>, mettez l'adresse d'écoute à <code>0.0.0.0</code>, définissez un jeton, puis redémarrez Localive.</p></li>
        <li><strong>Lancez une conversation.</strong>
          <p class="muted">Elle apparaît sur l'<a href="#home">accueil</a>. En cas de souci, le <a href="#log">journal</a> dit pourquoi.</p></li>
      </ol>
    </div>`;
  bindCopy();
}

// --------------------------------------------------------------- settings
async function loadSchema() {
  state.schema = await api("/api/settings");
  renderNav();
}

function valueOf(setting) {
  return state.dirty.has(setting.key) ? state.dirty.get(setting.key) : setting.value;
}

const CHOICE_LABELS = { female: "Féminine", male: "Masculine", off: "Désactivée", searxng: "SearXNG", auto: "Automatique", cuda: "NVIDIA (CUDA)", cpu: "Processeur",
  phrases: "Formules prédéfinies", d3: "Décideur local (d3-mini)",
  localive: "Localive (llama.cpp intégré)", ollama: "Ollama", lmstudio: "LM Studio", openai: "Autre serveur compatible OpenAI" };

const choiceLabel = (setting, value) => setting.choice_labels?.[value] ?? CHOICE_LABELS[value] ?? value;

function control(setting) {
  const value = valueOf(setting);
  const id = `s-${setting.key.replace(/\W/g, "-")}`;
  const describedBy = setting.help ? ` aria-describedby="${id}-help"` : "";
  const attrs = `id="${id}" data-key="${esc(setting.key)}" aria-label="${esc(setting.label)}"${describedBy}`;
  if (setting.dynamic === "external_models") {
    const options = [...new Set([value, ...(state.externalModels || [])].filter(Boolean))];
    return `<div class="row"><select ${attrs}>${options.length ? "" : '<option value="">—</option>'}${options.map((m) => `<option value="${esc(m)}" ${m === value ? "selected" : ""}>${esc(m)}</option>`).join("")}</select>
      <button class="btn small" type="button" data-refresh-models>Actualiser</button></div>`;
  }
  if (setting.choices.length) {
    return `<select ${attrs}>${setting.choices.map((c) => `<option value="${esc(c)}" ${c === value ? "selected" : ""}>${esc(choiceLabel(setting, c))}</option>`).join("")}</select>`;
  }
  if (setting.type === "bool") {
    return `<label class="switch"><input type="checkbox" ${attrs} ${value ? "checked" : ""}><span></span></label>`;
  }
  if (setting.type === "int" || setting.type === "float") {
    const step = setting.step ?? (setting.type === "int" ? 1 : 0.05);
    const range = setting.min !== null && setting.max !== null && (setting.max - setting.min) / step <= 400
      ? `<input type="range" min="${setting.min}" max="${setting.max}" step="${step}" value="${value}" data-mirror="${id}" tabindex="-1" aria-hidden="true">`
      : "";
    return `<div class="number">${range}<input type="number" ${attrs} min="${setting.min ?? ""}" max="${setting.max ?? ""}" step="${step}" value="${value}"><span class="unit">${esc(setting.unit)}</span></div>`;
  }
  if (setting.secret) {
    const placeholder = setting.is_set ? "•••••••• (défini)" : "non défini";
    return `<input type="password" ${attrs} value="${state.dirty.has(setting.key) ? esc(value) : ""}" placeholder="${placeholder}" autocomplete="new-password">`;
  }
  return `<input type="text" ${attrs} value="${esc(value)}" spellcheck="false">`;
}

function describeDefault(setting) {
  if (setting.secret) return setting.is_set ? '<button class="linklike" data-clear>Effacer</button>' : "";
  const d = setting.default;
  const shown = typeof d === "boolean" ? (d ? "activé" : "désactivé")
    : d === "" ? "vide"
    : setting.choices.length ? choiceLabel(setting, d)
    : `${fmtNumber(d)}${setting.unit ? ` ${setting.unit}` : ""}`;
  const modified = JSON.stringify(setting.value) !== JSON.stringify(d);
  return `<span>Par défaut : <span class="mono">${esc(shown)}</span></span>${modified ? '<button class="linklike" data-reset>Valeur par défaut</button>' : ""}`;
}

function settingRow(setting, pending) {
  const changed = state.dirty.has(setting.key);
  const id = `s-${setting.key.replace(/\W/g, "-")}`;
  return `<div class="setting ${changed ? "changed" : ""}" data-row="${esc(setting.key)}">
    <div class="setting-text">
      <label class="setting-title" for="${id}">${esc(setting.label)}</label>
      ${setting.help ? `<p class="setting-help" id="${id}-help">${esc(setting.help)}</p>` : ""}
      <div class="setting-meta">${describeDefault(setting)}
        ${setting.restart ? `<span class="tag">${pending.has(setting.key) ? "redémarrage en attente" : "au redémarrage"}</span>` : ""}
        <span class="mono">${esc(setting.key)}</span></div>
    </div>
    <div class="setting-control">${control(setting)}</div>
  </div>`;
}

// A setting that only applies to one choice of another (brain.backend=localive) is hidden
// otherwise, following the value being edited.
function shown(setting, group) {
  const match = /^([\w.]+)(!?=)(.*)$/.exec(setting.only || "");
  if (!match) return true;
  const other = group.settings.find((s) => s.key === match[1]);
  if (!other) return true;
  const equal = String(valueOf(other)) === match[3];
  return match[2] === "=" ? equal : !equal;
}

function renderSection(group) {
  const pending = new Set(state.status?.restart_pending || []);
  const visible = group.settings.filter((s) => shown(s, group));
  const basic = visible.filter((s) => !s.advanced);
  const advanced = visible.filter((s) => s.advanced);
  view.innerHTML = `
    <div class="page-head">
      <span class="eyebrow">Réglages</span>
      <h1>${esc(group.title)}</h1>
      <p class="muted">${esc(group.intro)}</p>
    </div>
    ${advanced.length ? `<div class="toolbar"><span class="muted">${basic.length} réglage${basic.length > 1 ? "s" : ""} essentiel${basic.length > 1 ? "s" : ""}, ${advanced.length} avancé${advanced.length > 1 ? "s" : ""}</span>
      <label class="row"><span class="muted">Réglages avancés</span><span class="switch"><input type="checkbox" id="show-advanced" ${state.advanced ? "checked" : ""}><span></span></span></label></div>` : ""}
    <div class="card">${basic.map((s) => settingRow(s, pending)).join("")}</div>
    ${advanced.length && state.advanced ? `<div class="card"><span class="eyebrow" style="margin-bottom:6px">Avancé</span>${advanced.map((s) => settingRow(s, pending)).join("")}</div>` : ""}
    ${group.id === "brain" && shown({ only: "brain.backend=localive" }, group) ? `<div class="card" id="models-card"><span class="eyebrow">Télécharger d'autres modèles</span>
      <p class="muted">Pour le moteur Localive (llama.cpp). Vérifiés comme à l'installation et repris en cas de coupure. Une fois téléchargé, un modèle apparaît dans « Modèle de raisonnement » ci-dessus : choisissez-le, enregistrez, puis redémarrez Localive. La taille est celle du fichier : il faut un peu plus de mémoire vidéo.</p>
      <div id="models-list"><p class="muted">Chargement…</p></div>
      <div class="setting">
        <div class="setting-text">
          <span class="setting-title">Autre modèle GGUF de Hugging Face</span>
          <p class="setting-help">Le nom d'un dépôt public, par exemple <span class="mono">unsloth/Qwen3.5-4B-GGUF</span>, puis le fichier à télécharger (Q4_K_M : bon compromis).</p>
          <div id="hf-files"></div>
        </div>
        <div class="setting-control"><div class="row"><input type="text" id="hf-repo" placeholder="auteur/nom" spellcheck="false" aria-label="Dépôt Hugging Face"><button class="btn small" id="hf-list">Voir les fichiers</button></div></div>
      </div></div>` : ""}`;
  bindSection(group);
  if (group.id === "brain") {
    renderModels();
    bindExternalModels(group);
  }
}

// ------------------------------------------------- Ollama / LM Studio brain
function bindExternalModels(group) {
  const byKey = Object.fromEntries(group.settings.map((s) => [s.key, s]));
  const row = view.querySelector('[data-row="brain.external_model"]');
  if (!row || !byKey["brain.backend"]) return;
  const help = row.querySelector(".setting-text");
  const status = document.createElement("p");
  status.className = "setting-help";
  status.id = "external-models-status";
  help.append(status);
  const refresh = () => refreshExternalModels(byKey);
  row.querySelector("[data-refresh-models]").onclick = refresh;
  view.querySelector('[data-key="brain.external_url"]')?.addEventListener("change", refresh);
  refresh();
}

async function refreshExternalModels(byKey) {
  const select = view.querySelector('[data-key="brain.external_model"]');
  const status = $("#external-models-status");
  if (!select || !status) return;
  const backend = valueOf(byKey["brain.backend"]);
  select.disabled = backend === "localive";
  if (backend === "localive") {
    status.textContent = "Moteur Localive : le modèle se choisit dans « Modèle de raisonnement ».";
    return;
  }
  status.textContent = "Recherche des modèles…";
  try {
    const result = await api("/api/brain/models", {
      method: "POST",
      body: JSON.stringify({ backend, url: valueOf(byKey["brain.external_url"]) }),
    });
    if (result.error) {
      status.textContent = result.error;
      return;
    }
    state.externalModels = result.models;
    const current = valueOf(byKey["brain.external_model"]);
    const options = [...new Set([current, ...result.models].filter(Boolean))];
    select.innerHTML = options.map((m) => `<option value="${esc(m)}" ${m === current ? "selected" : ""}>${esc(m)}${result.models.includes(m) ? "" : " (absent du serveur)"}</option>`).join("");
    if (!current && result.models.length) {
      select.value = result.models[0];
      select.dispatchEvent(new Event("input"));
    }
    status.textContent = result.models.length
      ? `${result.models.length} modèle${result.models.length > 1 ? "s" : ""} sur ${result.url}.`
      : `Aucun modèle sur ${result.url} : téléchargez-en un dans ce logiciel.`;
  } catch (error) {
    status.textContent = error.message;
  }
}

// ------------------------------------------------------------ model downloads
const gigabytes = (bytes) => (bytes / 1e9).toLocaleString("fr-FR", { maximumFractionDigits: 1 });

function modelAction(m) {
  if (m.state === "downloading" || m.state === "verifying") {
    const percent = m.total ? Math.floor((100 * m.done) / m.total) : 0;
    return m.state === "verifying"
      ? '<span class="muted">Vérification…</span>'
      : `<div class="progress" role="progressbar" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"><span style="width:${percent}%"></span></div>
         <span class="muted mono">${m.total ? `${gigabytes(m.done)} / ${gigabytes(m.total)} Go` : "Connexion…"}</span>`;
  }
  if (m.installed) return `<span class="tag">${m.selected ? "utilisé" : "installé"}</span>`;
  return `<button class="btn small" data-download="${esc(m.id)}">Télécharger</button>`;
}

function modelRow(m) {
  const error = m.state === "error" ? `<p class="setting-help" style="color:var(--destructive)">${esc(m.error)}</p>` : "";
  const size = m.size_gb ? ` <span class="muted mono">${fmtNumber(m.size_gb)} Go</span>` : "";
  return `<div class="setting">
    <div class="setting-text">
      <span class="setting-title">${esc(m.label)}${size}</span>
      <p class="setting-help">${esc(m.note)}</p>${error}
    </div>
    <div class="setting-control model-action">${modelAction(m)}</div>
  </div>`;
}

const openFamilies = new Set();

function familyBlock(family, rows) {
  const busy = rows.some((m) => m.state === "downloading" || m.state === "verifying" || m.state === "error");
  const installed = rows.filter((m) => m.installed).length;
  const open = openFamilies.has(family) || busy;
  return `<details class="family" data-family="${esc(family)}" ${open ? "open" : ""}>
    <summary><span class="setting-title">${esc(family)}</span> <span class="muted">${rows.length} modèle${rows.length > 1 ? "s" : ""}${installed ? `, ${installed} installé${installed > 1 ? "s" : ""}` : ""}</span></summary>
    ${rows.map(modelRow).join("")}
  </details>`;
}

async function startDownload(body) {
  try {
    await api("/api/models/download", { method: "POST", body: JSON.stringify(body) });
  } catch (error) {
    toast(error.message, true);
  }
  renderModels();
}

async function renderModels() {
  const box = $("#models-list");
  if (!box) return;
  bindCustomModel();
  let list;
  try {
    list = (await api("/api/models")).models;
  } catch (error) {
    box.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    return;
  }
  const active = list.some((m) => m.state === "downloading" || m.state === "verifying");
  const families = [...new Set(list.map((m) => m.family))];
  box.innerHTML = families.map((f) => familyBlock(f, list.filter((m) => m.family === f))).join("");
  box.querySelectorAll("details.family").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open) openFamilies.add(details.dataset.family);
      else openFamilies.delete(details.dataset.family);
    });
  });
  box.querySelectorAll("[data-download]").forEach((button) => {
    button.onclick = () => {
      button.disabled = true;
      startDownload({ id: button.dataset.download });
    };
  });
  clearTimeout(renderModels.timer);
  const models = state.schema?.groups.find((g) => g.id === "brain")?.settings
    .find((s) => s.key === "brain.model");
  const finished = list.filter((m) => m.state === "done" && m.installed);
  const hasNewFile = models && finished.some((m) => !models.choices.includes(m.path)
    && !renderModels.finished?.has(m.path));
  renderModels.finished = new Set(finished.map((m) => m.path));
  if (hasNewFile) {
    await loadSchema();
    if (location.hash.slice(1) === "brain") {
      toast("Modèle téléchargé : il est maintenant dans la liste.");
      route();
    }
    return;
  }
  if (active) renderModels.timer = setTimeout(() => (location.hash.slice(1) === "brain") && renderModels(), 1500);
}

function bindCustomModel() {
  const button = $("#hf-list");
  const input = $("#hf-repo");
  if (!button || button.dataset.bound) return;
  button.dataset.bound = "1";
  input.addEventListener("keydown", (event) => event.key === "Enter" && button.click());
  button.onclick = async () => {
    const repo = input.value.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/$/, "");
    const box = $("#hf-files");
    if (!repo) return input.focus();
    button.disabled = true;
    box.innerHTML = '<p class="setting-help">Lecture du dépôt…</p>';
    try {
      const result = await api("/api/models/files", { method: "POST", body: JSON.stringify({ repo }) });
      box.innerHTML = result.files.map((f) => `<div class="row" style="justify-content:space-between;margin-top:6px">
          <span class="mono">${esc(f.file)} <span class="muted">${fmtNumber(f.size_gb)} Go</span></span>
          <button class="btn small" data-repo="${esc(result.repo)}" data-file="${esc(f.file)}">Télécharger</button></div>`).join("");
      box.querySelectorAll("[data-file]").forEach((b) => {
        b.onclick = () => {
          b.disabled = true;
          startDownload({ repo: b.dataset.repo, file: b.dataset.file });
        };
      });
    } catch (error) {
      box.innerHTML = `<p class="setting-help" style="color:var(--destructive)">${esc(error.message)}</p>`;
    }
    button.disabled = false;
  };
}

function bindSection(group) {
  const toggle = $("#show-advanced");
  if (toggle) toggle.onchange = () => {
    state.advanced = toggle.checked;
    localStorage.setItem("localive-advanced", state.advanced ? "1" : "0");
    renderSection(group);
  };
  const byKey = Object.fromEntries(group.settings.map((s) => [s.key, s]));
  view.querySelectorAll("[data-key]").forEach((input) => {
    const setting = byKey[input.dataset.key];
    const update = () => {
      let value;
      if (setting.type === "bool") value = input.checked;
      else if (setting.type === "int") value = Number.parseInt(input.value, 10);
      else if (setting.type === "float") value = Number.parseFloat(input.value);
      else value = input.value;
      if ((setting.type === "int" || setting.type === "float") && Number.isNaN(value)) return;
      if (setting.secret ? value === "" : JSON.stringify(value) === JSON.stringify(setting.value)) state.dirty.delete(setting.key);
      else state.dirty.set(setting.key, value);
      input.closest(".setting").classList.toggle("changed", state.dirty.has(setting.key));
      const range = view.querySelector(`[data-mirror="${input.id}"]`);
      if (range && range.value !== input.value) range.value = input.value;
      renderSavebar();
      // Other settings depend on this one (the brain's engine): show the right ones.
      if (group.settings.some((s) => (s.only || "").startsWith(`${setting.key}=`) || (s.only || "").startsWith(`${setting.key}!=`))) {
        renderSection(group);
      }
    };
    input.addEventListener(setting.type === "bool" || setting.choices.length ? "change" : "input", update);
  });
  view.querySelectorAll("[data-mirror]").forEach((range) => {
    range.addEventListener("input", () => {
      const target = document.getElementById(range.dataset.mirror);
      target.value = range.value;
      target.dispatchEvent(new Event("input"));
    });
  });
  view.querySelectorAll("[data-reset]").forEach((button) => {
    button.onclick = () => resetSetting(button.closest("[data-row]").dataset.row);
  });
  view.querySelectorAll("[data-clear]").forEach((button) => {
    button.onclick = () => {
      const key = button.closest("[data-row]").dataset.row;
      if (confirm("Effacer ce jeton ? Il ne pourra pas être relu.")) {
        state.dirty.set(key, "");
        save();
      }
    };
  });
}

function renderSavebar() {
  const bar = $("#savebar");
  bar.hidden = state.dirty.size === 0;
  $("#savebar-text").textContent = `${state.dirty.size} modification${state.dirty.size > 1 ? "s" : ""} non enregistrée${state.dirty.size > 1 ? "s" : ""}`;
}

async function save() {
  const changes = Object.fromEntries(state.dirty);
  try {
    const result = await api("/api/settings", { method: "PUT", body: JSON.stringify(changes) });
    state.dirty.clear();
    state.schema = result;
    if (state.status) state.status.restart_pending = result.restart_pending;
    renderNav();
    renderBanner();
    renderSavebar();
    route();
    toast(result.restart_required.length
      ? "Enregistré. Certains réglages s'appliqueront au redémarrage."
      : "Enregistré et appliqué.");
  } catch (error) {
    toast(error.message, true);
  }
}

async function resetSetting(key) {
  try {
    const result = await api("/api/settings/reset", { method: "POST", body: JSON.stringify({ keys: [key] }) });
    state.dirty.delete(key);
    state.schema = result;
    if (state.status) state.status.restart_pending = result.restart_pending;
    renderNav();
    renderBanner();
    renderSavebar();
    route();
    toast("Valeur par défaut rétablie.");
  } catch (error) {
    toast(error.message, true);
  }
}

// -------------------------------------------------------------------- log
async function renderLog() {
  view.innerHTML = `
    <div class="page-head">
      <span class="eyebrow">Suivi</span>
      <h1>Journal</h1>
      <p class="muted">Ce que Localive entend, décide et répond, en direct. Les avertissements sont en orange.</p>
    </div>
    <div class="toolbar"><label class="row"><span class="muted">Avertissements seulement</span><span class="switch"><input type="checkbox" id="only-warnings"><span></span></span></label>
      <button class="btn small" id="log-clear">Effacer l'affichage</button></div>
    <pre class="log" id="log" aria-live="off"></pre>`;
  state.logIndex = 0;
  const box = $("#log");
  const onlyWarnings = $("#only-warnings");
  $("#log-clear").onclick = () => (box.textContent = "");
  onlyWarnings.onchange = () => {
    box.textContent = "";
    state.logIndex = 0;
    poll();
  };
  const poll = async () => {
    let lines;
    try {
      lines = (await api(`/api/logs?after=${state.logIndex}`)).lines;
    } catch {
      return;
    }
    const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
    for (const line of lines) {
      state.logIndex = line.n;
      if (onlyWarnings.checked && line.level === "INFO") continue;
      const span = document.createElement("span");
      span.className = line.level;
      span.textContent = `${line.text}\n`;
      box.append(span);
    }
    if (stick) box.scrollTop = box.scrollHeight;
  };
  await poll();
  state.timers.push(setInterval(poll, 1500));
}

// ------------------------------------------------------------------- boot
$("#save").onclick = save;
$("#discard").onclick = () => {
  state.dirty.clear();
  renderSavebar();
  route();
};
window.addEventListener("hashchange", (event) => {
  if (state.dirty.size && !confirm("Abandonner les modifications non enregistrées ?")) {
    history.replaceState(history.state, "", event.oldURL);
    return;
  }
  closeNavigation();
  state.dirty.clear();
  renderSavebar();
  route();
});
window.addEventListener("beforeunload", (event) => {
  if (state.dirty.size) event.preventDefault();
});

(async () => {
  await refreshStatus();
  try {
    await loadSchema();
  } catch (error) {
    toast(error.message, true);
  }
  renderNav();
  route();
})();
