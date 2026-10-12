/* Google SWE II OS - the desktop window. Plain JavaScript, no libraries, talks only to 127.0.0.1.
   Every screen answers Krug's three questions: where am I (sidebar + breadcrumbs), what is this
   (one title), what do I do next (one primary button, always bound to Enter). */
"use strict";

const params = new URLSearchParams(location.search);
const TOKEN = params.get("t") || safeGet("prep-token") || "";
safeSet("prep-token", TOKEN);
const DEBUG = params.get("debug") === "1";
const COLORS = { blue: "#4285F4", green: "#34A853", yellow: "#FBBC05", red: "#EA4335" };
const FOLLOW_UP_AT = 35 / 45;
const RUBRIC_AT = 40 / 45;

function safeGet(key) { try { return sessionStorage.getItem(key); } catch (_) { return null; } }
function safeSet(key, value) { try { if (value) sessionStorage.setItem(key, value); } catch (_) { /* private mode */ } }

// ---------------------------------------------------------------- DOM helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const SVG_NS = "http://www.w3.org/2000/svg";

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  addKids(el, kids);
  return el;
}
function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  addKids(el, kids);
  return el;
}
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = Boolean(v);
    else if (k === "vars") for (const [name, val] of Object.entries(v)) el.style.setProperty(name, val);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}
function addKids(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}
const kbd = (key) => h("kbd", null, key);
const btn = (label, onclick, opts = {}) => h("button", {
  type: "button", class: `btn ${opts.cls || ""}`.trim(), onclick, "data-primary": opts.primary || null,
  disabled: opts.disabled || null, "data-test": opts.test || null, title: opts.title || null,
}, label, opts.key ? kbd(opts.key) : null);

/** `**bold**` and `code` inline markup from the shared lesson/today text, built as safe nodes. */
function rich(text) {
  const out = [];
  const re = /\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  let m;
  text = String(text ?? "");
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(m[1] != null ? h("b", null, rich(m[1])) : h("code", { class: "inline" }, m[2]));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Small syntax colouring for lessons and notes (the whiteboard stays plain, like a real interview).
const KEYWORDS = {
  python: new Set(("False None True and as assert async await break class continue def del elif else except finally "
    + "for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" ")),
  cpp: new Set(("auto bool break case catch char class const constexpr continue default delete do double else enum "
    + "explicit false float for friend if inline int long mutable namespace new noexcept nullptr operator private "
    + "protected public return short signed sizeof static struct switch template this throw true try typedef "
    + "typename unsigned using virtual void volatile while std size_t").split(" ")),
};
function highlight(code, lang) {
  const isCpp = lang === "cpp" || lang === "c++";
  const kw = isCpp ? KEYWORDS.cpp : KEYWORDS.python;
  const comment = isCpp ? String.raw`//[^\n]*|/\*[\s\S]*?\*/|#[ \t]*[a-z]+[^\n]*` : String.raw`#[^\n]*`;
  const re = new RegExp([
    `(${comment})`,
    String.raw`("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')`,
    String.raw`(\b0[xXbBoO][0-9a-fA-F_]+\b|\b\d[\d_']*(?:\.\d+)?(?:[eE][+-]?\d+)?[a-zA-Z]*\b)`,
    String.raw`([A-Za-z_]\w*)(?=\s*\()`,
    String.raw`([A-Za-z_]\w*)`,
  ].join("|"), "g");
  const frag = document.createDocumentFragment();
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    if (m.index > last) frag.append(code.slice(last, m.index));
    let cls = null;
    if (m[1]) cls = "tok-com";
    else if (m[2]) cls = "tok-str";
    else if (m[3]) cls = "tok-num";
    else if (m[4]) cls = kw.has(m[4]) ? "tok-kw" : "tok-fn";
    else if (kw.has(m[5])) cls = "tok-kw";
    frag.append(cls ? h("span", { class: cls }, m[0]) : m[0]);
    last = re.lastIndex;
  }
  frag.append(code.slice(last));
  return frag;
}
const codeBlock = (code, lang) => h("pre", { class: "code" }, h("code", null, highlight(code, lang)));

function ring(value, { size = 120, stroke = 10, color = COLORS.blue, label = "", sub = "" } = {}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  const mid = size / 2;
  return svg("svg", { class: "ring", viewBox: `0 0 ${size} ${size}`, role: "img", "aria-label": `${label} ${sub}`.trim() },
    svg("circle", { class: "track", cx: mid, cy: mid, r, fill: "none", "stroke-width": stroke }),
    svg("circle", {
      class: "fill", cx: mid, cy: mid, r, fill: "none", stroke: color, "stroke-width": stroke, "stroke-linecap": "round",
      "stroke-dasharray": c.toFixed(2), "stroke-dashoffset": (c * (1 - v)).toFixed(2), transform: `rotate(-90 ${mid} ${mid})`,
    }),
    svg("text", { x: "50%", y: sub ? "46%" : "52%", "text-anchor": "middle", "dominant-baseline": "middle", "font-size": (size * 0.24).toFixed(1) }, label),
    sub ? svg("text", { class: "sub", x: "50%", y: "68%", "text-anchor": "middle", "dominant-baseline": "middle", "font-size": (size * 0.105).toFixed(1) }, sub) : null);
}
function bar(frac, tone = "") {
  const pct = Math.round(Math.max(0, Math.min(1, Number(frac) || 0)) * 100);
  return h("div", { class: `bar ${tone}`.trim(), role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(pct) },
    h("i", { vars: { "--w": `${pct}%` } }));
}
const toneFor = (frac) => (frac >= 0.7 ? "green" : frac >= 0.4 ? "yellow" : "red");
const mmss = (secs) => {
  const s = Math.max(0, Math.floor(secs));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const fmtDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });

// ---------------------------------------------------------------- server calls
class ApiError extends Error {}
async function api(name, payload = {}, opts = {}) {
  let res;
  try {
    res = await fetch(`/api/${name}`, {
      method: "POST", keepalive: Boolean(opts.keepalive),
      headers: { "Content-Type": "application/json", "X-Prep-Token": TOKEN },
      body: JSON.stringify(payload),
    });
  } catch (_) {
    throw new ApiError("The app stopped. Start Google SWE II OS again: your work is saved.");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (res.status === 403) throw new ApiError("This window lost its key. Close it and start the app again.");
  if (!res.ok || (data && data.error)) throw new ApiError((data && data.error) || `Request failed (${res.status})`);
  return data;
}

// ---------------------------------------------------------------- frame: nav, crumbs, meta
const ICONS = {
  today: "M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  fasttrack: "M13 2L4 14h7l-1 8 9-12h-7z",
  recall: "M4 7h12v13H4z M8 4h12v13",
  mock: "M12 21a8 8 0 1 0 0-16 8 8 0 0 0 0 16z M12 9v4l3 2 M9 2h6",
  roadmap: "M3 20h18 M6 16v-5 M11 16V6 M16 16v-8 M21 16V11",
  star: "M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z",
  vault: "M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z M5 17a3 3 0 0 1 3-3h10",
};
const SECTIONS = [
  { name: "today", label: "Today", key: "t" },
  { name: "fasttrack", label: "Fast-Track", key: "f" },
  { name: "recall", label: "Recall Drills", key: "d" },
  { name: "mock", label: "Mock & Solve", key: "w" },
  { name: "roadmap", label: "Roadmap", key: "r" },
  { name: "star", label: "STAR Stories", key: "s" },
  { name: "vault", label: "Concept Vault", key: "v" },
];
const NAV_KEYS = Object.fromEntries(SECTIONS.map((x) => [x.key, x.name]));
const icon = (name) => svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.8", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" },
  svg("path", { d: ICONS[name] }));

const S = { state: null, boardStart: null };

function buildNav() {
  $("#nav").replaceChildren(...SECTIONS.map((x) => h("button", {
    type: "button", "data-section": x.name, onclick: () => go(`#/${x.name}`),
  }, icon(x.name), h("span", null, x.label), h("span", { class: "nav-end" }, h("span", { class: "badge", hidden: true }), kbd(x.key)))));
  $("#settings-btn").addEventListener("click", openSettings);
  $("#help-btn").addEventListener("click", openHelp);
}
function markNav(name) {
  const section = name === "board" ? "mock" : name;
  for (const b of $$("#nav button")) {
    if (b.dataset.section === section) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
}
function setCrumbs(parts, ctx) {
  if (ctx && !ctx.alive()) return; // a slow, abandoned page must not relabel the one you moved to
  const nodes = [];
  (parts || []).forEach((p, i) => {
    if (i) nodes.push(h("span", { class: "sep", "aria-hidden": "true" }, "›"));
    const cls = i === 0 ? "app-name" : i === parts.length - 1 ? "here" : "";
    nodes.push(h("span", { class: cls || null, "aria-current": i === parts.length - 1 ? "location" : null }, p));
  });
  $("#crumbs").replaceChildren(...nodes);
  document.title = parts && parts.length > 1 ? `${parts[parts.length - 1]} · Google SWE II OS` : "Google SWE II OS";
}
async function refreshState() {
  const st = await api("state");
  S.state = st;
  const ready = st.readiness / 100;
  $("#mini").replaceChildren(
    ring(ready, { size: 54, stroke: 6, label: `${Math.round(st.readiness)}`, color: COLORS.blue }),
    h("div", { class: "mini-lines" },
      h("b", null, `Day ${st.day} of ${st.plan_days}`), h("br"),
      `Phase ${st.phase.number} · ${st.phase.short}`, h("br"),
      `Readiness ${Math.round(st.readiness)}%`));
  $("#brand-sub").textContent = st.settings.target || "Interview prep";
  const pills = [h("span", { class: "pill", "data-test": "streak" }, `Streak ${plural(st.streak, "day")}`)];
  pills.push(h("span", { class: `pill ${st.days_left <= 14 ? "warn" : ""}`.trim() },
    st.days_left >= 0 ? `${plural(st.days_left, "day")} to interview` : "Interview date passed"));
  $("#meta").replaceChildren(...pills);
  const ftBadge = $('#nav button[data-section="fasttrack"] .badge');
  const pending = st.fasttrack.pending;
  ftBadge.hidden = !pending;
  ftBadge.textContent = pending ? `Day ${pending}` : "";
  return st;
}

// ---------------------------------------------------------------- toasts and modals
function toast(message, bad = false) {
  const el = h("div", { class: bad ? "bad" : null, role: bad ? "alert" : null }, message);
  $("#toast").append(el);
  setTimeout(() => el.remove(), bad ? 7000 : 3500);
}
let modalState = null;
/** Opens a dialog; resolves with the clicked button's value, or null on Esc / backdrop. */
function openModal(title, body, buttons) {
  if (modalState) modalState.close(null);
  return new Promise((resolve) => {
    const prev = document.activeElement;
    const close = (value) => {
      $("#modal-root").replaceChildren();
      $(".app").inert = false;
      modalState = null;
      if (prev && prev.focus && document.contains(prev)) prev.focus();
      resolve(value);
    };
    const nodes = buttons.map((b) => h("button", {
      type: "button", class: `btn ${b.primary ? "primary" : ""} ${b.danger ? "danger" : ""}`.trim(),
      onclick: () => close(b.value), "data-test": b.test || null,
    }, b.label, b.key ? kbd(b.key) : null));
    const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("h2", null, title), body, h("div", { class: "row" }, h("div", { class: "spacer" }), nodes));
    const bg = h("div", { class: "modal-bg", onclick: (e) => { if (e.target === bg) close(null); } }, box);
    $("#modal-root").replaceChildren(bg);
    $(".app").inert = true; // focus and clicks cannot reach the page behind the dialog
    modalState = { close, buttons, box };
    const focus = nodes[buttons.findIndex((b) => b.primary)] || nodes[0];
    if (focus) focus.focus();
  });
}
const confirmBox = (title, text, yes, no = "Cancel", danger = false) => openModal(title, h("p", { class: "muted" }, text), danger
  ? [{ label: no, value: false, key: "Enter" }, { label: yes, value: true, danger: true }] // safe choice is the default
  : [{ label: no, value: false, key: "Esc" }, { label: yes, value: true, primary: true, key: "Enter" }]);

/** recall_start / mock_start answer {conflict} when another session is unfinished: ask, never overwrite it. */
function askAboutUnfinished(conflict) {
  return openModal("You have an unfinished session", h("div", null,
    h("p", null, conflict.text.replace(/^Session restored: /, "")),
    h("p", { class: "muted" }, "Starting something new discards it. Drills you already graded still count for the day.")), [
    { label: "Cancel", value: null },
    { label: "Discard it and start", value: "discard", danger: true, test: "conflict-discard" },
    { label: "Continue it", value: "continue", primary: true, key: "Enter", test: "conflict-continue" },
  ]);
}

function openHelp() {
  const rows = [
    ["Enter", "Do the highlighted main action"], ["t f d w r s v", "Today · Fast-Track · Drills · Mock · Roadmap · STAR · Vault"],
    ["m", "Micro-Mode (on Today)"], ["Space", "Reveal a drill answer"], ["1 2 3", "Missed · Effort · Instant (also answers lesson exercises)"],
    ["y / n", "Yes / No on the scorecard"], ["Ctrl+S", "Save now (whiteboard) · save a STAR story"], ["Ctrl+Enter", "Submit the whiteboard and grade it"],
    ["Alt+1…4", "Tick the live checklist while coding"], ["Tab", "Indent 4 spaces in the code pad"],
    ["Esc", "Leave the code pad · close a dialog · back to Today"], ["?", "This list"],
  ];
  openModal("Keyboard", h("div", { class: "keys-help" }, rows.map(([k, t]) => [h("span", null, k.split(" ").map((x) => [kbd(x), " "])), h("span", null, t)])),
    [{ label: "Close", value: null, primary: true, key: "Esc" }]);
}

async function openSettings() {
  const st = S.state || await refreshState();
  const field = (label, name, value, type = "text") => h("div", { class: "field" },
    h("label", { for: `set-${name}` }, label), h("input", { id: `set-${name}`, name, type, value: value || "" }));
  const form = h("form", { class: "settings-form", onsubmit: (e) => e.preventDefault() },
    field("Your name", "name", st.settings.name), field("Target role", "target", st.settings.target),
    h("div", { class: "grid-2" }, field("Plan start (Day 1)", "start_date", st.settings.start_date, "date"),
      field("Interview date", "interview_date", st.settings.interview_date, "date")),
    h("p", { class: "muted small" }, "Day 1–2 are the Python fast-track; the Google problem bank starts on Day 3."),
    h("p", { class: "muted small", "data-test": "data-dir" }, "Your progress is saved automatically in ", h("code", { class: "inline" }, st.data_dir),
      ". Updating or uninstalling the app never deletes it."));
  const ok = await openModal("Settings", form, [{ label: "Cancel", value: false, key: "Esc" }, { label: "Save", value: true, primary: true, test: "settings-save" }]);
  if (!ok) return;
  const data = Object.fromEntries(new FormData(form).entries());
  try {
    await api("settings_save", data);
    toast("Settings saved.");
    await refreshState();
    if (!["board", "recall", "star"].includes(S.routeName)) route(); // never restart in-progress work
  } catch (err) { toast(err.message, true); }
}

// ---------------------------------------------------------------- router
const ROUTES = {};
let current = null;
let navSeq = 0;

async function go(hash, opts = {}) {
  if (current && current.beforeLeave && !opts.force && !(await current.beforeLeave())) return;
  if (current && current.beforeLeave) current.beforeLeave = null;
  if (location.hash === hash) route();
  else if (opts.replace) { history.replaceState(null, "", hash); route(); }
  else location.hash = hash;
}

async function route() {
  const seq = ++navSeq;
  if (current && current.cleanup) { try { current.cleanup(); } catch (_) { /* keep going */ } }
  current = null;
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const name = ROUTES[parts[0]] ? parts[0] : "today";
  S.lastHash = S.currentHash;
  S.currentHash = location.hash || "#/today";
  S.routeName = name;
  markNav(name);
  const view = $("#view");
  view.classList.toggle("board-mode", name === "board");
  const ctx = { parts, query: new URLSearchParams(qs || ""), alive: () => seq === navSeq };
  try {
    const handle = await ROUTES[name](ctx);
    if (seq !== navSeq) { if (handle && handle.cleanup) handle.cleanup(); return; }
    current = handle || {};
    const active = document.activeElement;
    if (!active || active === document.body || active.closest(".sidebar")) view.focus({ preventScroll: true });
  } catch (err) {
    if (seq !== navSeq) return;
    mount(ctx, h("div", { class: "page" }, h("div", { class: "empty" }, h("p", null, err.message || String(err)),
      btn("Back to Today", () => go("#/today", { force: true }), { cls: "primary", primary: true, key: "Enter" }))));
    current = {};
  }
}
/** Leave a page whose start was cancelled: back where you came from. */
function goBack() {
  const back = S.lastHash && S.lastHash !== S.currentHash && !/^#\/(recall|board)/.test(S.lastHash) ? S.lastHash : "#/today";
  return go(back, { replace: true, force: true });
}
function mount(ctx, ...nodes) {
  if (!ctx.alive()) return false;
  const view = $("#view");
  view.replaceChildren(...nodes);
  view.scrollTop = 0;
  return true;
}
const pageTitle = (title, sub, ...right) => h("div", { class: "page-title" },
  h("div", null, h("h1", null, title), sub ? h("p", null, sub) : null), right.length ? h("div", { class: "row" }, right) : null);

// ---------------------------------------------------------------- Today
ROUTES.today = async (ctx) => {
  const mode = ctx.query.get("mode") || "normal";
  const [d, st] = await Promise.all([api("today", { mode }), refreshState()]);
  setCrumbs(d.crumbs, ctx);
  const tone = { green: "green", yellow: "yellow" }[d.tone] || "";
  const unfinished = Boolean(st.session); // finishing it comes first, like the terminal app's launch
  const kicker = { fasttrack: "Phase 0 · C++ → Python", warmup: "Next", deep: "Next", micro: "Never zero", review: "Re-solve", done: "Today" }[d.kind] || "Next";
  const actions = [];
  if (d.action) {
    actions.push(btn(d.cta, () => runAction(d), unfinished ? { cls: "big", test: "primary" }
      : { cls: "primary big", primary: true, key: "Enter", test: "primary" }));
  }
  if (mode === "micro") actions.push(btn("Back to the full plan", () => go("#/today?mode=full")));
  else if (d.kind !== "fasttrack") actions.push(btn("Long day? Micro-Mode · 10 min", () => go("#/today?mode=micro"), { key: "m" }));
  const hero = h("section", { class: `hero ${tone}`.trim(), "data-test": "next-card" },
    h("div", { class: "kicker" }, kicker), h("h2", null, d.title),
    h("ul", { class: "bullets" }, d.bullets.map((b) => h("li", null, rich(b)))),
    actions.length ? h("div", { class: "actions" }, actions) : null);

  const resume = st.session ? h("div", { class: "banner resume", "data-test": "resume" },
    h("b", null, "Unfinished"), h("span", { class: "text" }, st.session.text.replace(/^Session restored: /, "")),
    btn("Continue", () => resumeSession(st.session), { cls: "primary", primary: true, key: "Enter", test: "resume-continue" }),
    btn("Discard", async () => {
      const ok = await confirmBox("Discard the unfinished session?", `${st.session.text.replace(/^Session restored: /, "")} It cannot be resumed afterwards. Drills you already graded still count.`, "Discard", "Keep it", true);
      if (!ok) return;
      await api("session_discard");
      toast("Discarded.");
      route();
    }, { cls: "ghost" })) : null;

  const statusRows = d.status.length ? h("div", { class: "status-rows" }, d.status.map((row) => {
    const cls = { Done: "ok-t", Open: "warn-t" }[row.tag] || "muted";
    const open = row.tag === "Open" ? btn("Open", () => go("#/fasttrack"), { cls: "ghost small" }) : null;
    return h("div", { class: "status-row" }, h("span", { class: `tag ${cls}` }, row.tag), h("span", { class: "spacer" }, rich(row.text)), open);
  })) : null;

  const pace = st.pace;
  let paceSub;
  if (st.day < 3) paceSub = "Problem bank starts on Day 3";
  else if (pace.solved >= pace.total) paceSub = "Bank done";
  else if (pace.behind) paceSub = `Behind by ${pace.behind} · need ${pace.per_day.toFixed(1)}/day`;
  else paceSub = "On track";
  const change = st.weekly_change;
  const tiles = h("div", { class: "grid-4" },
    tile(ring(st.readiness / 100, { size: 58, stroke: 7, label: `${Math.round(st.readiness)}` }), "Readiness", `${Math.round(st.readiness)}%`,
      change == null ? "Weekly change after 7 days" : `${change >= 0 ? "+" : ""}${change.toFixed(0)} this week`),
    tile(null, "Streak", plural(st.streak, "day"), "Micro-Mode keeps it alive"),
    tile(null, "Interview", st.days_left >= 0 ? plural(st.days_left, "day") : "Passed", fmtDate(st.interview_date)),
    tile(ring(pace.total ? pace.solved / pace.total : 0, { size: 58, stroke: 7, color: COLORS.green, label: `${pace.solved}` }),
      "Problems solved", `${pace.solved}/${pace.total}`, paceSub));
  mount(ctx, h("div", { class: "page" }, pageTitle(mode === "micro" ? "Micro-Mode" : "Today",
    `${new Date(st.now).toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })} · Day ${st.day} of ${st.plan_days} · Phase ${st.phase.number}: ${st.phase.name}`),
  resume, hero, statusRows, tiles));
  return {
    onKey(e, typing) {
      if (!typing && !e.ctrlKey && !e.altKey && !e.metaKey && e.key === "m" && mode !== "micro" && d.kind !== "fasttrack") { go("#/today?mode=micro"); return true; }
      return false;
    },
  };
};
function tile(visual, label, value, sub) {
  return h("div", { class: "tile" }, visual, h("div", null, h("div", { class: "label" }, label), h("div", { class: "value" }, value), h("div", { class: "sub" }, sub)));
}
function runAction(d) {
  if (d.action === "warmup") return go("#/recall?mode=warmup");
  if (d.action === "micro_start") return go("#/recall?mode=micro");
  if (d.action === "fasttrack") return go(`#/fasttrack/${d.day || ""}`);
  if (d.action === "deep" || d.action === "review_start") {
    S.boardStart = { problem_id: d.problem_id, practice: d.action === "review_start" ? true : Boolean(d.practice) };
    return go("#/board");
  }
  return null;
}
function resumeSession(sess) {
  if (sess.kind === "recall") return go(`#/recall?mode=${encodeURIComponent(sess.mode || "warmup")}&resume=1`);
  S.boardStart = { resume: true };
  return go("#/board");
}

// ---------------------------------------------------------------- Phase 0: Python fast-track
ROUTES.fasttrack = async (ctx) => {
  const data = await api("fasttrack");
  const dayNo = Number(ctx.parts[1]) || data.current;
  const lesson = data.days.find((x) => x.day === dayNo) || data.days[0];
  setCrumbs([...data.crumbs, `Day ${lesson.day}: ${lesson.title}`], ctx);
  const started = performance.now();
  const results = Object.fromEntries(lesson.exercises.map((e) => [e.id, e.result]));
  const fresh = new Set(); // exercises answered during this visit
  const exNodes = {};

  const answeredCount = () => Object.values(results).filter((v) => v !== null && v !== undefined).length;
  const rightCount = () => Object.values(results).filter((v) => v === true).length;

  const daytabs = h("div", { class: "daytabs", role: "tablist" }, data.days.map((x) => {
    const p = x.progress;
    const status = p.done ? `Done · ${p.correct}/${p.total} right` : p.answered ? `${p.answered}/${p.total} answered` : `${x.minutes} min · not started`;
    return h("button", { type: "button", role: "tab", "aria-selected": String(x.day === lesson.day), onclick: () => go(`#/fasttrack/${x.day}`) },
      h("span", { class: "t" }, `Day ${x.day} · ${x.title}`), h("span", { class: "s" }, status));
  }));

  const sectionCards = lesson.sections.map((sec, i) => h("section", { class: "card lesson", id: `sec-${sec.id}` },
    h("div", { class: "kicker" }, `Section ${i + 1} of ${lesson.sections.length}`),
    h("h3", null, sec.title),
    h("ul", { class: "bullets" }, sec.points.map((pt) => h("li", null, rich(pt)))),
    h("div", { class: "pair" },
      h("div", { class: "pane cpp" }, h("div", { class: "head" }, h("span", null, "C++ (what you know)")), codeBlock(sec.cpp, "cpp")),
      h("div", { class: "pane py" }, h("div", { class: "head" }, h("span", null, "Python (what to write)")), codeBlock(sec.python, "python"))),
    sec.gotcha ? h("div", { class: "gotcha" }, h("b", null, "Gotcha"), h("div", null, rich(sec.gotcha))) : null));

  const exCards = lesson.exercises.map((ex, i) => {
    const feedback = h("div", { class: "feedback-slot" });
    const opts = ex.options.map((opt, j) => h("button", {
      type: "button", class: "option", "data-choice": String(j), onclick: () => answer(ex, j),
    }, kbd(String(j + 1)), h("span", null, opt)));
    const before = ex.result == null ? null : h("p", { class: `small ${ex.result ? "ok-t" : "warn-t"}` },
      ex.result ? "You got this right before. Again for speed?" : "Missed last time. Try again.");
    const node = h("div", { class: "exercise", id: `ex-${ex.id}`, "data-test": "exercise" },
      h("div", { class: "row" }, h("span", { class: "chip blue" }, ex.topic), h("span", { class: "muted small" }, `Exercise ${i + 1} of ${lesson.exercises.length}`)),
      h("p", { class: "ex-prompt" }, rich(ex.prompt)), codeBlock(ex.code, "python"), before,
      h("div", { class: "options" }, opts), feedback);
    exNodes[ex.id] = { node, opts, feedback };
    return node;
  });

  const finishSlot = h("div", null);
  const tocExercises = h("div", { class: "toc-ex" });
  const counter = h("span", { class: "chip" });

  function renderProgress() {
    const answered = answeredCount();
    counter.textContent = `${answered}/${lesson.exercises.length} answered · ${rightCount()} right`;
    tocExercises.replaceChildren(...lesson.exercises.map((ex, i) => {
      const r = results[ex.id];
      return h("button", { type: "button", onclick: () => exNodes[ex.id].node.scrollIntoView({ behavior: "smooth", block: "center" }) },
        h("span", { class: `n ${r === true ? "ok" : r === false ? "bad" : ""}`.trim() }, String(i + 1)), h("span", null, ex.topic));
    }));
    const all = answered === lesson.exercises.length;
    const done = lesson.progress.done;
    finishSlot.replaceChildren(h("section", { class: `card finish ${done ? "done" : ""}`.trim() },
      h("h3", null, done ? `Day ${lesson.day} is done` : `Finish Day ${lesson.day}`),
      h("p", { class: "muted" }, all ? `${rightCount()} of ${lesson.exercises.length} right. Finishing logs today as a full study day (streak safe).`
        : `Answer the remaining ${lesson.exercises.length - answered} exercise${lesson.exercises.length - answered === 1 ? "" : "s"} first. Use 1, 2, 3.`),
      h("div", { class: "row" },
        done ? null : btn(`Finish Day ${lesson.day}`, finish, { cls: "primary", primary: all, disabled: !all, key: all ? "Enter" : null, test: "finish" }),
        btn("Open the full note in the Vault", () => go(`#/vault/${encodeURIComponent(lesson.vault)}`), { cls: "ghost" }))));
  }

  async function answer(ex, choice) {
    const slot = exNodes[ex.id];
    if (fresh.has(ex.id)) return;
    let res;
    try { res = await api("fasttrack_answer", { exercise_id: ex.id, choice }); } catch (err) { toast(err.message, true); return; }
    fresh.add(ex.id);
    results[ex.id] = res.correct;
    slot.opts.forEach((b, j) => {
      b.disabled = true;
      if (j === res.answer) b.classList.add("right");
      else if (j === choice) b.classList.add("wrong");
    });
    slot.feedback.replaceChildren(h("div", { class: `feedback ${res.correct ? "right" : "wrong"}`, "data-test": "feedback" },
      h("p", null, h("b", { class: res.correct ? "ok-t" : "bad-t" }, res.correct ? "Correct." : `Not quite. Answer: ${ex.options[res.answer]}`)),
      h("p", null, rich(res.explanation)),
      res.output ? [h("div", { class: "muted small" }, "Python actually prints:"), h("pre", { class: "code out" }, res.output)] : null));
    renderProgress();
    const next = lesson.exercises.find((x) => !fresh.has(x.id));
    if (next) setTimeout(() => exNodes[next.id].node.scrollIntoView({ behavior: "smooth", block: "center" }), 900);
    else setTimeout(() => finishSlot.scrollIntoView({ behavior: "smooth", block: "center" }), 900);
  }

  async function finish() {
    const minutes = Math.max(1, Math.round((performance.now() - started) / 60000));
    try {
      const res = await api("fasttrack_complete", { day: lesson.day, minutes });
      lesson.progress.done = true;
      await refreshState();
      const next = data.days.find((x) => x.day > lesson.day && !x.progress.done);
      openModal(`Day ${lesson.day} done`, h("div", null,
        h("p", null, `${rightCount()} of ${lesson.exercises.length} right · streak ${plural(res.streak, "day")}.`),
        h("p", { class: "muted" }, next ? `Day ${next.day} (${next.title}) is next. It opens on Today tomorrow, or start it now.` : "The Google problem bank starts on Day 3. Today will show your first warmup.")),
      [{ label: "Back to Today", value: "today", primary: true, key: "Enter" }, next ? { label: `Start Day ${next.day} now`, value: "next" } : null].filter(Boolean))
        .then((v) => { if (v === "next") go(`#/fasttrack/${next.day}`); else go("#/today"); });
    } catch (err) { toast(err.message, true); }
  }

  const toc = h("nav", { class: "toc", "aria-label": "Lesson contents" },
    h("h4", null, "Sections"),
    lesson.sections.map((sec, i) => h("button", { type: "button", onclick: () => $(`#sec-${sec.id}`).scrollIntoView({ behavior: "smooth" }) },
      h("span", { class: "n" }, String(i + 1)), h("span", null, sec.title))),
    h("h4", null, "Exercises"), tocExercises);

  renderProgress();
  mount(ctx, h("div", { class: "page wide" },
    pageTitle(`Day ${lesson.day} · ${lesson.title}`, lesson.subtitle, counter),
    daytabs,
    h("div", { class: "ft" }, toc, h("div", { class: "lesson-col" }, sectionCards,
      h("section", { class: "card", id: "exercises" }, h("h2", null, "Predict the output"),
        h("p", { class: "muted" }, "Read the code, pick what Python prints. Each answer shows the real output from running it."),
        h("div", { class: "ex-list" }, exCards)),
      finishSlot))));
  return {
    onKey(e, typing) {
      if (typing || e.ctrlKey || e.altKey || e.metaKey || !["1", "2", "3"].includes(e.key)) return false;
      const ex = lesson.exercises.find((x) => !fresh.has(x.id));
      if (!ex) return false;
      exNodes[ex.id].node.scrollIntoView({ behavior: "smooth", block: "center" });
      answer(ex, Number(e.key) - 1);
      return true;
    },
  };
};

// ---------------------------------------------------------------- Recall drills
ROUTES.recall = async (ctx) => {
  const mode = ctx.query.get("mode") || "warmup";
  const request = { mode, count: 5, resume: ctx.query.get("resume") === "1" };
  let data = await api("recall_start", request);
  if (data.conflict) {
    const choice = await askAboutUnfinished(data.conflict);
    if (!ctx.alive()) return {};
    if (choice === "continue") { resumeSession(data.conflict); return {}; }
    if (choice !== "discard") { goBack(); return {}; }
    data = await api("recall_start", { ...request, discard: true });
  }
  setCrumbs(data.crumbs, ctx);
  const title = data.mode === "micro" ? "Micro-Mode drills" : "Morning warmup";
  if (!data.drills.length) {
    mount(ctx, h("div", { class: "page" }, pageTitle(title),
      h("div", { class: "empty" }, h("p", null, "No drills are due right now."), btn("Back to Today", () => go("#/today"), { cls: "primary", primary: true, key: "Enter" }))));
    return {};
  }
  let index = data.index;
  const counts = { ...data.counts };
  const carried = Number(data.seconds) || 0;
  const sessionStart = performance.now();
  let timer = null;
  let phase = "think";
  let drillStart = 0;
  let revealSeconds = 0;
  let timedOut = false;
  let busy = false;

  const stop = () => { if (timer) clearInterval(timer); timer = null; };

  function showDrill() {
    stop();
    phase = "think";
    timedOut = false;
    const drill = data.drills[index];
    const label = drill.type === "bug_hunt" ? "Bug hunt" : "Pattern drill";
    const dots = h("div", { class: "dots", "aria-hidden": "true" }, data.drills.map((_, i) => h("i", { class: i < index ? "done" : i === index ? "now" : null })));
    const [lead, ...rest] = drill.prompt.split("\n\n");
    const promptNodes = drill.prompt.includes("\n")
      ? [rest.length ? h("p", { class: "prompt-text" }, lead) : null, h("pre", { class: "prompt-code" }, rest.length ? rest.join("\n\n") : drill.prompt)]
      : [h("p", { class: "prompt-text" }, drill.prompt)];
    const ringSlot = h("div", { class: "ring-slot" });
    const work = h("div", { class: "work" },
      h("div", { class: "think" }, h("div", null, "1. Which pattern?"), h("div", null, "2. Time and space?"), h("div", null, "3. One edge case?")),
      h("div", { class: "actions" }, btn("Reveal answer", reveal, { cls: "primary big", primary: true, key: "Space", test: "reveal" }),
        btn("Stop here", stopEarly, { cls: "ghost", key: "Esc", test: "stop" })));
    const left = h("div", null, h("div", { class: "row" }, h("span", { class: "kicker" }, label), h("span", { class: "muted small" }, `Drill ${index + 1} of ${data.drills.length}`)),
      dots, promptNodes, work);
    mount(ctx, h("div", { class: "page" }, pageTitle(title, "Name the pattern and complexity out loud, then reveal. Be honest when you grade."),
      h("section", { class: "card drill" }, left, h("div", { class: "timer-side" }, ringSlot,
        h("p", { class: "muted small center" }, `Instant Reflex = reveal within ${data.instant_seconds}s`)))));
    drillStart = performance.now();
    const paint = () => {
      const left2 = Math.max(0, data.seconds_per_drill - (performance.now() - drillStart) / 1000);
      ringSlot.replaceChildren(ring(left2 / data.seconds_per_drill, {
        size: 170, stroke: 12, label: String(Math.ceil(left2)), sub: "seconds",
        color: left2 > 20 ? COLORS.blue : left2 > 10 ? COLORS.yellow : COLORS.red,
      }));
      if (left2 <= 0 && phase === "think") { timedOut = true; reveal(); }
    };
    paint();
    timer = setInterval(paint, 200);

    async function reveal() {
      if (phase !== "think") return;
      phase = "reveal";
      stop();
      revealSeconds = (performance.now() - drillStart) / 1000;
      let ans;
      try { ans = await api("recall_reveal", { drill_id: drill.id }); } catch (err) { toast(err.message, true); phase = "think"; return; }
      const rows = [["Pattern", ans.pattern, ans.label], ["Complexity", ans.complexity], ["Edge cases", ans.edge_cases.join(" · ")]];
      if (ans.key_insight) rows.push(["Key insight", ans.key_insight]);
      if (ans.problem) rows.push(["Problem", `${ans.problem.id} · ${ans.problem.title} (LeetCode ${ans.problem.leetcode_id})`]);
      const grades = h("div", { class: "grades" }, data.grades.map((g) => h("button", {
        type: "button", class: `grade g${g.key}`, "data-grade": g.key, onclick: () => grade(g.key),
      }, h("span", { class: "top" }, kbd(g.key), g.label), h("span", { class: "again" }, g.again))));
      work.replaceChildren(
        h("p", { class: timedOut ? "warn-t" : "muted" }, timedOut ? "Time's up: graded as at most Effort." : `Revealed after ${Math.round(revealSeconds)}s.`),
        h("dl", { class: "reveal", "data-test": "answer" }, rows.map(([k, v, l]) => [h("dt", null, l || k), h("dd", null, v)])),
        h("p", { class: "muted small" }, "How did you do?"), grades);
      ringSlot.replaceChildren(ring(0, { size: 170, stroke: 12, label: `${Math.round(revealSeconds)}s`, sub: "to reveal", color: COLORS.blue }));
    }
  }

  async function grade(choice) {
    if (phase !== "reveal" || busy) return;
    busy = true;
    const drill = data.drills[index];
    try {
      const res = await api("recall_grade", { drill_id: drill.id, choice, seconds: revealSeconds, timed_out: timedOut });
      counts[res.counted_as] = (counts[res.counted_as] || 0) + 1;
      if (res.note) toast(`Saved · ${res.note}`);
      index += 1;
      if (index < data.drills.length) showDrill();
      else await finish();
    } catch (err) { toast(err.message, true); } finally { busy = false; }
  }

  async function stopEarly() {
    if (phase === "done") return;
    if (Object.values(counts).some((n) => n > 0)) { await finish(); return; } // log what was done, like [q] in the terminal
    stop();
    phase = "done";
    await api("session_discard");
    go("#/today", { force: true });
  }

  async function finish() {
    if (phase === "done") return;
    stop();
    phase = "done";
    const seconds = carried + (performance.now() - sessionStart) / 1000;
    const res = await api("recall_finish", { mode: data.mode, counts, seconds });
    await refreshState();
    const lines = [["Missed", counts["1"] || 0, "bad-t", "back today"], ["Effort", counts["2"] || 0, "warn-t", "in 3 days"], ["Instant", counts["3"] || 0, "ok-t", "next week"]];
    mount(ctx, h("div", { class: "page" }, pageTitle(`${title}: done`),
      h("section", { class: "hero green", "data-test": "recall-done" }, h("div", { class: "kicker" }, "Logged"),
        h("h2", null, `${plural(res.done, "drill")} in ${plural(res.minutes, "minute")} · streak ${plural(res.streak, "day")}`),
        h("div", { class: "grid-3" }, lines.map(([name, n, cls, when]) => h("div", { class: "tile" },
          h("div", null, h("div", { class: "label" }, name), h("div", { class: `value ${cls}` }, String(n)), h("div", { class: "sub" }, when))))),
        h("div", { class: "actions" }, btn("Back to Today", () => go("#/today"), { cls: "primary big", primary: true, key: "Enter" })))));
  }

  showDrill();
  return {
    cleanup: stop,
    onKey(e, typing) {
      if (typing || e.ctrlKey || e.altKey || e.metaKey) return false;
      if (e.key === " " && phase === "think") { $('[data-test="reveal"]').click(); return true; }
      if (e.key === "Escape" && phase !== "done") { stopEarly(); return true; }
      if (phase === "reveal" && ["1", "2", "3"].includes(e.key)) { grade(e.key); return true; }
      return false;
    },
  };
};

// ---------------------------------------------------------------- Mock: pick a problem
ROUTES.mock = async (ctx) => {
  const topic = ctx.query.get("topic") || "all";
  const [target, list] = await Promise.all([api("mock_target", { topic }), api("problems")]);
  setCrumbs(target.crumbs, ctx);
  let chosen = target.problem;
  let filter = ctx.query.get("show") || "all";
  let search = "";
  const heroSlot = h("div");
  const tbody = h("tbody");

  function renderHero() {
    const p = chosen;
    heroSlot.replaceChildren(h("section", { class: "hero", "data-test": "mock-target" },
      h("div", { class: "kicker" }, p.id === target.problem.id ? `Recommended · weakest topic${topic === "all" ? "" : " in this filter"}` : "Your pick"),
      h("h2", null, `${p.title}`),
      h("div", { class: "row" }, h("span", { class: `chip ${p.difficulty.toLowerCase()}` }, p.difficulty), h("span", { class: "chip blue" }, p.category_name),
        p.recency_label ? h("span", { class: "chip" }, p.recency_label) : null, h("span", { class: "muted small" }, `${p.id} · LeetCode ${p.leetcode_id}`)),
      h("p", { class: "statement" }, p.statement),
      p.time_target ? h("p", { class: "muted" }, "Aim for ", h("b", null, p.time_target), " time and ", h("b", null, p.space_target), " space.") : null,
      h("div", { class: "actions" },
        btn("Start 45-min mock", () => start(false), { cls: "primary big", primary: true, key: "Enter", test: "start-mock" }),
        btn("60-min timed solve", () => start(true), { key: "p", test: "start-solve" }))));
  }
  function start(practice) {
    S.boardStart = { problem_id: chosen.id, practice };
    go("#/board");
  }
  function renderRows() {
    const q = search.trim().toLowerCase();
    const rows = list.problems.filter((p) => (topic === "all" || p.category === topic)
      && (filter === "all" || (filter === "due" && p.due) || (filter === "unseen" && p.status === "unseen"))
      && (!q || `${p.id} ${p.title}`.toLowerCase().includes(q)));
    tbody.replaceChildren(...rows.map((p) => h("tr", {
      class: p.id === chosen.id ? "sel" : null, tabindex: "0",
      onclick: () => pick(p.id), onkeydown: (e) => { if (e.key === "Enter") { e.stopPropagation(); pick(p.id); } },
    }, h("td", { class: "mono small" }, p.id), h("td", null, p.title), h("td", null, h("span", { class: `chip ${p.difficulty.toLowerCase()}` }, p.difficulty)),
    h("td", { class: "muted" }, p.category_name), h("td", { class: "muted small" }, p.recency || "–"),
    h("td", null, p.due ? h("span", { class: "chip medium" }, "Due") : h("span", { class: `small ${p.status === "mastered" ? "ok-t" : "muted"}` }, p.status)))));
    if (!rows.length) tbody.replaceChildren(h("tr", null, h("td", { colspan: "6", class: "muted" }, "Nothing matches.")));
  }
  async function pick(id) {
    const row = list.problems.find((p) => p.id === id);
    if (!row) return;
    chosen = { ...row, statement: "", time_target: null };
    try {
      const card = await api("problem_card", { problem_id: id });
      chosen = card.problem;
    } catch (_) { /* preview is optional */ }
    renderHero();
    renderRows();
    heroSlot.scrollIntoView({ behavior: "smooth" });
  }
  const topicChips = h("div", { class: "filters" }, [{ key: "all", name: "All topics" }, ...list.categories].map((c) => h("button", {
    type: "button", "aria-pressed": String(c.key === topic), onclick: () => go(`#/mock?topic=${encodeURIComponent(c.key)}`),
  }, c.name)));
  const showChips = h("div", { class: "filters" }, [["all", "All"], ["due", "Due for re-solve"], ["unseen", "Not tried"]].map(([k, label]) => h("button", {
    type: "button", "aria-pressed": String(k === filter), onclick: (e) => {
      filter = k;
      $$("button", showChips).forEach((b) => b.setAttribute("aria-pressed", "false"));
      e.currentTarget.setAttribute("aria-pressed", "true");
      renderRows();
    },
  }, label)));
  const searchBox = h("input", { type: "search", class: "search", placeholder: "Find a problem", "aria-label": "Find a problem", oninput: (e) => { search = e.target.value; renderRows(); } });
  renderHero();
  renderRows();
  mount(ctx, h("div", { class: "page" },
    pageTitle("Mock & timed solves", "Press Enter for the recommended problem, or pick any problem below."),
    heroSlot, topicChips,
    h("section", { class: "card" }, h("div", { class: "row" }, h("h2", null, `Google problem bank · ${list.problems.length}`), h("div", { class: "spacer" }), showChips, searchBox),
      h("table", { class: "table" }, h("thead", null, h("tr", null, ["ID", "Problem", "Level", "Topic", "Asked", "Status"].map((x) => h("th", null, x)))), tbody))));
  return {
    onKey(e, typing) {
      if (!typing && !e.ctrlKey && !e.altKey && !e.metaKey && e.key === "p") { start(true); return true; }
      return false;
    },
  };
};

// ---------------------------------------------------------------- Whiteboard (45-min mock / 60-min solve)
ROUTES.board = async (ctx) => {
  let opts = S.boardStart;
  S.boardStart = null;
  if (!opts) {
    const sess = (await api("session")).session;
    if (!sess || sess.kind !== "mock") { go("#/mock", { replace: true, force: true }); return {}; }
    opts = { resume: true };
  }
  let data = await api("mock_start", opts);
  if (data.conflict) {
    const choice = await askAboutUnfinished(data.conflict);
    if (!ctx.alive()) return {};
    if (choice === "continue") { resumeSession(data.conflict); return {}; }
    if (choice !== "discard") { goBack(); return {}; }
    data = await api("mock_start", { ...opts, discard: true });
  }
  const p = data.problem;
  const B = {
    problem_id: p.id, practice: data.practice, minutes: data.minutes, elapsed: data.elapsed,
    ticks: data.ticks.slice(), shown: data.shown, paused: false, done: false,
  };
  const total = B.minutes * 60;
  let saveTimer = null;
  let dirty = false;
  let last = performance.now();
  let lastSaved = 0;
  if (data.submitted) return scorecard(ctx, B, data);
  setCrumbs(data.crumbs, ctx);

  const clockText = h("span");
  const pausedTag = h("span", { class: "paused", hidden: true }, "paused");
  const clock = h("div", { class: "clock", "data-test": "clock" }, clockText, pausedTag);
  const segs = data.phases.map((ph) => {
    const fill = h("i");
    const seg = h("div", { class: "seg", title: `${ph.label} · ${ph.start}–${ph.end} min (of 45)` }, fill, h("span", null, ph.label));
    seg.style.setProperty("flex", `${ph.end - ph.start} 1 0`);
    return { ph, seg, fill };
  });
  const savedTag = h("span", { class: "saved", "data-test": "saved" }, "Autosave on");
  const banner = h("div", { class: "banner", hidden: true, "data-test": "follow-up" });
  const code = h("textarea", { class: "code", spellcheck: "false", "aria-label": "Code pad (Python)", "data-test": "code" });
  code.value = data.code;
  const trace = h("textarea", { class: "code", spellcheck: "false", hidden: true, "aria-label": "Dry-run trace", "data-test": "trace" });
  trace.value = data.trace;
  const tabCode = h("button", { type: "button", role: "tab", "aria-selected": "true", onclick: () => showTab("code") }, "Code");
  const tabTrace = h("button", { type: "button", role: "tab", "aria-selected": "false", onclick: () => showTab("trace") }, "Dry-run trace");
  const checks = data.checklist.map((item, i) => {
    const box = h("input", { type: "checkbox", checked: B.ticks[i], onchange: (e) => { B.ticks[i] = e.target.checked; save(false); } });
    return { box, label: h("label", null, box, h("span", null, h("b", null, `Alt+${i + 1}`), "  ", item)) };
  });
  const pauseBtn = btn("Pause", togglePause, { cls: "ghost", test: "pause" });

  function showTab(which) {
    const isCode = which === "code";
    code.hidden = !isCode;
    trace.hidden = isCode;
    tabCode.setAttribute("aria-selected", String(isCode));
    tabTrace.setAttribute("aria-selected", String(!isCode));
    (isCode ? code : trace).focus();
  }
  function renderBanner() {
    if (B.shown < 0) { banner.hidden = true; return; }
    banner.hidden = false;
    banner.replaceChildren(h("b", null, "Google follow-up"), h("span", { class: "text" }, data.follow_ups[B.shown % data.follow_ups.length]),
      btn("Another one", () => { B.shown = (B.shown + 1) % data.follow_ups.length; renderBanner(); save(false); }, { cls: "ghost small" }));
  }
  function paint() {
    const left = total - B.elapsed;
    clockText.textContent = left >= 0 ? mmss(left) : `+${mmss(-left)}`;
    clock.classList.toggle("late", B.elapsed > total * FOLLOW_UP_AT); // past the 35-min coding target
    const mins = (B.elapsed / total) * 45;
    for (const { ph, seg, fill } of segs) {
      const frac = Math.max(0, Math.min(1, (mins - ph.start) / (ph.end - ph.start)));
      fill.style.setProperty("--w", `${Math.round(frac * 100)}%`);
      seg.classList.toggle("now", mins >= ph.start && mins < ph.end);
      seg.classList.toggle("past", mins >= ph.end);
    }
  }
  function tick() {
    const now = performance.now();
    if (!B.paused && !B.done) B.elapsed += (now - last) / 1000;
    last = now;
    const frac = B.elapsed / total;
    if (B.shown < 0 && frac >= FOLLOW_UP_AT) {
      B.shown = 0;
      renderBanner();
      toast("Minute 35: the interviewer adds a follow-up. Talk through the trade-offs.");
      save(false);
    }
    paint();
    if (!B.done && ((!B.practice && frac >= RUBRIC_AT) || (B.practice && frac >= 1))) {
      toast(B.practice ? "Time. Grade yourself." : "40:00. Stop coding: scorecard time.");
      submit(true);
      return;
    }
    if (now - lastSaved > 5000) save(dirty);
  }
  async function save(withText, keepalive = false) {
    lastSaved = performance.now();
    const payload = {
      problem_id: B.problem_id, practice: B.practice, minutes: B.minutes, elapsed: B.elapsed, ticks: B.ticks, shown: B.shown,
    };
    if (withText) { payload.code = code.value; payload.trace = trace.value; dirty = false; }
    try {
      await api("mock_save", payload, { keepalive });
      savedTag.textContent = `Saved ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
      return true;
    } catch (err) { savedTag.textContent = "Not saved"; toast(err.message, true); return false; }
  }
  function onInput() {
    dirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => save(true), 600);
  }
  function togglePause() {
    B.paused = !B.paused;
    pausedTag.hidden = !B.paused;
    pauseBtn.firstChild.textContent = B.paused ? "Resume" : "Pause";
    save(false);
  }
  function stopAll() {
    clearInterval(loop);
    clearTimeout(saveTimer);
    document.removeEventListener("visibilitychange", onVisibility);
  }
  async function submit(auto = false) {
    if (B.done) return;
    const limit = B.practice ? total : total * RUBRIC_AT;
    if (!auto && B.elapsed < limit) {
      const ok = await confirmBox("Submit and grade now?", `${mmss(limit - B.elapsed)} of coding time is left. After grading you cannot go back to this attempt.`, "Submit & grade", "Keep coding", true);
      if (!ok || B.done) return;
    }
    B.done = true;
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); // typing must not answer the scorecard
    stopAll();
    await save(true);
    if (current) current.beforeLeave = null;
    scorecard(ctx, B, data).then((handle) => { if (ctx.alive()) current = handle; })
      .catch((err) => { toast(err.message, true); if (ctx.alive()) current = {}; });
  }
  async function abandon() {
    const ok = await confirmBox("Abandon this round?", "The attempt is not logged and the timer resets. Your code file stays on disk.", "Abandon", "Keep going", true);
    if (!ok) return;
    B.done = true;
    stopAll();
    await api("mock_abandon");
    go("#/mock", { force: true });
  }
  for (const pad of [code, trace]) {
    pad.addEventListener("input", onInput);
    pad.addEventListener("keydown", (e) => editorKeys(e, pad, onInput));
  }

  const left = h("div", { class: "wb-left" },
    h("section", { class: "card" },
      h("div", { class: "row" }, h("span", { class: `chip ${p.difficulty.toLowerCase()}` }, p.difficulty), h("span", { class: "chip blue" }, p.category_name),
        p.recency_label ? h("span", { class: "chip" }, p.recency_label) : null),
      h("h3", { class: "problem-title" }, p.title),
      h("p", { class: "statement" }, p.statement),
      h("p", { class: "muted small" }, "First 5 minutes: ask about input size, ranges, empty input and duplicates. Say the complexity before you code.")),
    h("section", { class: "card checklist" }, h("h2", null, "Live interview checklist"), checks.map((c) => c.label)),
    DEBUG ? btn("+5 min (debug)", () => { B.elapsed += 300; tick(); }, { cls: "ghost small", test: "ff" }) : null);
  const right = h("div", { class: "wb-right" },
    h("div", { class: "tabs", role: "tablist" }, tabCode, tabTrace, h("span", { class: "hint" }, savedTag, " · Ctrl+S saves · Tab = 4 spaces · Esc leaves the pad")),
    code, trace);

  mount(ctx, h("div", { class: "page wb" },
    h("div", { class: "wb-top" },
      h("div", null, h("div", { class: "muted small" }, B.practice ? `Timed solve · ${B.minutes} min` : "Mock interview · 45 min"), clock),
      h("div", { class: "phases" }, segs.map((x) => x.seg)),
      h("div", { class: "row" }, pauseBtn, btn("Abandon", abandon, { cls: "ghost" }),
        btn("Submit & grade", () => submit(), { cls: "primary", key: "Ctrl+Enter", test: "submit" }))),
    banner, h("div", { class: "wb-body" }, left, right)));
  renderBanner();
  paint();
  const loop = setInterval(tick, 250);
  const onVisibility = () => { if (document.visibilityState === "hidden") save(true, true); };
  document.addEventListener("visibilitychange", onVisibility);
  if (!data.code.trim() || data.elapsed === 0) code.focus();

  return {
    cleanup() {
      if (!B.done) { stopAll(); save(true, true); }
    },
    async beforeLeave() {
      if (B.done) return true;
      return confirmBox("Leave the whiteboard?", "Your code, trace and clock are saved. Continue later from Today.", "Leave", "Stay");
    },
    onKey(e, typing) {
      if (B.done) return true; // handing over to the scorecard: swallow keys still being typed
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { clearTimeout(saveTimer); save(true).then((ok) => { if (ok) toast("Saved."); }); return true; }
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { submit(); return true; }
      if (e.altKey && ["1", "2", "3", "4"].includes(e.key)) {
        const i = Number(e.key) - 1;
        B.ticks[i] = !B.ticks[i];
        checks[i].box.checked = B.ticks[i];
        save(false);
        return true;
      }
      if (e.altKey && e.key.toLowerCase() === "p") { togglePause(); return true; }
      if (typing && e.key === "Escape") { document.activeElement.blur(); return true; }
      if (!typing && e.key === "Escape") { go("#/today"); return true; }
      return false;
    },
  };
};

/** Code pad keys: Tab indents 4 spaces, Shift+Tab dedents, Enter keeps indentation (+4 after ':'). */
function editorKeys(e, pad, changed) {
  const { selectionStart: a, selectionEnd: b, value } = pad;
  const lineStart = value.lastIndexOf("\n", a - 1) + 1;
  if (e.key === "Tab" && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    if (e.shiftKey) {
      const spaces = /^ {1,4}/.exec(value.slice(lineStart))?.[0].length || 0;
      if (spaces) { pad.setRangeText("", lineStart, lineStart + spaces, "preserve"); pad.selectionStart = pad.selectionEnd = Math.max(lineStart, a - spaces); }
    } else {
      pad.setRangeText("    ", a, b, "end");
    }
    changed();
  } else if (e.key === "Enter" && !e.ctrlKey && !e.altKey && !e.shiftKey && !e.metaKey) {
    const line = value.slice(lineStart, a);
    let indent = /^[ \t]*/.exec(line)[0];
    if (/:\s*$/.test(line) || /[{([]\s*$/.test(line)) indent += "    ";
    e.preventDefault();
    pad.setRangeText(`\n${indent}`, a, b, "end");
    changed();
  }
}

// ---------------------------------------------------------------- Scorecard (40:00-45:00) and result
async function scorecard(ctx, B, data) {
  const sc = await api("mock_scorecard", {
    problem_id: B.problem_id, practice: B.practice, minutes: B.minutes, elapsed: B.elapsed, ticks: B.ticks, shown: B.shown,
  });
  if (!ctx.alive()) return {};
  setCrumbs(sc.crumbs, ctx);
  const answers = { outcome: null, checks: {}, mistake: null };
  let step = 0;
  let needsMistake = false;
  const steps = () => ["outcome", ...sc.questions.map((q) => q.field), ...(needsMistake ? ["mistake"] : [])];
  const main = h("section", { class: "card score-main" });
  const side = h("section", { class: "card score-side" });
  let keyHandler = () => false;

  async function advance() {
    if (step === 1 + sc.questions.length) {
      // outcome + every check answered: decide whether to ask what went wrong
      const res = await api("mock_needs_mistake", { outcome: answers.outcome, dry_run_passed: Boolean(answers.checks.dry_run_passed), elapsed: B.elapsed, minutes: B.minutes });
      needsMistake = res.needs;
    }
    if (step >= steps().length) return finish();
    render();
  }
  function choose(value) {
    const name = steps()[step];
    if (name === "outcome") answers.outcome = value;
    else if (name === "mistake") answers.mistake = value;
    else answers.checks[name] = value;
    step += 1;
    advance();
  }
  function back() {
    if (step === 0) return;
    step -= 1;
    render();
  }
  function render() {
    const all = steps();
    const name = all[step];
    const header = h("div", { class: "row" }, h("span", { class: "kicker" }, `Step ${step + 1} of ${all.length}`),
      h("div", { class: "spacer" }), step ? btn("Back", back, { cls: "ghost small", key: "Backspace" }) : null);
    if (name === "outcome") {
      main.replaceChildren(header, h("p", { class: "q" }, "How did it go?"),
        sc.over_time ? h("p", { class: "note" }, "Over the 35-min coding target: Instant Reflex will count as Effort.") : null,
        h("div", { class: "grades" }, sc.outcomes.map((o) => h("button", {
          type: "button", class: `grade g${o.key}`, "data-grade": o.key, onclick: () => choose(o.key),
        }, h("span", { class: "top" }, kbd(o.key), o.label), h("span", { class: "again" }, o.again)))));
      keyHandler = (k) => (["1", "2", "3"].includes(k) ? (choose(k), true) : false);
    } else if (name === "mistake") {
      main.replaceChildren(header, h("p", { class: "q" }, "What cost you the most?"),
        h("p", { class: "muted" }, "One tap. This tunes tomorrow's warmup to your weak spot."),
        h("div", { class: "mistakes" }, sc.mistakes.map((m, i) => h("button", { type: "button", class: "grade", onclick: () => choose(m.key) },
          h("span", { class: "top" }, kbd(String(i + 1)), m.label)))));
      keyHandler = (k) => {
        const m = sc.mistakes[Number(k) - 1];
        return m ? (choose(m.key), true) : false;
      };
    } else {
      const q = sc.questions.find((x) => x.field === name);
      main.replaceChildren(header,
        sc.practice ? null : h("div", { class: "muted small" }, `Pillar · ${q.pillar_name}`),
        h("p", { class: "q" }, `${q.question}?`),
        q.ticked ? h("p", { class: "note good" }, "You ticked this during the round.") : null,
        name === "dry_run_passed" ? h("p", { class: "note" }, sc.trace_rows ? `Your trace table has ${plural(sc.trace_rows, "filled row")}.` : "Your trace table is empty. Be strict.") : null,
        name === "follow_up_handled" && sc.follow_up ? h("p", { class: "note" }, `Follow-up: ${sc.follow_up}`) : null,
        h("div", { class: "yn" },
          btn("Yes", () => choose(true), { cls: `good ${q.ticked ? "primary" : ""}`, key: "y", primary: q.ticked, test: "yes" }),
          btn("No", () => choose(false), { cls: "bad", key: "n", test: "no" })));
      keyHandler = (k) => (k === "y" ? (choose(true), true) : k === "n" ? (choose(false), true) : false);
    }
    side.replaceChildren(h("h2", null, B.practice ? "Self-grade" : "Scorecard · 4 pillars"),
      h("p", null, "Coding time ", h("b", null, mmss(sc.used)), ` of ${B.minutes}:00`, sc.over_time ? h("span", { class: "warn-t" }, " · over the 35-min target") : null),
      h("ul", { class: "answers" }, all.map((field, i) => {
        let label;
        let val;
        if (field === "outcome") { label = "Outcome"; val = answers.outcome ? sc.outcomes.find((o) => o.key === answers.outcome).label : null; }
        else if (field === "mistake") { label = "Biggest cost"; val = answers.mistake ? sc.mistakes.find((m) => m.key === answers.mistake).label : null; }
        else { label = sc.questions.find((q) => q.field === field).question; val = field in answers.checks ? (answers.checks[field] ? "Yes" : "No") : null; }
        const cls = val === "Yes" ? "ok-t" : val === "No" ? "bad-t" : "";
        return h("li", { class: i === step ? "now" : null }, h("span", null, label), h("b", { class: cls || null }, val || (i === step ? "…" : "")));
      })));
  }
  let finishing = false;
  async function finish() {
    if (finishing) return; // a double click or a repeated key must not log the attempt twice
    finishing = true;
    keyHandler = () => false;
    main.replaceChildren(h("p", { class: "muted" }, "Saving…"));
    let res;
    try {
      res = await api("mock_grade", {
        problem_id: B.problem_id, practice: B.practice, minutes: B.minutes, elapsed: B.elapsed, outcome: answers.outcome,
        checks: answers.checks, mistake: answers.mistake, shown: B.shown,
      });
    } catch (err) { toast(err.message, true); finishing = false; step -= 1; render(); return; }
    await refreshState();
    showResult(ctx, res, B, data);
    handle.onKey = () => false;
  }
  mount(ctx, h("div", { class: "page" },
    pageTitle(B.practice ? "Self-grade" : "Scorecard", B.practice ? "Be strict: only a trace that matched counts." : "40:00–45:00 · Google's four pillars, yes or no.", h("span", { class: "chip" }, data.problem.title)),
    h("div", { class: "score-wrap" }, main, side)));
  render();
  const keysFrom = performance.now() + 800; // keystrokes still in flight from the code pad are not answers
  const handle = {
    onKey(e, typing) {
      if (typing || e.ctrlKey || e.altKey || e.metaKey) return false;
      if (performance.now() < keysFrom) return true;
      if (e.key === "Backspace") { back(); return true; }
      return keyHandler(e.key.toLowerCase());
    },
  };
  return handle;
}

function showResult(ctx, res, B, data) {
  const solved = res.solved;
  const pillars = Object.entries(res.pillars || {});
  mount(ctx, h("div", { class: "page" },
    pageTitle(B.practice ? "Solve logged" : "Mock logged", data.problem.title),
    h("section", { class: `hero ${solved ? "green" : "yellow"}`, "data-test": "result" },
      h("div", { class: "kicker" }, res.again),
      h("h2", null, res.outcome_label, B.practice ? null : h("span", { class: "signal blue-t" }, ` · ${res.signal}`)),
      h("ul", { class: "bullets" },
        h("li", null, "Coding time ", h("b", null, mmss(res.used_seconds)), ` of ${B.minutes}:00`, res.over_time && solved ? h("span", { class: "warn-t" }, " (over the 35-min coding target)") : null),
        h("li", null, "Dry-run matched: ", h("b", { class: res.dry_run_ok ? "ok-t" : "bad-t" }, res.dry_run_ok ? "yes" : "no")),
        res.mistake_label ? h("li", null, "Tomorrow's warmup leans on: ", h("b", null, res.mistake_label)) : null),
      B.practice ? null : h("div", { class: "pillars" }, pillars.map(([name, value]) => h("div", { class: "meter" },
        h("span", null, name), bar(value / 25, toneFor(value / 25)), h("span", { class: "val" }, `${Math.round(value)}/25`))),
      h("div", { class: "meter total" }, h("b", null, "Score"), bar(res.score / 100, toneFor(res.score / 100)), h("span", { class: "val" }, `${Math.round(res.score)}/100`))),
      h("div", { class: "actions" },
        btn("Back to Today", () => go("#/today"), { cls: "primary big", primary: true, key: "Enter" }),
        btn("Check it on LeetCode", () => api("open_link", { url: data.problem.url }).catch((err) => toast(err.message, true))),
        btn("Another round", () => go("#/mock"), { cls: "ghost", key: "w" })))));
}

// ---------------------------------------------------------------- Roadmap
ROUTES.roadmap = async (ctx) => {
  const r = await api("roadmap");
  setCrumbs(r.crumbs, ctx);
  const comps = h("div", null, r.components.map((c) => h("div", { class: "meter" },
    h("span", null, c.name, h("span", { class: "muted small" }, ` · ${c.weight}%`)), bar(c.value, toneFor(c.value)), h("span", { class: "val" }, `${Math.round(c.value * 100)}%`))));
  const change = r.weekly_change;
  const gauge = h("section", { class: "card gauge-card" },
    ring(r.index / 100, { size: 210, stroke: 16, label: `${Math.round(r.index)}%`, sub: "readiness", color: r.index >= 70 ? COLORS.green : r.index >= 40 ? COLORS.yellow : COLORS.blue }),
    h("div", null, h("h2", null, "Google Readiness Index"), comps, h("div", { class: "formula" }, r.formula),
      h("p", { class: "muted small" }, change == null ? "Weekly change shows after a week of history." : `${change >= 0 ? "+" : ""}${change.toFixed(1)} points vs. 7 days ago.`)));
  const history = r.history.length > 1 ? sparkline(r.history) : h("p", { class: "muted" }, "The trend line starts after your second day.");
  const phases = h("div", { class: "timeline" }, r.phases.map((ph) => h("div", { class: "tl-row" },
    h("b", null, `Phase ${ph.number}`),
    h("div", null, h("div", { class: "small" }, ph.name), bar(ph.progress, ph.state === "done" ? "green" : "")),
    h("span", { class: "muted small" }, ph.first === ph.last ? `Day ${ph.first}` : `Days ${ph.first}–${ph.last}`),
    h("span", { class: `state ${ph.state === "done" ? "ok-t" : ph.state === "now" ? "blue-t" : "muted"}` }, ph.state === "now" ? `Day ${r.day}` : ph.state))));
  const topics = h("div", null, r.topics.map((t) => h("div", { class: "meter" },
    h("span", null, t.name), bar(t.score, toneFor(t.score)), h("span", { class: "val" }, `${t.solved}/${t.total}`))));
  const maxMistake = Math.max(1, ...r.mistakes.map((m) => m.count));
  const mistakes = r.mistakes.some((m) => m.count) ? h("div", null, r.mistakes.map((m) => h("div", { class: "meter" },
    h("span", null, m.label), bar(m.count / maxMistake, "red"), h("span", { class: "val" }, String(m.count)))))
    : h("p", { class: "muted" }, "No misses logged yet. Each failed or slow solve asks one question and lands here.");
  const pace = r.pace;
  mount(ctx, h("div", { class: "page" },
    pageTitle("Roadmap", `Day ${r.day} of ${r.plan_days} · interview ${fmtDate(r.interview_date)}`),
    gauge,
    h("div", { class: "grid-2" },
      h("section", { class: "card" }, h("h2", null, "Readiness trend · 14 days"), history),
      h("section", { class: "card" }, h("h2", null, "8-week plan"), phases)),
    h("div", { class: "grid-2" },
      h("section", { class: "card" }, h("h2", null, "Topic mastery"), topics,
        h("p", { class: "muted small" }, `Problems solved ${pace.solved}/${pace.total}`, pace.behind ? ` · behind plan by ${pace.behind}` : " · on plan")),
      h("section", { class: "card" }, h("h2", null, "What costs you points"), mistakes,
        h("h2", { class: "mt" }, "STAR stories"), h("div", { class: "meter" }, h("span", null, "Stories ready"), bar(r.star.value, toneFor(r.star.value)), h("span", { class: "val" }, `${r.star.have}/${r.star.target}`))))));
  return {};
};
function sparkline(values) {
  const w = 300;
  const hgt = 90;
  const max = Math.max(100, ...values);
  const step = w / Math.max(1, values.length - 1);
  const pts = values.map((v, i) => [i * step, hgt - 6 - (v / max) * (hgt - 12)]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const [lx, ly] = pts[pts.length - 1];
  return svg("svg", { class: "spark", viewBox: `0 0 ${w} ${hgt}`, preserveAspectRatio: "none", role: "img", "aria-label": `Readiness trend, latest ${Math.round(values[values.length - 1])}%` },
    svg("path", { class: "area", d: `${line} L${w},${hgt} L0,${hgt} Z` }), svg("path", { class: "line", d: line }), svg("circle", { class: "dot", cx: lx, cy: ly, r: 4 }));
}

// ---------------------------------------------------------------- STAR stories
ROUTES.star = async (ctx) => {
  let data = await api("star_list");
  setCrumbs(data.crumbs, ctx);
  let selected = ctx.parts[1] ? Number(ctx.parts[1]) : null;
  let dim = (data.dimensions.find((d) => d.count < data.need) || data.dimensions[0]).key;
  const leftCol = h("div", { class: "star-left" });
  const editor = h("section", { class: "card star-editor" });
  let form = null;
  let dirty = false;
  const okToDrop = async () => !dirty
    || confirmBox("Discard your unsaved changes?", "This story has changes that are not saved yet.", "Discard", "Keep editing", true);
  async function open(id) {
    if (!(await okToDrop())) return false;
    selected = id;
    renderLeft();
    renderEditor();
    return true;
  }

  function renderLeft() {
    leftCol.replaceChildren(
      h("section", { class: "card" }, h("h2", null, `Coverage · ${data.need} per dimension`), data.dimensions.map((d) => h("div", { class: "meter" },
        h("span", null, d.key), bar(Math.min(1, d.count / data.need), d.count >= data.need ? "green" : "yellow"), h("span", { class: "val" }, `${d.count}/${data.need}`)))),
      h("div", { class: "row" }, btn("New story", () => open(null), { key: "n", cls: "primary" }), btn("Practice a prompt", practice, { key: "p" })),
      data.stories.length ? h("div", { class: "story-list" }, data.stories.map((x) => h("button", {
        type: "button", class: "story", "aria-current": String(x.id === selected), onclick: () => open(x.id),
      }, h("span", { class: "chip blue" }, x.dimension), h("span", { class: "title" }, x.title))))
        : h("div", { class: "empty" }, "No stories yet. Google asks about ambiguity, ownership, disagreement and humility: write one per dimension first."));
  }
  function renderEditor() {
    const story = data.stories.find((x) => x.id === selected) || null;
    if (story) dim = story.dimension;
    const seg = h("div", { class: "seg-ctl", role: "group", "aria-label": "Dimension" }, data.dimensions.map((d) => h("button", {
      type: "button", "aria-pressed": String(d.key === dim), title: d.label,
      onclick: (e) => { dim = d.key; dirty = true; $$("button", seg).forEach((b) => b.setAttribute("aria-pressed", "false")); e.currentTarget.setAttribute("aria-pressed", "true"); },
    }, d.key)));
    const field = (name, label, hint, rows) => h("div", { class: "field" }, h("label", { for: `star-${name}` }, label, h("span", { class: "muted" }, ` · ${hint}`)),
      rows ? h("textarea", { id: `star-${name}`, name, rows: String(rows), value: story ? story[name] : "" }) : h("input", { id: `star-${name}`, name, value: story ? story.title : "" }));
    form = h("form", { class: "star-form", onsubmit: (e) => { e.preventDefault(); saveStory(); } },
      seg,
      field("title", "Title", "a short name you will remember", 0),
      field("situation", "Situation", "context in 1–2 sentences", 2),
      field("task", "Task", "what you owned", 2),
      field("action", "Action", "what YOU did; use I, not we", 4),
      field("result", "Result", "numbers: latency, revenue, incidents, time saved", 2),
      h("div", { class: "row" }, btn(story ? "Save changes" : "Save story", saveStory, { cls: "primary", key: "Ctrl+S", test: "star-save" }),
        story ? btn("Delete", deleteStory, { cls: "danger" }) : null));
    form.addEventListener("input", () => { dirty = true; });
    dirty = false;
    editor.replaceChildren(h("h2", null, story ? "Edit story" : "New story"), form);
  }
  async function saveStory() {
    const fd = Object.fromEntries(new FormData(form).entries());
    try {
      data = await api("star_save", { dimension: dim, ...fd, id: selected });
      if (selected == null) selected = Math.max(...data.stories.map((x) => x.id));
      toast("Story saved.");
      renderLeft();
      renderEditor();
      refreshState();
    } catch (err) { toast(err.message, true); }
  }
  async function deleteStory() {
    if (!(await confirmBox("Delete this story?", "This cannot be undone.", "Delete", "Keep", true))) return;
    data = await api("star_delete", { id: selected });
    selected = null;
    dirty = false;
    renderLeft();
    renderEditor();
  }
  async function practice() {
    const pr = await api("star_practice", { dimension: dim });
    let left = pr.seconds;
    const clock = h("div", { class: "clock" }, mmss(left));
    const timer = setInterval(() => { left -= 1; clock.textContent = mmss(left); if (left <= 0) { clearInterval(timer); clock.classList.add("late"); } }, 1000);
    await openModal(`${pr.dimension} · ${pr.label}`, h("div", null, h("p", { class: "q" }, pr.prompt), clock,
      h("p", { class: "muted" }, "Answer out loud in under 2 minutes: Situation, Task, Action (I), Result (numbers)."),
      pr.stories.length ? h("p", { class: "small" }, "Your stories for this: ", pr.stories.map((x) => x.title).join(" · ")) : h("p", { class: "warn-t small" }, "You have no story for this dimension yet.")),
    [{ label: "Done", value: true, primary: true, key: "Enter" }]);
    clearInterval(timer);
  }
  renderLeft();
  renderEditor();
  mount(ctx, h("div", { class: "page" }, pageTitle("STAR stories", "Googleyness & leadership: two ready stories per dimension."),
    h("div", { class: "star-layout" }, leftCol, editor)));
  return {
    beforeLeave: okToDrop,
    isDirty: () => dirty,
    onKey(e, typing) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { saveStory(); return true; }
      if (typing || e.ctrlKey || e.altKey || e.metaKey) return false;
      if (e.key === "n") { open(null).then((ok) => { if (ok) $("#star-title").focus(); }); return true; }
      if (e.key === "p") { practice(); return true; }
      return false;
    },
  };
};

// ---------------------------------------------------------------- Concept vault
ROUTES.vault = async (ctx) => {
  const list = await api("vault_list");
  if (!list.notes.length) {
    setCrumbs(list.crumbs, ctx);
    mount(ctx, h("div", { class: "page" }, pageTitle("Concept vault"), h("div", { class: "empty" }, "No notes bundled.")));
    return {};
  }
  const file = ctx.parts[1] || list.notes[0].file;
  const note = await api("vault_note", { file });
  setCrumbs(note.crumbs, ctx);
  const body = h("article", { class: "card md", "data-test": "note" });
  body.innerHTML = note.html; // rendered server-side by markdown-it with raw HTML disabled
  for (const codeEl of $$("pre > code", body)) {
    const lang = (/language-([\w+]+)/.exec(codeEl.className) || [])[1] || "";
    if (/^(py|python|cpp|c\+\+|cc)$/.test(lang)) codeEl.replaceChildren(highlight(codeEl.textContent, lang.startsWith("c") ? "cpp" : "python"));
  }
  for (const a of $$("a[href]", body)) {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const href = a.getAttribute("href");
      if (/^https:\/\//.test(href)) api("open_link", { url: href }).catch((err) => toast(err.message, true));
      else if (href.endsWith(".md")) go(`#/vault/${encodeURIComponent(href.split("/").pop())}`);
    });
  }
  const side = h("nav", { class: "toc", "aria-label": "Notes" }, h("h4", null, "Notes"), list.notes.map((n) => h("button", {
    type: "button", "aria-current": n.file === file ? "step" : null, onclick: () => go(`#/vault/${encodeURIComponent(n.file)}`),
  }, h("span", { class: "n" }, n.file.slice(0, 2)), h("span", null, n.title.replace(/^\d+\s*[·:.-]?\s*/, "")))));
  mount(ctx, h("div", { class: "page wide" }, pageTitle("Concept vault", "C++ next to Python, complexity tables and traps for each pattern."),
    h("div", { class: "vault" }, side, body)));
  return {};
};

// ---------------------------------------------------------------- keys and start-up
const isTyping = (el) => el && (el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable
  || (el.tagName === "INPUT" && !["checkbox", "radio", "button", "submit"].includes(el.type)));

document.addEventListener("keydown", (e) => {
  if (modalState) {
    if (e.key === "Escape") { e.preventDefault(); modalState.close(null); return; }
    if (e.key === "Enter" && !(e.target instanceof HTMLButtonElement) && e.target.tagName !== "TEXTAREA") {
      const primary = modalState.buttons.find((b) => b.primary);
      if (primary) { e.preventDefault(); modalState.close(primary.value); }
    }
    return;
  }
  const typing = isTyping(e.target);
  if (current && current.onKey && current.onKey(e, typing)) { e.preventDefault(); return; }
  if (typing || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === "?") { e.preventDefault(); openHelp(); return; }
  if (e.key === "Escape") { if (!location.hash.startsWith("#/today")) { e.preventDefault(); go("#/today"); } return; }
  if (e.key === "Enter") {
    const active = document.activeElement;
    if (active && active !== document.body && active.id !== "view" && active.matches("button, a, input, select, textarea, [tabindex]")) return;
    const primary = $("#view [data-primary]:not(:disabled)");
    if (primary) { e.preventDefault(); primary.click(); }
    return;
  }
  if (NAV_KEYS[e.key]) { e.preventDefault(); go(`#/${NAV_KEYS[e.key]}`); }
});
window.addEventListener("hashchange", () => {
  // Back/Forward (Alt+Left, mouse back button) must ask before dropping unsaved work, like the sidebar does
  if (current && current.beforeLeave && S.currentHash && location.hash !== S.currentHash) {
    const target = location.hash;
    history.replaceState(null, "", S.currentHash);
    go(target);
    return;
  }
  route();
});
window.addEventListener("beforeunload", (e) => {
  if (current && current.isDirty && current.isDirty()) e.preventDefault(); // closing with an unsaved STAR story
});

async function boot() {
  buildNav();
  if (!TOKEN) {
    $("#view").replaceChildren(h("div", { class: "page" }, h("div", { class: "empty" }, "Open this window from the Google SWE II OS app.")));
    return;
  }
  try { await refreshState(); } catch (err) {
    $("#view").replaceChildren(h("div", { class: "page" }, h("div", { class: "empty" }, err.message)));
    return;
  }
  if (!location.hash) history.replaceState(null, "", `${location.search}#/today`);
  await route(); // an unfinished drill or mock shows as the first thing on Today (auto-recovery)
  setInterval(() => api("heartbeat").catch(() => {}), 30000);
}
boot();
