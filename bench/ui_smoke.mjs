/**
 * Frontend smoke test — runs web/index.html's module in Node against a stub DOM.
 *
 * This exists because of a bug that shipped: `setTheme()` was called at the top
 * of the module and reached a `let PAL` declared thirty lines further down.
 * A `let` is in its temporal dead zone until its own declaration runs, so the
 * call threw `ReferenceError: Cannot access 'PAL' before initialization`, which
 * aborts module evaluation — `connect()` and `loadTopics()` never ran and the
 * page came up completely empty. It is valid syntax, so `node --check` passed,
 * and every element id resolved, so a static reference check passed too. The
 * only thing that catches it is *running* it.
 *
 * So this evaluates the real module, then drives it through a scripted
 * interview to check the part that cannot be eyeballed either: that captions
 * are revealed on the audio clock, in order, with the question replacing the
 * feedback rather than being appended to it.
 *
 * Run:  node bench/ui_smoke.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(ROOT, "web", "index.html"), "utf8");

// --------------------------------------------------------------- stub DOM --
class El {
  constructor(id = "") {
    this.id = id; this.children = []; this._text = ""; this._html = "";
    this.style = {}; this.dataset = {}; this.disabled = false;
    this.scrollTop = 0; this.scrollHeight = 0;
    this._classes = new Set();
    this.classList = {
      toggle: (c, on) => on ? this._classes.add(c) : this._classes.delete(c),
      add: (c) => this._classes.add(c),
      remove: (c) => this._classes.delete(c),
      contains: (c) => this._classes.has(c),
    };
  }
  set textContent(v) { this._text = String(v); this._html = ""; }
  get textContent() { return this._html ? stripTags(this._html) : this._text; }
  set innerHTML(v) { this._html = String(v); this._text = ""; if (!v) this.children = []; }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { this[k === "aria-pressed" ? "pressed" : k] = v; }
  getAttribute(k) { return this[k === "aria-pressed" ? "pressed" : k]; }
  append(...kids) { this.children.push(...kids); }
  getBoundingClientRect() { return { width: 196, height: 196 }; }
  getContext() { return CANVAS_CTX; }
}

const stripTags = (s) => s.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();

const CANVAS_CTX = new Proxy({}, {
  get: (_t, k) => {
    if (k === "createRadialGradient")
      return () => ({ addColorStop() {} });
    return () => {};
  },
});

const els = new Map();
const el = (id) => {
  if (!els.has(id)) els.set(id, new El(id));
  return els.get(id);
};
// Every id the markup declares, so getElementById never returns null.
for (const m of html.matchAll(/\bid="([\w-]+)"/g)) el(m[1]);

const views = ["setup", "live", "report", "history"].map(el);

globalThis.document = {
  documentElement: new El("html"),
  getElementById: (id) => els.get(id) ?? null,
  querySelectorAll: (sel) => (sel === ".view" ? views : []),
  createElement: () => new El(),
  createTextNode: (t) => ({ nodeText: String(t) }),
};
globalThis.getComputedStyle = () => ({
  // Real values, so anything that concatenates a colour with an alpha suffix
  // (the orb does) produces something plausible rather than "undefined33".
  getPropertyValue: (n) => ({
    "--bg": "#e9edf5", "--dark": "#c2c9d8", "--light": "#ffffff",
    "--accent": "#5866e0", "--faint": "#98a0b5",
    "--ok": "#1f9d6b", "--mid": "#c98a12",
  }[n] ?? "#000000"),
});

const store = {};
globalThis.localStorage = {
  getItem: (k) => store[k] ?? null,
  setItem: (k, v) => { store[k] = String(v); },
};
globalThis.window = globalThis;
globalThis.devicePixelRatio = 1;
globalThis.requestAnimationFrame = () => 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = ((f, ms = 0, ...a) =>
  timers.push({ at: CLOCK.now + ms, f, a }) && 0);

const timers = [];
const CLOCK = { now: 0 };
const AUDIO = { currentTime: 0 };
// The wall clock and the audio clock advance together, as they do in a browser.
// Advancing only the timers made every caption look late, which is a bug in the
// harness rather than in the page.
function advance(ms) {
  CLOCK.now += ms;
  AUDIO.currentTime += ms / 1000;
  const due = timers.filter((t) => t.at <= CLOCK.now).sort((a, b) => a.at - b.at);
  for (const t of due) { timers.splice(timers.indexOf(t), 1); t.f(...t.a); }
}

// A fake audio clock. Buffers are scheduled ahead on `playhead`, exactly as in
// the browser, so the caption timing under test is the real timing.
globalThis.AudioContext = class {
  get currentTime() { return AUDIO.currentTime; }
  createBuffer(_ch, n, rate) { return { duration: n / rate, getChannelData: () => new Float32Array(n) }; }
  createBufferSource() { return { buffer: null, connect() {}, start() {} }; }
  createAnalyser() { return { fftSize: 0, frequencyBinCount: 8, connect() {}, getByteTimeDomainData() {} }; }
  createGain() { return { gain: {}, connect: () => ({ connect() {} }) }; }
  createMediaStreamSource() { return { connect() {} }; }
  get audioWorklet() { return { addModule: async () => {} }; }
  close() {}
};
globalThis.AudioWorkletNode = class {
  constructor() { this.port = {}; }
  connect() { return { connect() {} }; }
  disconnect() {}
};
// Node 22 defines navigator as a getter-only global, so it has to be replaced
// rather than assigned.
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
  configurable: true,
});

let SOCKET = null;
globalThis.WebSocket = class {
  static OPEN = 1;
  constructor() { this.readyState = 1; SOCKET = this; this.sent = []; }
  send(d) { this.sent.push(d); }
};
globalThis.location = { host: "127.0.0.1:8000" };

const FETCHED = [];
globalThis.fetch = async (url) => {
  FETCHED.push(url);
  const body = url === "/api/topics" ? {
    topics: [{ key: "python", label: "Python", blurb: "Language semantics.", available: 66 },
             { key: "dsa", label: "DSA", blurb: "Complexity.", available: 41 },
             { key: "resume", label: "Résumé", blurb: "Needs a CV.", available: 0 }],
    coverage: { python: { count: 66 }, dsa: { count: 41 } },
    lengths: [5, 8, 12],
    difficulties: [{ value: 2, label: "Junior" }, { value: 3, label: "Mid-level" },
                   { value: 4, label: "Senior" }],
  } : { stats: { sessions: 0 }, sessions: [], topics: [] };
  return { json: async () => body };
};

// ------------------------------------------------------------------ run it --
const src = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const results = [];
const check = (label, ok, detail = "") => {
  results.push([label, ok]);
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${label}${detail ? ` — ${detail}` : ""}`);
};

console.log("\n1. THE MODULE EVALUATES\n" + "=".repeat(72));
try {
  await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
  check("module runs to completion without throwing", true);
} catch (e) {
  check("module runs to completion without throwing", false, `${e.name}: ${e.message}`);
  console.log("\n  (nothing below can run)\n");
  process.exit(2);
}

// The two things the module is supposed to do on load. Both were silently
// skipped by the bug this file exists for.
await new Promise((r) => process.nextTick(r));
await new Promise((r) => process.nextTick(r));
check("opened a WebSocket", SOCKET !== null);
check("fetched the topic list", FETCHED.includes("/api/topics"));

console.log("\n2. THE SETUP SCREEN POPULATES\n" + "=".repeat(72));
const chips = el("topics").children;
console.log(`  topic chips: ${chips.map((c) => stripTags(c.innerHTML)).join(", ") || "(none)"}`);
check("a chip per topic", chips.length === 3, `${chips.length} chips`);
check("an empty topic is disabled", chips[2]?.disabled === true);
check("level options rendered", el("difficulty").children.length === 3);
check("length options rendered", el("length").children.length === 3);
check("bank size shown", /197|107|questions in the bank/.test(el("note").textContent),
      el("note").textContent);

console.log("\n2b. THE THEME TOGGLE\n" + "=".repeat(72));
check("starts light", document.documentElement.dataset.theme === "light",
      document.documentElement.dataset.theme);
el("theme").onclick();
check("toggles to dark without throwing",
      document.documentElement.dataset.theme === "dark");
check("the choice is persisted", store["coach-theme"] === "dark");
el("theme2").onclick();
check("the in-interview toggle works too",
      document.documentElement.dataset.theme === "light");

console.log("\n3. STARTING AN INTERVIEW\n" + "=".repeat(72));
chips[0].onclick();
check("selecting a topic enables Start", el("go").disabled === false);
await el("go").onclick();
await new Promise((r) => setImmediate(r));
const start = JSON.parse(SOCKET.sent.at(-1));
console.log(`  sent: ${JSON.stringify(start)}`);
check("start carries the chosen topic and settings",
      start.type === "start" && start.topics[0] === "python" &&
      start.length === 8 && start.difficulty === 3);

console.log("\n4. CAPTIONS FOLLOW THE AUDIO CLOCK\n" + "=".repeat(72));
const send = (o) => SOCKET.onmessage({ data: JSON.stringify(o) });
const RATE = 22050;
// 1 second of audio per span, so the clock maths is legible.
const audio = () => {
  const buf = new ArrayBuffer(4 + RATE * 2);
  new DataView(buf).setUint32(0, RATE, true);
  SOCKET.onmessage({ data: buf });
};
const caption = () => stripTags(el("caption").innerHTML);

send({ type: "blueprint", title: "Python", questions: new Array(8),
       difficulty_curve: [1] });
send({ type: "state", state: "speaking" });

send({ type: "span", text: "That's right." });   audio();
send({ type: "span", text: "A tuple is fixed." }); audio();

check("nothing is shown before its audio starts", caption() === "", `"${caption()}"`);
advance(10);
check("the first span appears when its audio starts",
      caption() === "That's right.", `"${caption()}"`);
console.log(`  t=0.0s  "${caption()}"`);

// The second span was scheduled a full second later, on the audio clock.
advance(500);
check("the second span has NOT appeared half a second in",
      caption() === "That's right.", `"${caption()}"`);
advance(600);
check("the second span appears when its own audio starts",
      caption() === "That's right. A tuple is fixed.", `"${caption()}"`);
console.log(`  t=1.1s  "${caption()}"`);

// A question replaces the feedback rather than being appended to it.
send({ type: "question", text: "Let's start. What's a closure?", position: 1,
       total: 8, difficulty: 1, decision: { move: "harder", source: "llm" } });
send({ type: "span", text: "Let's start." }); audio();
send({ type: "span", text: "What's a closure?" }); audio();
advance(2500);
check("the question replaces the feedback on screen",
      caption() === "Let's start. What's a closure?", `"${caption()}"`);
console.log(`  t=3.6s  "${caption()}"`);
check("the question is styled as the question, not as feedback",
      /class="q"/.test(el("caption").innerHTML));
check("the level pill shows the difficulty and the move",
      /Basics/.test(el("level").innerHTML) && /↑/.test(el("level").innerHTML),
      stripTags(el("level").innerHTML));
check("progress updated", el("progress").textContent === "1 / 8",
      el("progress").textContent);

console.log("\n5. A DROPPED TURN DOES NOT DESYNC THE CAPTION\n" + "=".repeat(72));
// Spans whose audio never arrives must be discarded, or every later caption
// is offset by one and shows the wrong text against the wrong voice.
send({ type: "span", text: "orphaned, no audio follows" });
send({ type: "state", state: "thinking" });
send({ type: "state", state: "speaking" });
send({ type: "span", text: "Next question." }); audio();
advance(2000);
check("an orphaned span is discarded rather than shown later",
      caption() === "Next question.", `"${caption()}"`);

console.log("\nUI SMOKE EXIT CRITERIA\n" + "=".repeat(72));
for (const [label, ok] of results) console.log(`  [${ok ? "PASS" : "FAIL"}] ${label}`);
process.exit(results.every(([, ok]) => ok) ? 0 : 2);
