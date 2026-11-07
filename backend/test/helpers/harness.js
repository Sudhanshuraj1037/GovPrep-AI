"use strict";
/**
 * Test harness shared by the backend tests (Node's built-in test runner; no extra dependencies).
 *
 * startBackend() runs a COPY of the backend in a temp folder, so the real govprep-db.json,
 * uploads/ and .env are never read or written, and the user's real GROQ key is never used.
 * The child process gets a minimal, explicit environment. Groq is redirected to a local stub by a
 * preload, so no test needs network access.
 */
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");

const BACKEND_ROOT = path.resolve(__dirname, "..", "..");
// Which backend source tree to test. Defaults to this one; point it at a pristine copy to compare.
const SOURCE_DIR = path.resolve(process.env.GOVPREP_BACKEND_DIR || BACKEND_ROOT);
const PRELOAD = path.join(__dirname, "preload-openai.js");

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** A programmable HTTP stub. handler(req, bodyText) -> {status, body, headers, delayMs, hang, destroy}. */
async function startStub(handler) {
  const stub = { requests: [], handler };
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      const bodyText = Buffer.concat(chunks).toString("utf8");
      const url = new URL(req.url, "http://stub");
      const record = {
        method: req.method, path: url.pathname, query: Object.fromEntries(url.searchParams),
        rawQuery: url.search, headers: req.headers, bodyText,
        body: bodyText ? safeJson(bodyText) : undefined,
      };
      stub.requests.push(record);
      const out = (await stub.handler(record, req)) || {};
      if (out.destroy) return req.socket.destroy();
      if (out.delayMs) await sleep(out.delayMs);
      if (out.hang) { res.writeHead(out.status || 200, { "Content-Type": "application/json" }); res.write('{"partial":'); return; }
      const body = typeof out.body === "string" || Buffer.isBuffer(out.body) ? out.body : JSON.stringify(out.body ?? {});
      res.writeHead(out.status || 200, { "Content-Type": "application/json", ...(out.headers || {}) });
      res.end(body);
    });
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  stub.port = server.address().port;
  stub.url = `http://127.0.0.1:${stub.port}`;
  stub.close = () => new Promise((r) => { server.closeAllConnections?.(); server.close(r); });
  stub.reset = () => { stub.requests.length = 0; };
  return stub;
}

function safeJson(text) {
  try { return JSON.parse(text); } catch { return undefined; }
}

/** A stub that speaks enough of the OpenAI chat-completions protocol for the Groq routes. */
async function startGroqStub() {
  const state = { mode: "ok", questions: defaultQuestions(), reply: "Stub assistant reply." };
  const stub = await startStub((req) => {
    if (req.method !== "POST" || !req.path.endsWith("/chat/completions")) return { status: 404, body: { error: "not found" } };
    const sys = req.body?.messages?.[0]?.content || "";
    const isGeneration = sys.includes("strict JSON API");
    if (state.mode === "rate_limit") {
      return { status: 429, headers: { "retry-after-ms": "1" }, body: { error: { message: "Rate limit reached for model", type: "rate_limit_exceeded" } } };
    }
    let content;
    if (isGeneration) content = state.mode === "bad_json" ? "this is definitely not json" : JSON.stringify(state.questions);
    else content = state.reply;
    return { body: { id: "chatcmpl-stub", object: "chat.completion", created: 0, model: req.body?.model || "stub",
      choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } } };
  });
  stub.state = state;
  return stub;
}

function defaultQuestions() {
  return [
    { question: "Who founded the Maurya Empire?", options: ["Chandragupta", "Ashoka", "Bindusara", "Akbar"], correct: 0, difficulty: "easy", explanation: "Chandragupta Maurya.", pageRef: 1, topic: "Mauryan Empire" },
    { question: "Which battle was fought in 1526?", options: ["Plassey", "Panipat", "Buxar", "Haldighati"], correct: 1, difficulty: "medium", explanation: "First Battle of Panipat.", pageRef: 1, topic: "Medieval India" },
    { question: "Who wrote the Arthashastra?", options: ["Kautilya", "Kalidasa", "Banabhatta", "Panini"], correct: 0, difficulty: "hard", explanation: "Kautilya.", pageRef: 1, topic: "Mauryan Empire" },
  ];
}

function copyBackend(sourceDir, dest) {
  for (const f of ["server.js", "auth.js", "database.js", "package.json"]) fs.copyFileSync(path.join(sourceDir, f), path.join(dest, f));
  const nlpDir = path.join(sourceDir, "nlp");
  if (fs.existsSync(nlpDir)) fs.cpSync(nlpDir, path.join(dest, "nlp"), { recursive: true });
  const modules = [path.join(sourceDir, "node_modules"), path.join(BACKEND_ROOT, "node_modules")].find((p) => fs.existsSync(p));
  if (!modules) throw new Error("backend/node_modules not found: run `npm install` in backend/ first.");
  fs.symlinkSync(modules, path.join(dest, "node_modules"), "junction");
}

/**
 * Start a copy of the backend. ``env`` entries override the defaults; a value of undefined removes it.
 * Returns {url, dir, logs(), stop()}.
 */
async function startBackend({ env = {}, sourceDir = SOURCE_DIR, groqUrl } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "govprep-backend-"));
  copyBackend(sourceDir, dir);
  const port = await freePort();
  const childEnv = {
    PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: process.env.TEMP, TMP: process.env.TMP,
    HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE,
    PORT: String(port),
    JWT_SECRET: "test-secret-not-a-real-secret",
    GROQ_API_KEY: "test-key-not-real",
    ...(groqUrl ? { TEST_GROQ_BASE_URL: groqUrl } : {}),
    NODE_OPTIONS: `--require "${PRELOAD}"`,
    ...env,
  };
  for (const k of Object.keys(childEnv)) if (childEnv[k] === undefined) delete childEnv[k];

  const out = [];
  const child = spawn(process.execPath, ["server.js"], { cwd: dir, env: childEnv, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", (d) => out.push(String(d)));
  child.stderr.on("data", (d) => out.push(String(d)));
  let exited = false;
  child.on("exit", () => { exited = true; });

  const url = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 20000;
  for (;;) {
    if (exited) throw new Error(`backend exited during start-up:\n${out.join("")}`);
    try { if ((await fetch(`${url}/api/health`)).ok) break; } catch { /* not up yet */ }
    if (Date.now() > deadline) { child.kill(); throw new Error(`backend did not start:\n${out.join("")}`); }
    await sleep(100);
  }
  return {
    url, dir, port, logs: () => out.join(""),
    async stop() {
      if (!exited) { child.kill(); await new Promise((r) => child.once("exit", r)); }
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}

/** fetch wrapper: returns {status, headers, text, json}. */
async function request(base, method, route, { token, body, headers = {}, raw } = {}) {
  const h = { ...headers };
  if (token) h.Authorization = `Bearer ${token}`;
  let payload = raw;
  if (body !== undefined) { h["Content-Type"] = h["Content-Type"] || "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(base + route, { method, headers: h, body: payload });
  const text = await res.text();
  return { status: res.status, headers: res.headers, text, json: safeJson(text) };
}

let counter = 0;
async function signup(base, name = `user${Date.now()}_${counter++}`, password = "secret1") {
  const r = await request(base, "POST", "/api/auth/signup", { body: { name, password } });
  if (r.status !== 200) throw new Error(`signup failed: ${r.status} ${r.text}`);
  return { token: r.json.token, user: r.json.user, name, password };
}

/** A real 3-page PDF made by fpdf2 (test/fixtures/history-notes.pdf).
 *  It must stay above ~4 KB: the old pdf.js bundled in pdf-parse rejects tiny files ("bad XRef entry"). */
const FIXTURE_PDF = path.join(__dirname, "..", "fixtures", "history-notes.pdf");
function readFixturePdf() {
  return fs.readFileSync(FIXTURE_PDF);
}

async function uploadPdf(base, token, pdf, { name = "history-notes.pdf", subject = "History", chapter = "Ancient India", type = "application/pdf", subjectMode } = {}) {
  const form = new FormData();
  form.append("pdf", new Blob([pdf], { type }), name);
  form.append("subject", subject);
  if (subjectMode) form.append("subjectMode", subjectMode);
  form.append("chapter", chapter);
  const res = await fetch(`${base}/api/upload`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form });
  const text = await res.text();
  return { status: res.status, text, json: safeJson(text) };
}

module.exports = {
  BACKEND_ROOT, SOURCE_DIR, startBackend, startStub, startGroqStub, request, signup, readFixturePdf, uploadPdf, freePort, sleep,
};
