"use strict";
/**
 * Regression tests for the EXISTING GovPrep-AI backend features (auth, PDF upload, question
 * generation with Groq, test scoring, results, assistant). They are black-box HTTP tests against
 * a copy of the backend, with Groq replaced by a local stub, so they need no network.
 *
 * They are written so they can run unchanged against the original (pre-NLP) backend:
 *     GOVPREP_BACKEND_DIR=/path/to/original/backend node test/run-tests.js regression
 * and must pass on both: that is how "the NLP integration changed nothing else" is checked.
 */
const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const jwt = require("jsonwebtoken");
const { startBackend, startGroqStub, request, signup, readFixturePdf, uploadPdf, SOURCE_DIR } = require("./helpers/harness");

const SECRET = "test-secret-not-a-real-secret";
let groq, app, appNoKey;

before(async () => {
  groq = await startGroqStub();
  app = await startBackend({ groqUrl: groq.url });
  // KNOWN PRE-EXISTING QUIRK (also true of the original server.js): with GROQ_API_KEY unset the OpenAI SDK falls
  // back to OPENAI_API_KEY and throws at start-up if that is missing too, so the server only boots without a Groq
  // key when a placeholder OPENAI_API_KEY exists. Used here to exercise the route-level "key not configured" checks.
  appNoKey = await startBackend({ env: { GROQ_API_KEY: undefined, OPENAI_API_KEY: "placeholder-so-the-sdk-can-start" } });
});
after(async () => {
  await app?.stop();
  await appNoKey?.stop();
  await groq?.close();
});

async function userWithPdf(base = app.url) {
  const u = await signup(base);
  const up = await uploadPdf(base, u.token, readFixturePdf());
  assert.equal(up.status, 200, up.text);
  return { ...u, pdfId: up.json.pdf.id };
}

async function generate(u, extra = {}) {
  return request(app.url, "POST", "/api/generate-questions", { token: u.token, body: { pdfId: u.pdfId, questionCount: 3, ...extra } });
}

test("regression target: which backend source is being tested", () => {
  console.log(`# backend under test: ${SOURCE_DIR}`);
});

// ------------------------------------------------------------------ health + generic behaviour
test("GET /api/health reports status, Groq key presence and user count", async () => {
  const r = await request(app.url, "GET", "/api/health");
  assert.equal(r.status, 200);
  assert.equal(r.json.status, "ok");
  assert.equal(r.json.groqKeyConfigured, true);
  assert.equal(typeof r.json.users, "number");
  assert.equal((await request(appNoKey.url, "GET", "/api/health")).json.groqKeyConfigured, false);
});

test("CORS preflight on an existing route still allows any origin", async () => {
  const r = await request(app.url, "OPTIONS", "/api/health", { headers: { Origin: "http://localhost:3000", "Access-Control-Request-Method": "GET" } });
  assert.ok(r.status === 204 || r.status === 200);
  assert.equal(r.headers.get("access-control-allow-origin"), "*");
});

test("unknown routes are 404", async () => {
  assert.equal((await request(app.url, "GET", "/api/does-not-exist")).status, 404);
});

test("the default 100 KB JSON limit still applies to the existing routes", async () => {
  const u = await signup(app.url);
  const r = await request(app.url, "POST", "/api/assistant/chat", { token: u.token, body: { message: "x".repeat(150 * 1024) } });
  assert.equal(r.status, 413);
});

// ------------------------------------------------------------------ authentication
test("signup creates an account and returns a verifiable token", async () => {
  const r = await request(app.url, "POST", "/api/auth/signup", { body: { name: "  Asha  ", password: "pw1234", examTarget: "SSC CGL Tier 1" } });
  assert.equal(r.status, 200);
  assert.equal(r.json.success, true);
  assert.equal(r.json.user.name, "Asha");                       // trimmed
  assert.equal(r.json.user.examTarget, "SSC CGL Tier 1");
  const claims = jwt.verify(r.json.token, SECRET);
  assert.equal(claims.id, r.json.user.id);
  assert.equal(claims.name, "Asha");
  const defaulted = await signup(app.url);
  assert.equal(defaulted.user.examTarget, "Railway RRB NTPC");  // default exam target
});

test("signup validation: name required, password length, duplicate names", async () => {
  const bad = (body) => request(app.url, "POST", "/api/auth/signup", { body });
  assert.deepEqual([(await bad({ password: "pw1234" })).status, (await bad({ name: "   ", password: "pw1234" })).json.error], [400, "Name is required."]);
  const short = await bad({ name: "Short", password: "abc" });
  assert.deepEqual([short.status, short.json.error], [400, "Password must be at least 4 characters."]);
  assert.equal((await bad({ name: "Dup", password: "pw1234" })).status, 200);
  const dup = await bad({ name: "Dup", password: "other1" });
  assert.deepEqual([dup.status, dup.json.error], [409, "Name already taken. Choose another or log in."]);
});

test("login: success, unknown account, wrong password, missing fields", async () => {
  const u = await signup(app.url, "LoginUser", "mypass1");
  const ok = await request(app.url, "POST", "/api/auth/login", { body: { name: "LoginUser", password: "mypass1" } });
  assert.equal(ok.status, 200);
  assert.equal(ok.json.user.id, u.user.id);
  assert.ok(jwt.verify(ok.json.token, SECRET));
  const cases = [
    [{ name: "Nobody", password: "x1234" }, 401, "No account with this name."],
    [{ name: "LoginUser", password: "wrong" }, 401, "Incorrect password."],
    [{ name: "LoginUser" }, 400, "Name and password required."],
    [{}, 400, "Name and password required."],
  ];
  for (const [body, status, error] of cases) {
    const r = await request(app.url, "POST", "/api/auth/login", { body });
    assert.deepEqual([r.status, r.json.error], [status, error]);
  }
});

test("GET /api/auth/me returns the profile; bad, expired and foreign tokens are 401", async () => {
  const u = await signup(app.url);
  const me = await request(app.url, "GET", "/api/auth/me", { token: u.token });
  assert.equal(me.status, 200);
  assert.equal(me.json.user.id, u.user.id);
  assert.equal(me.json.user.name, u.name);
  for (const token of [undefined, "garbage", jwt.sign({ id: u.user.id }, "another-secret"), jwt.sign({ id: u.user.id }, SECRET, { expiresIn: -10 })]) {
    assert.equal((await request(app.url, "GET", "/api/auth/me", { token })).status, 401, String(token).slice(0, 20));
  }
  const ghost = jwt.sign({ id: "no-such-user", name: "x" }, SECRET);
  assert.equal((await request(app.url, "GET", "/api/auth/me", { token: ghost })).status, 404);
});

test("every protected route answers 401 without a token", async () => {
  const routes = [
    ["GET", "/api/auth/me"], ["POST", "/api/upload"], ["GET", "/api/pdfs"], ["DELETE", "/api/pdfs/abc"],
    ["POST", "/api/generate-questions"], ["POST", "/api/submit-test"], ["GET", "/api/my-tests"],
    ["POST", "/api/assistant/chat"], ["GET", "/api/assistant/history"], ["DELETE", "/api/assistant/history"],
  ];
  for (const [method, route] of routes) {
    const r = await request(app.url, method, route, method === "POST" ? { body: {} } : {});
    assert.equal(r.status, 401, `${method} ${route}`);
  }
});

// ------------------------------------------------------------------ PDF upload
test("upload: a real PDF is parsed and stored; listing shows it; users are isolated", async () => {
  const a = await signup(app.url);
  const b = await signup(app.url);
  const up = await uploadPdf(app.url, a.token, readFixturePdf(), { name: "ancient-india.pdf", subject: "History", chapter: "Mauryas" });
  assert.equal(up.status, 200, up.text);
  assert.equal(up.json.success, true);
  assert.equal(up.json.pdf.name, "ancient-india.pdf");
  assert.equal(up.json.pdf.totalPages, 3);
  assert.equal(up.json.pdf.subject, "History");
  assert.equal(up.json.pdf.chapter, "Mauryas");
  assert.ok(up.json.pdf.wordCount > 100);

  const mine = await request(app.url, "GET", "/api/pdfs", { token: a.token });
  assert.equal(mine.json.pdfs.length, 1);
  assert.equal(mine.json.pdfs[0].id, up.json.pdf.id);
  assert.equal((await request(app.url, "GET", "/api/pdfs", { token: b.token })).json.pdfs.length, 0);
  // another user cannot delete it; the owner can
  assert.equal((await request(app.url, "DELETE", `/api/pdfs/${up.json.pdf.id}`, { token: b.token })).status, 404);
  const del = await request(app.url, "DELETE", `/api/pdfs/${up.json.pdf.id}`, { token: a.token });
  assert.deepEqual([del.status, del.json.success], [200, true]);
  assert.equal((await request(app.url, "GET", "/api/pdfs", { token: a.token })).json.pdfs.length, 0);
  assert.equal((await request(app.url, "DELETE", `/api/pdfs/${up.json.pdf.id}`, { token: a.token })).json.error, "PDF not found");
});

test("upload: missing file is 400 and a non-PDF is refused without being stored", async () => {
  const u = await signup(app.url);
  const none = await request(app.url, "POST", "/api/upload", { token: u.token, raw: new FormData() });
  assert.equal(none.status, 400);
  assert.equal(none.json.error, "No PDF provided");
  const txt = await uploadPdf(app.url, u.token, Buffer.from("just some text, not a pdf"), { name: "notes.txt", type: "text/plain" });
  assert.ok(txt.status >= 400, `a text file must be refused, got ${txt.status}`);
  assert.equal((await request(app.url, "GET", "/api/pdfs", { token: u.token })).json.pdfs.length, 0);
});

// ------------------------------------------------------------------ question generation (Groq stubbed)
test("generate-questions: input validation and missing Groq key", async () => {
  const u = await userWithPdf();
  const post = (body, base = app.url, token = u.token) => request(base, "POST", "/api/generate-questions", { token, body });
  assert.deepEqual([(await post({})).status, (await post({})).json.error], [400, "pdfId required"]);
  assert.equal((await post({ pdfId: "nope" })).status, 404);
  const range = await post({ pdfId: u.pdfId, pageStart: 3, pageEnd: 1 });
  assert.equal(range.status, 400);
  assert.match(range.json.error, /Invalid page range/);
  const other = await signup(app.url);
  assert.equal((await post({ pdfId: u.pdfId }, app.url, other.token)).status, 404);   // not their PDF
  // without a Groq key (separate backend) the route reports it clearly
  const v = await signup(appNoKey.url);
  const up = await uploadPdf(appNoKey.url, v.token, readFixturePdf());
  const nokey = await post({ pdfId: up.json.pdf.id }, appNoKey.url, v.token);
  assert.deepEqual([nokey.status, nokey.json.error], [500, "GROQ_API_KEY not configured."]);
});

test("generate-questions: builds the Groq prompt from the PDF and stores the test", async () => {
  groq.state.mode = "ok";
  groq.reset();
  const u = await userWithPdf();
  const r = await generate(u, { pageStart: 1, pageEnd: 2, difficulty: "hard", examTemplate: "SSC CGL Tier 1" });
  assert.equal(r.status, 200, r.text);
  assert.equal(r.json.success, true);
  assert.equal(r.json.questions.length, 3);
  assert.ok(r.json.testId);
  for (const q of r.json.questions) {
    assert.ok(q.id);
    assert.equal(q.options.length, 4);
    assert.ok(Number.isInteger(q.correct));
  }
  assert.equal(r.json.meta.pdfName, "history-notes.pdf");
  assert.equal(r.json.meta.pageRange, "1–2");
  const sent = groq.requests.at(-1).body;
  assert.match(sent.messages[0].content, /strict JSON API/);
  assert.match(sent.messages[1].content, /Chandragupta/);        // text from the PDF page
  assert.match(sent.messages[1].content, /SSC CGL Tier 1/);
  assert.match(sent.messages[1].content, /All hard\./);
  assert.match(sent.messages[1].content, /Pages: 1-2/);
  assert.doesNotMatch(sent.messages[1].content, /Delhi Sultanate/);   // page 3 was not requested
});

test("generate-questions: Hindi instruction is sent to Groq", async () => {
  groq.reset();
  const u = await userWithPdf();
  assert.equal((await generate(u, { language: "hindi" })).status, 200);
  assert.match(groq.requests.at(-1).body.messages[1].content, /HINDI/);
});

test("generate-questions: unusable AI output and rate limits give clear errors", async () => {
  const u = await userWithPdf();
  groq.state.mode = "bad_json";
  const bad = await generate(u);
  assert.equal(bad.status, 500);
  assert.match(bad.json.error, /did not return valid JSON/);
  groq.state.mode = "rate_limit";
  const limited = await generate(u);
  assert.equal(limited.status, 429);
  assert.equal(typeof limited.json.error, "string");
  groq.state.mode = "ok";
});

// ------------------------------------------------------------------ scoring and results
async function generatedTest(u) {
  groq.state.mode = "ok";
  const g = await generate(u);
  assert.equal(g.status, 200, g.text);
  return g.json;
}

test("submit-test scores with the exam's marking scheme and reports weak topics", async () => {
  const u = await userWithPdf();
  const t = await generatedTest(u);
  const [q1, q2] = t.questions;
  // q1 right (Chandragupta=0), q2 wrong, q3 unattempted. Railway scheme: +1 / -0.33
  const r = await request(app.url, "POST", "/api/submit-test", { token: u.token, body: {
    testId: t.testId, answers: { [q1.id]: q1.correct, [q2.id]: (q2.correct + 1) % 4 }, timeTaken: 95 } });
  assert.equal(r.status, 200, r.text);
  const s = r.json.scorecard;
  assert.deepEqual([s.correct, s.wrong, s.unattempted], [1, 1, 1]);
  assert.equal(s.rawScore, 0.67);
  assert.equal(s.maxScore, 3);
  assert.equal(s.percentage, 22);
  assert.equal(s.timeTaken, 95);
  assert.match(s.recommendation, /Focus on: .*Mauryan Empire/);
  assert.match(s.recommendation, /Medieval India/);

  // SSC CGL scheme: +2 / -0.5 -> 2 - 0.5 = 1.5 of 6 = 25%
  const ssc = await request(app.url, "POST", "/api/submit-test", { token: u.token, body: {
    testId: t.testId, examTemplate: "SSC CGL Tier 1", answers: { [q1.id]: q1.correct, [q2.id]: (q2.correct + 1) % 4 } } });
  assert.deepEqual([ssc.json.scorecard.rawScore, ssc.json.scorecard.maxScore, ssc.json.scorecard.percentage], [1.5, 6, 25]);
});

test("submit-test: unknown test and other users' tests are 404; results are private", async () => {
  const u = await userWithPdf();
  const t = await generatedTest(u);
  const other = await signup(app.url);
  assert.equal((await request(app.url, "POST", "/api/submit-test", { token: u.token, body: { testId: "nope" } })).status, 404);
  assert.equal((await request(app.url, "POST", "/api/submit-test", { token: other.token, body: { testId: t.testId } })).status, 404);
  assert.equal((await request(app.url, "POST", "/api/submit-test", { token: u.token, body: { testId: t.testId, answers: {} } })).status, 200);
  const mine = await request(app.url, "GET", "/api/my-tests", { token: u.token });
  assert.equal(mine.json.results.length, 1);
  assert.equal(mine.json.results[0].unattempted, 3);
  assert.equal(mine.json.results[0].pdf_name, "history-notes.pdf");
  assert.equal((await request(app.url, "GET", "/api/my-tests", { token: other.token })).json.results.length, 0);
});

// ------------------------------------------------------------------ assistant
test("assistant: message required, replies via Groq, keeps per-user history that can be cleared", async () => {
  const u = await userWithPdf();
  const other = await signup(app.url);
  const chat = (token, body) => request(app.url, "POST", "/api/assistant/chat", { token, body });
  assert.deepEqual([(await chat(u.token, { message: "   " })).status, (await chat(u.token, {})).json.error], [400, "Message required."]);

  groq.reset();
  groq.state.reply = "Revise the Mauryan timeline first.";
  const first = await chat(u.token, { message: "How should I start?" });
  assert.deepEqual([first.status, first.json.success, first.json.reply], [200, true, "Revise the Mauryan timeline first."]);
  assert.match(groq.requests.at(-1).body.messages[0].content, /GovPrep Assistant/);
  assert.equal(groq.requests.at(-1).body.messages.at(-1).content, "How should I start?");

  const second = await chat(u.token, { message: "And the pdf?", context: { type: "pdf", pdfId: u.pdfId, pageStart: 1, pageEnd: 1 } });
  assert.equal(second.status, 200);
  const sent = groq.requests.at(-1).body.messages;
  assert.match(sent[0].content, /history-notes\.pdf/);                       // PDF context reached the prompt
  assert.match(sent[0].content, /Chandragupta/);
  assert.ok(sent.some((m) => m.role === "user" && m.content === "How should I start?"));   // earlier turn is replayed
  assert.ok(sent.some((m) => m.role === "assistant" && m.content === "Revise the Mauryan timeline first."));

  const hist = await request(app.url, "GET", "/api/assistant/history", { token: u.token });
  assert.deepEqual(hist.json.messages.map((m) => m.role), ["user", "assistant", "user", "assistant"]);
  assert.equal((await request(app.url, "GET", "/api/assistant/history", { token: other.token })).json.messages.length, 0);
  assert.equal((await request(app.url, "DELETE", "/api/assistant/history", { token: other.token })).status, 200);
  assert.equal((await request(app.url, "GET", "/api/assistant/history", { token: u.token })).json.messages.length, 4);   // untouched
  assert.equal((await request(app.url, "DELETE", "/api/assistant/history", { token: u.token })).json.success, true);
  assert.equal((await request(app.url, "GET", "/api/assistant/history", { token: u.token })).json.messages.length, 0);
});

test("assistant: rate limits and a missing key give clear errors", async () => {
  const u = await signup(app.url);
  groq.state.mode = "rate_limit";
  assert.equal((await request(app.url, "POST", "/api/assistant/chat", { token: u.token, body: { message: "hi" } })).status, 429);
  groq.state.mode = "ok";
  const v = await signup(appNoKey.url);
  const r = await request(appNoKey.url, "POST", "/api/assistant/chat", { token: v.token, body: { message: "hi" } });
  assert.deepEqual([r.status, r.json.error], [500, "GROQ_API_KEY not configured."]);
});
