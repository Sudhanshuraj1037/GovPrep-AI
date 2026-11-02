"use strict";
/**
 * Integration tests for the authenticated NLP proxy (/api/nlp/*): a real backend process talks over
 * real HTTP to a programmable stand-in for the Python NLP service. The stand-in lets us check the
 * exact request that reaches the service and produce every failure mode on demand.
 * (nlp-real.test.js repeats the key checks against the real FastAPI service.)
 */
const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const jwt = require("jsonwebtoken");
const { startBackend, startStub, request, signup, freePort, sleep } = require("./helpers/harness");

const SECRET = "test-secret-not-a-real-secret";
let stub, app, token;

const echo = (req) => ({ body: { stub: true, method: req.method, path: req.path, query: req.query, body: req.body ?? null } });

before(async () => {
  stub = await startStub(echo);
  app = await startBackend({ env: { NLP_SERVICE_URL: stub.url, NLP_TIMEOUT_MS: "1500", NLP_SLOW_TIMEOUT_MS: "4000" } });
  token = (await signup(app.url)).token;
});
after(async () => {
  await app?.stop();
  await stub?.close();
});

async function withHandler(handler, fn) {
  const previous = stub.handler;
  stub.handler = handler;
  stub.reset();
  try { return await fn(); } finally { stub.handler = previous; stub.reset(); }
}
const call = (method, route, opts = {}) => request(app.url, method, route, { token, ...opts });

// Every proxied route: [name, method, local path, request options, expected upstream request]
const CASES = [
  ["health", "GET", "/api/nlp/health", {}, { method: "GET", path: "/health" }],
  ["preprocess", "POST", "/api/nlp/preprocess", { body: { text: "Which is NOT correct?", normalizer: "none", remove_stopwords: false } },
    { method: "POST", path: "/preprocess", body: { text: "Which is NOT correct?", normalizer: "none", remove_stopwords: false } }],
  ["tfidf classify", "POST", "/api/nlp/classify/tfidf", { body: { texts: ["river delta", "ancient empire"], top_k: 2 } },
    { method: "POST", path: "/classify/tfidf", body: { texts: ["river delta", "ancient empire"], top_k: 2 } }],
  ["lstm classify", "POST", "/api/nlp/classify/lstm", { body: { texts: ["a"] } }, { method: "POST", path: "/classify/lstm", body: { texts: ["a"] } }],
  ["bilstm classify", "POST", "/api/nlp/classify/bilstm", { body: { texts: ["a"], top_k: 5 } }, { method: "POST", path: "/classify/bilstm", body: { texts: ["a"], top_k: 5 } }],
  ["bert classify", "POST", "/api/nlp/classify/bert", { body: { texts: ["a"] } }, { method: "POST", path: "/classify/bert", body: { texts: ["a"] } }],
  ["tfidf metrics", "GET", "/api/nlp/models/tfidf/metrics", {}, { method: "GET", path: "/models/tfidf/metrics" }],
  ["lstm status", "GET", "/api/nlp/models/lstm/status", {}, { method: "GET", path: "/models/lstm/status" }],
  ["lstm metrics", "GET", "/api/nlp/models/lstm/metrics", {}, { method: "GET", path: "/models/lstm/metrics" }],
  ["bilstm status", "GET", "/api/nlp/models/bilstm/status", {}, { method: "GET", path: "/models/bilstm/status" }],
  ["bilstm metrics", "GET", "/api/nlp/models/bilstm/metrics", {}, { method: "GET", path: "/models/bilstm/metrics" }],
  ["bert status", "GET", "/api/nlp/models/bert/status", {}, { method: "GET", path: "/models/bert/status" }],
  ["bert metrics", "GET", "/api/nlp/models/bert/metrics", {}, { method: "GET", path: "/models/bert/metrics" }],
  ["word2vec status", "GET", "/api/nlp/models/word2vec/status", {}, { method: "GET", path: "/models/word2vec/status" }],
  ["word2vec similar", "GET", "/api/nlp/models/word2vec/similar?word=river&top_k=5&variant=cbow", {},
    { method: "GET", path: "/models/word2vec/similar", query: { word: "river", top_k: "5", variant: "cbow" } }],
  ["word2vec similarity", "GET", "/api/nlp/models/word2vec/similarity?word_a=river&word_b=delta", {},
    { method: "GET", path: "/models/word2vec/similarity", query: { word_a: "river", word_b: "delta" } }],
  ["word2vec analogy", "GET", "/api/nlp/models/word2vec/analogy?a=man&b=king&c=woman&top_k=3", {},
    { method: "GET", path: "/models/word2vec/analogy", query: { a: "man", b: "king", c: "woman", top_k: "3" } }],
  ["word2vec vector", "GET", "/api/nlp/models/word2vec/vector?word=river", {},
    { method: "GET", path: "/models/word2vec/vector", query: { word: "river" } }],
  ["word2vec visualize", "POST", "/api/nlp/models/word2vec/visualize", { body: { method: "tsne", words: ["river", "delta"], neighbors_per_word: 2, perplexity: 5 } },
    { method: "POST", path: "/models/word2vec/visualize", body: { method: "tsne", words: ["river", "delta"], neighbors_per_word: 2, perplexity: 5 } }],
  ["comparison", "GET", "/api/nlp/models/comparison?n_bootstrap=200", {}, { method: "GET", path: "/models/comparison", query: { n_bootstrap: "200" } }],
  ["comparison rows", "GET", "/api/nlp/models/comparison/rows", {}, { method: "GET", path: "/models/comparison/rows" }],
];

// ================================================================== authentication
test("every NLP route requires a valid login (no token, garbage, foreign secret, expired) and reaches nothing", async () => {
  stub.reset();
  const me = jwt.decode(token);
  const bad = [undefined, "garbage", jwt.sign({ id: me.id }, "some-other-secret"), jwt.sign({ id: me.id }, SECRET, { expiresIn: -60 })];
  for (const [, method, route, opts] of CASES) {
    for (const t of bad) {
      const r = await request(app.url, method, route, { token: t, body: opts.body });
      assert.equal(r.status, 401, `${method} ${route} with ${String(t).slice(0, 12)}`);
      assert.equal(r.headers.get("x-nlp-proxy"), null);
    }
  }
  assert.equal((await request(app.url, "GET", "/api/nlp")).status, 401);          // the index is protected too
  assert.equal((await request(app.url, "GET", "/api/nlp/anything-unknown")).status, 401);   // unknown routes do not leak either
  assert.equal(stub.requests.length, 0, "an unauthenticated request must never reach the NLP service");
});

// ================================================================== index
test("GET /api/nlp lists the routes and reports configuration, without leaking the service address", async () => {
  const r = await call("GET", "/api/nlp");
  assert.equal(r.status, 200);
  assert.equal(r.json.configured, true);
  assert.deepEqual(r.json.timeoutsMs, { normal: 1500, slow: 4000 });
  assert.equal(r.json.routes.length, CASES.length);
  assert.ok(r.json.routes.every((x) => x.path.startsWith("/api/nlp/") && ["GET", "POST"].includes(x.method)));
  assert.deepEqual(r.json.routes.filter((x) => x.slow).map((x) => x.path).sort(), [
    "/api/nlp/classify/bert", "/api/nlp/models/comparison", "/api/nlp/models/comparison/rows", "/api/nlp/models/word2vec/visualize"]);
  assert.ok(!r.text.includes(String(stub.port)), "the internal service address must not be exposed");
});

// ================================================================== forwarding
for (const [name, method, route, opts, upstream] of CASES) {
  test(`proxies ${name}: forwards exactly the validated request and returns the service's answer`, async () => {
    stub.reset();
    const r = await call(method, route, opts);
    assert.equal(r.status, 200, r.text);
    assert.equal(r.headers.get("x-nlp-proxy"), "forwarded");
    assert.match(r.headers.get("content-type"), /^application\/json/);
    assert.equal(stub.requests.length, 1);
    const got = stub.requests[0];
    assert.equal(got.method, upstream.method);
    assert.equal(got.path, upstream.path);
    assert.deepEqual(got.query, upstream.query ?? {});
    assert.deepEqual(got.body, upstream.body);
    // the client receives the stand-in's own answer (which echoes what it received), nothing added
    assert.deepEqual(r.json, { stub: true, method: upstream.method, path: upstream.path, query: upstream.query ?? {}, body: upstream.body ?? null });
  });
}

test("the answer is forwarded byte for byte, not re-serialised", async () => {
  const raw = '{"b":1e-05,"a":[1.0,2.50,100000000000000000000],"unicode":"\\u00e9 भारत","nested":{"k":null}}';
  await withHandler(() => ({ body: raw }), async () => {
    const r = await call("GET", "/api/nlp/models/tfidf/metrics");
    assert.equal(r.text, raw);
  });
});

test("every status code the service answers with is forwarded unchanged (not_trained, OOV, validation, server error)", async () => {
  const answers = [
    [503, { detail: { status: "not_trained_yet", model: "bert", message: "BERT not trained yet" } }],
    [404, { detail: { status: "word_not_in_vocabulary", words: ["zzz"], variant: "skipgram" } }],
    [422, { detail: [{ loc: ["body", "texts"], msg: "Field required", type: "missing" }] }],
    [404, { detail: { status: "variant_not_trained", variant: "cbow", available: ["skipgram"] } }],
    [500, { detail: "Internal failure inside the NLP service" }],
    [200, { status: "ok", models_found: [] }],
  ];
  for (const [status, body] of answers) {
    await withHandler(() => ({ status, body }), async () => {
      const r = await call("GET", "/api/nlp/models/word2vec/vector?word=zzz");
      assert.equal(r.status, status);
      assert.deepEqual(r.json, body);
      assert.equal(r.headers.get("x-nlp-proxy"), "forwarded");
      assert.equal(r.json.source, undefined, "a forwarded answer must not be mixed with proxy fields");
    });
  }
});

test("optional fields sent as null are dropped, not forwarded as null (the service applies its own default)", async () => {
  stub.reset();
  const r = await call("POST", "/api/nlp/classify/tfidf", { body: { texts: ["a"], top_k: null } });
  assert.equal(r.status, 200);
  assert.deepEqual(stub.requests[0].body, { texts: ["a"] });
  assert.ok(!("top_k" in stub.requests[0].body));
  stub.reset();
  await call("POST", "/api/nlp/models/word2vec/visualize", { body: { method: "pca", variant: null, words: null, top_n: 10, perplexity: null } });
  assert.deepEqual(stub.requests[0].body, { method: "pca", top_n: 10 });
});

test("the service sees no credentials: the user's token and cookies are never forwarded", async () => {
  stub.reset();
  await call("POST", "/api/nlp/classify/tfidf", { body: { texts: ["a"] }, headers: { Cookie: "session=secret-cookie", "X-Api-Key": "k", "X-Forwarded-For": "6.6.6.6" } });
  const h = stub.requests[0].headers;
  assert.equal(h.authorization, undefined);
  assert.equal(h.cookie, undefined);
  assert.equal(h["x-api-key"], undefined);
  assert.equal(h["x-forwarded-for"], undefined);
  assert.equal(h.accept, "application/json");
  assert.equal(h["content-type"], "application/json");
  assert.ok(!stub.requests[0].bodyText.includes(token));
});

test("non-ASCII and special characters survive the query string intact", async () => {
  stub.reset();
  const words = ["भारत", "empire & more=1#x", "a b+c", "ünï", "100%"];
  for (const w of words) {
    const r = await call("GET", `/api/nlp/models/word2vec/vector?word=${encodeURIComponent(w)}`);
    assert.equal(r.status, 200);
    assert.equal(r.json.query.word, w);
  }
});

test("large valid requests pass: 32 texts of 5000 ASCII, Hindi or emoji characters", async () => {
  for (const ch of ["x", "भ", "😀"]) {
    stub.reset();
    const texts = Array(32).fill(ch.repeat(5000));
    const r = await call("POST", "/api/nlp/classify/tfidf", { body: { texts } });
    assert.equal(r.status, 200, `${ch}: ${r.text.slice(0, 120)}`);
    assert.equal(stub.requests[0].body.texts.length, 32);
    assert.equal([...stub.requests[0].body.texts[0]].length, 5000);
  }
});

test("concurrent requests are answered independently", async () => {
  stub.reset();
  await withHandler(async (req) => { await sleep(80); return echo(req); }, async () => {
    const results = await Promise.all(Array.from({ length: 12 }, (_, i) => call("POST", "/api/nlp/classify/tfidf", { body: { texts: [`text-${i}`] } })));
    results.forEach((r, i) => { assert.equal(r.status, 200); assert.deepEqual(r.json.body.texts, [`text-${i}`]); });
  });
});

// ================================================================== invalid input
const INVALID = [
  ["POST", "/api/nlp/classify/tfidf", { texts: [] }, /texts/],
  ["POST", "/api/nlp/classify/tfidf", {}, /'texts' is required/],
  ["POST", "/api/nlp/classify/tfidf", { texts: "one text" }, /array/],
  ["POST", "/api/nlp/classify/tfidf", { texts: ["a"], top_k: 11 }, /top_k/],
  ["POST", "/api/nlp/classify/tfidf", { texts: ["a"], top_k: "3" }, /top_k/],
  ["POST", "/api/nlp/classify/tfidf", { texts: ["a"], extra: 1 }, /Unknown field 'extra'/],
  ["POST", "/api/nlp/classify/tfidf", { texts: ["x".repeat(5001)] }, /5000/],
  ["POST", "/api/nlp/classify/tfidf", { texts: Array(33).fill("a") }, /1-32/],
  ["POST", "/api/nlp/classify/bert", { texts: [123] }, /must be a string/],
  ["POST", "/api/nlp/classify/lstm", { texts: [""] }, /1-5000/],
  ["POST", "/api/nlp/classify/bilstm", [], /JSON object/],
  ["POST", "/api/nlp/preprocess", { text: "" }, /1-20000/],
  ["POST", "/api/nlp/preprocess", { text: "a", normalizer: "bogus" }, /normalizer/],
  ["POST", "/api/nlp/models/word2vec/visualize", { method: "umap" }, /method/],
  ["POST", "/api/nlp/models/word2vec/visualize", { words: [] }, /words/],
  ["POST", "/api/nlp/models/word2vec/visualize", { perplexity: 1 }, /perplexity/],
  ["POST", "/api/nlp/models/word2vec/visualize", { top_n: 501 }, /top_n/],
  ["GET", "/api/nlp/models/word2vec/similar", undefined, /'word' is required/],
  ["GET", "/api/nlp/models/word2vec/similar?word=", undefined, /1-100/],
  ["GET", `/api/nlp/models/word2vec/similar?word=${"x".repeat(101)}`, undefined, /1-100/],
  ["GET", "/api/nlp/models/word2vec/similar?word=a&top_k=0", undefined, /top_k/],
  ["GET", "/api/nlp/models/word2vec/similar?word=a&top_k=1.5", undefined, /top_k/],
  ["GET", "/api/nlp/models/word2vec/similar?word=a&variant=glove", undefined, /variant/],
  ["GET", "/api/nlp/models/word2vec/similar?word=a&word=b", undefined, /exactly once/],
  ["GET", "/api/nlp/models/word2vec/similar?word[x]=1", undefined, /exactly once/],
  ["GET", "/api/nlp/models/word2vec/similar?word=a&evil=1", undefined, /Unknown query parameter 'evil'/],
  ["GET", "/api/nlp/models/word2vec/similarity?word_a=a", undefined, /word_b/],
  ["GET", "/api/nlp/models/word2vec/analogy?a=a&b=b", undefined, /'c' is required/],
  ["GET", "/api/nlp/models/word2vec/vector?word=a&top_k=1", undefined, /Unknown query parameter/],
  ["GET", "/api/nlp/models/comparison?n_bootstrap=5001", undefined, /n_bootstrap/],
  ["GET", "/api/nlp/models/comparison/rows?n_bootstrap=-1", undefined, /n_bootstrap/],
  ["GET", "/api/nlp/health?x=1", undefined, /this route takes none/],
  ["GET", "/api/nlp/models/bert/status?verbose=1", undefined, /this route takes none/],
];

test("invalid input is refused with a clear 400 and never reaches the NLP service", async () => {
  stub.reset();
  for (const [method, route, body, expected] of INVALID) {
    const r = await call(method, route, { body });
    assert.equal(r.status, 400, `${method} ${route} ${JSON.stringify(body)?.slice(0, 60)} -> ${r.status} ${r.text.slice(0, 100)}`);
    assert.equal(r.json.code, "invalid_request");
    assert.equal(r.json.source, "node-proxy");
    assert.match(r.json.error, expected);
    assert.equal(r.headers.get("x-nlp-proxy"), null);
  }
  assert.equal(stub.requests.length, 0);
});

test("malformed JSON, wrong content type, empty body and oversized body are refused", async () => {
  stub.reset();
  const json = await call("POST", "/api/nlp/classify/tfidf", { headers: { "Content-Type": "application/json" }, raw: "{bad json" });
  assert.deepEqual([json.status, json.json.code, json.json.source], [400, "invalid_json", "node-proxy"]);
  const text = await call("POST", "/api/nlp/classify/tfidf", { headers: { "Content-Type": "text/plain" }, raw: '{"texts":["a"]}' });
  assert.deepEqual([text.status, text.json.code], [415, "unsupported_media_type"]);
  const empty = await call("POST", "/api/nlp/classify/tfidf", {});
  assert.deepEqual([empty.status, empty.json.code], [415, "unsupported_media_type"]);
  const huge = await call("POST", "/api/nlp/classify/tfidf", { body: { texts: ["a"], pad: "x".repeat(1200 * 1024) } });
  assert.deepEqual([huge.status, huge.json.code], [413, "payload_too_large"]);
  assert.equal(stub.requests.length, 0);
});

test("unknown routes and wrong HTTP methods are a JSON 404 (after login)", async () => {
  for (const [method, route, body] of [["GET", "/api/nlp/classify/tfidf"], ["POST", "/api/nlp/health", {}], ["DELETE", "/api/nlp/models/comparison"],
    ["GET", "/api/nlp/models/unknown/status"], ["POST", "/api/nlp/classify/gpt", { texts: ["a"] }], ["PUT", "/api/nlp/preprocess", { text: "a" }]]) {
    const r = await call(method, route, { body });
    assert.equal(r.status, 404, `${method} ${route}`);
    assert.deepEqual([r.json.code, r.json.source], ["not_found", "node-proxy"]);
  }
});

test("CORS preflight to /api/nlp works for the browser (before login, as browsers do)", async () => {
  const r = await request(app.url, "OPTIONS", "/api/nlp/classify/tfidf", { headers: {
    Origin: "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type" } });
  assert.ok(r.status === 204 || r.status === 200);
  assert.equal(r.headers.get("access-control-allow-origin"), "*");
  assert.match(r.headers.get("access-control-allow-headers") || "", /authorization/i);
});

// ================================================================== the service misbehaves
test("a slow answer from the service is a 504 timeout, within the configured limit", async () => {
  await withHandler(async (req) => { await sleep(2500); return echo(req); }, async () => {
    const t0 = Date.now();
    const r = await call("GET", "/api/nlp/models/word2vec/status");
    assert.equal(r.status, 504);
    assert.deepEqual([r.json.code, r.json.source], ["nlp_service_timeout", "node-proxy"]);
    assert.equal(r.json.timeoutMs, 1500);
    const elapsed = Date.now() - t0;
    assert.ok(elapsed >= 1400 && elapsed < 2400, `timed out after ${elapsed} ms`);
  });
});

test("slow routes (BERT, t-SNE, comparison) get the longer timeout", async () => {
  await withHandler(async (req) => { await sleep(2500); return echo(req); }, async () => {
    for (const [method, route, body] of [["POST", "/api/nlp/classify/bert", { texts: ["a"] }], ["POST", "/api/nlp/models/word2vec/visualize", { method: "tsne" }],
      ["GET", "/api/nlp/models/comparison"], ["GET", "/api/nlp/models/comparison/rows"]]) {
      const r = await call(method, route, { body });
      assert.equal(r.status, 200, `${route}: ${r.text.slice(0, 100)}`);       // 2.5 s > 1.5 s normal limit, < 4 s slow limit
    }
    assert.equal((await call("POST", "/api/nlp/classify/tfidf", { body: { texts: ["a"] } })).status, 504);   // normal route: too slow
  });
  await withHandler(async (req) => { await sleep(4500); return echo(req); }, async () => {
    assert.equal((await call("GET", "/api/nlp/models/comparison")).status, 504);   // even slow routes have a limit
  });
});

test("a service that starts answering and then stalls is also a 504", async () => {
  await withHandler(() => ({ hang: true }), async () => {
    const r = await call("GET", "/api/nlp/models/tfidf/metrics");
    assert.deepEqual([r.status, r.json.code], [504, "nlp_service_timeout"]);
  });
});

test("a dropped connection is a 503 nlp_service_unavailable", async () => {
  await withHandler(() => ({ destroy: true }), async () => {
    const r = await call("GET", "/api/nlp/health");
    assert.equal(r.status, 503);
    assert.deepEqual([r.json.code, r.json.source], ["nlp_service_unavailable", "node-proxy"]);
    assert.ok(r.json.reason);
  });
});

test("answers that are not valid JSON become a 502 and are never passed on", async () => {
  const garbage = [
    { status: 500, headers: { "Content-Type": "text/plain" }, body: "Internal Server Error" },
    { status: 200, body: "<html><body>proxy error</body></html>" },
    { status: 200, body: "" },
    { status: 200, body: '{"truncated": ' },
  ];
  for (const g of garbage) {
    await withHandler(() => g, async () => {
      const r = await call("GET", "/api/nlp/health");
      assert.equal(r.status, 502, r.text.slice(0, 100));
      assert.deepEqual([r.json.code, r.json.source], ["nlp_service_bad_response", "node-proxy"]);
      assert.equal(r.headers.get("x-nlp-proxy"), null);
      assert.ok(!r.text.includes("Internal Server Error") && !r.text.includes("proxy error"));
    });
  }
});

test("an oversized answer (more than 5 MB) is refused with 502", async () => {
  await withHandler(() => ({ body: JSON.stringify({ blob: "x".repeat(6 * 1024 * 1024) }) }), async () => {
    const r = await call("GET", "/api/nlp/models/comparison");
    assert.deepEqual([r.status, r.json.code], [502, "nlp_service_bad_response"]);
    assert.match(r.json.error, /too large/);
  });
});

test("redirects from the service are not followed", async () => {
  await withHandler((req) => (req.path === "/health"
    ? { status: 302, headers: { Location: `${stub.url}/elsewhere` }, body: {} }
    : { body: { reached: true } }), async () => {
    const r = await call("GET", "/api/nlp/health");
    assert.equal(r.status, 503);
    assert.equal(r.json.code, "nlp_service_unavailable");
    assert.ok(stub.requests.every((q) => q.path !== "/elsewhere"), "the redirect target must never be requested");
  });
});

test("user text is not written to the server log", async () => {
  const secret = "SENSITIVE-STUDENT-TEXT-12345";
  await withHandler(() => ({ destroy: true }), async () => {
    await call("POST", "/api/nlp/classify/tfidf", { body: { texts: [secret] } });
  });
  assert.ok(!app.logs().includes(secret));
  assert.match(app.logs(), /\[nlp\] POST \/classify\/tfidf -> nlp_service_unavailable/);
});

// ================================================================== service not reachable / not configured
test("service stopped: 503 nlp_service_unavailable, and the rest of the app keeps working", async () => {
  const dead = await freePort();                                   // nothing listens here
  const b = await startBackend({ env: { NLP_SERVICE_URL: `http://127.0.0.1:${dead}`, NLP_TIMEOUT_MS: "2000" } });
  try {
    const u = await signup(b.url);
    const r = await request(b.url, "POST", "/api/nlp/classify/tfidf", { token: u.token, body: { texts: ["a"] } });
    assert.equal(r.status, 503);
    assert.deepEqual([r.json.code, r.json.source, r.json.reason], ["nlp_service_unavailable", "node-proxy", "ECONNREFUSED"]);
    assert.match(r.json.error, /not reachable/);
    for (const route of ["/api/nlp/health", "/api/nlp/models/word2vec/status", "/api/nlp/models/comparison"]) {
      assert.equal((await request(b.url, "GET", route, { token: u.token })).status, 503, route);
    }
    // invalid input is still a 400 even when the service is down
    assert.equal((await request(b.url, "POST", "/api/nlp/classify/tfidf", { token: u.token, body: { texts: [] } })).status, 400);
    // existing features are unaffected
    assert.equal((await request(b.url, "GET", "/api/auth/me", { token: u.token })).status, 200);
    assert.equal((await request(b.url, "GET", "/api/pdfs", { token: u.token })).status, 200);
    assert.equal((await request(b.url, "GET", "/api/health")).json.status, "ok");
  } finally { await b.stop(); }
});

test("NLP_SERVICE_URL not set or invalid: 503 nlp_service_not_configured, startup is not blocked", async () => {
  for (const url of [undefined, "", "not a url", "ftp://example.org", "localhost:8001"]) {
    const b = await startBackend({ env: { NLP_SERVICE_URL: url } });
    try {
      const u = await signup(b.url);
      const r = await request(b.url, "GET", "/api/nlp/models/word2vec/status", { token: u.token });
      assert.equal(r.status, 503, `url=${url}`);
      assert.deepEqual([r.json.code, r.json.source], ["nlp_service_not_configured", "node-proxy"]);
      assert.match(r.json.error, /NLP_SERVICE_URL/);
      const index = await request(b.url, "GET", "/api/nlp", { token: u.token });
      assert.equal(index.json.configured, false);
      assert.ok(index.json.problem);
      assert.match(b.logs(), /NLP service not configured/);
      assert.equal((await request(b.url, "GET", "/api/health")).status, 200);
    } finally { await b.stop(); }
  }
});

test("the service URL may carry a path prefix and a trailing slash; timeouts come from the environment", async () => {
  const prefixed = await startStub(echo);
  const b = await startBackend({ env: { NLP_SERVICE_URL: `${prefixed.url}/nlp-api///`, NLP_TIMEOUT_MS: "abc", NLP_SLOW_TIMEOUT_MS: "5" } });
  try {
    const u = await signup(b.url);
    const r = await request(b.url, "GET", "/api/nlp/models/word2vec/similar?word=x", { token: u.token });
    assert.equal(r.status, 200);
    assert.equal(prefixed.requests[0].path, "/nlp-api/models/word2vec/similar");
    const index = await request(b.url, "GET", "/api/nlp", { token: u.token });
    assert.deepEqual(index.json.timeoutsMs, { normal: 15000, slow: 60000 });     // invalid / out-of-range values fall back to the defaults
  } finally { await b.stop(); await prefixed.close(); }
});
