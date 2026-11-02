"use strict";
/**
 * End-to-end tests against the REAL Python NLP service (FastAPI + trained TF-IDF and Word2Vec models).
 *
 * The test trains tiny models on synthetic data in a temp folder (never in nlp_service/artifacts),
 * starts the real service, then compares what the Node proxy returns with what the service returns
 * when called directly: they must be identical. It also checks that Node's input validation and the
 * service's own validation agree at every limit.
 *
 * Skipped (with a stated reason) when Python or the service's packages are not available. To point at a
 * specific interpreter:  NLP_PYTHON=/path/to/python node test/run-tests.js real
 */
const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { startBackend, request, signup, freePort, BACKEND_ROOT, sleep } = require("./helpers/harness");

const NLP_DIR = path.resolve(BACKEND_ROOT, "..", "nlp_service");
const REQUIRED = "fastapi, uvicorn, sklearn, gensim, pandas, nltk, joblib";

function findPython() {
  if (!fs.existsSync(path.join(NLP_DIR, "app", "main.py"))) return { reason: `nlp_service not found at ${NLP_DIR}` };
  const candidates = [
    process.env.NLP_PYTHON,
    path.join(NLP_DIR, ".venv-nlp", "Scripts", "python.exe"),
    path.join(NLP_DIR, ".venv-nlp", "bin", "python"),
    "python3", "python", "py",
  ].filter(Boolean);
  for (const cmd of candidates) {
    const r = spawnSync(cmd, ["-c", `import ${REQUIRED}`], { encoding: "utf8", timeout: 60000 });
    if (r.status === 0) return { cmd };
  }
  return { reason: `no Python with the NLP service packages (${REQUIRED}); set NLP_PYTHON to the nlp_service virtualenv's python` };
}
const PY = findPython();
const SKIP = PY.cmd ? false : PY.reason;

let work, service, servicePort, app, token;

const POOLS = {
  History: "empire dynasty emperor battle ancient kingdom medieval treaty colony revolt".split(" "),
  Geography: "river mountain plateau monsoon delta latitude climate soil desert coast".split(" "),
  Science: "atom molecule enzyme voltage velocity photon cell reaction gravity orbit".split(" "),
  Polity: "constitution article parliament amendment judiciary federal preamble election bill statute".split(" "),
};
const FILLER = "which of the following is correct about the".split(" ");

function syntheticCsv() {
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const pick = (arr, n) => { const a = [...arr]; const out = []; while (out.length < n) out.push(a.splice(Math.floor(rnd() * a.length), 1)[0]); return out; };
  const rows = ["text,subject,topic,difficulty"];
  for (const [subject, pool] of Object.entries(POOLS)) {
    for (let i = 0; i < 100; i++) {
      const words = [...pick(pool, 4), ...pick(FILLER, 3)].sort(() => rnd() - 0.5);
      rows.push(`${words.join(" ")} case${subject.slice(0, 2).toLowerCase()}${i},${subject},,`);
    }
  }
  return rows.join("\n") + "\n";
}

function runPython(args, env) {
  const r = spawnSync(PY.cmd, args, { cwd: NLP_DIR, env: { ...process.env, ...env, PYTHONPATH: NLP_DIR }, encoding: "utf8", timeout: 240000 });
  if (r.status !== 0) throw new Error(`python ${args.join(" ")} failed (${r.status}):\n${r.stdout}\n${r.stderr}`);
}

before(async () => {
  if (SKIP) return;
  work = fs.mkdtempSync(path.join(os.tmpdir(), "govprep-nlp-e2e-"));
  const env = { NLP_DATA_DIR: path.join(work, "data"), NLP_ARTIFACTS_DIR: path.join(work, "artifacts") };
  fs.mkdirSync(path.join(work, "data", "public"), { recursive: true });
  fs.writeFileSync(path.join(work, "data", "public", "synthetic.csv"), syntheticCsv());
  runPython(["-m", "training.prepare_data", "--input", path.join(work, "data", "public", "synthetic.csv")], env);
  runPython(["-m", "training.train_tfidf"], env);
  runPython(["-m", "training.train_word2vec", "--epochs", "100", "--vector-size", "50"], env);

  servicePort = await freePort();
  const out = [];
  service = spawn(PY.cmd, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(servicePort)], {
    cwd: NLP_DIR, env: { ...process.env, ...env, PYTHONPATH: NLP_DIR }, stdio: ["ignore", "pipe", "pipe"] });
  service.stdout.on("data", (d) => out.push(String(d)));
  service.stderr.on("data", (d) => out.push(String(d)));
  const deadline = Date.now() + 90000;
  for (;;) {
    try { if ((await fetch(`http://127.0.0.1:${servicePort}/health`)).ok) break; } catch { /* starting */ }
    if (Date.now() > deadline) throw new Error(`NLP service did not start:\n${out.join("")}`);
    await sleep(250);
  }
  app = await startBackend({ env: { NLP_SERVICE_URL: `http://127.0.0.1:${servicePort}`, NLP_TIMEOUT_MS: "30000", NLP_SLOW_TIMEOUT_MS: "90000" } });
  token = (await signup(app.url)).token;
});

after(async () => {
  await app?.stop();
  if (service) { service.kill(); await new Promise((r) => service.once("exit", r)); }
  if (work) fs.rmSync(work, { recursive: true, force: true });
});

const direct = async (method, route, body) => {
  const res = await fetch(`http://127.0.0.1:${servicePort}${route}`, {
    method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  const text = await res.text();
  return { status: res.status, text, json: JSON.parse(text) };
};
const viaNode = (method, route, body) => request(app.url, method, `/api/nlp${route}`, { token, body });
const T = { skip: SKIP, timeout: 120000 };

// Same answer through the proxy as straight from the service (byte for byte), for deterministic routes.
const IDENTICAL = [
  ["preprocess", "POST", "/preprocess", { text: "Which of the following is NOT a Mauryan ruler? (Page 3 of 70)" }],
  ["tfidf classify", "POST", "/classify/tfidf", { texts: ["the river delta and monsoon climate", "ancient empire battle dynasty", "zzz qqq"], top_k: 3 }],
  ["tfidf metrics", "GET", "/models/tfidf/metrics"],
  ["word2vec status", "GET", "/models/word2vec/status"],
  ["word2vec similar", "GET", "/models/word2vec/similar?word=river&top_k=4"],
  ["word2vec similar (cbow)", "GET", "/models/word2vec/similar?word=empire&top_k=3&variant=cbow"],
  ["word2vec similarity", "GET", "/models/word2vec/similarity?word_a=river&word_b=delta"],
  ["word2vec analogy", "GET", "/models/word2vec/analogy?a=river&b=delta&c=empire&top_k=3"],
  ["word2vec vector", "GET", "/models/word2vec/vector?word=river"],
  ["word2vec OOV (404)", "GET", "/models/word2vec/vector?word=zzznotaword"],
  ["word2vec visualize (PCA)", "POST", "/models/word2vec/visualize", { method: "pca", top_n: 20 }],
  ["word2vec visualize (t-SNE)", "POST", "/models/word2vec/visualize", { method: "tsne", words: ["river", "delta", "empire", "dynasty", "atom", "photon", "constitution", "parliament"], perplexity: 2 }],
  ["comparison (one model)", "GET", "/models/comparison?n_bootstrap=100"],
  ["comparison rows", "GET", "/models/comparison/rows?n_bootstrap=0"],
  ["lstm status (not trained)", "GET", "/models/lstm/status"],
  ["lstm metrics (404)", "GET", "/models/lstm/metrics"],
  ["lstm classify (503 not trained)", "POST", "/classify/lstm", { texts: ["river"] }],
  ["bilstm classify (503 not trained)", "POST", "/classify/bilstm", { texts: ["river"] }],
  ["bert status (not trained)", "GET", "/models/bert/status"],
  ["bert classify (503 not trained)", "POST", "/classify/bert", { texts: ["river"] }],
];

for (const [name, method, route, body] of IDENTICAL) {
  test(`real service: ${name} is identical through the proxy and direct`, T, async () => {
    const d = await direct(method, route, body);
    const n = await viaNode(method, route, body);
    assert.equal(n.status, d.status, `${route}: ${n.text.slice(0, 200)}`);
    assert.equal(n.text, d.text);                                        // byte for byte
    assert.equal(n.headers.get("x-nlp-proxy"), "forwarded");
  });
}

test("real service: the proxied predictions are the model's, not placeholders", T, async () => {
  const r = await viaNode("POST", "/classify/tfidf", { texts: ["the river delta and monsoon climate", "ancient empire battle dynasty", "atom photon voltage"], top_k: 2 });
  assert.equal(r.status, 200);
  assert.equal(r.json.model, "tfidf");
  assert.deepEqual(r.json.results.map((x) => x.predictions.subject.label), ["Geography", "History", "Science"]);
  assert.ok(r.json.results.every((x) => x.predictions.subject.confidence > 0 && x.predictions.subject.top.length === 2));
  const metrics = await viaNode("GET", "/models/tfidf/metrics");
  assert.equal(metrics.json.model, "tfidf");
  assert.equal(metrics.json.seed, 42);
  assert.ok(metrics.json.targets.subject.splits.test.confusion_matrix);
  const status = await viaNode("GET", "/models/word2vec/status");
  assert.deepEqual([status.json.trained, status.json.variants_available], [true, ["cbow", "skipgram"]]);
  const vec = await viaNode("GET", "/models/word2vec/vector?word=river");
  assert.equal(vec.json.dimension, 50);
  assert.equal(vec.json.vector.length, 50);
});

test("real service: models that are not trained report it truthfully, nothing is invented", T, async () => {
  for (const m of ["lstm", "bilstm", "bert"]) {
    const s = await viaNode("GET", `/models/${m}/status`);
    assert.deepEqual([s.status, s.json.trained, s.json.targets_trained], [200, false, []]);
    const c = await viaNode("POST", `/classify/${m}`, { texts: ["river"] });
    assert.equal(c.status, 503);
    assert.equal(c.json.detail.status, "not_trained_yet");
    assert.equal(c.json.source, undefined);                               // the service's own answer, forwarded
    assert.equal((await viaNode("GET", `/models/${m}/metrics`)).status, 404);
  }
  const cmp = (await viaNode("GET", "/models/comparison?n_bootstrap=0")).json;
  assert.deepEqual([cmp.status, cmp.models_found, cmp.models_missing], ["single_model", ["tfidf"], ["lstm", "bilstm", "bert"]]);
});

test("real service: /health passes through (volatile fields aside) and reflects reality", T, async () => {
  const d = (await direct("GET", "/health")).json;
  const n = (await viaNode("GET", "/health")).json;
  for (const f of ["time", "uptime_seconds"]) { delete d[f]; delete n[f]; }
  assert.deepEqual(n, d);
  const trained = Object.fromEntries(n.models.map((m) => [m.name, m.trained]));
  assert.deepEqual(trained, { tfidf: true, word2vec: true, lstm: false, bilstm: false, bert: false });
});

// ---- Node validation and the service's own validation must agree at every limit
const text = (n, ch = "x") => ch.repeat(n);
const AGREEMENT = [
  // [label, method, route, body, expected]  accepted = service answers 200; refused = service 422 and Node 400
  ["classify: 1 text", "POST", "/classify/tfidf", { texts: ["river"] }, "accepted"],
  ["classify: 32 texts", "POST", "/classify/tfidf", { texts: Array(32).fill("river") }, "accepted"],
  ["classify: 33 texts", "POST", "/classify/tfidf", { texts: Array(33).fill("river") }, "refused"],
  ["classify: no texts", "POST", "/classify/tfidf", { texts: [] }, "refused"],
  ["classify: 5000 characters", "POST", "/classify/tfidf", { texts: [text(5000)] }, "accepted"],
  ["classify: 5001 characters", "POST", "/classify/tfidf", { texts: [text(5001)] }, "refused"],
  ["classify: 5000 emoji (10000 UTF-16 units)", "POST", "/classify/tfidf", { texts: [text(5000, "😀")] }, "accepted"],
  ["classify: 5001 emoji", "POST", "/classify/tfidf", { texts: [text(5001, "😀")] }, "refused"],
  ["classify: 5000 Hindi letters", "POST", "/classify/tfidf", { texts: [text(5000, "भ")] }, "accepted"],
  ["classify: empty string", "POST", "/classify/tfidf", { texts: [""] }, "refused"],
  ["classify: top_k 1", "POST", "/classify/tfidf", { texts: ["river"], top_k: 1 }, "accepted"],
  ["classify: top_k 10", "POST", "/classify/tfidf", { texts: ["river"], top_k: 10 }, "accepted"],
  ["classify: top_k 0", "POST", "/classify/tfidf", { texts: ["river"], top_k: 0 }, "refused"],
  ["classify: top_k 11", "POST", "/classify/tfidf", { texts: ["river"], top_k: 11 }, "refused"],
  ["preprocess: 20000 characters", "POST", "/preprocess", { text: text(20000) }, "accepted"],
  ["preprocess: 20001 characters", "POST", "/preprocess", { text: text(20001) }, "refused"],
  ["preprocess: empty", "POST", "/preprocess", { text: "" }, "refused"],
  ["preprocess: normalizer none", "POST", "/preprocess", { text: "river", normalizer: "none" }, "accepted"],
  ["preprocess: bad normalizer", "POST", "/preprocess", { text: "river", normalizer: "porter" }, "refused"],
  ["visualize: top_n 10 (PCA)", "POST", "/models/word2vec/visualize", { method: "pca", top_n: 10 }, "accepted"],
  ["visualize: top_n 501", "POST", "/models/word2vec/visualize", { method: "pca", top_n: 501 }, "refused"],
  ["visualize: top_n 0", "POST", "/models/word2vec/visualize", { method: "pca", top_n: 0 }, "refused"],
  ["visualize: bad method", "POST", "/models/word2vec/visualize", { method: "umap" }, "refused"],
  ["visualize: neighbors 21", "POST", "/models/word2vec/visualize", { words: ["river", "delta", "empire"], neighbors_per_word: 21 }, "refused"],
  ["visualize: neighbors 20", "POST", "/models/word2vec/visualize", { words: ["river", "delta", "empire"], neighbors_per_word: 20 }, "accepted"],
  ["visualize: perplexity 1", "POST", "/models/word2vec/visualize", { method: "tsne", top_n: 20, perplexity: 1 }, "refused"],
  ["visualize: perplexity 100.5", "POST", "/models/word2vec/visualize", { method: "tsne", top_n: 20, perplexity: 100.5 }, "refused"],
  ["visualize: empty words", "POST", "/models/word2vec/visualize", { words: [] }, "refused"],
  ["similar: top_k 50", "GET", "/models/word2vec/similar?word=river&top_k=50", undefined, "accepted"],
  ["similar: top_k 51", "GET", "/models/word2vec/similar?word=river&top_k=51", undefined, "refused"],
  ["similar: top_k 0", "GET", "/models/word2vec/similar?word=river&top_k=0", undefined, "refused"],
  ["similar: word 100 characters is valid input (not in vocabulary: 404)", "GET", `/models/word2vec/similar?word=${text(100)}`, undefined, "reaches-service"],
  ["similar: word 101 characters", "GET", `/models/word2vec/similar?word=${text(101)}`, undefined, "refused"],
  ["similar: no word", "GET", "/models/word2vec/similar", undefined, "refused"],
  ["similar: bad variant", "GET", "/models/word2vec/similar?word=river&variant=glove", undefined, "refused"],
  ["analogy: top_k 20", "GET", "/models/word2vec/analogy?a=river&b=delta&c=empire&top_k=20", undefined, "accepted"],
  ["analogy: top_k 21", "GET", "/models/word2vec/analogy?a=river&b=delta&c=empire&top_k=21", undefined, "refused"],
  ["comparison: n_bootstrap 0", "GET", "/models/comparison?n_bootstrap=0", undefined, "accepted"],
  ["comparison: n_bootstrap 5000", "GET", "/models/comparison?n_bootstrap=5000", undefined, "accepted"],
  ["comparison: n_bootstrap 5001", "GET", "/models/comparison?n_bootstrap=5001", undefined, "refused"],
  ["comparison: n_bootstrap -1", "GET", "/models/comparison?n_bootstrap=-1", undefined, "refused"],
];

for (const [label, method, route, body, expected] of AGREEMENT) {
  test(`validation agrees with the service: ${label}`, T, async () => {
    const d = await direct(method, route, body);
    const n = await viaNode(method, route, body);
    if (expected === "accepted") {
      assert.equal(d.status, 200, `service: ${d.text.slice(0, 160)}`);
      assert.equal(n.status, 200, `node: ${n.text.slice(0, 160)}`);
      assert.equal(n.text, d.text);
    } else if (expected === "refused") {
      assert.equal(d.status, 422, `the service should refuse this: ${d.status} ${d.text.slice(0, 160)}`);
      assert.equal(n.status, 400, `node should refuse this: ${n.status} ${n.text.slice(0, 160)}`);
      assert.equal(n.json.source, "node-proxy");
    } else {                                                             // valid input that the model then answers itself (e.g. unknown word)
      assert.equal(n.status, d.status);
      assert.equal(n.text, d.text);
      assert.notEqual(n.status, 400);
    }
  });
}
