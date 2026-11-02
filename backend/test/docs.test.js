"use strict";
// Keeps the documentation honest: the route table in docs/NLP_INTEGRATION.md must match the router, and
// backend/.env.example must cover every environment variable the backend reads, with the code's defaults.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { buildRoutes } = require("../nlp/router");
const { DEFAULTS } = require("../nlp/config");

const ROOT = path.resolve(__dirname, "..", "..");
const read = (...p) => fs.readFileSync(path.join(ROOT, ...p), "utf8");
const DOC = read("docs", "NLP_INTEGRATION.md");
const ENV_EXAMPLE = read("backend", ".env.example");

function docRoutes() {
  const rows = [];
  for (const line of DOC.split("\n")) {
    const cells = line.split("|").map((c) => c.trim());
    if (cells.length < 6 || !/^(GET|POST)$/.test(cells[1])) continue;
    const route = (cells[2].match(/^`([^`]+)`$/) || [])[1];
    if (!route || !route.startsWith("/api/nlp")) continue;
    rows.push({ method: cells[1], route, forwards: (cells[3].match(/^`([^`]+)`$/) || [])[1] || null, notes: cells[4] });
  }
  return rows;
}

test("every route in the guide exists in the router, and every router route is in the guide", () => {
  const documented = docRoutes();
  const actual = buildRoutes().map((r) => ({ method: r.method.toUpperCase(), route: `/api/nlp${r.path}`, forwards: r.upstream, slow: r.slow }));
  const key = (r) => `${r.method} ${r.route}`;
  assert.deepEqual(documented.map(key).filter((k) => k !== "GET /api/nlp").sort(), actual.map(key).sort());
  assert.ok(documented.some((r) => key(r) === "GET /api/nlp"), "the index route is documented");
  for (const a of actual) {
    const d = documented.find((r) => key(r) === key(a));
    assert.equal(d.forwards, a.forwards, `${key(a)}: the guide must name the real upstream route`);
    assert.equal(/\bslow\b/i.test(d.notes), a.slow, `${key(a)}: the guide must mark exactly the slow routes as slow`);
  }
});

test("the error table lists every code the proxy can produce", () => {
  const source = ["router.js", "client.js"].map((f) => fs.readFileSync(path.join(__dirname, "..", "nlp", f), "utf8")).join("\n");
  const codes = new Set([...source.matchAll(/"((?:nlp_service|invalid|not_found|payload|unsupported)[a-z_]*)"/g)].map((m) => m[1]));
  for (const code of ["invalid_request", "invalid_json", "not_found", "payload_too_large", "unsupported_media_type", "nlp_service_not_configured",
    "nlp_service_unavailable", "nlp_service_timeout", "nlp_service_bad_response"]) {
    assert.ok(codes.has(code), `${code} should exist in the code`);
    assert.ok(DOC.includes(`\`${code}\``), `${code} should be documented`);
  }
});

test(".env.example documents every environment variable the backend reads, with the code's defaults", () => {
  const used = new Set();
  for (const f of ["server.js", "auth.js", "database.js", path.join("nlp", "config.js"), path.join("nlp", "router.js")]) {
    for (const m of fs.readFileSync(path.join(__dirname, "..", f), "utf8").matchAll(/process\.env\.([A-Z_]+)|env\.([A-Z_]{4,})/g)) used.add(m[1] || m[2]);
  }
  used.delete("NODE_OPTIONS");
  assert.ok(used.has("GROQ_API_KEY") && used.has("JWT_SECRET") && used.has("PORT") && used.has("NLP_SERVICE_URL"), [...used].join(","));
  for (const name of used) assert.match(ENV_EXAMPLE, new RegExp(`^#?\\s*${name}=`, "m"), `${name} must appear in .env.example`);

  const values = Object.fromEntries([...ENV_EXAMPLE.matchAll(/^([A-Z_]+)=(.*)$/gm)].map((m) => [m[1], m[2].trim()]));
  assert.equal(Number(values.NLP_TIMEOUT_MS), DEFAULTS.timeoutMs);
  assert.equal(Number(values.NLP_SLOW_TIMEOUT_MS), DEFAULTS.slowTimeoutMs);
  assert.equal(Number(values.NLP_MAX_BODY_KB), DEFAULTS.maxBodyKb);
  assert.equal(new URL(values.NLP_SERVICE_URL).protocol, "http:");
  assert.match(values.JWT_SECRET, /change-me/);                       // a placeholder, not a secret
  assert.match(values.GROQ_API_KEY, /your-/);
});
