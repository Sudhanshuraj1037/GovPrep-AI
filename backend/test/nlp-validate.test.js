"use strict";
// Unit tests for the proxy's input validation (backend/nlp/validate.js). Pure functions, no server.
// The limits must mirror the FastAPI service (nlp_service/app/api_nlp.py and api_word2vec.py).
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { validateBody, validateQuery, specs } = require("../nlp/validate");

const body = (spec, value) => validateBody(value, specs[spec]);
const query = (spec, value) => validateQuery(value, specs[spec]);
const ok = (r) => { assert.equal(r.ok, true, r.error); return r.value; };
const bad = (r, fragment) => { assert.equal(r.ok, false, "expected a validation error"); assert.match(r.error, fragment); };

// ------------------------------------------------------------------ classify body
test("classify: texts is required and must hold 1-32 strings", () => {
  assert.deepEqual(ok(body("classifyBody", { texts: ["a"] })), { texts: ["a"] });
  ok(body("classifyBody", { texts: Array(32).fill("x") }));
  bad(body("classifyBody", {}), /'texts' is required/);
  bad(body("classifyBody", { texts: [] }), /'texts' must be an array of 1-32 items/);
  bad(body("classifyBody", { texts: Array(33).fill("x") }), /1-32 items/);
  bad(body("classifyBody", { texts: "abc" }), /must be an array/);
  bad(body("classifyBody", { texts: [1] }), /item 0 must be a string/);
  bad(body("classifyBody", { texts: ["ok", null] }), /item 1 must be a string/);
});

test("classify: each text is 1-5000 characters, counted in code points like Python", () => {
  ok(body("classifyBody", { texts: ["x".repeat(5000)] }));
  bad(body("classifyBody", { texts: ["x".repeat(5001)] }), /1-5000 characters/);
  bad(body("classifyBody", { texts: [""] }), /1-5000 characters/);
  ok(body("classifyBody", { texts: ["भ".repeat(5000)] }));                  // Hindi
  ok(body("classifyBody", { texts: ["😀".repeat(5000)] }));                 // 5000 code points (10000 UTF-16 units)
  bad(body("classifyBody", { texts: ["😀".repeat(5001)] }), /1-5000 characters/);
});

test("classify: top_k is an integer from 1 to 10; null means 'use the default'", () => {
  assert.equal(ok(body("classifyBody", { texts: ["a"], top_k: 1 })).top_k, 1);
  assert.equal(ok(body("classifyBody", { texts: ["a"], top_k: 10 })).top_k, 10);
  for (const v of [0, 11, 1.5, "3", -1, NaN, Infinity, true]) bad(body("classifyBody", { texts: ["a"], top_k: v }), /'top_k' must be an integer from 1 to 10/);
  assert.deepEqual(ok(body("classifyBody", { texts: ["a"], top_k: null })), { texts: ["a"] });   // dropped, not forwarded as null
});

test("bodies must be plain objects with known fields only", () => {
  for (const v of [null, undefined, [], "x", 5, true]) bad(body("classifyBody", v), /must be a JSON object/);
  bad(body("classifyBody", { texts: ["a"], model: "bert" }), /Unknown field 'model'/);
  bad(body("classifyBody", { texts: ["a"], text: "typo" }), /Unknown field 'text'/);
  // inherited / prototype keys are not "known fields"
  bad(body("classifyBody", JSON.parse('{"texts":["a"],"__proto__":{"x":1}}')), /Unknown field '__proto__'/);
  bad(body("classifyBody", { texts: ["a"], constructor: 1 }), /Unknown field 'constructor'/);
  bad(body("classifyBody", { texts: ["a"], toString: 1 }), /Unknown field 'toString'/);
});

test("only validated, whitelisted fields come out", () => {
  const v = ok(body("visualizeBody", { method: "tsne", words: ["river"], top_n: null, perplexity: 5 }));
  assert.deepEqual(v, { method: "tsne", words: ["river"], perplexity: 5 });
});

// ------------------------------------------------------------------ preprocess body
test("preprocess: text 1-20000 characters, optional normalizer and remove_stopwords", () => {
  assert.deepEqual(ok(body("preprocessBody", { text: "Hello", normalizer: "stem", remove_stopwords: false })),
    { text: "Hello", normalizer: "stem", remove_stopwords: false });
  ok(body("preprocessBody", { text: "x".repeat(20000) }));
  bad(body("preprocessBody", { text: "x".repeat(20001) }), /1-20000/);
  bad(body("preprocessBody", { text: "" }), /1-20000/);
  bad(body("preprocessBody", {}), /'text' is required/);
  bad(body("preprocessBody", { text: "a", normalizer: "porter" }), /one of: none, stem, lemma/);
  bad(body("preprocessBody", { text: "a", remove_stopwords: "yes" }), /true or false/);
});

// ------------------------------------------------------------------ word2vec visualize body
test("visualize: method, variant, words, top_n, neighbors_per_word, perplexity", () => {
  assert.deepEqual(ok(body("visualizeBody", {})), {});                          // everything optional
  for (const m of ["pca", "tsne"]) ok(body("visualizeBody", { method: m }));
  bad(body("visualizeBody", { method: "umap" }), /one of: pca, tsne/);
  bad(body("visualizeBody", { variant: "glove" }), /one of: cbow, skipgram/);
  ok(body("visualizeBody", { words: Array(500).fill("w") }));
  bad(body("visualizeBody", { words: Array(501).fill("w") }), /1-500 items/);
  bad(body("visualizeBody", { words: [] }), /1-500 items/);
  bad(body("visualizeBody", { words: ["x".repeat(101)] }), /item 0 must be 1-100/);
  ok(body("visualizeBody", { top_n: 500 }));
  for (const v of [0, 501, 2.5]) bad(body("visualizeBody", { top_n: v }), /integer from 1 to 500/);
  ok(body("visualizeBody", { neighbors_per_word: 0 }));
  ok(body("visualizeBody", { neighbors_per_word: 20 }));
  bad(body("visualizeBody", { neighbors_per_word: 21 }), /integer from 0 to 20/);
  bad(body("visualizeBody", { neighbors_per_word: -1 }), /integer from 0 to 20/);
  ok(body("visualizeBody", { perplexity: 1.01 }));
  ok(body("visualizeBody", { perplexity: 100 }));
  for (const v of [1, 0.5, 100.1, "30", NaN]) bad(body("visualizeBody", { perplexity: v }), /greater than 1 and at most 100/);
});

// ------------------------------------------------------------------ query strings
test("similar: word required (1-100), top_k 1-50, variant", () => {
  assert.deepEqual(ok(query("similarQuery", { word: "river", top_k: "5", variant: "cbow" })), { word: "river", top_k: "5", variant: "cbow" });
  bad(query("similarQuery", {}), /'word' is required/);
  bad(query("similarQuery", { word: "" }), /'word' must be 1-100/);
  ok(query("similarQuery", { word: "x".repeat(100) }));
  bad(query("similarQuery", { word: "x".repeat(101) }), /1-100/);
  ok(query("similarQuery", { word: "w", top_k: "50" }));
  for (const v of ["0", "51", "1.5", "abc", "", "1e1", " 5", "5 ", "--1"]) bad(query("similarQuery", { word: "w", top_k: v }), /'top_k' must be an integer from 1 to 50/);
  bad(query("similarQuery", { word: "w", variant: "x" }), /one of: cbow, skipgram/);
  ok(query("similarQuery", { word: "भारत" }));                                  // Hindi
});

test("similarity, analogy, vector: required words and bounds", () => {
  ok(query("similarityQuery", { word_a: "a", word_b: "b" }));
  bad(query("similarityQuery", { word_a: "a" }), /'word_b' is required/);
  ok(query("analogyQuery", { a: "a", b: "b", c: "c", top_k: "20" }));
  bad(query("analogyQuery", { a: "a", b: "b", c: "c", top_k: "21" }), /integer from 1 to 20/);
  bad(query("analogyQuery", { a: "a", b: "b" }), /'c' is required/);
  ok(query("vectorQuery", { word: "w", variant: "skipgram" }));
  bad(query("vectorQuery", {}), /'word' is required/);
});

test("comparison: n_bootstrap is an integer from 0 to 5000", () => {
  assert.deepEqual(ok(query("comparisonQuery", {})), {});
  for (const v of ["0", "1000", "5000"]) ok(query("comparisonQuery", { n_bootstrap: v }));
  for (const v of ["-1", "5001", "x", "10.5"]) bad(query("comparisonQuery", { n_bootstrap: v }), /integer from 0 to 5000/);
});

test("query strings: unknown, repeated and nested parameters are refused", () => {
  bad(query("similarQuery", { word: "w", extra: "1" }), /Unknown query parameter 'extra'/);
  bad(query("noQuery", { anything: "1" }), /this route takes none/);
  ok(query("noQuery", {}));
  bad(query("similarQuery", { word: ["a", "b"] }), /exactly once/);               // ?word=a&word=b
  bad(query("similarQuery", { word: { x: "1" } }), /exactly once/);               // ?word[x]=1 (qs nesting)
  bad(query("similarQuery", { constructor: "x", word: "w" }), /Unknown query parameter 'constructor'/);
  bad(query("similarQuery", null), /Invalid query string/);
});
