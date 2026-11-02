"use strict";
/**
 * Input validation for the NLP proxy. The limits mirror the FastAPI service exactly
 * (nlp_service/app/api_nlp.py and api_word2vec.py), so a request that would be refused there is
 * refused here first, with a clear 400 and without a network round trip.
 *
 * Only validated, whitelisted fields are ever forwarded. Unknown fields are rejected, never
 * passed through. Lengths are counted in Unicode code points (as Python does), so Hindi and
 * emoji are measured the same way on both sides.
 */

const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key); // not "in": "constructor" etc. are not fields
const codePoints = (s) => { let n = 0; for (const _ of s) n++; return n; }; // eslint-disable-line no-unused-vars

// A rule returns an error text, or null when the value is acceptable.
const rules = {
  str: (min, max) => (v) =>
    typeof v !== "string" ? "must be a string"
      : codePoints(v) < min || codePoints(v) > max ? `must be ${min}-${max} characters long` : null,
  int: (min, max) => (v) =>
    typeof v !== "number" || !Number.isInteger(v) || v < min || v > max ? `must be an integer from ${min} to ${max}` : null,
  numGt: (gt, le) => (v) =>
    typeof v !== "number" || !Number.isFinite(v) || v <= gt || v > le ? `must be a number greater than ${gt} and at most ${le}` : null,
  bool: () => (v) => (typeof v === "boolean" ? null : "must be true or false"),
  oneOf: (values) => (v) => (values.includes(v) ? null : `must be one of: ${values.join(", ")}`),
  array: (min, max, item) => (v) => {
    if (!Array.isArray(v) || v.length < min || v.length > max) return `must be an array of ${min}-${max} items`;
    for (let i = 0; i < v.length; i++) {
      const bad = item(v[i]);
      if (bad) return `item ${i} ${bad}`;
    }
    return null;
  },
};

const fail = (error) => ({ ok: false, error });

/** Validate a JSON body against ``spec = { field: { required, rule } }``. */
function validateBody(body, spec) {
  if (!isPlainObject(body)) return fail("Request body must be a JSON object.");
  for (const key of Object.keys(body)) {
    if (!has(spec, key)) return fail(`Unknown field '${key}'. Allowed: ${Object.keys(spec).join(", ")}.`);
  }
  const value = {};
  for (const [field, { required, rule }] of Object.entries(spec)) {
    const v = body[field];
    if (v === undefined || v === null) {
      if (required) return fail(`'${field}' is required.`);
      continue; // optional and absent: let the NLP service apply its own default
    }
    const bad = rule(v);
    if (bad) return fail(`'${field}' ${bad}.`);
    value[field] = v;
  }
  return { ok: true, value };
}

/** Validate a query string against ``spec = { field: { required, type, ...} }``; values stay strings. */
function validateQuery(query, spec) {
  if (!isPlainObject(query)) return fail("Invalid query string.");
  for (const key of Object.keys(query)) {
    if (!has(spec, key)) {
      const allowed = Object.keys(spec);
      return fail(allowed.length ? `Unknown query parameter '${key}'. Allowed: ${allowed.join(", ")}.` : `Unknown query parameter '${key}'; this route takes none.`);
    }
  }
  const value = {};
  for (const [field, def] of Object.entries(spec)) {
    const raw = query[field];
    if (raw === undefined) {
      if (def.required) return fail(`Query parameter '${field}' is required.`);
      continue;
    }
    if (typeof raw !== "string") return fail(`Query parameter '${field}' must be given exactly once as a plain value.`);
    let parsed = raw;
    if (def.type === "int") {
      if (!/^-?\d+$/.test(raw)) return fail(`Query parameter '${field}' must be an integer from ${def.min} to ${def.max}.`);
      parsed = Number(raw);
    }
    const bad = def.rule(parsed);
    if (bad) return fail(`Query parameter '${field}' ${bad}.`);
    value[field] = raw;
  }
  return { ok: true, value };
}

const word = rules.str(1, 100);
const variant = rules.oneOf(["cbow", "skipgram"]);

const specs = {
  classifyBody: {
    texts: { required: true, rule: rules.array(1, 32, rules.str(1, 5000)) },
    top_k: { rule: rules.int(1, 10) },
  },
  preprocessBody: {
    text: { required: true, rule: rules.str(1, 20000) },
    normalizer: { rule: rules.oneOf(["none", "stem", "lemma"]) },
    remove_stopwords: { rule: rules.bool() },
  },
  visualizeBody: {
    method: { rule: rules.oneOf(["pca", "tsne"]) },
    variant: { rule: variant },
    words: { rule: rules.array(1, 500, word) },
    top_n: { rule: rules.int(1, 500) },
    neighbors_per_word: { rule: rules.int(0, 20) },
    perplexity: { rule: rules.numGt(1, 100) },
  },
  noQuery: {},
  similarQuery: {
    word: { required: true, type: "str", rule: word },
    top_k: { type: "int", min: 1, max: 50, rule: rules.int(1, 50) },
    variant: { type: "str", rule: variant },
  },
  similarityQuery: {
    word_a: { required: true, type: "str", rule: word },
    word_b: { required: true, type: "str", rule: word },
    variant: { type: "str", rule: variant },
  },
  analogyQuery: {
    a: { required: true, type: "str", rule: word },
    b: { required: true, type: "str", rule: word },
    c: { required: true, type: "str", rule: word },
    top_k: { type: "int", min: 1, max: 20, rule: rules.int(1, 20) },
    variant: { type: "str", rule: variant },
  },
  vectorQuery: {
    word: { required: true, type: "str", rule: word },
    variant: { type: "str", rule: variant },
  },
  comparisonQuery: {
    n_bootstrap: { type: "int", min: 0, max: 5000, rule: rules.int(0, 5000) },
  },
};

module.exports = { validateBody, validateQuery, specs, rules };
