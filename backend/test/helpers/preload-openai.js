"use strict";
// TEST-ONLY preload (used through NODE_OPTIONS by the harness, never by the real server).
// It redirects the OpenAI client that server.js creates for Groq to a local stub, so the existing
// Groq routes can be exercised without network access and without changing server.js.
// Active only when TEST_GROQ_BASE_URL is set.
const Module = require("node:module");

const originalLoad = Module._load;
let patched = null;
Module._load = function (request) {
  const loaded = originalLoad.apply(this, arguments);
  const base = process.env.TEST_GROQ_BASE_URL;
  if (request !== "openai" || !base) return loaded;
  if (!patched) {
    const Base = loaded;
    class Patched extends Base {
      constructor(opts = {}) {
        super({ ...opts, baseURL: base });
      }
    }
    Object.assign(Patched, Base);
    Patched.default = Patched;
    patched = Patched;
  }
  return patched;
};
