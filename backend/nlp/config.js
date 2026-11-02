"use strict";
/**
 * Configuration of the NLP proxy, read from environment variables (never hard-coded).
 *
 *   NLP_SERVICE_URL        base URL of the Python NLP service, e.g. http://127.0.0.1:8001.
 *                          If it is not set, /api/nlp/* answers 503 "nlp_service_not_configured"
 *                          and the rest of GovPrep-AI is unaffected.
 *   NLP_TIMEOUT_MS         timeout for normal calls                     (default 15000)
 *   NLP_SLOW_TIMEOUT_MS    timeout for slow calls: BERT, t-SNE, comparison (default 60000)
 *   NLP_MAX_BODY_KB        maximum JSON request body accepted by /api/nlp (default 1024)
 */

const DEFAULTS = { timeoutMs: 15000, slowTimeoutMs: 60000, maxBodyKb: 1024 };
const MAX_RESPONSE_BYTES = 5 * 1024 * 1024;

function boundedInt(raw, fallback, min, max) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return fallback;
  const n = Number(raw);
  return Number.isInteger(n) && n >= min && n <= max ? n : fallback;
}

function readConfig(env = process.env) {
  const raw = String(env.NLP_SERVICE_URL || "").trim();
  const cfg = {
    configured: false,
    baseUrl: null,
    origin: null,        // what is safe to log (never includes credentials or path)
    problem: null,
    timeoutMs: boundedInt(env.NLP_TIMEOUT_MS, DEFAULTS.timeoutMs, 100, 300000),
    slowTimeoutMs: boundedInt(env.NLP_SLOW_TIMEOUT_MS, DEFAULTS.slowTimeoutMs, 100, 600000),
    jsonLimit: `${boundedInt(env.NLP_MAX_BODY_KB, DEFAULTS.maxBodyKb, 16, 8192)}kb`,
    maxResponseBytes: MAX_RESPONSE_BYTES,
  };
  if (!raw) {
    cfg.problem = "NLP_SERVICE_URL is not set.";
    return cfg;
  }
  let url;
  try {
    url = new URL(raw);
  } catch {
    cfg.problem = "NLP_SERVICE_URL is not a valid URL.";
    return cfg;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    cfg.problem = "NLP_SERVICE_URL must start with http:// or https://.";
    return cfg;
  }
  cfg.configured = true;
  cfg.origin = url.origin;
  cfg.baseUrl = url.origin + url.pathname.replace(/\/+$/, "");
  return cfg;
}

module.exports = { readConfig, DEFAULTS, MAX_RESPONSE_BYTES };
