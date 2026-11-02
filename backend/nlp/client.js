"use strict";
/**
 * HTTP client for the Python NLP service (Node 18+ global fetch).
 *
 * Contract: it returns the upstream response UNCHANGED (status and raw JSON text) whenever the
 * service answered with JSON, whatever the status code. It never fabricates a body. Everything
 * that is not a real upstream answer becomes an NlpProxyError with a stable ``code``:
 *
 *   nlp_service_unavailable  503  could not connect / connection dropped
 *   nlp_service_timeout      504  no complete answer within the timeout
 *   nlp_service_bad_response 502  the service answered with something that is not valid JSON, or too large
 *
 * Security: only the configured base URL plus a fixed, whitelisted path is ever requested;
 * redirects are not followed; the caller's Authorization header and cookies are never forwarded.
 */

class NlpProxyError extends Error {
  constructor(code, status, message, extra = {}) {
    super(message);
    this.name = "NlpProxyError";
    this.code = code;
    this.status = status;
    this.extra = extra;
  }
}

function describeNetworkError(err) {
  return (err && (err.cause?.code || err.code)) || (err && err.name) || "unknown";
}

async function readLimited(res, maxBytes) {
  if (!res.body) return "";
  const reader = res.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > maxBytes) {
      try { await reader.cancel(); } catch { /* ignore */ }
      throw new NlpProxyError("nlp_service_bad_response", 502, "The NLP service response is too large.");
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * @param {object} config    from readConfig()
 * @param {{method:string, path:string, query?:object, body?:object, slow?:boolean}} call
 * @returns {Promise<{status:number, text:string}>}
 */
async function callUpstream(config, { method, path, query, body, slow = false }) {
  if (typeof fetch !== "function") {
    throw new NlpProxyError("nlp_service_unavailable", 503, "This Node.js version has no fetch(); Node 18 or newer is required.");
  }
  const timeoutMs = slow ? config.slowTimeoutMs : config.timeoutMs;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let url = config.baseUrl + path;
  if (query && Object.keys(query).length) url += "?" + new URLSearchParams(query).toString();

  const headers = { Accept: "application/json" };
  const init = { method, headers, signal: controller.signal, redirect: "error" };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  try {
    let res, text;
    try {
      res = await fetch(url, init);
      text = await readLimited(res, config.maxResponseBytes);
    } catch (err) {
      if (err instanceof NlpProxyError) throw err;
      if (controller.signal.aborted) {
        throw new NlpProxyError("nlp_service_timeout", 504,
          `The NLP service did not answer within ${timeoutMs} ms.`, { timeoutMs });
      }
      throw new NlpProxyError("nlp_service_unavailable", 503,
        "The NLP service is not reachable. Is it running and is NLP_SERVICE_URL correct?",
        { reason: describeNetworkError(err) });
    }
    try {
      JSON.parse(text);
    } catch {
      throw new NlpProxyError("nlp_service_bad_response", 502,
        "The NLP service answered with something that is not valid JSON.", { upstreamStatus: res.status });
    }
    return { status: res.status, text };
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { callUpstream, NlpProxyError };
