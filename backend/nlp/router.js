"use strict";
/**
 * Authenticated Express proxy to the Python NLP service (Phase 8).
 *
 *   React  ->  Node/Express (/api/nlp/*, JWT required)  ->  FastAPI NLP service
 *
 * Every route maps to ONE fixed FastAPI route. The request is validated here (limits mirror the
 * FastAPI service), only whitelisted fields are forwarded, and the service's answer is passed
 * back byte for byte with its own status code: this layer never invents a prediction, a metric
 * or a "not trained" message of its own. Errors that are NOT an answer from the NLP service
 * carry ``source: "node-proxy"`` and a stable ``code`` so a client can tell them apart:
 *
 *   400 invalid_request | invalid_json        the request was refused here
 *   401                                       not logged in (existing auth middleware)
 *   404 not_found                             unknown /api/nlp route or wrong HTTP method
 *   413 payload_too_large | 415 unsupported_media_type
 *   503 nlp_service_not_configured            NLP_SERVICE_URL is not set / invalid
 *   503 nlp_service_unavailable               the service cannot be reached
 *   504 nlp_service_timeout                   the service did not answer in time
 *   502 nlp_service_bad_response              the service answered with invalid JSON
 *
 * An answer FROM the service (200, 404 word_not_in_vocabulary, 503 not_trained_yet, 422 ...) is
 * forwarded unchanged and marked with the response header ``X-NLP-Proxy: forwarded``.
 */
const express = require("express");
const { readConfig } = require("./config");
const { callUpstream, NlpProxyError } = require("./client");
const { validateBody, validateQuery, specs } = require("./validate");

const SLOW = true;

function buildRoutes() {
  const routes = [
    { method: "get", path: "/health", query: specs.noQuery },
    { method: "post", path: "/preprocess", body: specs.preprocessBody },
    { method: "post", path: "/classify/tfidf", body: specs.classifyBody },
    { method: "get", path: "/models/tfidf/metrics", query: specs.noQuery },
    { method: "get", path: "/models/word2vec/status", query: specs.noQuery },
    { method: "get", path: "/models/word2vec/similar", query: specs.similarQuery },
    { method: "get", path: "/models/word2vec/similarity", query: specs.similarityQuery },
    { method: "get", path: "/models/word2vec/analogy", query: specs.analogyQuery },
    { method: "get", path: "/models/word2vec/vector", query: specs.vectorQuery },
    { method: "post", path: "/models/word2vec/visualize", body: specs.visualizeBody, slow: SLOW },
    { method: "get", path: "/models/comparison", query: specs.comparisonQuery, slow: SLOW },
    { method: "get", path: "/models/comparison/rows", query: specs.comparisonQuery, slow: SLOW },
  ];
  for (const model of ["lstm", "bilstm", "bert"]) {
    routes.push(
      { method: "post", path: `/classify/${model}`, body: specs.classifyBody, slow: model === "bert" },
      { method: "get", path: `/models/${model}/status`, query: specs.noQuery },
      { method: "get", path: `/models/${model}/metrics`, query: specs.noQuery },
    );
  }
  return routes.map((r) => ({ ...r, upstream: r.path, slow: !!r.slow }));
}

function sendError(res, status, code, message, extra = {}) {
  return res.status(status).json({ error: message, code, source: "node-proxy", ...extra });
}

function createNlpRouter({ requireAuth, env = process.env, logger = console } = {}) {
  if (typeof requireAuth !== "function") throw new Error("createNlpRouter needs the requireAuth middleware.");
  const config = readConfig(env);
  const routes = buildRoutes();

  if (config.configured) {
    logger.log(`[nlp] NLP service: ${config.origin} (timeouts ${config.timeoutMs}/${config.slowTimeoutMs} ms)`);
  } else {
    logger.warn(`[nlp] NLP service not configured (${config.problem}) - /api/nlp/* will answer 503 until NLP_SERVICE_URL is set.`);
  }

  const router = express.Router();
  router.use(requireAuth);                              // existing JWT auth: nothing below is reachable without it
  router.use(express.json({ limit: config.jsonLimit }));

  // GET /api/nlp : what this proxy offers. Pure configuration facts, no model output.
  router.get("/", (req, res) => {
    res.json({
      configured: config.configured,
      problem: config.configured ? null : config.problem,
      timeoutsMs: { normal: config.timeoutMs, slow: config.slowTimeoutMs },
      routes: routes.map((r) => ({ method: r.method.toUpperCase(), path: `/api/nlp${r.path}`, slow: r.slow })),
    });
  });

  for (const route of routes) {
    router[route.method](route.path, async (req, res) => {
      if (route.method === "post" && !req.is("application/json")) {
        return sendError(res, 415, "unsupported_media_type", "Send a JSON body with Content-Type: application/json.");
      }
      const checked = route.body ? validateBody(req.body, route.body) : validateQuery(req.query, route.query);
      if (!checked.ok) return sendError(res, 400, "invalid_request", `Invalid request: ${checked.error}`);

      if (!config.configured) {
        return sendError(res, 503, "nlp_service_not_configured",
          `The NLP service is not configured on this server (${config.problem}) Set NLP_SERVICE_URL and restart the backend.`);
      }
      try {
        const up = await callUpstream(config, {
          method: route.method.toUpperCase(),
          path: route.upstream,
          query: route.body ? undefined : checked.value,
          body: route.body ? checked.value : undefined,
          slow: route.slow,
        });
        res.status(up.status).set("Content-Type", "application/json; charset=utf-8").set("X-NLP-Proxy", "forwarded").send(up.text);
      } catch (err) {
        if (err instanceof NlpProxyError) {
          logger.error(`[nlp] ${route.method.toUpperCase()} ${route.upstream} -> ${err.code}${err.extra.reason ? ` (${err.extra.reason})` : ""}`);
          return sendError(res, err.status, err.code, err.message, err.extra);
        }
        logger.error("[nlp] unexpected error:", err);
        return sendError(res, 500, "nlp_proxy_error", "Unexpected error in the NLP proxy.");
      }
    });
  }

  router.use((req, res) => sendError(res, 404, "not_found", "Unknown NLP route, or wrong HTTP method for it."));
  router.use((err, req, res, next) => { // body-parser errors, as JSON like every other API error
    if (err && err.type === "entity.parse.failed") return sendError(res, 400, "invalid_json", "The request body is not valid JSON.");
    if (err && err.type === "entity.too.large") return sendError(res, 413, "payload_too_large", `The request body is larger than ${config.jsonLimit}.`);
    return next(err);
  });

  return router;
}

module.exports = { createNlpRouter, buildRoutes };
