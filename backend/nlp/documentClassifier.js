"use strict";

const { readConfig } = require("./config");
const { callUpstream, NlpProxyError } = require("./client");

const CLASSIFIER_SUBJECTS = Object.freeze(["History", "Geography", "Polity", "Science"]);
const DEFAULTS = Object.freeze({
  maxChunks: 8,
  chunkChars: 1200,
  overlapChars: 160,
  minDocumentChars: 180,
  minConfidence: 0.55,
  minMargin: 0.10,
  minAgreement: 0.60,
});

function boundedNumber(raw, fallback, min, max) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min && n <= max ? n : fallback;
}

function getOptions(env = process.env) {
  return {
    maxChunks: Math.round(boundedNumber(env.NLP_SUBJECT_MAX_CHUNKS, DEFAULTS.maxChunks, 1, 32)),
    chunkChars: Math.round(boundedNumber(env.NLP_SUBJECT_CHUNK_CHARS, DEFAULTS.chunkChars, 250, 4500)),
    overlapChars: Math.round(boundedNumber(env.NLP_SUBJECT_OVERLAP_CHARS, DEFAULTS.overlapChars, 0, 1000)),
    minDocumentChars: Math.round(boundedNumber(env.NLP_SUBJECT_MIN_DOCUMENT_CHARS, DEFAULTS.minDocumentChars, 50, 5000)),
    minConfidence: boundedNumber(env.NLP_SUBJECT_MIN_SCORE, DEFAULTS.minConfidence, 0.25, 0.99),
    minMargin: boundedNumber(env.NLP_SUBJECT_MIN_MARGIN, DEFAULTS.minMargin, 0, 0.8),
    minAgreement: boundedNumber(env.NLP_SUBJECT_MIN_AGREEMENT, DEFAULTS.minAgreement, 0.25, 1),
  };
}

function evenlySample(items, limit) {
  if (items.length <= limit) return items;
  if (limit <= 1) return [items[Math.floor((items.length - 1) / 2)]];
  const chosen = [];
  const used = new Set();
  for (let i = 0; i < limit; i++) {
    const index = Math.round((i * (items.length - 1)) / (limit - 1));
    if (!used.has(index)) {
      chosen.push(items[index]);
      used.add(index);
    }
  }
  return chosen;
}

/** Build bounded, page-aware representative chunks. Source text is never persisted in detection metadata. */
function createTextChunks(pageTexts, options = {}) {
  const cfg = { ...DEFAULTS, ...options };
  const all = [];
  const pages = Array.isArray(pageTexts) ? pageTexts : [];
  let totalChars = 0;

  pages.forEach((raw, index) => {
    const text = String(raw || "").replace(/\u0000/g, " ").replace(/[ \t]+/g, " ").trim();
    totalChars += text.length;
    if (!text) return;

    const step = Math.max(1, cfg.chunkChars - Math.min(cfg.overlapChars, cfg.chunkChars - 1));
    for (let start = 0; start < text.length; start += step) {
      const piece = text.slice(start, start + cfg.chunkChars).trim();
      if (piece.length >= 80) all.push({ page: index + 1, text: piece });
      if (start + cfg.chunkChars >= text.length) break;
    }
  });

  return {
    chunks: evenlySample(all, cfg.maxChunks),
    totalChars,
    candidateChunks: all.length,
    pagesWithText: pages.filter((p) => String(p || "").trim().length > 0).length,
  };
}

function manualRequired(reasonCode, extra = {}) {
  return {
    status: "manual_required",
    model: "bert",
    detectedSubject: extra.detectedSubject || null,
    score: Number.isFinite(extra.score) ? extra.score : null,
    margin: Number.isFinite(extra.margin) ? extra.margin : null,
    agreement: Number.isFinite(extra.agreement) ? extra.agreement : null,
    chunksAnalyzed: extra.chunksAnalyzed || 0,
    pagesAnalyzed: extra.pagesAnalyzed || [],
    reasonCode,
    predictions: extra.predictions || [],
  };
}

/** Aggregate actual BERT class probabilities across chunks; scores are not calibrated confidence. */
function aggregateBertPredictions(response, chunks, options = {}) {
  const cfg = { ...DEFAULTS, ...options };
  const results = response && Array.isArray(response.results) ? response.results : null;
  if (!results || results.length !== chunks.length || results.length === 0) {
    return manualRequired("invalid_classifier_response", { chunksAnalyzed: 0 });
  }

  const usable = [];
  for (let i = 0; i < results.length; i++) {
    const prediction = results[i]?.predictions?.subject;
    if (!prediction || prediction.warning || !CLASSIFIER_SUBJECTS.includes(prediction.label) || !Array.isArray(prediction.top)) continue;

    const scores = {};
    for (const item of prediction.top) {
      if (!CLASSIFIER_SUBJECTS.includes(item?.label)) continue;
      const probability = Number(item.probability);
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) continue;
      scores[item.label] = probability;
    }
    if (!CLASSIFIER_SUBJECTS.every((label) => Object.prototype.hasOwnProperty.call(scores, label))) continue;
    usable.push({
      page: chunks[i].page,
      label: prediction.label,
      confidence: Number.isFinite(Number(prediction.confidence)) ? Number(prediction.confidence) : scores[prediction.label],
      scores,
    });
  }

  const minimumUsable = Math.max(1, Math.ceil(chunks.length * 0.6));
  if (usable.length < minimumUsable) {
    return manualRequired("insufficient_classifiable_chunks", {
      chunksAnalyzed: usable.length,
      pagesAnalyzed: [...new Set(usable.map((r) => r.page))],
    });
  }

  const averaged = Object.fromEntries(CLASSIFIER_SUBJECTS.map((label) => [
    label,
    usable.reduce((sum, row) => sum + row.scores[label], 0) / usable.length,
  ]));
  const ranked = Object.entries(averaged).sort((a, b) => b[1] - a[1]);
  const [detectedSubject, score] = ranked[0];
  const margin = score - ranked[1][1];
  const agreement = usable.filter((r) => r.label === detectedSubject).length / usable.length;
  const predictions = ranked.map(([label, probability]) => ({ label, probability }));
  const common = {
    detectedSubject,
    score,
    margin,
    agreement,
    chunksAnalyzed: usable.length,
    pagesAnalyzed: [...new Set(usable.map((r) => r.page))].sort((a, b) => a - b),
    predictions,
    chunkResults: usable.map((r) => ({ page: r.page, label: r.label, score: r.confidence })),
  };

  let reasonCode = null;
  if (score < cfg.minConfidence) reasonCode = "low_subject_score";
  else if (margin < cfg.minMargin) reasonCode = "ambiguous_subject_scores";
  else if (agreement < cfg.minAgreement) reasonCode = "mixed_or_inconsistent_document";

  return reasonCode
    ? { ...manualRequired(reasonCode, common), predictions: common.predictions, chunkResults: common.chunkResults }
    : { status: "awaiting_confirmation", model: "bert", reasonCode: null, ...common };
}

async function detectDocumentSubject(pageTexts, env = process.env) {
  const cfg = getOptions(env);
  const built = createTextChunks(pageTexts, cfg);
  if (built.totalChars < cfg.minDocumentChars || built.chunks.length === 0) {
    return manualRequired("insufficient_extractable_text", {
      chunksAnalyzed: 0,
      pagesAnalyzed: [],
    });
  }

  const config = readConfig(env);
  if (!config.configured) return manualRequired("nlp_service_not_configured", { chunksAnalyzed: 0 });

  try {
    const upstream = await callUpstream(config, {
      method: "POST",
      path: "/classify/bert",
      body: { texts: built.chunks.map((chunk) => chunk.text), top_k: 4 },
      slow: true,
    });
    let body;
    try { body = JSON.parse(upstream.text); }
    catch { return manualRequired("invalid_classifier_response", { chunksAnalyzed: 0 }); }

    if (upstream.status < 200 || upstream.status >= 300) {
      const detail = body?.detail;
      const code = detail?.status || detail?.code || body?.code || `classifier_http_${upstream.status}`;
      return manualRequired(String(code).slice(0, 80), { chunksAnalyzed: 0 });
    }
    if (body?.status === "not_trained_yet" || body?.status === "dependency_missing") {
      return manualRequired(body.status, { chunksAnalyzed: 0 });
    }

    const result = aggregateBertPredictions(body, built.chunks, cfg);
    return {
      ...result,
      pagesWithText: built.pagesWithText,
      candidateChunks: built.candidateChunks,
      pagesAnalyzed: result.pagesAnalyzed || [],
    };
  } catch (err) {
    if (err instanceof NlpProxyError) return manualRequired(err.code, { chunksAnalyzed: 0 });
    return manualRequired("classification_failed", { chunksAnalyzed: 0 });
  }
}

module.exports = {
  CLASSIFIER_SUBJECTS,
  DEFAULTS,
  getOptions,
  createTextChunks,
  aggregateBertPredictions,
  detectDocumentSubject,
};
