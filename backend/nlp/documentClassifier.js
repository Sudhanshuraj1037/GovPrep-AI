
"use strict";

const { readConfig } = require("./config");
const { callUpstream, NlpProxyError } = require("./client");

const CLASSIFIER_SUBJECTS = Object.freeze([
  "History",
  "Geography",
  "Polity",
  "Science",
]);

const DEFAULTS = Object.freeze({
  maxChunks: 24,
  chunkChars: 450,
  overlapChars: 60,
  minDocumentChars: 180,
  minConfidence: 0.55,
  minMargin: 0.10,
  minAgreement: 0.60,
});

function boundedNumber(raw, fallback, min, max) {
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    return fallback;
  }

  const n = Number(raw);
  return Number.isFinite(n) && n >= min && n <= max ? n : fallback;
}

function getOptions(env = process.env) {
  const chunkChars = Math.round(
    boundedNumber(env.NLP_SUBJECT_CHUNK_CHARS, DEFAULTS.chunkChars, 250, 4500)
  );

  const overlapChars = Math.round(
    boundedNumber(env.NLP_SUBJECT_OVERLAP_CHARS, DEFAULTS.overlapChars, 0, 1000)
  );

  return {
    maxChunks: Math.round(
      boundedNumber(env.NLP_SUBJECT_MAX_CHUNKS, DEFAULTS.maxChunks, 1, 32)
    ),
    chunkChars,
    overlapChars: Math.min(overlapChars, chunkChars - 1),
    minDocumentChars: Math.round(
      boundedNumber(
        env.NLP_SUBJECT_MIN_DOCUMENT_CHARS,
        DEFAULTS.minDocumentChars,
        50,
        5000
      )
    ),
    minConfidence: boundedNumber(
      env.NLP_SUBJECT_MIN_SCORE,
      DEFAULTS.minConfidence,
      0.25,
      0.99
    ),
    minMargin: boundedNumber(
      env.NLP_SUBJECT_MIN_MARGIN,
      DEFAULTS.minMargin,
      0,
      0.8
    ),
    minAgreement: boundedNumber(
      env.NLP_SUBJECT_MIN_AGREEMENT,
      DEFAULTS.minAgreement,
      0.25,
      1
    ),
  };
}

function evenlySample(items, limit) {
  if (items.length <= limit) return items;

  if (limit <= 1) {
    return [items[Math.floor((items.length - 1) / 2)]];
  }

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

/**
 * Remove characters produced by custom-encoded PDF fonts.
 * These characters are not useful semantic input for BERT.
 */
function cleanPdfText(raw) {
  return String(raw || "")
    .replace(/\u0000/g, " ")
    .replace(/[\uE000-\uF8FF]/g, " ")
    .replace(/[\u{F0000}-\u{FFFFD}\u{100000}-\u{10FFFD}]/gu, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\r/g, "\n")
    .trim();
}

/**
 * Conservative front-matter detection.
 * Avoid removing normal educational content pages.
 */
function isFrontMatterPage(raw) {
  const text = String(raw || "").trim();

  if (!text) return true;

  const lines = text
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 5) return false;

  const normalized = text.toLowerCase();

  const tocTerms = [
    "table of contents",
    "contents",
    "index",
    "chapter contents",
    "list of figures",
    "list of tables",
  ];

  const hasTocHeading = tocTerms.some((term) =>
    new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(normalized)
  );

  const bareNumberLines = lines.filter((line) =>
    /^[\s.\-–—]*\d{1,4}[\s.\-–—]*$/.test(line)
  ).length;

  const numberRatio = bareNumberLines / lines.length;

  // Drop pages explicitly identified as a contents/index page.
  if (hasTocHeading && lines.length >= 5) return true;

  // Catch TOC pages made mostly of standalone page numbers.
  if (numberRatio >= 0.35 && lines.length >= 8) return true;

  return false;
}

/**
 * Build bounded, page-aware representative chunks.
 *
 * Filename and chapter metadata are deliberately not used.
 * Source text is never persisted in detection metadata.
 */
function createTextChunks(pageTexts, options = {}) {
  const cfg = { ...DEFAULTS, ...options };
  const pages = Array.isArray(pageTexts) ? pageTexts : [];

  const all = [];
  let totalChars = 0;
  let pagesWithText = 0;
  let pagesSkipped = 0;

  pages.forEach((raw, index) => {
    const text = cleanPdfText(raw);

    totalChars += text.length;

    if (!text) return;

    pagesWithText++;

    if (isFrontMatterPage(text)) {
      pagesSkipped++;
      return;
    }

    const step = Math.max(
      1,
      cfg.chunkChars - Math.min(cfg.overlapChars, cfg.chunkChars - 1)
    );

    for (let start = 0; start < text.length; start += step) {
      const piece = text.slice(start, start + cfg.chunkChars).trim();

      if (piece.length >= 80) {
        all.push({
          page: index + 1,
          text: piece,
        });
      }

      if (start + cfg.chunkChars >= text.length) break;
    }
  });

  return {
    chunks: evenlySample(all, cfg.maxChunks),
    totalChars,
    candidateChunks: all.length,
    pagesWithText,
    pagesSkipped,
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
    ...(extra.chunkResults ? { chunkResults: extra.chunkResults } : {}),
  };
}

/**
 * Kept for backward compatibility with any existing imports.
 * Metadata must NEVER determine the subject.
 */
function detectStrongMetadataSubject() {
  return null;
}

/**
 * Aggregate actual BERT predictions using:
 * 1. Average class probabilities.
 * 2. Majority chunk voting.
 *
 * A high average score alone is not sufficient to approve a subject.
 */
function aggregateBertPredictions(response, chunks, options = {}) {
  const cfg = { ...DEFAULTS, ...options };

  const results =
    response && Array.isArray(response.results)
      ? response.results
      : null;

  if (!results || results.length !== chunks.length || results.length === 0) {
    return manualRequired("invalid_classifier_response");
  }

  const usable = [];

  for (let i = 0; i < results.length; i++) {
    const prediction = results[i]?.predictions?.subject;

    if (
      !prediction ||
      prediction.warning ||
      !CLASSIFIER_SUBJECTS.includes(prediction.label) ||
      !Array.isArray(prediction.top)
    ) {
      continue;
    }

    const scores = {};

    for (const item of prediction.top) {
      if (!CLASSIFIER_SUBJECTS.includes(item?.label)) continue;

      const probability = Number(item.probability);

      if (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
      ) {
        continue;
      }

      scores[item.label] = probability;
    }

    if (
      !CLASSIFIER_SUBJECTS.every((label) =>
        Object.prototype.hasOwnProperty.call(scores, label)
      )
    ) {
      continue;
    }

    usable.push({
      page: chunks[i].page,
      label: prediction.label,
      confidence: Number.isFinite(Number(prediction.confidence))
        ? Number(prediction.confidence)
        : scores[prediction.label],
      scores,
    });
  }

  const minimumUsable = Math.max(1, Math.ceil(chunks.length * 0.6));

  if (usable.length < minimumUsable) {
    return manualRequired("insufficient_classifiable_chunks", {
      chunksAnalyzed: usable.length,
      pagesAnalyzed: [...new Set(usable.map((row) => row.page))],
    });
  }

  // Average probabilities across all usable chunks.
  const averaged = Object.fromEntries(
    CLASSIFIER_SUBJECTS.map((label) => [
      label,
      usable.reduce((sum, row) => sum + row.scores[label], 0) / usable.length,
    ])
  );

  const ranked = Object.entries(averaged).sort((a, b) => b[1] - a[1]);

  const [detectedSubject, score] = ranked[0];
  const margin = score - ranked[1][1];

  // Count actual per-chunk winning labels.
  const votes = Object.fromEntries(
    CLASSIFIER_SUBJECTS.map((label) => [
      label,
      usable.filter((row) => row.label === label).length,
    ])
  );

  const rankedVotes = Object.entries(votes).sort((a, b) => b[1] - a[1]);

  const [voteSubject, voteCount] = rankedVotes[0];
  const topVoteFraction = voteCount / usable.length;
  const runnerUpVoteFraction = rankedVotes[1][1] / usable.length;

  const agreement =
    usable.filter((row) => row.label === detectedSubject).length / usable.length;

  const predictions = ranked.map(([label, probability]) => ({
    label,
    probability,
    votes: votes[label],
    voteFraction: votes[label] / usable.length,
  }));

  const common = {
    detectedSubject,
    score,
    margin,
    agreement,
    chunksAnalyzed: usable.length,
    pagesAnalyzed: [...new Set(usable.map((row) => row.page))].sort(
      (a, b) => a - b
    ),
    predictions,
    chunkResults: usable.map((row) => ({
      page: row.page,
      label: row.label,
      score: row.confidence,
    })),
  };

  let reasonCode = null;

  if (score < cfg.minConfidence) {
    reasonCode = "low_subject_score";
  } else if (margin < cfg.minMargin) {
    reasonCode = "ambiguous_subject_scores";
  } else if (
    topVoteFraction < cfg.minAgreement ||
    voteSubject !== detectedSubject
  ) {
    reasonCode = "mixed_or_inconsistent_document";
  } else if (runnerUpVoteFraction >= 0.30) {
    reasonCode = "mixed_or_inconsistent_document";
  }

  if (reasonCode) {
    return manualRequired(reasonCode, common);
  }

  return {
    status: "awaiting_confirmation",
    model: "bert",
    reasonCode: null,
    ...common,
  };
}

/**
 * Detect document subject from extracted PDF content.
 *
 * Filename and chapter are intentionally ignored.
 * BERT is the only automatic subject classifier.
 */
async function detectDocumentSubject(
  pageTexts,
  env = process.env,
  metadata = {}
) {
  // metadata is accepted for API compatibility only.
  void metadata;

  const cfg = getOptions(env);
  const built = createTextChunks(pageTexts, cfg);

  if (
    built.totalChars < cfg.minDocumentChars ||
    built.chunks.length === 0
  ) {
    return manualRequired("insufficient_extractable_text", {
      chunksAnalyzed: 0,
      pagesAnalyzed: [],
    });
  }

  const config = readConfig(env);

  if (!config.configured) {
    return manualRequired("nlp_service_not_configured", {
      chunksAnalyzed: 0,
    });
  }

  try {
    const upstream = await callUpstream(config, {
      method: "POST",
      path: "/classify/bert",
      body: {
        texts: built.chunks.map((chunk) => chunk.text),
        top_k: 4,
      },
      slow: true,
    });

    let body;

    try {
      body = JSON.parse(upstream.text);
    } catch {
      return manualRequired("invalid_classifier_response", {
        chunksAnalyzed: 0,
      });
    }

    if (upstream.status < 200 || upstream.status >= 300) {
      const detail = body?.detail;

      const code =
        detail?.status ||
        detail?.code ||
        body?.code ||
        `classifier_http_${upstream.status}`;

      return manualRequired(String(code).slice(0, 80), {
        chunksAnalyzed: 0,
      });
    }

    if (
      body?.status === "not_trained_yet" ||
      body?.status === "dependency_missing"
    ) {
      return manualRequired(body.status, {
        chunksAnalyzed: 0,
      });
    }

    const result = aggregateBertPredictions(body, built.chunks, cfg);

    console.log(
      "[BERT DOCUMENT DIAGNOSTICS]",
      JSON.stringify(
        {
          status: result.status,
          reasonCode: result.reasonCode,
          detectedSubject: result.detectedSubject,
          score: result.score,
          margin: result.margin,
          agreement: result.agreement,
          totalChunks: built.candidateChunks,
          chunksSent: built.chunks.length,
          chunksAnalyzed: result.chunksAnalyzed,
          pagesWithText: built.pagesWithText,
          pagesSkipped: built.pagesSkipped,
          chunkResults: result.chunkResults || [],
          predictions: result.predictions || [],
        },
        null,
        2
      )
    );

    return {
      ...result,
      pagesWithText: built.pagesWithText,
      pagesSkipped: built.pagesSkipped,
      candidateChunks: built.candidateChunks,
      pagesAnalyzed: result.pagesAnalyzed || [],
    };
  } catch (err) {
    if (err instanceof NlpProxyError) {
      return manualRequired(err.code, {
        chunksAnalyzed: 0,
      });
    }

    console.error("[BERT DOCUMENT CLASSIFICATION ERROR]", err);

    return manualRequired("classification_failed", {
      chunksAnalyzed: 0,
    });
  }
}

module.exports = {
  CLASSIFIER_SUBJECTS,
  DEFAULTS,
  getOptions,
  cleanPdfText,
  isFrontMatterPage,
  createTextChunks,
  aggregateBertPredictions,
  detectStrongMetadataSubject,
  detectDocumentSubject,
};
