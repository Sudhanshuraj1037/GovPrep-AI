"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {
  CLASSIFIER_SUBJECTS,
  createTextChunks,
  aggregateBertPredictions,
  getOptions,
} = require("../nlp/documentClassifier");

function bertResponse(texts, label = "History", probs = { History: 0.91, Geography: 0.04, Polity: 0.03, Science: 0.02 }) {
  return {
    model: "bert",
    results: texts.map(() => ({
      predictions: {
        subject: {
          label,
          confidence: probs[label],
          top: Object.entries(probs).map(([name, probability]) => ({ label: name, probability })),
        },
      },
    })),
  };
}

test("document chunker keeps page references, respects character limits and samples across long documents", () => {
  const pages = Array.from({ length: 20 }, (_, i) => `PAGE_${i + 1} ` + ("Indian constitution parliament history geography science. ".repeat(80)));
  const built = createTextChunks(pages, { maxChunks: 8, chunkChars: 900, overlapChars: 100 });
  assert.equal(built.chunks.length, 8);
  assert.ok(built.chunks.every((chunk) => chunk.text.length <= 900 && chunk.page >= 1 && chunk.page <= 20));
  assert.ok(new Set(built.chunks.map((chunk) => chunk.page)).size >= 6);
  assert.equal(built.pagesWithText, 20);
});

test("document chunker returns no chunks for empty or whitespace-only pages", () => {
  const built = createTextChunks(["", "  ", null]);
  assert.equal(built.chunks.length, 0);
  assert.equal(built.totalChars, 0);
});

test("BERT probability aggregation returns awaiting_confirmation for a clear single-subject document", () => {
  const chunks = [{ page: 1, text: "history text" }, { page: 2, text: "history text" }];
  const result = aggregateBertPredictions(bertResponse(chunks), chunks, getOptions({}));
  assert.equal(result.status, "awaiting_confirmation");
  assert.equal(result.model, "bert");
  assert.equal(result.detectedSubject, "History");
  assert.ok(result.score > 0.9);
  assert.equal(result.agreement, 1);
  assert.equal(result.chunksAnalyzed, 2);
});

test("BERT aggregation requests manual selection for inconsistent chunk predictions", () => {
  const chunks = [{ page: 1 }, { page: 2 }, { page: 3 }, { page: 4 }];
  const responses = [
    bertResponse(["1"], "History"),
    bertResponse(["2"], "History"),
    bertResponse(["3"], "History"),
    bertResponse(["4"], "Polity", { History: 0.03, Geography: 0.02, Polity: 0.92, Science: 0.03 }),
  ];
  const response = { results: responses.flatMap((item) => item.results) };
  const result = aggregateBertPredictions(response, chunks, getOptions({ NLP_SUBJECT_MIN_AGREEMENT: "0.9" }));
  assert.equal(result.status, "manual_required");
  assert.equal(result.reasonCode, "mixed_or_inconsistent_document");
});

test("BERT aggregation rejects malformed or incomplete responses without inventing a subject", () => {
  const chunks = [{ page: 1 }, { page: 2 }];
  const result = aggregateBertPredictions({ results: [] }, chunks);
  assert.equal(result.status, "manual_required");
  assert.equal(result.reasonCode, "invalid_classifier_response");
  assert.equal(result.detectedSubject, null);
});

test("classifier supports only the four trained subjects", () => {
  assert.deepEqual(CLASSIFIER_SUBJECTS, ["History", "Geography", "Polity", "Science"]);
});
