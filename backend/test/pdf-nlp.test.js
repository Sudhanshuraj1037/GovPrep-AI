"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {
  startBackend, startGroqStub, startStub, request, signup, readFixturePdf, uploadPdf,
} = require("./helpers/harness");

function classificationResponse(texts, label = "History") {
  const base = { History: 0.92, Geography: 0.03, Polity: 0.03, Science: 0.02 };
  if (label !== "History") {
    for (const key of Object.keys(base)) base[key] = key === label ? 0.92 : 0.08 / 3;
  }
  return {
    model: "bert",
    results: texts.map((text) => ({
      text_preview: text.slice(0, 120),
      predictions: {
        subject: {
          label,
          confidence: base[label],
          top: Object.entries(base).map(([name, probability]) => ({ label: name, probability })),
        },
      },
    })),
  };
}

test("auto PDF upload calls the BERT classifier, requires confirmation, and then permits grounded question generation", async () => {
  const groq = await startGroqStub();
  const nlp = await startStub((req) => {
    if (req.method !== "POST" || req.path !== "/classify/bert") return { status: 404, body: { detail: "not found" } };
    assert.equal(req.body.top_k, 4);
    assert.ok(req.body.texts.length >= 1 && req.body.texts.length <= 8);
    assert.ok(req.body.texts.every((text) => text.length <= 1200));
    return { status: 200, body: classificationResponse(req.body.texts, "History") };
  });
  const app = await startBackend({ groqUrl: groq.url, env: { NLP_SERVICE_URL: nlp.url } });
  try {
    const user = await signup(app.url, "AutoDetectUser");
    const uploaded = await uploadPdf(app.url, user.token, readFixturePdf(), { subjectMode: "auto", subject: "Other" });
    assert.equal(uploaded.status, 200, uploaded.text);
    assert.equal(uploaded.json.pdf.subject, "History");
    assert.equal(uploaded.json.pdf.subjectDetectionStatus, "awaiting_confirmation");
    assert.equal(uploaded.json.pdf.subjectDetected, "History");
    assert.equal(uploaded.json.pdf.subjectDetectionModel, "bert");
    assert.ok(uploaded.json.pdf.subjectDetectionScore > 0.9);
    assert.equal(nlp.requests.length, 1);
    assert.equal(nlp.requests[0].path, "/classify/bert");

    const blocked = await request(app.url, "POST", "/api/generate-questions", {
      token: user.token, body: { pdfId: uploaded.json.pdf.id },
    });
    assert.equal(blocked.status, 409);
    assert.equal(blocked.json.code, "subject_confirmation_required");

    const confirmed = await request(app.url, "PATCH", `/api/pdfs/${uploaded.json.pdf.id}/subject`, {
      token: user.token, body: { subject: "History" },
    });
    assert.equal(confirmed.status, 200, confirmed.text);
    assert.equal(confirmed.json.pdf.subjectDetectionStatus, "confirmed");

    const generated = await request(app.url, "POST", "/api/generate-questions", {
      token: user.token,
      body: { pdfId: uploaded.json.pdf.id, pageStart: 1, pageEnd: 2, questionCount: 3 },
    });
    assert.equal(generated.status, 200, generated.text);
    assert.equal(generated.json.meta.subject, "History");
    assert.match(groq.requests.at(-1).body.messages[1].content, /Subject: History/);
    assert.match(groq.requests.at(-1).body.messages[1].content, /Chandragupta/);

    const listed = await request(app.url, "GET", "/api/pdfs", { token: user.token });
    assert.equal(listed.json.pdfs[0].subjectDetectionStatus, "confirmed");
  } finally {
    await app.stop(); await nlp.close(); await groq.close();
  }
});

test("NLP service failure does not lose uploaded PDF; user can manually select subject", async () => {
  const app = await startBackend({ env: { NLP_SERVICE_URL: undefined } });
  try {
    const user = await signup(app.url, "ManualFallbackUser");
    const uploaded = await uploadPdf(app.url, user.token, readFixturePdf(), { subjectMode: "auto" });
    assert.equal(uploaded.status, 200, uploaded.text);
    assert.equal(uploaded.json.pdf.subjectDetectionStatus, "manual_required");
    assert.equal(uploaded.json.pdf.subjectDetectionReason, "nlp_service_not_configured");
    assert.equal(uploaded.json.pdf.subject, "Other");

    const blocked = await request(app.url, "POST", "/api/generate-questions", {
      token: user.token, body: { pdfId: uploaded.json.pdf.id },
    });
    assert.equal(blocked.status, 409);

    const selected = await request(app.url, "PATCH", `/api/pdfs/${uploaded.json.pdf.id}/subject`, {
      token: user.token, body: { subject: "History" },
    });
    assert.equal(selected.status, 200, selected.text);
    assert.equal(selected.json.pdf.subjectDetectionStatus, "manual_confirmed");
    assert.equal(selected.json.pdf.subject, "History");
  } finally {
    await app.stop();
  }
});

test("PDF subject correction is owner-scoped and validates allowed subjects", async () => {
  const app = await startBackend({ env: { NLP_SERVICE_URL: undefined } });
  try {
    const owner = await signup(app.url, "PdfOwner");
    const other = await signup(app.url, "OtherPdfUser");
    const uploaded = await uploadPdf(app.url, owner.token, readFixturePdf(), { subjectMode: "manual", subject: "History" });
    assert.equal(uploaded.status, 200);

    const foreign = await request(app.url, "PATCH", `/api/pdfs/${uploaded.json.pdf.id}/subject`, {
      token: other.token, body: { subject: "Science" },
    });
    assert.equal(foreign.status, 404);

    const invalid = await request(app.url, "PATCH", `/api/pdfs/${uploaded.json.pdf.id}/subject`, {
      token: owner.token, body: { subject: "Astronomy" },
    });
    assert.equal(invalid.status, 400);

    const unauth = await request(app.url, "PATCH", `/api/pdfs/${uploaded.json.pdf.id}/subject`, {
      body: { subject: "Science" },
    });
    assert.equal(unauth.status, 401);
  } finally {
    await app.stop();
  }
});
