require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { v4: uuidv4 } = require("uuid");
const pdfParse = require("pdf-parse");
const OpenAI = require("openai");
const db = require("./database");
const { hashPassword, verifyPassword, signToken, requireAuth } = require("./auth");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => cb(null, uuidv4() + "-" + file.originalname),
});
const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") cb(null, true);
    else cb(new Error("Only PDF files are allowed"), false);
  },
  limits: { fileSize: 50 * 1024 * 1024 },
});

if (!process.env.GROQ_API_KEY) console.warn("\n⚠️  GROQ_API_KEY not set in .env\n");
const groq = new OpenAI({ apiKey: process.env.GROQ_API_KEY, baseURL: "https://api.groq.com/openai/v1" });
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

// ── Helpers ──────────────────────────────────────────────────
function splitIntoPages(rawText, totalPages) {
  let pages = rawText.split(/\f/);
  if (pages.length < 2) {
    const cpp = Math.ceil(rawText.length / Math.max(totalPages, 1));
    pages = Array.from({ length: totalPages }, (_, i) => rawText.slice(i * cpp, (i + 1) * cpp));
  }
  return pages;
}

function tryParse(s) { try { return JSON.parse(s); } catch (_) { return undefined; } }

function findArrayField(p) {
  if (Array.isArray(p)) return p;
  if (p && typeof p === "object") {
    if (Array.isArray(p.questions)) return p.questions;
    const f = Object.values(p).find(Array.isArray);
    if (f) return f;
  }
  return undefined;
}

function splitTopLevel(text) {
  const vals = []; let depth = 0, start = -1, inStr = false, esc = false, qc = null;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inStr) { if (esc) { esc = false; continue; } if (c === "\\") { esc = true; continue; } if (c === qc) inStr = false; continue; }
    if (c === '"' || c === "'") { inStr = true; qc = c; continue; }
    if (c === "[" || c === "{") { if (!depth) start = i; depth++; }
    else if (c === "]" || c === "}") { if (!--depth && start !== -1) { vals.push(text.slice(start, i + 1)); start = -1; } }
  }
  return vals;
}

function extractJsonArray(raw) {
  if (!raw) throw new Error("Empty AI response");
  let text = raw.trim().replace(/```(?:json)?/gi, "").trim();
  let p = tryParse(text); if (p) { const a = findArrayField(p); if (a) return a; }
  const cands = splitTopLevel(text);
  if (cands.length) {
    let merged = [], ok = false;
    for (const c of cands) {
      const x = tryParse(c) ?? tryParse(c.replace(/,\s*([\]}])/g, "$1"));
      if (!x) continue;
      const a = findArrayField(x); if (a) { merged = merged.concat(a); ok = true; }
    }
    if (ok && merged.length) return merged;
  }
  const f = text.indexOf("["), l = text.lastIndexOf("]");
  if (f !== -1 && l > f) { const sl = text.slice(f, l + 1); const x = tryParse(sl) ?? tryParse(sl.replace(/,\s*([\]}])/g, "$1")); if (x && Array.isArray(x)) return x; }
  throw new Error("AI did not return valid JSON. Try fewer questions or a shorter page range.");
}

function isRateLimit(err) { return err?.status === 429 || /rate limit/i.test(err?.error?.message || err?.message || ""); }
function groqErr(err) {
  const msg = err?.error?.message || err?.message || "AI service error";
  return isRateLimit(err) ? `Groq free-tier daily limit reached. ${msg} Try again later.` : msg;
}

// ════════════════════════════════════════════════════════════
//  AUTH
// ════════════════════════════════════════════════════════════
app.post("/api/auth/signup", async (req, res) => {
  try {
    const { name, password, examTarget } = req.body;
    if (!name?.trim()) return res.status(400).json({ error: "Name is required." });
    if (!password || password.length < 4) return res.status(400).json({ error: "Password must be at least 4 characters." });
    if (db.prepare("SELECT id FROM users WHERE name = ?").get(name.trim())) return res.status(409).json({ error: "Name already taken. Choose another or log in." });
    const id = uuidv4();
    db.prepare("INSERT INTO users (id, name, password_hash, exam_target, created_at) VALUES (?, ?, ?, ?, ?)")
      .run(id, name.trim(), await hashPassword(password), examTarget || "Railway RRB NTPC", new Date().toISOString());
    res.json({ success: true, token: signToken({ id, name: name.trim() }), user: { id, name: name.trim(), examTarget: examTarget || "Railway RRB NTPC" } });
  } catch (e) { console.error(e); res.status(500).json({ error: "Failed to create account." }); }
});

app.post("/api/auth/login", async (req, res) => {
  try {
    const { name, password } = req.body;
    if (!name || !password) return res.status(400).json({ error: "Name and password required." });
    const row = db.prepare("SELECT * FROM users WHERE name = ?").get(name.trim());
    if (!row) return res.status(401).json({ error: "No account with this name." });
    if (!await verifyPassword(password, row.password_hash)) return res.status(401).json({ error: "Incorrect password." });
    res.json({ success: true, token: signToken({ id: row.id, name: row.name }), user: { id: row.id, name: row.name, examTarget: row.exam_target } });
  } catch (e) { console.error(e); res.status(500).json({ error: "Login failed." }); }
});

app.get("/api/auth/me", requireAuth, (req, res) => {
  const row = db.prepare("SELECT id, name, exam_target, created_at FROM users WHERE id = ?").get(req.userId);
  if (!row) return res.status(404).json({ error: "User not found." });
  res.json({ user: { id: row.id, name: row.name, examTarget: row.exam_target, createdAt: row.created_at } });
});

// ════════════════════════════════════════════════════════════
//  PDFs
// ════════════════════════════════════════════════════════════
app.post("/api/upload", requireAuth, upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No PDF provided" });
    const { subject = "General", chapter = "Chapter 1" } = req.body;
    const buf = fs.readFileSync(req.file.path);
    const parsed = await pdfParse(buf);
    const pageTexts = splitIntoPages(parsed.text, parsed.numpages);
    const id = uuidv4();
    db.prepare("INSERT INTO pdfs (id,user_id,name,file_path,text,page_texts,total_pages,subject,chapter,word_count,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
      .run(id, req.userId, req.file.originalname, req.file.path, parsed.text, JSON.stringify(pageTexts), parsed.numpages, subject, chapter, parsed.text.split(/\s+/).filter(Boolean).length, new Date().toISOString());
    res.json({ success: true, pdf: { id, name: req.file.originalname, totalPages: parsed.numpages, subject, chapter, wordCount: parsed.text.split(/\s+/).filter(Boolean).length, uploadedAt: new Date().toISOString() } });
  } catch (e) { console.error(e); res.status(500).json({ error: e.message || "Upload failed" }); }
});

app.get("/api/pdfs", requireAuth, (req, res) => {
  const rows = db.prepare("SELECT id,name,total_pages,subject,chapter,word_count,uploaded_at FROM pdfs WHERE user_id=? ORDER BY uploaded_at DESC").all(req.userId);
  res.json({ pdfs: rows.map(p => ({ id: p.id, name: p.name, totalPages: p.total_pages, subject: p.subject, chapter: p.chapter, wordCount: p.word_count, uploadedAt: p.uploaded_at })) });
});

app.delete("/api/pdfs/:id", requireAuth, (req, res) => {
  const pdf = db.prepare("SELECT * FROM pdfs WHERE id=? AND user_id=?").get(req.params.id, req.userId);
  if (!pdf) return res.status(404).json({ error: "PDF not found" });
  try { if (fs.existsSync(pdf.file_path)) fs.unlinkSync(pdf.file_path); } catch (_) {}
  db.prepare("DELETE FROM pdfs WHERE id=?").run(req.params.id);
  res.json({ success: true });
});

// ════════════════════════════════════════════════════════════
//  GENERATE QUESTIONS
// ════════════════════════════════════════════════════════════
app.post("/api/generate-questions", requireAuth, async (req, res) => {
  const { pdfId, pageStart = 1, pageEnd, questionCount = 10, difficulty = "mixed", examTemplate = "Railway RRB NTPC", questionTypes = ["MCQ"], language = "english" } = req.body;
  if (!pdfId) return res.status(400).json({ error: "pdfId required" });
  if (!process.env.GROQ_API_KEY) return res.status(500).json({ error: "GROQ_API_KEY not configured." });

  const pdf = db.prepare("SELECT * FROM pdfs WHERE id=? AND user_id=?").get(pdfId, req.userId);
  if (!pdf) return res.status(404).json({ error: "PDF not found. Please upload first." });

  const pages = JSON.parse(pdf.page_texts);
  const s = Math.max(1, parseInt(pageStart) || 1);
  const e = Math.min(pdf.total_pages, parseInt(pageEnd) || pdf.total_pages);
  if (s > e) return res.status(400).json({ error: `Invalid page range ${s}–${e}. PDF has ${pdf.total_pages} pages.` });

  const ctx = pages.slice(s - 1, e).join("\n\n").trim();
  if (!ctx || ctx.length < 50) return res.status(400).json({ error: `Not enough text on pages ${s}–${e}. Use a text-based PDF.` });

  const truncated = ctx.length > 9000 ? ctx.slice(0, 9000) + "\n...[truncated]" : ctx;
  const n = Math.min(parseInt(questionCount) || 10, 20);
  const diffInstr = difficulty === "mixed" ? "Mix easy(30%), medium(50%), hard(20%)." : `All ${difficulty}.`;

  // Language instruction
  const isHindi = language === "hindi";
  const langInstr = isHindi
    ? `IMPORTANT: Write ALL questions, ALL options, ALL explanations, and ALL topic names in HINDI (Devanagari script). Do NOT use English for question text, options, or explanations. Even if source PDF is in English, translate everything to Hindi. Keep proper nouns/names as-is if needed.`
    : `Write all questions, options, and explanations in English.`;

  const sysMsg = "You are a strict JSON API. Output ONLY a valid JSON array. No markdown, no explanation. Output [] if you cannot comply.";
  const userMsg = `Generate ${n} MCQs for ${examTemplate} exam STRICTLY from the content below.
Subject: ${pdf.subject} | Chapter: ${pdf.chapter} | Pages: ${s}-${e}
Language instruction: ${langInstr}
CONTENT:\n---\n${truncated}\n---
Rules: Questions from this content ONLY. 4 options. ${diffInstr} "correct"=0-based index. "pageRef"=page between ${s}-${e}.
Output ONLY this JSON array:
[{"question":"...","options":["A","B","C","D"],"correct":0,"difficulty":"easy","explanation":"...","pageRef":${s},"topic":"..."}]`;

  try {
    let comp = await groq.chat.completions.create({ model: GROQ_MODEL, messages: [{ role: "system", content: sysMsg }, { role: "user", content: userMsg }], temperature: 0.3, max_tokens: 4000 });
    let raw = comp.choices?.[0]?.message?.content?.trim() || "";
    let questions;
    try { questions = extractJsonArray(raw); }
    catch (e1) {
      console.error("RAW (1st attempt):\n", raw);
      comp = await groq.chat.completions.create({ model: GROQ_MODEL, messages: [{ role: "system", content: sysMsg }, { role: "user", content: userMsg }, { role: "assistant", content: raw }, { role: "user", content: "Not valid JSON. Reply ONLY with a JSON array starting [ and ending ]. Nothing else." }], temperature: 0.2, max_tokens: 4000 });
      raw = comp.choices?.[0]?.message?.content?.trim() || "";
      try { questions = extractJsonArray(raw); } catch (e2) { console.error("RAW (2nd attempt):\n", raw); throw e2; }
    }

    questions = questions.filter(q => q?.question && Array.isArray(q.options) && q.options.length >= 2).slice(0, n)
      .map(q => ({ id: uuidv4(), question: String(q.question), options: q.options.slice(0, 4).map(String), correct: typeof q.correct === "number" ? q.correct : 0, difficulty: q.difficulty || "medium", explanation: q.explanation || "", pageRef: q.pageRef || s, topic: q.topic || pdf.chapter, source: `${pdf.name} (pages ${s}-${e})`, pdfId, pdfName: pdf.name, subject: pdf.subject, chapter: pdf.chapter }));

    if (!questions.length) return res.status(500).json({ error: "AI couldn't generate valid questions. Try a larger page range or fewer questions." });

    const testId = uuidv4();
    db.prepare("INSERT INTO tests (id,user_id,pdf_id,pdf_name,exam_template,page_start,page_end,questions_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)")
      .run(testId, req.userId, pdfId, pdf.name, examTemplate, s, e, JSON.stringify(questions), new Date().toISOString());

    res.json({ success: true, testId, questions, meta: { pdfName: pdf.name, subject: pdf.subject, chapter: pdf.chapter, pageRange: `${s}–${e}`, totalPages: pdf.total_pages, questionsGenerated: questions.length } });
  } catch (err) {
    console.error("Generation error:", err);
    res.status(isRateLimit(err) ? 429 : 500).json({ error: groqErr(err) });
  }
});

// ════════════════════════════════════════════════════════════
//  SUBMIT TEST
// ════════════════════════════════════════════════════════════
app.post("/api/submit-test", requireAuth, (req, res) => {
  const { testId, answers = {}, timeTaken = 0, examTemplate = "Railway RRB NTPC" } = req.body;
  const test = db.prepare("SELECT * FROM tests WHERE id=? AND user_id=?").get(testId, req.userId);
  if (!test) return res.status(404).json({ error: "Test not found" });

  const questions = JSON.parse(test.questions_json);
  const schemes = { "Railway RRB NTPC": { correct: 1, wrong: -0.33 }, "SSC CGL Tier 1": { correct: 2, wrong: -0.5 }, "UPSC Prelims": { correct: 2, wrong: -0.66 }, "Banking IBPS PO": { correct: 1, wrong: -0.25 }, Custom: { correct: 1, wrong: -0.25 } };
  const scheme = schemes[examTemplate] || schemes["Railway RRB NTPC"];

  let correct = 0, wrong = 0, unattempted = 0;
  const questionResults = questions.map(q => {
    const ua = answers[q.id];
    let status = "unattempted";
    if (ua == null) unattempted++;
    else if (ua === q.correct) { correct++; status = "correct"; }
    else { wrong++; status = "wrong"; }
    return { ...q, userAnswer: ua, status };
  });

  const rawScore = correct * scheme.correct + wrong * scheme.wrong;
  const maxScore = questions.length * scheme.correct;
  const percentage = Math.round(Math.max(0, rawScore) / maxScore * 100);

  const topicMap = {};
  questionResults.forEach(q => {
    if (!topicMap[q.topic]) topicMap[q.topic] = { correct: 0, total: 0 };
    topicMap[q.topic].total++;
    if (q.status === "correct") topicMap[q.topic].correct++;
  });
  const topicBreakdown = Object.entries(topicMap).map(([topic, d]) => ({ topic, correct: d.correct, total: d.total, percentage: Math.round(d.correct / d.total * 100) }));
  const weak = topicBreakdown.filter(t => t.percentage < 60).map(t => t.topic);
  const recommendation = weak.length ? `Focus on: ${weak.join(", ")}. Review pages ${test.page_start}–${test.page_end} of ${test.pdf_name}.` : "Great performance! Try a harder page range or more questions next time.";

  const resultId = uuidv4();
  db.prepare("INSERT INTO test_results (id,user_id,test_id,exam_template,correct,wrong,unattempted,raw_score,max_score,percentage,time_taken,topic_breakdown_json,question_results_json,pdf_name,page_range,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    .run(resultId, req.userId, testId, examTemplate, correct, wrong, unattempted, rawScore, maxScore, percentage, timeTaken, JSON.stringify(topicBreakdown), JSON.stringify(questionResults), test.pdf_name, `${test.page_start}–${test.page_end}`, new Date().toISOString());

  res.json({ success: true, scorecard: { testId, resultId, examTemplate, correct, wrong, unattempted, rawScore: parseFloat(rawScore.toFixed(2)), maxScore, percentage, timeTaken, scheme, topicBreakdown, questionResults, recommendation, pdfName: test.pdf_name, pageRange: `${test.page_start}–${test.page_end}` } });
});

app.get("/api/my-tests", requireAuth, (req, res) => {
  const rows = db.prepare("SELECT id,test_id,exam_template,correct,wrong,unattempted,raw_score,max_score,percentage,time_taken,pdf_name,page_range,created_at FROM test_results WHERE user_id=? ORDER BY created_at DESC LIMIT 50").all(req.userId);
  res.json({ results: rows });
});

// ════════════════════════════════════════════════════════════
//  AI ASSISTANT
// ════════════════════════════════════════════════════════════
app.post("/api/assistant/chat", requireAuth, async (req, res) => {
  const { message, context } = req.body;
  if (!message?.trim()) return res.status(400).json({ error: "Message required." });
  if (!process.env.GROQ_API_KEY) return res.status(500).json({ error: "GROQ_API_KEY not configured." });

  let ctxBlock = "";
  try {
    if (context?.type === "pdf" && context.pdfId) {
      const pdf = db.prepare("SELECT * FROM pdfs WHERE id=? AND user_id=?").get(context.pdfId, req.userId);
      if (pdf) {
        const pages = JSON.parse(pdf.page_texts);
        const s = Math.max(1, parseInt(context.pageStart) || 1);
        const e = Math.min(pdf.total_pages, parseInt(context.pageEnd) || pdf.total_pages);
        const slice = pages.slice(s - 1, e).join("\n\n").slice(0, 6000);
        ctxBlock = `\n\nStudent is asking about their PDF: "${pdf.name}" (${pdf.subject} - ${pdf.chapter}), pages ${s}-${e}.\nContent:\n---\n${slice}\n---\nAnswer from this content when the question is about the PDF.`;
      }
    } else if (context?.type === "question") {
      const opts = (context.options || []).map((o, i) => `${["A","B","C","D"][i]}. ${o}`).join(" | ");
      ctxBlock = `\n\nStudent is asking about this question they got wrong:\nQ: ${context.questionText}\nOptions: ${opts}\nCorrect answer: ${["A","B","C","D"][context.correctAnswer]}\nStudent's answer: ${context.userAnswer != null ? ["A","B","C","D"][context.userAnswer] : "Not answered"}\nExplanation on file: ${context.explanation || "None"}\nExplain clearly why the correct answer is right.`;
    }
  } catch (e) { console.error("Context error:", e); }

  const history = db.prepare("SELECT role,content FROM chat_messages WHERE user_id=? ORDER BY created_at DESC LIMIT 10").all(req.userId).reverse();

  const sysMsg = `You are "GovPrep Assistant" — a friendly, encouraging AI study helper for Indian government exam aspirants (Railway, SSC, Banking, UPSC, State PSC).
- Be concise, warm, and exam-focused.
- Use simple English. Give specific, actionable advice.
- For PDF-related questions, use only the provided content.
- For exam concepts, syllabus, strategy — be accurate and helpful.
- Always end with an encouraging note when relevant.${ctxBlock}`;

  try {
    const comp = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: [{ role: "system", content: sysMsg }, ...history.map(h => ({ role: h.role, content: h.content })), { role: "user", content: message }],
      temperature: 0.5, max_tokens: 800,
    });
    const reply = comp.choices?.[0]?.message?.content?.trim() || "Sorry, I couldn't respond. Please try again.";
    const now = new Date().toISOString();
    db.prepare("INSERT INTO chat_messages (id,user_id,role,content,created_at) VALUES (?,?,?,?,?)").run(uuidv4(), req.userId, "user", message, now);
    db.prepare("INSERT INTO chat_messages (id,user_id,role,content,created_at) VALUES (?,?,?,?,?)").run(uuidv4(), req.userId, "assistant", reply, new Date().toISOString());
    res.json({ success: true, reply });
  } catch (err) {
    console.error("Assistant error:", err);
    res.status(isRateLimit(err) ? 429 : 500).json({ error: groqErr(err) });
  }
});

app.get("/api/assistant/history", requireAuth, (req, res) => {
  const rows = db.prepare("SELECT role,content,created_at FROM chat_messages WHERE user_id=? ORDER BY created_at ASC LIMIT 100").all(req.userId);
  res.json({ messages: rows });
});

app.delete("/api/assistant/history", requireAuth, (req, res) => {
  db.prepare("DELETE FROM chat_messages WHERE user_id=?").run(req.userId);
  res.json({ success: true });
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", groqKeyConfigured: !!process.env.GROQ_API_KEY, users: db.prepare("SELECT COUNT(*) as c FROM users").get().c });
});

app.listen(PORT, () => {
  console.log(`\n✅ GovPrep AI v2 running at http://localhost:${PORT}`);
  console.log(`   Groq model: ${GROQ_MODEL}`);
  console.log(`   GROQ_API_KEY: ${!!process.env.GROQ_API_KEY}`);
  console.log(`   Database: ${path.join(__dirname, "govprep.db")}\n`);
});
