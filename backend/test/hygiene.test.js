"use strict";
/**
 * Tests for scripts/repo-hygiene.js, run against real throwaway git repositories.
 * Secrets used as test data are assembled at run time, so this file itself contains none.
 */
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const SCRIPT = path.join(REPO_ROOT, "scripts", "repo-hygiene.js");
const HAS_GIT = spawnSync("git", ["--version"]).status === 0;
const T = { skip: HAS_GIT ? false : "git is not installed" };

const GROQ = "gsk_" + "a1B2c3D4e5".repeat(4);
const OPENAI = "sk-" + "Zy9Xw8Vu7T".repeat(3);
const AWS = "AKIA" + "ABCDEFGHIJKLMNOP";
const PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIB\n-----END " + "RSA PRIVATE KEY-----";
const REAL_JWT = "JWT_SECRET=" + "9f8a7b6c5d4e3f2a1b0c9d8e";

const git = (cwd, ...args) => spawnSync("git", args, { cwd, encoding: "utf8" });
const sha = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const tracked = (cwd) => git(cwd, "ls-files").stdout.split("\n").filter(Boolean);

function makeRepo(files, { commit = true, scripts = true } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "govprep-hygiene-"));
  git(dir, "init", "-q");
  git(dir, "config", "user.email", "test@example.org");
  git(dir, "config", "user.name", "Test");
  git(dir, "config", "core.autocrlf", "false");
  if (scripts) {
    fs.mkdirSync(path.join(dir, "scripts"));
    fs.copyFileSync(SCRIPT, path.join(dir, "scripts", "repo-hygiene.js"));
  }
  for (const [rel, content] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.writeFileSync(path.join(dir, rel), content);
  }
  if (commit) { git(dir, "add", "-A", "-f"); git(dir, "commit", "-q", "-m", "init"); }
  return dir;
}
function run(cwd, ...args) {
  const r = spawnSync(process.execPath, [SCRIPT, ...args], { cwd, encoding: "utf8" });
  return { code: r.status, out: r.stdout, err: r.stderr };
}
const violations = (cwd, ...extra) => JSON.parse(run(cwd, "check", "--json", ...extra).out).violations;
const cleanup = (dir) => fs.rmSync(dir, { recursive: true, force: true });

const CLEAN = {
  "README.md": "# GovPrep-AI\n",
  "backend/server.js": 'const secret = process.env.JWT_SECRET || "govprep-dev-secret-change-in-production";\n',
  "backend/.env.example": "GROQ_API_KEY=your-groq-key-here\nJWT_SECRET=change-me-to-a-long-random-string\nNLP_SERVICE_URL=http://127.0.0.1:8001\n",
  "backend/uploads/.gitkeep": "",
  "nlp_service/.env.example": "NLP_PORT=8001\n",
  "nlp_service/artifacts/bert/metrics.json": "{}",
  "nlp_service/artifacts/bert/tokenizer/tokenizer.json": "{}",
  "nlp_service/artifacts/comparison/comparison.md": "# report\n",
  "nlp_service/data/README.md": "# data\n",
  "nlp_service/data/public/.gitkeep": "",
  "nlp_service/data/public/SOURCES.md": "licence notes\n",
  "docs/notes.md": "Use a .venv-nlp virtualenv and keep govprep-db.json out of git.\n",
};

// ================================================================== check
test("a clean repository passes (templates, .gitkeep, metrics.json, tokenizer and licence notes are allowed)", T, () => {
  const dir = makeRepo(CLEAN);
  try {
    const r = run(dir, "check");
    assert.equal(r.code, 0, r.out);
    assert.match(r.out, /Repository hygiene OK/);
  } finally { cleanup(dir); }
});

const FORBIDDEN = [
  ["backend/govprep-db.json", "local-database"], ["govprep-db.json.bak", "local-database"],
  [".env", "env-file"], ["backend/.env", "env-file"], ["frontend/.env.local", "env-file"], ["backend/.env.production", "env-file"],
  [".venv/lib/site.py", "virtualenv"], ["nlp_service/.venv-nlp/pyvenv.cfg", "virtualenv"], ["venv/bin/activate", "virtualenv"],
  ["backend/node_modules/x/index.js", "node-modules"], ["frontend/node_modules/y/a.js", "node-modules"],
  ["nlp_service/app/__pycache__/main.cpython-312.pyc", "python-cache"],
  ["backend/uploads/notes.pdf", "user-uploads"], ["uploads/notes.pdf", "user-uploads"],
  ["nlp_service/artifacts/bert/subject.pt", "model-artifact"], ["nlp_service/artifacts/tfidf/subject.joblib", "model-artifact"],
  ["nlp_service/artifacts/word2vec/cbow.model", "model-artifact"], ["nlp_service/artifacts/word2vec/cbow.model.wv.vectors.npy", "model-artifact"],
  ["nlp_service/artifacts/bert/model.safetensors", "model-artifact"],
  ["nlp_service/data/public/questions.csv", "dataset"], ["nlp_service/data/processed/public_train.csv", "dataset"],
  ["nlp_service/data/pdf_heldout/pdf.csv", "dataset"],
  [".hygiene-backup/2026-01-01/backend/govprep-db.json", "hygiene-backup"],
];

test("every kind of file that must never be committed is detected, with its reason", T, () => {
  const dir = makeRepo({ ...CLEAN, ...Object.fromEntries(FORBIDDEN.map(([p]) => [p, "x"])) });
  try {
    const found = violations(dir);
    for (const [file, rule] of FORBIDDEN) {
      assert.ok(found.some((v) => v.file === file && v.rule === rule), `${file} should be flagged as ${rule}; got ${JSON.stringify(found.filter((v) => v.file === file))}`);
    }
    assert.ok(found.every((v) => v.why && v.why.length > 10));
    assert.equal(run(dir, "check").code, 1);
  } finally { cleanup(dir); }
});

test("secrets inside text files are detected; placeholders and the existing dev-default string are not", T, () => {
  const bad = {
    "a/keys.js": `const k = "${GROQ}";`, "a/openai.txt": `token ${OPENAI}`, "a/aws.cfg": `id=${AWS}`,
    "a/id_rsa.txt": PEM, "a/config.txt": REAL_JWT, "a/groq.conf": `GROQ_API_KEY = "${GROQ}"`,
  };
  const dir = makeRepo({ ...CLEAN, ...bad });
  try {
    const found = violations(dir);
    const rules = {};
    for (const v of found.filter((x) => x.rule.startsWith("secret:"))) (rules[v.file] ||= []).push(v.rule);
    for (const k of Object.keys(rules)) rules[k].sort();
    assert.deepEqual(rules, {
      "a/keys.js": ["secret:groq-key"], "a/openai.txt": ["secret:openai-style-key"], "a/aws.cfg": ["secret:aws-access-key"],
      "a/id_rsa.txt": ["secret:private-key"], "a/config.txt": ["secret:secret-assignment"],
      "a/groq.conf": ["secret:groq-key", "secret:secret-assignment"],     // a real key in a GROQ_API_KEY line trips both rules
    });
    // the clean files (templates with placeholders, code that reads process.env with a dev default) are not flagged
    assert.ok(!found.some((v) => v.file.startsWith("backend/")), JSON.stringify(found.filter((v) => v.file.startsWith("backend/"))));
  } finally { cleanup(dir); }
});

test("binary files are not scanned for secrets; the size limit is configurable", T, () => {
  const bin = Buffer.concat([Buffer.from([0, 1, 2, 3]), Buffer.from(GROQ)]);
  const dir = makeRepo({ ...CLEAN, "assets/blob.bin": bin, "assets/big.dat": Buffer.alloc(2 * 1024 * 1024) });
  try {
    assert.deepEqual(violations(dir), []);                                   // 2 MB is fine under the default 25 MB
    const small = violations(dir, "--max-mb", "1");
    assert.deepEqual(small.map((v) => [v.file, v.rule]), [["assets/big.dat", "large-file"]]);
  } finally { cleanup(dir); }
});

test("staged mode checks only what is about to be committed, and reads the staged content", T, () => {
  const dir = makeRepo(CLEAN);
  try {
    assert.equal(run(dir, "check", "--staged").code, 0);
    fs.writeFileSync(path.join(dir, ".env"), "JWT_SECRET=x\n");               // untracked and unstaged: not a problem yet
    assert.equal(run(dir, "check", "--staged").code, 0);
    assert.equal(run(dir, "check").code, 0);
    git(dir, "add", "-f", ".env");                                            // now it would be committed
    assert.equal(run(dir, "check", "--staged").code, 1);
    git(dir, "rm", "--cached", "-q", ".env");

    fs.writeFileSync(path.join(dir, "notes.txt"), "all fine\n");
    git(dir, "add", "notes.txt");
    fs.writeFileSync(path.join(dir, "notes.txt"), `leaked ${GROQ}\n`);        // only the working copy has the secret
    assert.equal(run(dir, "check", "--staged").code, 0, "the staged blob is clean");
    git(dir, "add", "notes.txt");
    const r = run(dir, "check", "--staged");
    assert.equal(r.code, 1);
    assert.match(r.out, /notes\.txt.*secret:groq-key/);
    assert.match(r.out, /git restore --staged/);
  } finally { cleanup(dir); }
});

test("outside a git repository the guard says so (exit 2) and bad usage prints help", T, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "govprep-nogit-"));
  try {
    const r = run(dir, "check");
    assert.equal(r.code, 2);
    assert.match(r.err, /Not inside a git repository/);
    assert.equal(run(dir, "bogus").code, 2);
    assert.match(run(dir).out, /Usage/);
  } finally { cleanup(dir); }
});

test("the project as shipped passes its own hygiene check, and git's ignore rules match the guard's rules", T, () => {
  // Copy the project (minus heavyweight/untracked folders) into a throwaway repo, drop in representative
  // files, let git apply the real .gitignore files with a plain `git add -A`, and check what would be committed.
  const skip = new Set(["node_modules", ".venv", ".venv-nlp", "venv", "__pycache__", "build", ".git", ".pytest_cache", ".hygiene-backup"]);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "govprep-project-"));
  try {
    fs.cpSync(REPO_ROOT, dir, { recursive: true, filter: (src) => !skip.has(path.basename(src)) });
    const put = (rel, text = "x") => { fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true }); fs.writeFileSync(path.join(dir, rel), text); };
    const SHOULD_COMMIT = ["nlp_service/data/public/SOURCES.md", "backend/.env.example", "nlp_service/artifacts/bert/metrics.json",
      "nlp_service/artifacts/bert/tokenizer/tokenizer.json", "nlp_service/artifacts/comparison/comparison.md"];
    const SHOULD_IGNORE = ["nlp_service/data/public/questions.csv", "nlp_service/data/processed/public_train.csv", "backend/.env", "backend/.env.local",
      "frontend/.env.production", "nlp_service/artifacts/bert/subject.pt", "nlp_service/artifacts/tfidf/subject.joblib", "nlp_service/artifacts/word2vec/cbow.model",
      "backend/govprep-db.json", "backend/uploads/student.pdf", ".hygiene-backup/2026/x", "venv/bin/activate", "nlp_service/.venv-nlp/pyvenv.cfg"];
    [...SHOULD_COMMIT, ...SHOULD_IGNORE].forEach((f) => put(f));
    git(dir, "init", "-q"); git(dir, "config", "user.email", "t@e.org"); git(dir, "config", "user.name", "T");
    git(dir, "add", "-A");
    const r = run(dir, "check");
    assert.equal(r.code, 0, r.out);
    const files = tracked(dir);
    for (const must of ["backend/server.js", "backend/nlp/router.js", "scripts/repo-hygiene.js", ".gitignore", ...SHOULD_COMMIT]) assert.ok(files.includes(must), `${must} should be committable`);
    for (const never of SHOULD_IGNORE) assert.ok(!files.includes(never), `${never} must be ignored by git`);
  } finally { cleanup(dir); }
});

// ================================================================== fix (safe)
const FIXABLE = {
  ...CLEAN,
  ".gitignore": ".hygiene-backup/\n",
  "backend/govprep-db.json": '{"users":[{"name":"real-student","passwordHash":"$2a$10$abc"}]}\n',
  "backend/.env": "GROQ_API_KEY=whatever\n",
  ".venv/lib/x.py": "print('venv')\n",
  "backend/uploads/student-notes.pdf": "%PDF-1.4 real user upload\n",
};

test("fix is a dry run by default: it changes nothing", T, () => {
  const dir = makeRepo(FIXABLE);
  try {
    const before = tracked(dir);
    const r = run(dir, "fix");
    assert.equal(r.code, 0);
    assert.match(r.out, /DRY RUN \(nothing changed\)/);
    assert.match(r.out, /backend\/govprep-db\.json/);
    assert.deepEqual(tracked(dir), before);
    assert.ok(!fs.existsSync(path.join(dir, ".hygiene-backup")));
  } finally { cleanup(dir); }
});

test("fix --apply backs up byte for byte, untracks, and never touches the working files", T, () => {
  const dir = makeRepo(FIXABLE);
  const watched = ["backend/govprep-db.json", "backend/.env", ".venv/lib/x.py", "backend/uploads/student-notes.pdf"];
  const originals = Object.fromEntries(watched.map((f) => [f, sha(path.join(dir, f))]));
  try {
    const r = run(dir, "fix", "--apply");
    assert.equal(r.code, 0, r.err);
    assert.match(r.out, /Backed up to \.hygiene-backup\//);
    const stampDirs = fs.readdirSync(path.join(dir, ".hygiene-backup"));
    assert.equal(stampDirs.length, 1);
    for (const f of watched) {
      assert.equal(sha(path.join(dir, f)), originals[f], `${f}: the working file must be untouched`);
      assert.equal(sha(path.join(dir, ".hygiene-backup", stampDirs[0], f)), originals[f], `${f}: the backup must be identical`);
    }
    const files = tracked(dir);
    for (const f of watched) assert.ok(!files.includes(f), `${f} should no longer be tracked`);
    assert.ok(files.includes("backend/server.js") && files.includes("backend/uploads/.gitkeep"));      // everything else stays tracked
    const status = git(dir, "status", "--porcelain").stdout;
    assert.match(status, /^D {2}backend\/govprep-db\.json$/m);                                          // staged removal from the index only
    assert.ok(!/^\?\? \.hygiene-backup/m.test(status), "the backup folder is ignored by git");
    assert.match(run(dir, "check").out, /OK/);
    assert.match(run(dir, "fix", "--apply").out, /Nothing to untrack/);                                  // idempotent
  } finally { cleanup(dir); }
});

test("fix does not touch secrets inside files (rotate them) and tolerates a file missing from the working tree", T, () => {
  const dir = makeRepo({ ...FIXABLE, "config/keys.txt": `key ${GROQ}\n` });
  try {
    fs.rmSync(path.join(dir, "backend", ".env"));                                                        // tracked but gone from disk
    const r = run(dir, "fix", "--apply");
    assert.equal(r.code, 0, r.err);
    assert.match(r.out, /MANUAL\s+config\/keys\.txt\s+\[secret:groq-key\]/);
    assert.match(r.out, /backend\/\.env.*not in the working tree/);
    assert.ok(tracked(dir).includes("config/keys.txt"));
    assert.equal(fs.readFileSync(path.join(dir, "config", "keys.txt"), "utf8"), `key ${GROQ}\n`);
  } finally { cleanup(dir); }
});

test("fix --apply refuses to run unless the backup folder is ignored, and then changes nothing", T, () => {
  const { ".gitignore": _omit, ...withoutIgnore } = FIXABLE;
  const dir = makeRepo(withoutIgnore);
  try {
    const before = tracked(dir);
    const r = run(dir, "fix", "--apply");
    assert.equal(r.code, 2);
    assert.match(r.err, /not ignored by git/);
    assert.deepEqual(tracked(dir), before);
    assert.ok(!fs.existsSync(path.join(dir, ".hygiene-backup")));
  } finally { cleanup(dir); }
});

// ================================================================== hook
test("the pre-commit hook blocks forbidden commits and allows clean ones", T, () => {
  const dir = makeRepo(CLEAN);
  try {
    const installed = run(dir, "install-hook");
    assert.equal(installed.code, 0, installed.err);
    const hook = path.join(dir, ".git", "hooks", "pre-commit");
    assert.ok(fs.existsSync(hook));
    assert.match(fs.readFileSync(hook, "utf8"), /govprep-hygiene-hook/);

    fs.writeFileSync(path.join(dir, "ok.txt"), "fine\n");
    git(dir, "add", "ok.txt");
    assert.equal(git(dir, "commit", "-q", "-m", "clean").status, 0, "a clean commit must go through");

    fs.mkdirSync(path.join(dir, "backend"), { recursive: true });
    fs.writeFileSync(path.join(dir, "backend", "govprep-db.json"), "{}");
    git(dir, "add", "-f", "backend/govprep-db.json");
    const blocked = git(dir, "commit", "-q", "-m", "oops");
    assert.notEqual(blocked.status, 0, "committing the local database must be blocked");
    assert.match(blocked.stdout + blocked.stderr, /govprep-db\.json.*local-database/);
    assert.ok(!git(dir, "log", "--oneline").stdout.includes("oops"));
  } finally { cleanup(dir); }
});

test("the hook installer never overwrites someone else's hook silently", T, () => {
  const dir = makeRepo(CLEAN);
  try {
    const hook = path.join(dir, ".git", "hooks", "pre-commit");
    fs.mkdirSync(path.dirname(hook), { recursive: true });
    fs.writeFileSync(hook, "#!/bin/sh\necho my own hook\n", { mode: 0o755 });
    const refused = run(dir, "install-hook");
    assert.equal(refused.code, 2);
    assert.match(refused.err, /different pre-commit hook already exists/);
    assert.equal(fs.readFileSync(hook, "utf8"), "#!/bin/sh\necho my own hook\n");

    assert.match(run(dir, "uninstall-hook").out, /not ours; left untouched/);   // and it will not remove it either
    assert.ok(fs.existsSync(hook));

    assert.equal(run(dir, "install-hook", "--force").code, 0);
    const backups = fs.readdirSync(path.dirname(hook)).filter((f) => f.startsWith("pre-commit.bak-"));
    assert.equal(backups.length, 1);
    assert.equal(fs.readFileSync(path.join(path.dirname(hook), backups[0]), "utf8"), "#!/bin/sh\necho my own hook\n");
    assert.match(run(dir, "uninstall-hook").out, /Hook removed/);
    assert.ok(!fs.existsSync(hook));
  } finally { cleanup(dir); }
});
