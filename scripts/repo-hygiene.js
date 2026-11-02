#!/usr/bin/env node
"use strict";
/**
 * Repository hygiene guard for GovPrep-AI (no dependencies; needs Node 18+ and git).
 *
 *   node scripts/repo-hygiene.js check [--staged] [--max-mb N] [--json]
 *   node scripts/repo-hygiene.js fix [--apply]
 *   node scripts/repo-hygiene.js install-hook [--force]   |   uninstall-hook
 *
 * Why: .gitignore cannot protect a file that is already tracked, nor `git add -f`. This guard checks
 * what git actually tracks (or what is staged for the next commit) for things that must never be
 * committed:
 *   - the local database (govprep-db.json): user accounts, password hashes, extracted PDF text
 *   - .env files (secrets; .env.example / .env.sample are fine)
 *   - virtual environments (.venv, venv, .venv-nlp, pyvenv.cfg), node_modules, __pycache__
 *   - user uploads (files in backend/uploads/) and NLP datasets (files in nlp_service/data/public, pdf_heldout, processed)
 *   - generated model weights (.pt .pth .joblib .safetensors .ckpt .onnx .npy .h5 .model ...)
 *   - files above the size limit (default 25 MB)
 *   - secrets inside text files (Groq / OpenAI-style keys, AWS keys, private keys, real-looking
 *     JWT_SECRET / GROQ_API_KEY assignments)
 *
 * `check` exits 0 (clean), 1 (violations) or 2 (not a git repository / bad usage).
 *
 * `fix` is SAFE BY DESIGN: it is a dry run unless you pass --apply. With --apply it first copies each
 * offending file that exists in the working tree into .hygiene-backup/<timestamp>/ (and verifies the
 * copy byte for byte), and only then runs `git rm --cached`, which removes the file from git's index
 * but leaves your working copy untouched. It never deletes or edits a working file. Secrets found
 * inside files are reported but not changed: rotate them (a leaked secret stays in git history).
 */
const { execFileSync, spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const BACKUP_DIR = ".hygiene-backup";
const ENV_TEMPLATES = new Set([".env.example", ".env.sample", ".env.template", ".env.dist"]);
const VENV_DIRS = new Set([".venv", "venv", ".venv-nlp", "virtualenv"]);
const MODEL_EXT = new Set([".pt", ".pth", ".joblib", ".safetensors", ".ckpt", ".onnx", ".npy", ".h5"]);
const HOOK_MARKER = "# govprep-hygiene-hook";

const PATH_RULES = [
  { id: "local-database", why: "the local database holds user accounts, password hashes and extracted PDF text",
    test: (p, b) => /^govprep-db\.json(\..+)?$/.test(b) },
  { id: "env-file", why: "environment files hold secrets (use .env.example for documentation)",
    test: (p, b) => (b === ".env" || b.startsWith(".env.")) && !ENV_TEMPLATES.has(b) },
  { id: "virtualenv", why: "virtual environments are machine-specific and huge",
    test: (p, b, segs) => segs.some((s) => VENV_DIRS.has(s)) || b === "pyvenv.cfg" },
  { id: "node-modules", why: "installed packages are reproducible from package-lock.json",
    test: (p, b, segs) => segs.includes("node_modules") },
  { id: "python-cache", why: "bytecode caches are generated",
    test: (p, b, segs) => segs.includes("__pycache__") || b.endsWith(".pyc") },
  { id: "user-uploads", why: "uploaded PDFs are user data",
    test: (p, b) => /^(backend\/)?uploads\//.test(p) && b !== ".gitkeep" },
  { id: "model-artifact", why: "generated model weights are large and reproducible (keep metrics.json instead)",
    test: (p, b) => MODEL_EXT.has(path.extname(b).toLowerCase()) || /\.model(\..+)?$/.test(b) },
  { id: "dataset", why: "NLP datasets may be licensed or contain user PDF text",
    test: (p, b) => /(^|\/)nlp_service\/data\/(public|pdf_heldout|processed)\//.test(p) && b !== ".gitkeep" && b !== "SOURCES.md" },
  { id: "hygiene-backup", why: "backups made by this tool must stay local",
    test: (p, b, segs) => segs.includes(BACKUP_DIR) },
];

const PLACEHOLDER = /^(your|change|example|placeholder|replace|xxx|<|\$\{|\{\{|process\.env|\.\.\.|todo|secret$|password$|test-)/i;
const SECRET_RULES = [
  { id: "groq-key", re: /\bgsk_[A-Za-z0-9]{20,}\b/ },
  { id: "openai-style-key", re: /\bsk-[A-Za-z0-9_-]{20,}\b/ },
  { id: "aws-access-key", re: /\bAKIA[0-9A-Z]{16}\b/ },
  { id: "private-key", re: /-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----/ },
  { id: "secret-assignment", re: /\b(GROQ_API_KEY|JWT_SECRET|OPENAI_API_KEY)\s*[:=]\s*["']?([^\s"'#,;)]{12,})/,
    accept: (m) => !PLACEHOLDER.test(m[2]) && !/change|your-|example|placeholder/i.test(m[2]) },
];

// ------------------------------------------------------------------ git helpers
function git(args, cwd, { input } = {}) {
  return execFileSync("git", args, { cwd, encoding: "utf8", input, stdio: ["pipe", "pipe", "pipe"], maxBuffer: 256 * 1024 * 1024 });
}
function repoRoot(cwd) {
  try { return git(["rev-parse", "--show-toplevel"], cwd).trim(); } catch { return null; }
}
function listFiles(root, staged) {
  const out = staged
    ? git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], root)
    : git(["ls-files", "-z"], root);
  return out.split("\0").filter(Boolean);
}
function readStaged(root, rel) {
  const r = spawnSync("git", ["show", `:${rel}`], { cwd: root, maxBuffer: 256 * 1024 * 1024 });
  return r.status === 0 ? r.stdout : null;
}
function readFile(root, rel, staged) {
  if (staged) return readStaged(root, rel);
  try { return fs.readFileSync(path.join(root, rel)); } catch { return null; }
}

// ------------------------------------------------------------------ checks
function pathViolations(rel) {
  const base = path.posix.basename(rel);
  const segs = rel.split("/");
  return PATH_RULES.filter((r) => r.test(rel, base, segs)).map((r) => ({ file: rel, rule: r.id, why: r.why }));
}

function isText(buf) {
  return !buf.subarray(0, 8000).includes(0);
}

function check(root, { staged = false, maxMb = 25 } = {}) {
  const violations = [];
  for (const rel of listFiles(root, staged)) {
    violations.push(...pathViolations(rel));
    const buf = readFile(root, rel, staged);
    if (!buf) continue;
    if (buf.length > maxMb * 1024 * 1024) {
      violations.push({ file: rel, rule: "large-file", why: `larger than ${maxMb} MB (${(buf.length / 1048576).toFixed(1)} MB)` });
    }
    if (buf.length > 2 * 1024 * 1024 || !isText(buf)) continue;            // scan reasonably small text files only
    const text = buf.toString("utf8");
    for (const rule of SECRET_RULES) {
      const m = text.match(rule.re);
      if (m && (!rule.accept || rule.accept(m))) {
        violations.push({ file: rel, rule: `secret:${rule.id}`, why: "looks like a real secret; rotate it if it was ever committed" });
      }
    }
  }
  return violations;
}

// ------------------------------------------------------------------ fix (safe)
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

function fix(root, { apply = false } = {}) {
  const all = check(root, { staged: false });
  const tracked = all.filter((v) => !v.rule.startsWith("secret:") && v.rule !== "large-file");
  const files = [...new Set(tracked.map((v) => v.file))];
  const manual = all.filter((v) => v.rule.startsWith("secret:") || v.rule === "large-file");
  const plan = { apply, files, manual, backupDir: null, actions: [] };
  if (!files.length || !apply) return plan;

  const ignored = spawnSync("git", ["check-ignore", "-q", `${BACKUP_DIR}/x`], { cwd: root }).status === 0;
  if (!ignored) {
    throw new Error(`${BACKUP_DIR}/ is not ignored by git. Add "${BACKUP_DIR}/" to .gitignore first, so the backup cannot be committed.`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupRoot = path.join(root, BACKUP_DIR, stamp);
  plan.backupDir = backupRoot;
  for (const rel of files) {
    const src = path.join(root, rel);
    if (fs.existsSync(src)) {                                                // 1) safe copy, verified
      const dest = path.join(backupRoot, rel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(src, dest);
      if (sha256(src) !== sha256(dest)) throw new Error(`Backup of ${rel} does not match the original; nothing was untracked.`);
      plan.actions.push({ file: rel, backedUp: true });
    } else {
      plan.actions.push({ file: rel, backedUp: false });
    }
  }
  for (const rel of files) git(["rm", "--cached", "-q", "--", rel], root);   // 2) index only: the working file stays
  return plan;
}

// ------------------------------------------------------------------ hook
function hookPath(root) {
  return path.resolve(root, git(["rev-parse", "--git-path", "hooks/pre-commit"], root).trim());
}
function installHook(root, { force = false } = {}) {
  const file = hookPath(root);
  const body = `#!/bin/sh\n${HOOK_MARKER}\nnode scripts/repo-hygiene.js check --staged || exit 1\n`;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  let backup = null;
  if (fs.existsSync(file) && !fs.readFileSync(file, "utf8").includes(HOOK_MARKER)) {
    if (!force) throw new Error(`A different pre-commit hook already exists at ${file}. Re-run with --force to back it up and replace it.`);
    backup = `${file}.bak-${Date.now()}`;
    fs.copyFileSync(file, backup);
  }
  fs.writeFileSync(file, body, { mode: 0o755 });
  try { fs.chmodSync(file, 0o755); } catch { /* not supported on this file system */ }
  return { file, backup };
}
function uninstallHook(root) {
  const file = hookPath(root);
  if (!fs.existsSync(file)) return { removed: false, reason: "no pre-commit hook" };
  if (!fs.readFileSync(file, "utf8").includes(HOOK_MARKER)) return { removed: false, reason: "the pre-commit hook is not ours; left untouched" };
  fs.rmSync(file);
  return { removed: true };
}

// ------------------------------------------------------------------ CLI
function main(argv, cwd = process.cwd()) {
  const [command, ...rest] = argv;
  const flag = (name) => rest.includes(name);
  const option = (name, fallback) => { const i = rest.indexOf(name); return i >= 0 && rest[i + 1] ? rest[i + 1] : fallback; };
  const say = (s = "") => process.stdout.write(s + "\n");
  const root = repoRoot(cwd);
  if (!["check", "fix", "install-hook", "uninstall-hook"].includes(command)) {
    say("Usage: node scripts/repo-hygiene.js <check [--staged] [--max-mb N] [--json] | fix [--apply] | install-hook [--force] | uninstall-hook>");
    return 2;
  }
  if (!root) { process.stderr.write("Not inside a git repository.\n"); return 2; }

  if (command === "check") {
    const maxMb = Number(option("--max-mb", 25));
    const staged = flag("--staged");
    const violations = check(root, { staged, maxMb });
    if (flag("--json")) say(JSON.stringify({ clean: !violations.length, violations }, null, 2));
    else if (!violations.length) say(`Repository hygiene OK (${staged ? "staged changes" : "tracked files"} checked).`);
    else {
      say(`Repository hygiene: ${violations.length} problem(s) in ${staged ? "staged changes" : "tracked files"}:`);
      for (const v of violations) say(`  FORBIDDEN  ${v.file}  [${v.rule}]  ${v.why}`);
      say("");
      say(staged ? "Commit blocked. Unstage with: git restore --staged <file>" : "To stop tracking these safely (backup first, working files kept):  node scripts/repo-hygiene.js fix --apply");
    }
    return violations.length ? 1 : 0;
  }
  if (command === "fix") {
    let plan;
    try { plan = fix(root, { apply: flag("--apply") }); } catch (e) { process.stderr.write(`${e.message}\n`); return 2; }
    if (!plan.files.length) say("Nothing to untrack: no tracked file breaks a path rule.");
    else if (!plan.apply) {
      say("DRY RUN (nothing changed). With --apply these tracked files would be backed up, then removed from git's index (your working copies stay):");
      plan.files.forEach((f) => say(`  ${f}`));
      say("Run again with --apply to do it.");
    } else {
      say(`Backed up to ${path.relative(root, plan.backupDir)}/ and untracked (working files untouched):`);
      plan.actions.forEach((a) => say(`  ${a.file}${a.backedUp ? "" : "  (not in the working tree: nothing to back up)"}`));
      say("Next: git commit -m \"Stop tracking local data\". Note: older commits still contain these files; rewriting history (git filter-repo) is a separate, deliberate step.");
    }
    for (const v of plan.manual) say(`  MANUAL  ${v.file}  [${v.rule}]  ${v.why}`);
    return 0;
  }
  if (command === "install-hook") {
    try { const r = installHook(root, { force: flag("--force") }); say(`Installed ${r.file}${r.backup ? ` (previous hook saved as ${r.backup})` : ""}.`); return 0; }
    catch (e) { process.stderr.write(`${e.message}\n`); return 2; }
  }
  const r = uninstallHook(root);
  say(r.removed ? "Hook removed." : r.reason);
  return 0;
}

module.exports = { check, fix, installHook, uninstallHook, pathViolations, PATH_RULES, SECRET_RULES, BACKUP_DIR, main };
if (require.main === module) process.exit(main(process.argv.slice(2)));
