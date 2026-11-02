"use strict";
// Cross-platform test runner (no globbing, so it works the same on Windows, macOS and Linux).
//   node test/run-tests.js              run every *.test.js file
//   node test/run-tests.js regression   only files whose name contains "regression"
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const filter = process.argv[2];
const files = fs
  .readdirSync(__dirname)
  .filter((f) => f.endsWith(".test.js") && (!filter || f.includes(filter)))
  .sort()
  .map((f) => path.join(__dirname, f));

if (!files.length) {
  console.error(`No test files match "${filter}".`);
  process.exit(1);
}
const result = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" });
process.exit(result.status === null ? 1 : result.status);
