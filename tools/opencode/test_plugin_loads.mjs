// Load check for the project plugins: can Node resolve @opencode/plugin from them WITHOUT a stub?
//
// Run from the repo root:
//   node --experimental-strip-types tools/opencode/test_plugin_loads.mjs
//
// WHY THIS EXISTS, and it is the check whose absence let two dead plugins get reported as working.
//
// On 2026-10-03 both plugins were committed with green logic tests. The opencode server log then said:
//
//   WARN failed to load plugin target=...\.opencode\plugins\claims-gate.ts
//     cause="Cause([Die(ResolveMessage: Cannot find package '@opencode/plugin' imported from ...)]"
//
// @opencode/plugin only resolves inside the global config dir, where thinking-guard.ts lives. The
// project plugins sit in this repo, which had no node_modules, so module resolution failed and
// opencode skipped both. The denial never fired and the injection never happened -- and the tests
// still passed, because tools/opencode/stub-opencode-plugin.mjs stubbed the very package that failed.
//
// That stub is still right for testing the DECISION LOGIC in isolation: it keeps the harness
// independent of an install that may be missing. It is the wrong instrument for asking "would the
// service load this", because it makes an unresolvable import resolvable. So there are two
// harnesses, and this is the one that answers that question.
//
// This script imports the real files with no resolver hook and no stub. A non-zero exit here means
// opencode will silently skip the plugin -- it does NOT crash the service, so nothing else will tell
// you.

import { pathToFileURL } from "node:url"
import { resolve } from "node:path"
import { existsSync } from "node:fs"

const REPO = "C:\\AI\\ninfer-v3-windows"
const PLUGIN_DIR = resolve(REPO, ".opencode/plugins")

let pass = 0
let fail = 0
function check(name, ok, detail = "") {
  if (ok) pass++
  else fail++
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : `  ${detail}`}`)
}

// The dependency has to be installed and committed as a lockfile, or this is the same bug again.
check(
  "package.json declares @opencode/plugin",
  existsSync(resolve(PLUGIN_DIR, "package.json")),
)
check(
  "package-lock.json is committed (resolves reproducibly)",
  existsSync(resolve(PLUGIN_DIR, "package-lock.json")),
  "run: npm install --prefix .opencode/plugins",
)
check(
  "@opencode/plugin is actually installed under .opencode/plugins/node_modules",
  existsSync(resolve(PLUGIN_DIR, "node_modules/@opencode/plugin/package.json")),
  "run: npm install --prefix .opencode/plugins",
)

for (const file of ["rules-inject.ts", "claims-gate.ts"]) {
  try {
    const mod = await import(pathToFileURL(resolve(PLUGIN_DIR, file)).href)
    const plugin = mod.default ?? mod
    check(`${file} imports with the REAL @opencode/plugin (no stub)`, !!plugin?.id, `id=${plugin?.id}`)
    check(`${file} exports a setup function`, typeof plugin?.setup === "function")
  } catch (err) {
    check(`${file} imports with the REAL @opencode/plugin (no stub)`, false, String(err).slice(0, 200))
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) {
  console.log(
    "\n  opencode SKIPS a plugin that fails to load -- it does not crash, so the only signal is this.\n" +
      "  Check the server log for 'failed to load plugin' at\n" +
      "  ~/.local/share/opencode/log/opencode.log",
  )
}
process.exit(fail === 0 ? 0 : 1)
