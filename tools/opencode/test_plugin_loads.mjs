// Load check for the project plugins in .opencode/plugins/.
//
// Run from the repo root:
//   node --experimental-strip-types tools/opencode/test_plugin_loads.mjs
//
// THIS EXISTS BECAUSE TWO PLUGINS WERE GREEN AND DEAD.
//
// On 2026-10-03 both plugins committed with passing tests, and the opencode service logged:
//
//   WARN failed to load plugin target=...\.opencode\plugins\claims-gate.ts
//     cause="Cause([Die(ResolveMessage: Cannot find package '@opencode/plugin' imported from ...)]"
//
// opencode SKIPS a plugin that fails to load rather than crashing, so nothing else reports it. The
// harness that missed it (test_claims_gate.mjs) stubs @opencode/plugin, which is correct for testing
// decision logic and fatal for this question -- the stub makes an unresolvable import resolvable.
//
// The first fix was to install @opencode/plugin into .opencode/plugins/node_modules. That made `node`
// resolve it and did NOT make the service resolve it, across a restart, with byte-identical packages
// (2.0.11, same exports map) in both the repo and the global config directory. The service resolves
// that package only under the global config dir.
//
// So the plugins no longer import it at all. `Plugin.define({ id, setup })` is a typing and ergonomics
// helper over a plain object, and the default export is that object. The check below therefore
// asserts the ABSENCE of that import, because that absence is the property that makes these plugins
// loadable here -- and a future contributor reaching for the documented import would otherwise
// reintroduce a silent, load-time-only failure that every logic test still passes.

import { pathToFileURL } from "node:url"
import { resolve } from "node:path"
import { existsSync, readFileSync } from "node:fs"

const REPO = "C:\\AI\\ninfer-v3-windows"
const PLUGIN_DIR = resolve(REPO, ".opencode/plugins")
const PLUGINS = ["rules-inject.ts", "claims-gate.ts"]

let pass = 0
let fail = 0
function check(name, ok, detail = "") {
  if (ok) pass++
  else fail++
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : `  ${detail}`}`)
}

// ---- the loadability property itself ----
// Strip comments before scanning. The first version of this check matched the string
// `from "@opencode/plugin"` anywhere in the file, and the file's own explanatory comment contains
// that text -- so it failed on prose and would have driven someone to delete the explanation instead
// of the import. That is the repo's own rule about guarding with string presence over prose, and the
// rule existed before I broke it.
function codeOnly(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n")
    .map((l) => l.replace(/^\s*\/\/.*$/, ""))
    .join("\n")
}

for (const file of PLUGINS) {
  const code = codeOnly(readFileSync(resolve(PLUGIN_DIR, file), "utf8"))
  const bare = [...code.matchAll(/^\s*import\s+[^;]*?from\s+["']([^"']+)["']/gm)].map((m) => m[1])
  const external = bare.filter((s) => !s.startsWith("node:") && !s.startsWith(".") && !s.startsWith("/"))
  check(
    `${file} does not import @opencode/plugin`,
    !/from\s+["']@opencode\/plugin["']/.test(code),
    "that package resolves only under the global config dir; the service would skip this file silently",
  )
  check(
    `${file} imports only node builtins and relative paths`,
    external.length === 0,
    `non-builtin: ${external.join(", ")}`,
  )
}

// ---- and that the files really do load and export what opencode expects ----
for (const file of PLUGINS) {
  try {
    const mod = await import(pathToFileURL(resolve(PLUGIN_DIR, file)).href)
    const plugin = mod.default ?? mod
    check(`${file} imports with NO install and NO stub`, !!plugin?.id, `id=${plugin?.id}`)
    check(`${file} exports a setup function`, typeof plugin?.setup === "function")
  } catch (err) {
    check(`${file} imports with NO install and NO stub`, false, String(err).slice(0, 200))
  }
}

// ---- the install is no longer required, so record that it is gone on purpose ----
const hasInstall = existsSync(resolve(PLUGIN_DIR, "node_modules/@opencode/plugin/package.json"))
console.log(
  hasInstall
    ? `\n  NOTE: .opencode/plugins/node_modules is present (${PLUGIN_DIR}\\node_modules). It is no\n` +
        "  longer needed -- the plugins import nothing -- and it is 139 MiB. Safe to delete:\n" +
        "    npm uninstall --prefix .opencode/plugins @opencode/plugin"
    : "\n  no local node_modules: the plugins are self-contained, as intended.",
)

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) {
  console.log(
    "\n  opencode SKIPS a plugin that fails to load -- it does not crash, so the only signal is this.\n" +
      "  Server log: ~/.local/share/opencode/log/opencode.log, grep 'failed to load plugin'",
  )
}
process.exit(fail === 0 ? 0 : 1)
