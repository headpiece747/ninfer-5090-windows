// Standalone falsification harness for .opencode/plugins/claims-gate.ts
//
//   node --experimental-strip-types tools/opencode/test_claims_gate.mjs
//
// Drives the ONE enforcement path the plugin registers: the `tool.execute.before` hook with a
// throw. There is deliberately no second path to drive.
//
// The plugin previously also wrapped write/edit/patch/shell via `ctx.tool.transform`. It was removed
// because the hook alone already denies, confirmed live -- a Markdown write citing line 999999 of a
// 706-line file was refused while a write citing line 42 was created, and no BLOCK line was ever
// logged because the hook throws before execute is reached. Two mechanisms for one rule is harder to
// reason about than one, and the redundancy came from a misread allow as a swallowed deny rather than
// from caution. The `transform` stub below exists only to ASSERT the plugin does not use it again.
//
// Also encodes the earlier lesson: every version of this harness drove the plugin's own interface
// rather than the host's, each was fully green, and each gate refused nothing. So the suite asserts
// the DENIAL SIGNAL on the real input shapes the host sends (`write` -> {path, content},
// `shell` -> {command}), and covers a Markdown file that does not exist yet.

import { pathToFileURL, fileURLToPath } from "node:url"
import { readFileSync, existsSync } from "node:fs"
import { dirname, resolve } from "node:path"

// Derived from this file's own location, never a literal: a hardcoded repo path passes on the one
// machine that has it and fails everywhere else. Same fix as test_plugin_loads.mjs.
const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const mod = await import(pathToFileURL(resolve(REPO, ".opencode/plugins/claims-gate.ts")).href)
const plugin = mod.default ?? mod

let hook = null
const registered = []
let transformCalls = 0

const ctx = {
  location: { directory: REPO },
  tool: {
    hook: async (name, cb) => {
      registered.push(`hook:${name}`)
      if (name === "execute.before") hook = cb
      return { dispose() {} }
    },
    transform: async () => {
      transformCalls++
      return { dispose() {} }
    },
  },
  permission: {
    // If a permission hook is ever registered again, record it: that layer is the wrong one and
    // re-adding it is how this gate shipped inert the first time.
    hook: async (name) => {
      registered.push(`permission:${name}`)
      return { dispose() {} }
    },
  },
}

await plugin.setup(ctx)

if (typeof hook !== "function") throw new Error("tool.execute.before was never registered")

let pass = 0
let fail = 0

function check(name, got, wantDeny) {
  const denied = got instanceof Error
  const ok = denied === wantDeny
  if (ok) pass++
  else fail++
  const detail = denied ? `denied: ${String(got.message).slice(0, 62)}` : "allowed"
  console.log(`${ok ? "PASS" : "FAIL"}  ${name} -> ${detail}${ok ? "" : ` (wanted ${wantDeny ? "deny" : "allow"})`}`)
}

async function call(tool, input) {
  try {
    await hook({ tool, input, sessionID: "t", agent: "build", id: "c1" })
    return null
  } catch (err) {
    return err
  }
}

// ---- one mechanism, one layer ----
check(
  "no permission hook is registered (wrong layer)",
  registered.filter((r) => r.startsWith("permission")).length === 0 ? null : new Error("re-added"),
  false,
)
check(
  "ctx.tool.transform is NOT used (redundant second mechanism)",
  transformCalls === 0 ? null : new Error(`transform called ${transformCalls}x`),
  false,
)

// ---- a doc in docs/ citing a repo-relative path, plus a real past-EOF target ----
const DOC = resolve(REPO, "docs/maintainer/engine-architecture.md")
const CPP = "src/models/qwen3_5/frontend/native_render.cpp"
const CPP_LINES = readFileSync(resolve(REPO, CPP), "utf8").split("\n").length
const AGENTS = resolve(REPO, "AGENTS.md")

const BAD = { path: DOC, content: `see ${CPP}:${CPP_LINES + 5000}` }
const GOOD = { path: DOC, content: `see ${CPP}:42` }
const MISSING = { path: AGENTS, content: "as shown in src/ops/not_a_real_file.cu:12" }
const BOMS = { command: "Set-Content -Path AGENTS.md -Value $t -Encoding utf8" }

// A Markdown file that DOES NOT EXIST YET. This case exists because the live probe was exactly
// this -- a brand-new .md carrying a fabricated citation -- and the plugin allowed it. It had
// `if (!existsSync(target)) return null` under the comment "a new file cannot cite an existing line
// range wrongly", which is false and disabled the gate for every newly created document. Every other
// case used an existing file, so 33 green assertions never touched it.
const NEW_MD = resolve(REPO, "zz-new-document-that-does-not-exist-yet.md")
const NEW_BAD = { path: NEW_MD, content: `see ${CPP}:999999` }
const NEW_MISSING = { path: NEW_MD, content: "see src/ops/not_a_real_file.cu:12" }
const NEW_GOOD = { path: NEW_MD, content: `see ${CPP}:42` }
const assertNewAbsent = !existsSync(NEW_MD)

console.log("\n-- the one path --")
await check("deny  citation past EOF", await call("write", BAD), true)
await check("deny  citation to a missing file", await call("write", MISSING), true)
// The case that shipped a gate which allowed a fabricated citation in a brand-new document.
await check("deny  past EOF in a NEW markdown file", assertNewAbsent ? await call("write", NEW_BAD) : new Error("precondition: file exists"), true)
await check("deny  missing file in a NEW markdown file", assertNewAbsent ? await call("write", NEW_MISSING) : new Error("precondition: file exists"), true)
await check("deny  Set-Content -Encoding utf8", await call("shell", BOMS), true)
await check("deny  Get-Content | Set-Content", await call("shell", { command: "(Get-Content a.md) | Set-Content b.md" }), true)
await check("allow repo-relative citation in range", await call("write", GOOD), false)
await check("allow in-range citation in a NEW markdown file", await call("write", NEW_GOOD), false)
await check("allow no citation", await call("write", { path: AGENTS, content: "plain prose" }), false)
await check("allow external citation (target under docs/research)", await call("write", { path: resolve(REPO, "docs/research/x.md"), content: "see vllm/scheduler.py:900" }), false)
await check("allow non-markdown file", await call("write", { path: resolve(REPO, "zz.cpp"), content: "see x.cpp:999999" }), false)
await check("allow safe WriteAllText", await call("shell", { command: "[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))" }), false)
await check("allow ordinary git", await call("shell", { command: "git log --oneline -1" }), false)
await check("allow a read-only tool", await call("read", { path: AGENTS }), false)

// ---- the edit that removes a bad citation must be allowed ----
// payloadOf used to concatenate `oldString` as well as `newString`, so an edit whose newString
// dropped a stale `profiles.py:107` was refused for carrying that citation in the text it was
// replacing. Live on 2026-10-04: four attempts to correct one stale line number, all refused by
// the string being removed. `oldString` is no longer validated; these two cases pin that, and the
// second one is the control -- without it, "allow" here could pass because nothing is checked.
const EDIT_FIXES = {
  path: DOC,
  oldString: `see ${CPP}:${CPP_LINES + 5000}`,
  newString: `see ${CPP}:42`,
}
await check("allow an edit that REMOVES a past-EOF citation", await call("edit", EDIT_FIXES), false)
await check(
  "deny  an edit that ADDS a past-EOF citation (control)",
  await call("edit", { path: DOC, oldString: "plain prose", newString: `see ${CPP}:${CPP_LINES + 5000}` }),
  true,
)

// ---- fail open: a check that throws on its own bug would break the tool it protects ----
const hostile = [
  { tool: "write" },
  { tool: "write", input: null },
  { tool: "write", input: { path: AGENTS, content: 12345 } },
  { tool: "shell", input: {} },
  { tool: "shell", input: { command: null } },
  { tool: 42 },
  {},
  null,
]
for (const [i, ev] of hostile.entries()) {
  try {
    await hook(ev)
    pass++
    console.log(`PASS  hostile[${i}] survived the hook`)
  } catch (err) {
    fail++
    console.log(`FAIL  hostile[${i}] THREW: ${err}`)
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)