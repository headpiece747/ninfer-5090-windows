// Standalone falsification harness for .opencode/plugins/claims-gate.ts
//
// Drives the REAL plugin file through a fake ctx, so what is tested is the shipped artifact and not a
// transcription of it. @opencode/plugin is stubbed by stub-opencode-plugin.mjs because the package
// only resolves inside the opencode config dir; Plugin.define becomes identity, which is all setup()
// needs.
//
// This lives in the repo rather than in %TEMP% deliberately: a gate whose only test is a scratch
// script is unverifiable once the script is gone, and a reviewer cannot rerun it.
//
// Run from the repo root:
//   node --experimental-strip-types tools/opencode/test_claims_gate.mjs
//
// The rule being honoured: a gate that cannot fail is worse than no gate. Every case below asserts the
// EXPECTED effect, and both directions are covered -- allow AND deny -- because a hook that always
// denies and one that never denies look identical from the outside until they fire at the wrong time.

import { pathToFileURL } from "node:url"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

const REPO = "C:\\AI\\ninfer-v3-windows"
const PLUGIN = resolve(REPO, ".opencode/plugins/claims-gate.ts")

const mod = await import(pathToFileURL(PLUGIN).href)
const plugin = mod.default ?? mod

let permHook = null
let toolHook = null

const ctx = {
  location: { directory: REPO },
  permission: {
    hook: async (name, cb) => {
      if (name !== "evaluate") throw new Error(`unexpected permission hook ${name}`)
      permHook = cb
      return { dispose() {} }
    },
  },
  tool: {
    hook: async (name, cb) => {
      if (name !== "execute.before") throw new Error(`unexpected tool hook ${name}`)
      toolHook = cb
      return { dispose() {} }
    },
  },
}

await plugin.setup(ctx)

if (typeof permHook !== "function") throw new Error("permission.evaluate hook was never registered")
if (typeof toolHook !== "function") throw new Error("tool.execute.before hook was never registered")

function ev(overrides) {
  return { sessionID: "test", action: "edit", resources: [], effect: "allow", ...overrides }
}

async function evaluate(e) {
  await permHook(e)
  return e
}

const AGENTS_LINES = readFileSync(resolve(REPO, "AGENTS.md"), "utf8").split("\n").length

const cases = [
  // ---- Gate 1: citation correctness. DENY and ALLOW both required. ----
  {
    name: "deny  citation past EOF in a .md edit",
    event: ev({
      resources: [resolve(REPO, "AGENTS.md")],
      metadata: { content: `see native_render.cpp:${AGENTS_LINES + 5000} for the fallback` },
    }),
    want: "deny",
  },
  {
    name: "deny  citation to a file that does not exist",
    event: ev({
      resources: [resolve(REPO, "AGENTS.md")],
      metadata: { content: "as shown in src/ops/definitely_not_a_real_file.cu:12" },
    }),
    want: "deny",
  },
  {
    name: "allow citation that is in range",
    event: ev({
      resources: [resolve(REPO, "AGENTS.md")],
      metadata: { content: "as shown in src/models/qwen3_5/frontend/native_render.cpp:42" },
    }),
    want: "allow",
  },
  {
    name: "allow a .md edit with no citation at all",
    event: ev({
      resources: [resolve(REPO, "AGENTS.md")],
      metadata: { content: "the renderer records per-part end offsets" },
    }),
    want: "allow",
  },
  {
    name: "allow citation under docs/research (external by design)",
    event: ev({
      resources: [resolve(REPO, "docs/research/foo.md")],
      metadata: { content: "see vllm/project/vllm/core/scheduler.py:900 for the design" },
    }),
    want: "allow",
  },

  // ---- Gate 2: PowerShell re-encoding. DENY and ALLOW both required. ----
  {
    name: "deny  Set-Content -Encoding utf8",
    event: ev({
      action: "bash",
      metadata: { command: 'Set-Content -Path AGENTS.md -Value $text -Encoding utf8' },
    }),
    want: "deny",
  },
  {
    name: "deny  Get-Content piped into Set-Content",
    event: ev({
      action: "bash",
      metadata: { command: '(Get-Content x.md) | Set-Content y.md' },
    }),
    want: "deny",
  },
  {
    name: "allow Set-Content WITHOUT -Encoding utf8",
    event: ev({ action: "bash", metadata: { command: "Set-Content -Path a.txt -Value x" } }),
    want: "allow",
  },
  {
    name: "allow the safe WriteAllText form",
    event: ev({
      action: "bash",
      metadata: {
        command:
          "[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))",
      },
    }),
    want: "allow",
  },
  {
    name: "allow an ordinary git command",
    event: ev({ action: "bash", metadata: { command: "git log --oneline -1" } }),
    want: "allow",
  },
]

// ---- Fail-open: hostile input must never throw and never deny. ----
const hostile = [
  ev({ resources: [resolve(REPO, "AGENTS.md")], metadata: { content: "a.cpp:notanumber" } }),
  ev({ resources: [":::not a path:::"], metadata: { content: "x.cpp:1" } }),
  ev({ resources: [], metadata: undefined }),
  ev({ resources: [resolve(REPO, "AGENTS.md")], metadata: { content: 12345 } }),
]

let pass = 0
let fail = 0
for (const c of cases) {
  const got = await evaluate(c.event)
  const ok = got.effect === c.want
  if (ok) pass++
  else fail++
  console.log(`${ok ? "PASS" : "FAIL"}  ${c.name}  -> ${got.effect}${ok ? "" : ` (wanted ${c.want})`}`)
}

for (const [i, e] of hostile.entries()) {
  try {
    const got = await evaluate(e)
    const ok = got.effect === "allow"
    if (ok) pass++
    else fail++
    console.log(`${ok ? "PASS" : "FAIL"}  hostile[${i}] survived, effect=${got.effect}`)
  } catch (err) {
    fail++
    console.log(`FAIL  hostile[${i}] THREW: ${err}`)
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
