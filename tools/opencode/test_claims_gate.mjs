// Standalone falsification harness for .opencode/plugins/claims-gate.ts
//
//   node --experimental-strip-types tools/opencode/test_claims_gate.mjs
//
// This harness is on its THIRD contract, and the first two were wrong in the same way:
//
//   v1 drove `permission.evaluate` and asserted `event.effect === "deny"`. That hook is the
//      PERMISSION path -- external_directory, doom_loop, ask-resolved rules -- so an in-workspace
//      edit never reaches it. 14 green assertions on a gate that had never refused anything.
//   v2 asserted on a `metadata.files[].patch` shape inferred from one old log line.
//   v3 drives `tool.execute.before`, which fires for every tool call, and asserts a THROW.
//
// The lesson the versions encode: assert the DENIAL SIGNAL, and drive the payload shape the host
// really sends. `event.input` is the args -- the service log shows `tool=write keys=path,content`
// and `tool=shell keys=command,timeout` -- so the cases below use exactly those shapes.

import { pathToFileURL } from "node:url"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

const REPO = "C:\\AI\\ninfer-v3-windows"
const mod = await import(pathToFileURL(resolve(REPO, ".opencode/plugins/claims-gate.ts")).href)
const plugin = mod.default ?? mod

let beforeHook = null
const registered = []
const ctx = {
  location: { directory: REPO },
  tool: {
    hook: async (name, cb) => {
      registered.push(name)
      if (name === "execute.before") beforeHook = cb
      return { dispose() {} }
    },
  },
  // The permission hook must NOT be registered any more. If a future contributor re-adds it, this
  // harness fails rather than leaving unreachable enforcement behind.
  permission: {
    hook: async (name) => {
      registered.push(`permission.${name}`)
      return { dispose() {} }
    },
  },
}

await plugin.setup(ctx)

if (typeof beforeHook !== "function") throw new Error("tool.execute.before hook was never registered")

let pass = 0
let fail = 0

async function call(tool, input) {
  try {
    await beforeHook({ tool, input, sessionID: "test", agent: "build", id: "c1" })
    return null // no throw -> allowed
  } catch (err) {
    return err
  }
}

function check(name, got, wantDeny) {
  const denied = got !== null
  const ok = denied === wantDeny
  if (ok) pass++
  else fail++
  const detail = denied ? `denied: ${String(got.message).slice(0, 70)}` : "allowed"
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  -> ${detail}${ok ? "" : ` (wanted ${wantDeny ? "deny" : "allow"})`}`)
}

const AGENTS = resolve(REPO, "AGENTS.md")
const AGENTS_LINES = readFileSync(AGENTS, "utf8").split("\n").length
const md = (content) => ({ path: AGENTS, content })

// A document that lives in docs/ but cites a repository-relative path. Resolving citations against
// the document's own directory would deny this, and check_doc_citations.py accepts it -- so the hook
// and the gate have to agree. This case exists because the first harness run reported a past-EOF
// citation as "no such file", which meant the past-EOF branch had never actually executed.
const DOC = resolve(REPO, "docs/maintainer/engine-architecture.md")
const CPP = "src/models/qwen3_5/frontend/native_render.cpp"
const CPP_LINES = readFileSync(resolve(REPO, CPP), "utf8").split("\n").length

// ---- registration: the dead hook stays dead ----
check("registers no permission hook", registered.filter((r) => r.startsWith("permission")).length === 0 ? null : new Error("permission hook re-added"), false)

// ---- Gate 1: citations. Both directions. ----
await check("deny  citation PAST EOF (real file, correct relative path)", await call("write", { path: DOC, content: `see ${CPP}:${CPP_LINES + 5000}` }), true)
await check("deny  citation to a missing file", await call("write", md("as shown in src/ops/not_a_real_file.cu:12")), true)
await check("allow repo-relative citation from a docs/ file", await call("write", { path: DOC, content: `see ${CPP}:42` }), false)
await check("allow citation in range", await call("write", md("as shown in src/models/qwen3_5/frontend/native_render.cpp:42")), false)
await check("allow no citation at all", await call("write", md("the renderer records per-part end offsets")), false)
await check("allow external citation under docs/research", await call("write", { path: resolve(REPO, "docs/research/x.md"), content: "see vllm/scheduler.py:900" }), false)
await check("allow a NON-markdown file with a bad citation", await call("write", { path: resolve(REPO, "zz.cpp"), content: "see x.cpp:999999" }), false)

// ---- Gate 2: PowerShell re-encoding. Both directions. ----
await check("deny  Set-Content -Encoding utf8", await call("shell", { command: "Set-Content -Path AGENTS.md -Value $t -Encoding utf8" }), true)
await check("deny  Get-Content piped into Set-Content", await call("shell", { command: "(Get-Content a.md) | Set-Content b.md" }), true)
await check("allow Set-Content WITHOUT -Encoding utf8", await call("shell", { command: "Set-Content -Path a.txt -Value x" }), false)
await check("allow the safe WriteAllText form", await call("shell", { command: "[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))" }), false)
await check("allow an ordinary git command", await call("shell", { command: "git log --oneline -1" }), false)

// ---- fail open: a check that throws on its own bug would break the tool it protects ----
const hostile = [
  { tool: "write" },
  { tool: "write", input: null },
  { tool: "write", input: { path: AGENTS, content: 12345 } },
  { tool: "shell", input: {} },
  { tool: "shell", input: { command: null } },
  { tool: "read", input: { path: AGENTS } },
  {},
  null,
]
for (const [i, ev] of hostile.entries()) {
  try {
    await beforeHook(ev)
    pass++
    console.log(`PASS  hostile[${i}] survived without throwing`)
  } catch (err) {
    fail++
    console.log(`FAIL  hostile[${i}] THREW: ${err}`)
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)