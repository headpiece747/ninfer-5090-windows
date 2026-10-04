// Standalone falsification harness for .opencode/plugins/claims-gate.ts
//
//   node --experimental-strip-types tools/opencode/test_claims_gate.mjs
//
// Drives BOTH enforcement paths. The plugin registers a `tool.execute.before` hook AND a
// `ctx.tool.transform` wrapper, because the hook's throw was MEASURED not to abort a call: with the
// plugin loaded at 00:09:23Z, a Markdown write carrying a past-EOF citation was still created. A
// harness that tested only the hook would have been green on a gate that does nothing.
//
// Every previous version of this harness drove the plugin's own interface rather than the host's.
// Each was fully green and each gate refused nothing, so the rule encoded here is: assert the
// DENIAL SIGNAL, on the real input shape the host sends (`write` -> {path, content},
// `shell` -> {command}), and exercise every path the plugin registers.

import { pathToFileURL } from "node:url"
import { readFileSync, existsSync } from "node:fs"
import { resolve } from "node:path"

const REPO = "C:\\AI\\ninfer-v3-windows"
const mod = await import(pathToFileURL(resolve(REPO, ".opencode/plugins/claims-gate.ts")).href)
const plugin = mod.default ?? mod

let hook = null
const registered = []
// A stand-in for the real tool editor: enough surface for the plugin to wrap, and enough to prove
// the wrapper is what performs the denial rather than the hook.
const WRAPPED = {}
const editor = {
  get: (id) => (WRAPPED[id] ? { id, execute: WRAPPED[id] } : undefined),
  update: (id, mutate) => {
    const holder = { execute: WRAPPED[id] }
    mutate(holder)
    WRAPPED[id] = holder.execute
  },
  add: () => {},
  remove: () => {},
}

// A per-call counter, so "the original ran" is distinguishable from "the wrapper returned a literal
// that happens to look like the original's output". The first version of this assertion compared a
// string that the stub and the wrapper both produced, and it stayed GREEN when the wrapper stopped
// calling through entirely -- a control that agreed for a structural reason.
let seq = 0
for (const id of ["write", "edit", "patch", "shell"]) {
  WRAPPED[id] = async () => ({ content: `ORIGINAL ${id}`, seq: ++seq })
}

const ctx = {
  location: { directory: REPO },
  tool: {
    hook: async (name, cb) => {
      registered.push(`hook:${name}`)
      if (name === "execute.before") hook = cb
      return { dispose() {} }
    },
    transform: async (cb) => {
      registered.push("transform")
      cb(editor)
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
if (!registered.includes("transform")) throw new Error("ctx.tool.transform was never called")
for (const id of ["write", "edit", "patch", "shell"]) {
  if (typeof WRAPPED[id] !== "function") throw new Error(`${id} was never wrapped`)
}

let pass = 0
let fail = 0

async function viaHook(tool, input) {
  try {
    await hook({ tool, input, sessionID: "t", agent: "build", id: "c1" })
    return null
  } catch (err) {
    return err
  }
}

async function viaTransform(tool, input) {
  try {
    const out = await WRAPPED[tool](input, {})
    return { ran: true, out }
  } catch (err) {
    return err
  }
}

function check(name, got, wantDeny) {
  const denied = got instanceof Error
  const ok = denied === wantDeny
  if (ok) pass++
  else fail++
  const detail = denied ? `denied: ${String(got.message).slice(0, 62)}` : "allowed"
  console.log(`${ok ? "PASS" : "FAIL"}  ${name} -> ${detail}${ok ? "" : ` (wanted ${wantDeny ? "deny" : "allow"})`}`)
}

// ---- registration: the wrong layer stays dead ----
check(
  "no permission hook is registered",
  registered.filter((r) => r.startsWith("permission")).length === 0 ? null : new Error("re-added"),
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

for (const [pathName, call] of [
  ["hook", viaHook],
  ["transform", viaTransform],
]) {
  console.log(`\n-- via ${pathName} --`)
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
  await check("allow external citation (docs/research)", await call("write", { path: resolve(REPO, "docs/research/x.md"), content: "see vllm/scheduler.py:900" }), false)
  await check("allow non-markdown file", await call("write", { path: resolve(REPO, "zz.cpp"), content: "see x.cpp:999999" }), false)
  await check("allow safe WriteAllText", await call("shell", { command: "[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))" }), false)
  await check("allow ordinary git", await call("shell", { command: "git log --oneline -1" }), false)
  // `read` is deliberately NOT wrapped -- a citation check has nothing to say about reading a file,
  // and wrapping every tool would widen the blast radius for nothing. Asserted rather than assumed.
  await check("transform does not wrap a read-only tool", WRAPPED.read === undefined ? null : new Error("read was wrapped"), false)
}

// ---- the transform must actually reach the original on an allowed call ----
// `seq` increments ONLY inside the original, so an increment of exactly 1 proves the original
// function ran, and a wrapper returning a hand-built literal would leave it unchanged. The
// comparison is RELATIVE: earlier cases in the transform section already called through, so an
// absolute `seq === 1` is wrong -- it was, and it failed for exactly that reason.
const before = seq
const allowed = await viaTransform("write", GOOD)
check(
  "transform calls through to the ORIGINAL on an allowed write",
  allowed.out?.seq === before + 1 ? null : new Error(`original not invoked (seq ${before} -> ${allowed.out?.seq})`),
  false,
)
const allowed2 = await viaTransform("write", GOOD)
check(
  "and again -- the original is invoked per call, not memoised",
  allowed2.out?.seq === before + 2 ? null : new Error(`seq=${allowed2.out?.seq}`),
  false,
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