// Standalone falsification harness for .opencode/plugins/rules-inject.ts
//
// Run from the repo root:
//   node --experimental-strip-types tools/opencode/test_rules_inject.mjs
//
// What is being protected, because each of these was true of the plugin's predecessor and of the
// session that motivated it:
//   * the text must actually ARRIVE on the hook event, not merely be constructed
//   * it must arrive on EVERY call -- that is the entire difference from a rules file, which is read
//     once at session start, so a single-call test proves nothing
//   * the rule set must stay SMALL. "Small and followed beats complete and ignored" is a principle
//     until a test fails when it stops being true, so the cap is asserted here
//   * a malformed event must not throw. A throw inside `context` lands on every agent-loop request,
//     and a plugin that can wedge the loop is worse than no plugin

import { pathToFileURL } from "node:url"
import { resolve } from "node:path"

const REPO = "C:\\AI\\ninfer-v3-windows"
const mod = await import(pathToFileURL(resolve(REPO, ".opencode/plugins/rules-inject.ts")).href)
const plugin = mod.default ?? mod

let contextHook = null
const ctx = {
  session: {
    hook: async (name, cb) => {
      if (name !== "context") throw new Error(`unexpected session hook ${name}`)
      contextHook = cb
      return { dispose() {} }
    },
  },
}

await plugin.setup(ctx)

if (typeof contextHook !== "function") throw new Error("session.context hook was never registered")

function freshEvent() {
  return { system: [], tools: {}, options: {}, agent: "build" }
}

async function fire(e) {
  await contextHook(e)
  return e
}

let pass = 0
let fail = 0
function check(name, ok, detail = "") {
  if (ok) pass++
  else fail++
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : `  ${detail}`}`)
}

// ---- 1. the text arrives ----
const first = await fire(freshEvent())
const injected = first.system.filter((p) => p && p.type === "text")
check("pushes exactly one text part", injected.length === 1, `got ${injected.length}`)
const body = injected[0]?.text ?? ""
check("mentions the first rule's substance", /before the tool call/i.test(body))
check("mentions the second rule's substance", /seen fail|not evidence/i.test(body))
check("mentions the third rule's substance", /scope/i.test(body))

// ---- 2. it arrives EVERY time: three consecutive agent-loop requests ----
const second = await fire(freshEvent())
const third = await fire(freshEvent())
check(
  "re-injects on a second model call",
  second.system.length === 1 && second.system[0].text === body,
)
check(
  "re-injects on a third model call",
  third.system.length === 1 && third.system[0].text === body,
)
check(
  "does not accumulate across calls",
  second.system.length === 1 && third.system.length === 1,
  `${second.system.length}/${third.system.length}`,
)

// ---- 3. the cap, which is the design ----
const numbered = body.match(/^\d+\.\s/gm) ?? []
check("rule set is at most 3 rules", numbered.length > 0 && numbered.length <= 3, `found ${numbered.length}`)
check("rule set is not trivially empty", numbered.length >= 1)
check("body stays under 900 characters", body.length < 900, `${body.length} chars`)

// ---- 4. fails open on hostile events: must not throw, must not push ----
const hostile = [
  { system: null },
  { system: undefined },
  {},
  { system: [] },
  { system: "not-an-array" },
  null,
  undefined,
]
for (const [i, bad] of hostile.entries()) {
  try {
    await contextHook(bad)
    pass++
    console.log(`PASS  hostile[${i}] survived`)
  } catch (err) {
    fail++
    console.log(`FAIL  hostile[${i}] THREW: ${err}`)
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
