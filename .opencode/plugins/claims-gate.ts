// claims-gate -- a blocking OpenCode V2 plugin.
//
// Built from the V2 plugins guide (https://opencode.ai/v2/docs/build/plugins), not from type
// definitions, which is how the previous attempt got its mental model wrong.
//
// WHAT THE GUIDE SETTLES. There is exactly one hook that BLOCKS: `ctx.permission.hook("evaluate")`,
// where "a hook may change `effect` to `allow`, `ask`, or `deny`". `ctx.tool.hook("execute.before")`
// cannot block -- it inspects or REPLACES tool input and has no effect field -- so it is used here only
// to observe. `ctx.shell.hook("create.before")` also only mutates. An explicitly configured `deny` is
// final and never reaches this hook, so this gate can only ever ADD a denial.
//
// WHY DENY ONLY PROVABLE DEFECTS. Two rules were added to AGENTS.md this session to stop the agent
// asserting before reading. Neither was followed, because a rules file is read once at session start
// and then competes with everything else. A gate does not need to be remembered. But a gate that
// denies on judgement is a gate that gets muted, and a muted gate teaches the agent to route around
// it -- worse than no gate. So every condition below is arithmetic or an exact command shape:
//
//   1. A Markdown edit carrying a `path/file.ext:LINE` citation whose target does not exist, or whose
//      LINE is past EOF. This class is real: check_doc_citations.py found three (fp8_linear_add_a8.cu:215
//      in a 117-line file, plus two unqualified config.py references). Enforcing at write time is the
//      point -- a wrong claim in a document outlives the correction.
//
//   2. A PowerShell command matching Set-Content / Out-File / Add-Content with `-Encoding utf8`.
//      PowerShell 5.1 writes a BOM and Get-Content decodes with the console codepage; that pair
//      destroyed 181 lines of docs/active-work.md in one session and no gate caught it, because the
//      corruption is confined to prose. The command shape IS the defect.
//
// WHAT THIS CANNOT DO, and it is the part that matters: nothing here observes what the agent SAYS.
// No hook covers assistant prose, so "do not assert before reading" is not enforceable by any plugin.
// Do not pretend otherwise.
//
// FAILS OPEN, ALWAYS. Every callback is wrapped; any error logs and returns without touching `effect`.
// The only path that sets `deny` is a proven defect. This file must never be able to brick a session.

import { appendFileSync, existsSync, readFileSync } from "node:fs"
import { dirname, isAbsolute, resolve } from "node:path"
import { Plugin } from "@opencode/plugin"

const LOG = "C:/Users/tobia/AppData/Local/Temp/opencode/claims-gate.log"

// docs/research/ cites other projects by design, so a path in it is an external reference rather than
// a dead one. check_doc_citations.py skips the same directory.
const EXTERNAL = "docs/research/"

const CITE =
  /(?:^|[\s(`"'])((?:[\w.-]+\/)*[\w.-]+\.(?:md|h|hpp|cuh|cpp|cu|py|cmd|bat|json|txt|cmake)):(\d+)\b/g

function log(line: string): void {
  try {
    appendFileSync(LOG, `${new Date().toISOString()} ${line}\n`)
  } catch {
    // logging must never break the request path
  }
}

function lineCount(path: string): number {
  const text = readFileSync(path, "utf8")
  if (text.length === 0) return 0
  let n = 0
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++
  if (text.charCodeAt(text.length - 1) !== 10) n++
  return n
}

/** Citations that are provably wrong: the file is missing, or LINE is past EOF. */
function brokenCitations(text: string, base: string): string[] {
  const bad: string[] = []
  CITE.lastIndex = 0
  for (const m of text.matchAll(CITE)) {
    const raw = String(m[1])
    const line = Number(m[2])
    if (raw.includes(EXTERNAL)) continue
    const abs = isAbsolute(raw) ? raw : resolve(base, raw)
    let total: number
    try {
      if (!existsSync(abs)) {
        bad.push(`${raw}:${line} -- no such file`)
        continue
      }
      total = lineCount(abs)
    } catch {
      continue // unreadable: fail open rather than guess
    }
    if (line > total) bad.push(`${raw}:${line} -- file has ${total} lines`)
  }
  return bad
}

/** PowerShell shapes that re-encode a tracked file. Exact match, no heuristics. */
function reencodingCommand(cmd: string): string | null {
  if (/\b(Set-Content|Out-File|Add-Content)\b/.test(cmd) && /-Encoding\s+utf8/i.test(cmd)) {
    return (
      "Set-Content/Out-File/Add-Content with -Encoding utf8 writes a BOM under PowerShell 5.1. " +
      "Use [System.IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false)))."
    )
  }
  if (/Get-Content/.test(cmd) && /(Set-Content|WriteAllLines|Out-File)/.test(cmd)) {
    return (
      "Get-Content decodes with the console codepage and the write re-encodes it, compounding each " +
      "pass. Read and write bytes rather than text, and verify with the consumer that reads the file."
    )
  }
  return null
}

/** Pull the incoming payload out of whatever shape the tool call carries. */
function payloadOf(input: unknown): string {
  if (typeof input === "string") return input
  if (!input || typeof input !== "object") return ""
  const rec = input as Record<string, unknown>
  let out = ""
  for (const k of ["content", "newString", "oldString", "command", "patch", "fileText"]) {
    const v = rec[k]
    if (typeof v === "string") out += `\n${v}`
  }
  return out
}

export default Plugin.define({
  id: "ninfer.claims-gate",

  async setup(ctx) {
    const repo = ctx.location?.directory ?? process.cwd()
    log(`SETUP repo=${repo}`)

    // Observe only. This hook CANNOT block -- the guide says it inspects or replaces input -- so it is
    // here to record the vocabulary that actually arrives, which is what makes the gate's field
    // assumptions checkable instead of believed.
    let observed = 0
    await ctx.tool.hook("execute.before", (event) => {
      if (observed++ < 50) {
        log(`TOOL tool=${event.tool} keys=${Object.keys((event.input ?? {}) as object).join(",")}`)
      }
    })

    // The one blocking hook. `deny` here is the only way this plugin stops anything.
    await ctx.permission.hook("evaluate", (event) => {
      try {
        const meta = (event.metadata ?? {}) as Record<string, unknown>
        const blob = `${(event.resources ?? []).join(" ")}\n${JSON.stringify(meta)}`

        // Gate 2 -- shell command shape. Checked first because it needs no path resolution.
        const cmd = typeof meta.command === "string" ? meta.command : blob
        const shellProblem = reencodingCommand(cmd)
        if (shellProblem) {
          log(`DENY shell: ${shellProblem}`)
          event.effect = "deny"
          event.message = `claims-gate: ${shellProblem}`
          return
        }

        // Gate 1 -- provably wrong citation in a Markdown edit.
        const md = (event.resources ?? []).find((r) => r.toLowerCase().endsWith(".md"))
        if (!md) return
        const mdAbs = isAbsolute(md) ? md : resolve(repo, md)
        if (!existsSync(mdAbs)) return

        const incoming = payloadOf(meta.content ?? meta.newString ?? meta.input ?? meta)
        if (!incoming) return

        const bad = brokenCitations(incoming, dirname(mdAbs))
        if (bad.length) {
          log(`DENY cite: ${bad.join(" ; ")}`)
          event.effect = "deny"
          event.message =
            `claims-gate: citation(s) in this edit are provably wrong -- ${bad.join("; ")}. ` +
            `Open the line and read it before writing it down, or drop the citation.`
        }
      } catch (err) {
        log(`ERROR (allowed, failing open): ${String(err)}`)
      }
    })
  },
})
