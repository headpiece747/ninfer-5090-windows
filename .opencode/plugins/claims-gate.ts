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
// NO `import { Plugin } from "@opencode/plugin"` -- see rules-inject.ts, whose header explains why in
// full. Short version: the opencode service resolves that package only inside the global config
// directory, so a plugin in this repository is SKIPPED with a "Cannot find package" warning that
// nothing else reports, and installing the package locally did not fix it across a restart.

const ID = "ninfer.claims-gate"

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

/** Citations that are provably wrong: the file is missing, or LINE is past EOF.
 *
 *  Resolution is tried against BOTH the document's own directory and the repository root, because
 *  documents here cite repository-relative paths (`src/ops/foo.cu:215`) while sitting in `docs/`.
 *  Resolving only against the document's directory would deny every such citation; resolving only
 *  against the root would miss a document citing its own siblings. A citation is accepted if EITHER
 *  base resolves it -- which is what check_doc_citations.py already does for the committed tree, and
 *  the two must agree, or the hook denies what the gate allows.
 */
function brokenCitations(text: string, docDir: string, repoRoot: string): string[] {
  const bad: string[] = []
  CITE.lastIndex = 0
  for (const m of text.matchAll(CITE)) {
    const raw = String(m[1])
    const line = Number(m[2])
    if (raw.includes(EXTERNAL)) continue
    const candidates = isAbsolute(raw) ? [raw] : [resolve(docDir, raw), resolve(repoRoot, raw)]
    let total: number | null = null
    for (const abs of candidates) {
      try {
        if (!existsSync(abs)) continue
        total = lineCount(abs)
        break
      } catch {
        continue // unreadable: try the next base rather than guess
      }
    }
    if (total === null) {
      bad.push(`${raw}:${line} -- no such file`)
      continue
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

export default {
  id: ID,

  async setup(ctx: any) {
    const repo = ctx.location?.directory ?? process.cwd()
    log(`SETUP repo=${repo}`)

    // THE ENFORCEMENT POINT. Two earlier designs got this wrong and were wrong in the same way.
    //
    // v1 hooked `ctx.permission.hook("evaluate")` and set `event.effect = "deny"`. It has a 14-case
    // test suite, and it has never once refused anything, because permission evaluation is the
    // PERMISSION path: external_directory, doom_loop, and rules that resolve to ask. An edit inside
    // the workspace defaults to allow and never routes there. Every PERM line the plugin ever logged
    // came from an edit to the global config directory -- outside the workspace, which is the one
    // case it does see.
    //
    // v2 hooked this one and only logged. The guide's own text -- "Inspect or replace tool input" --
    // reads as observe-only, and that reading is what produced a gate that observed.
    //
    // It is not observe-only. Throwing from `tool.execute.before` ABORTS the tool call. That is the
    // documented host behaviour, and it is how a third-party enforcement plugin (adlc, pinned against
    // the plugin SDK with a live-deny regression script) blocks tool use; upstream issue #37164 states
    // the same limit from the other side -- a hook can "only silently allow or throw a hard denial".
    // There is no effect field because throwing IS the denial.
    //
    // This hook fires for EVERY tool call, which the log now proves: TOOL lines appear for every
    // shell, write and read this session performs, including the in-workspace ones the permission
    // path never saw.
    await ctx.tool.hook("execute.before", (event: any) => {
      try {
        if (!event || typeof event.tool !== "string") return
        const input = (event.input ?? {}) as Record<string, unknown>

        // Gate 2 -- the PowerShell re-encoding shape. Checked first: it needs no path resolution.
        if (event.tool === "shell" || event.tool === "bash") {
          const problem = reencodingCommand(typeof input.command === "string" ? input.command : "")
          if (problem) {
            log(`DENY shell: ${problem}`)
            throw new Error(`claims-gate: ${problem}`)
          }
          return
        }

        // Gate 1 -- a provably wrong citation in a Markdown edit.
        if (event.tool !== "write" && event.tool !== "edit" && event.tool !== "patch") return
        const target =
          (typeof input.path === "string" && input.path) ||
          (typeof input.filePath === "string" && input.filePath) ||
          ""
        if (!target.toLowerCase().endsWith(".md")) return

        const mdAbs = isAbsolute(target) ? target : resolve(repo, target)
        if (!existsSync(mdAbs)) return // a new file cannot cite an existing line range wrongly

        const incoming = payloadOf(input)
        if (!incoming) return

        const bad = brokenCitations(incoming, dirname(mdAbs), repo)
        if (bad.length) {
          log(`DENY cite: ${bad.join(" ; ")}`)
          throw new Error(
            `claims-gate: refused -- ${bad.join("; ")}. ` +
              `Open the line and read it before writing it down, or drop the citation.`,
          )
        }
      } catch (err) {
        // Re-throw our own denial; swallow anything else. A check that throws on its own bug would
        // break the tool it is protecting, which is worse than not checking.
        if (err instanceof Error && err.message.startsWith("claims-gate:")) throw err
        log(`ERROR (allowed, failing open): ${String(err)}`)
      }
    })

    },
}
