// claims-gate -- blocks two provable defects at the tool layer.
//
// Built from the V2 plugins guide (https://opencode.ai/v2/docs/build/plugins), not from type
// definitions, which is how the discarded first attempt got its model wrong.
//
// WHY A GATE AND NOT A RULE. On 2026-10-03 a session asserted eleven wrong claims, all from stating
// a conclusion before the tool call that would establish it. Rules added to AGENTS.md to prevent it
// were not followed, because a rules file is read once at session start and then competes with
// everything else. A gate does not need to be remembered.
//
// WHAT IS DENIED, and both conditions are provable rather than judged -- a gate that denies on
// judgement is a gate that gets muted, and a muted gate teaches the agent to route around it:
//
//   1. A Markdown write/edit whose payload cites `file.ext:LINE` where the file is missing or LINE is
//      past EOF. check_doc_citations.py already found three of these in the committed tree
//      (fp8_linear_add_a8.cu:215 in a 117-line file, two unqualified config.py references). Enforcing
//      at write time rather than commit time is the point: a wrong claim in a document outlives the
//      correction.
//   2. A shell command matching Set-Content / Out-File / Add-Content with `-Encoding utf8`. Under
//      PowerShell 5.1 that writes a BOM and Get-Content decodes with the console codepage; the pair
//      destroyed 181 lines of docs/active-work.md in one session while every other gate passed.
//
// WHAT THIS CANNOT DO: nothing observes what the agent SAYS. No hook covers assistant prose, so "do
// not assert before reading" is not enforceable by any plugin.
//
// ------------------------------------------------------------------------------
// THREE ENFORCEMENT ATTEMPTS, and only the third is known to reach the tool. Read this before
// changing any of it -- each earlier version had a full green harness and refused nothing.
//
//   1. `ctx.permission.hook("evaluate")` with `effect = "deny"`. WRONG LAYER. That hook sits on the
//      PERMISSION path -- external_directory, doom_loop, ask-resolved rules. An edit inside the
//      workspace defaults to allow and never routes there, so the files this gate protects are
//      exactly the files it never sees. Every PERM line it logged came from an edit to the global
//      config directory, outside the workspace, the one case it does reach.
//
//   2. The same hook, fixed to read the payload shape the host actually sends
//      (`{files:[{file, patch}]}`). Still the wrong layer, so still inert for in-workspace edits.
//
//   3. `tool.execute.before` with a throw. Primary sources say a thrown error ABORTS the call --
//      adlc, pinned against the plugin SDK, states it as documented host behaviour, and upstream
//      issue #37164 says a hook can "only silently allow or throw a hard denial". MEASURED HERE AND
//      IT DID NOT ABORT: with claims-gate.ts loaded at 00:09:23Z and the throw present, a Markdown
//      write carrying a past-EOF citation was still created. Both of those sources describe the V1
//      plugin API (`@opencode-ai/plugin`, whose hook signature is `(input, output)`); the V2 callback
//      is `(event) => void`, with no output parameter and no documented abort-on-throw. The claim was
//      carried over across a major version boundary without being re-checked.
//
// So the throw stays -- it is correct, and it costs nothing -- and the enforcement that does not
// depend on hook semantics is the tool TRANSFORM below, which replaces the tool's own execute. That
// cannot be swallowed by hook semantics: the write simply never happens.

import { appendFileSync, existsSync, readFileSync } from "node:fs"
import { dirname, isAbsolute, resolve } from "node:path"

const ID = "ninfer.claims-gate"
const LOG = "C:/Users/tobia/AppData/Local/Temp/opencode/claims-gate.log"

// docs/research/ cites other projects by design, so a path there is an external reference rather than
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
 *  Resolution is tried against BOTH the document's directory and the repository root, because
 *  documents here cite repository-relative paths (`src/ops/foo.cu:215`) while sitting in `docs/`.
 *  Resolving only against the document's directory denies every such citation; resolving only against
 *  the root misses a document citing its own siblings. Accepting whichever base resolves it is what
 *  check_doc_citations.py already does, and the two must agree -- or this denies what the gate allows.
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

/** The incoming payload, from whichever key the tool used. */
function payloadOf(input: Record<string, unknown>): string {
  let out = ""
  for (const k of ["content", "newString", "oldString", "patch", "fileText"]) {
    const v = input[k]
    if (typeof v === "string") out += `\n${v}`
  }
  return out
}

// ---------------------------------------------------------------------------------------------
// The single decision. Both enforcement paths call this, so they cannot disagree.
// Returns a denial reason, or null to allow. NEVER throws: a check that throws on its own bug would
// break the tool it protects.
// ---------------------------------------------------------------------------------------------
function checkCall(tool: string, input: unknown, repo: string): string | null {
  try {
    if (typeof tool !== "string") return null
    const args = (input ?? {}) as Record<string, unknown>

    if (tool === "shell" || tool === "bash") {
      const problem = reencodingCommand(typeof args.command === "string" ? args.command : "")
      return problem
    }

    if (tool !== "write" && tool !== "edit" && tool !== "patch") return null
    const target =
      (typeof args.path === "string" && args.path) ||
      (typeof args.filePath === "string" && args.filePath) ||
      ""
    if (!target.toLowerCase().endsWith(".md")) return null

    const mdAbs = isAbsolute(target) ? target : resolve(repo, target)
    if (!existsSync(mdAbs)) return null // a new file cannot cite an existing line range wrongly

    const incoming = payloadOf(args)
    if (!incoming) return null

    const bad = brokenCitations(incoming, dirname(mdAbs), repo)
    if (bad.length) {
      return (
        `${bad.join("; ")}. Open the line and read it before writing it down, or drop the citation.`
      )
    }
    return null
  } catch (err) {
    log(`ERROR (allowed, failing open): ${String(err)}`)
    return null
  }
}

export default {
  id: ID,

  async setup(ctx: any) {
    const repo = ctx?.location?.directory ?? process.cwd()
    log(`SETUP repo=${repo}`)

    // Path A -- the hook. Kept because the throw is correct per primary sources, and because an
    // explicit `TOOL` line here is the only positive evidence that the hook fired at all. Without
    // it, "hook did not fire" and "hook fired and was swallowed" are indistinguishable -- which is
    // exactly how the previous version spent a session.
    await ctx.tool.hook("execute.before", (event: any) => {
      try {
        if (!event || typeof event.tool !== "string") return
        const reason = checkCall(event.tool, event.input, repo)
        log(`TOOL ${event.tool} -> ${reason ? "DENY" : "allow"}`)
        if (reason) throw new Error(`claims-gate: ${reason}`)
      } catch (err) {
        if (err instanceof Error && err.message.startsWith("claims-gate:")) {
          log(`DENY ${event.tool}: ${err.message.slice(9)}`)
          throw err
        }
        log(`ERROR (allowed, failing open): ${String(err)}`)
      }
    })

    // Path B -- the transform. This is the enforcement that does not rely on hook-throw semantics:
    // the wrapped execute returns the denial instead of calling the original, so the write does not
    // happen. Unregistering or disabling the plugin restores the original tool, which is the same
    // escape hatch every plugin override has.
    await ctx.tool.transform((editor: any) => {
      for (const id of ["write", "edit", "patch", "shell"]) {
        const existing = editor.get(id)
        if (!existing || typeof existing.execute !== "function") {
          log(`NOTE tool "${id}" not present at transform time; not wrapped`)
          continue
        }
        const original = existing.execute
        editor.update(id, (tool: any) => {
          tool.execute = async (input: any, toolCtx: any) => {
            const reason = checkCall(id, input, repo)
            if (reason) {
              log(`BLOCK ${id}: ${reason}`)
              throw new Error(`claims-gate: refused -- ${reason}`)
            }
            log(`ALLOW ${id} (transform)`)
            return original(input, toolCtx)
          }
        })
      }
    })

    log("SETUP complete: hook + transform registered")
  },
}