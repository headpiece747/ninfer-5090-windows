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
// FOUR ENFORCEMENT ATTEMPTS. Read this before changing any of it -- each earlier version had a full
// green harness and refused nothing.
//
//   1. `ctx.permission.hook("evaluate")` with `effect = "deny"`. WRONG LAYER. That hook sits on the
//      PERMISSION path -- external_directory, doom_loop, ask-resolved rules. An edit inside the
//      workspace defaults to allow and never routes there, so the files this gate protects are
//      exactly the files it never sees.
//
//   2. The same hook, reading the payload shape the host actually sends (`{files:[{file, patch}]}`).
//      Still the wrong layer.
//
//   3. A `ctx.tool.transform` wrapper replacing write/edit/patch/shell, added because I had concluded
//      that a throw from `tool.execute.before` does not abort a call. THAT CONCLUSION WAS WRONG, and
//      it is recorded here because the mistake is the instructive part. The evidence was a live probe
//      that was created while the plugin logged `TOOL write -> allow` -- an ALLOW, not a swallowed
//      denial. The check itself was returning allow because of a separate bug (below), so the throw
//      path had never been exercised. I read an allow as a swallowed deny and built a second
//      mechanism to compensate. Once the real bug was fixed, the hook alone denied the write, and no
//      BLOCK line was ever logged because the hook throws before execute is reached.
//
//   So: two mechanisms, one of them redundant, and the redundancy came from a misattribution rather
//   than from caution. Removed. A gate that enforces the same rule twice at two layers is harder to
//   reason about than one that enforces it once.
//
// The mechanism is `tool.execute.before` with a throw, and it is confirmed end to end in the running
// service: a Markdown write citing `native_render.cpp:999999` (the file has 706 lines) was refused
// and not created, while a write citing line 42 was created.
//
//   4. Not an attempt -- a bug in all of the above. `if (!existsSync(target)) return null`, commented
//      "a new file cannot cite an existing line range wrongly". False: a new file can cite an existing
//      file wrongly, and that is the easiest case. It disabled the gate for every newly created
//      Markdown file, which is why three live probes in a row were allowed, and why the evidence for
//      attempt 3 above looked the way it did.
//
// WHAT THIS CANNOT DO: nothing observes what the agent SAYS. No hook covers assistant prose, so "do
// not assert before reading" is not enforceable by any plugin.

import { appendFileSync, existsSync, readFileSync, statSync, writeFileSync } from "node:fs"
import { dirname, isAbsolute, resolve } from "node:path"

const ID = "ninfer.claims-gate"
const LOG = "C:/Users/tobia/AppData/Local/Temp/opencode/claims-gate.log"

// The log is BOUNDED, because an unbounded one is a defect I introduced. Logging every tool call was
// what made the hook observable at all -- without it, "hook did not fire" and "hook fired and was
// swallowed" stayed indistinguishable, which cost a session. But that diagnostic value is spent: the
// gate is confirmed working end to end, and a file that grows by a line per tool call forever is not
// something to leave running.
//
// So: denials always, in full, because they are the record of what the gate actually refused.
// Allow lines only for the first ALLOW_LOGGED calls after start, which covers a fresh start's
// evidence and then stops. The file is truncated if it exceeds LOG_MAX_BYTES, keeping the newest
// lines, so a long-lived service cannot grow it without limit either.
const ALLOW_LOGGED = 40
const LOG_MAX_BYTES = 256 * 1024

let allowsLogged = 0

function log(line: string): void {
  try {
    appendFileSync(LOG, `${new Date().toISOString()} ${line}\n`)
    const size = statSync(LOG).size
    if (size > LOG_MAX_BYTES) {
      // Keep the newest half. Truncation rather than deletion, so the most recent denials -- the
      // part with evidentiary value -- survive.
      const all = readFileSync(LOG, "utf8").split("\n")
      writeFileSync(LOG, `${all.slice(Math.floor(all.length / 2)).join("\n")}\n`, "utf8")
    }
  } catch {
    // logging must never break the request path
  }
}

// docs/research/ cites other projects by construction, so a document under it is exempt. The test is
// on the TARGET DOCUMENT's path, not on the citation text: an earlier version skipped citations
// containing "docs/research/", which never matches, because such a citation names some OTHER
// project's file (`vllm/project/vllm/core/scheduler.py:900`). check_doc_citations.py keys on the
// document, and the two must agree.
function isExternalDocument(target: string, repo: string): boolean {
  const abs = isAbsolute(target) ? target : resolve(repo, target)
  return abs.replace(/\\/g, "/").includes(EXTERNAL)
}
const EXTERNAL = "docs/research/"

const CITE =
  /(?:^|[\s(`"'])((?:[\w.-]+\/)*[\w.-]+\.(?:md|h|hpp|cuh|cpp|cu|py|cmd|bat|json|txt|cmake)):(\d+)\b/g

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
    if (isExternalDocument(target, repo)) return null

    // The target's existence is deliberately NOT checked, and an earlier version checked it with
    // `if (!existsSync(mdAbs)) return null` under the comment "a new file cannot cite an existing
    // line range wrongly". That comment is false, and it disabled the gate for every NEWLY CREATED
    // Markdown file -- which is exactly where a fabricated citation is most likely, and exactly
    // what the live probe was. It survived because every harness case used an existing file.
    //
    // Citation validity depends only on whether the CITED file exists and is long enough, and on
    // the target's directory for resolving a relative path. A file that does not exist yet has a
    // perfectly good dirname.
    const mdAbs = isAbsolute(target) ? target : resolve(repo, target)

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
    allowsLogged = 0
    log(`SETUP repo=${repo}`)

    // THE MECHANISM. One hook, one rule, one place. A thrown error here aborts the tool call, which
    // is confirmed end to end in the running service rather than inferred.
    //
    // The explicit TOOL line is load-bearing. An earlier version logged only on denial, which left
    // "hook did not fire" and "hook fired and was swallowed" indistinguishable -- and that is how a
    // whole session was spent diagnosing a transport that was working. A positive line per call is
    // what turned attempt 4 above from a mystery into one line of reading.
    await ctx.tool.hook("execute.before", (event: any) => {
      try {
        if (!event || typeof event.tool !== "string") return
        const reason = checkCall(event.tool, event.input, repo)
        if (reason) throw new Error(`claims-gate: ${reason}`)
        // Allow lines are capped: see ALLOW_LOGGED. The cap is per-process, so a fresh service start
        // re-arms it, which is exactly when the "did it fire?" evidence is worth having.
        if (allowsLogged++ < ALLOW_LOGGED) log(`TOOL ${event.tool} -> allow`)
      } catch (err) {
        if (err instanceof Error && err.message.startsWith("claims-gate:")) {
          log(`DENY ${event.tool}: ${err.message.slice(9)}`)
          throw err
        }
        log(`ERROR (allowed, failing open): ${String(err)}`)
      }
    })

    log("SETUP complete: execute.before hook registered")
  },
}