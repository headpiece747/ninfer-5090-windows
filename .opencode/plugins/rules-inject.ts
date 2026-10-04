// rules-inject -- re-injects three rules on EVERY model call.
//
// WHY THIS EXISTS, and why it is not the claims-gate.
//
// The 2026-10-03 session asserted eleven wrong claims about this codebase. Classified by cause:
// seven were a conclusion stated before the tool call that would have settled it, two were an
// instrument trusted without ever having been seen to fail, and two were a negative claim ("no
// caller", "nothing uses this") with no scope attached. Three rules were added to AGENTS.md to
// prevent exactly that. None was followed.
//
// The diagnosis was "AGENTS.md is read once at session start and then competes with everything else
// for attention", and the first response was to reach for a gate -- which is the wrong instrument.
// A gate denies. The failure here was not that the rules were unenforced; it was that they were
// forgotten. Denying a tool call cannot fix forgetting.
//
// This plugin uses the other mechanism. From the V2 plugins guide:
//
//     await ctx.session.hook("context", (event) => {
//       event.system.push({ type: "text", text: "..." })
//     })
//
// `context` runs for the agent loop INCLUDING tool-driven continuations, so these three lines are
// re-delivered on every model call rather than once per session. They cannot be forgotten, because
// forgetting them is not a state the model can be in when the request is assembled.
//
// THE HONEST LIMIT, stated in the file itself so the next reader has it too. Injection is not
// enforcement. An injected rule is still a rule the model can choose not to follow; what changes is
// that it cannot be absent from the context. This does not address asserting something false --
// only asserting it without having looked. Only `permission.evaluate` denies, and that is what
// claims-gate does.
//
// THREE, NOT SIXTY-THREE. A block that re-appears on every call is read once and then ignored, which
// reproduces the original problem at higher volume -- so the rule set is capped at three and the cap
// is asserted by tools/opencode/test_rules_inject.mjs. If a rule earns its place it replaces one,
// rather than joining the list.

// NO `import { Plugin } from "@opencode/plugin"`, and that is deliberate.
//
// It was there first, and it is the reason this plugin never ran. The opencode service resolves that
// package only inside the global config directory -- where thinking-guard.ts lives and loads fine -- so
// a plugin in this repository fails with:
//
//   WARN failed to load plugin cause="... Cannot find package '@opencode/plugin' imported from ..."
//
// and opencode SKIPS it rather than crashing, so the failure is silent. Installing the package into
// .opencode/plugins/node_modules fixed it for `node` and NOT for the service, across a restart, with
// an identical package (2.0.11, same exports map) sitting in both places.
//
// The import is optional. `Plugin.define({ id, setup })` is a typing and ergonomics helper over a
// plain object, and the default export is that object. So the plugins declare their shape directly.
// The cost is losing `Plugin.define`'s type-checking, which is what `test_plugin_loads.mjs` and the two
// decision harnesses are for -- and they now cover the thing that actually broke, which they did not.

export type RulesInjectPlugin = { id: string; setup: (ctx: any) => Promise<void> | void }
export const ID = "ninfer.rules-inject"

// Keep this short. Every line here is paid for on every model request.
const RULES: string[] = [
  "Do not state a conclusion about this codebase before the tool call that would settle it has returned. Read the result first; the answer is often already in the result you have.",
  "An instrument you have not seen fail is not evidence that it works. Falsify it in both directions before trusting it.",
  "Every negative claim -- 'no caller', 'nothing uses this', 'no launcher passes it' -- carries the scope it was searched over, because a scoped search that finds nothing is not a tree-wide fact.",
]

const HEADER =
  "Three rules, re-injected on every model call because a rules file is read once and then competes " +
  "for attention. They are not enforcement: they cannot be forgotten, and they can still be " +
  "disobeyed."

// The plugin object, declared inline rather than through Plugin.define -- see the header.
export default {
  id: ID,

  async setup(ctx: any) {
    await ctx.session.hook("context", (event) => {
      try {
        // Guard the shape rather than trusting it: a throw inside `context` would land on every
        // agent-loop request, and a plugin that can wedge the loop is worse than no plugin.
        if (!event || !Array.isArray(event.system)) return
        const text = `${HEADER}\n\n${RULES.map((r, i) => `${i + 1}. ${r}`).join("\n")}`
        event.system.push({ type: "text", text })
      } catch {
        // fail open, always
      }
    })
  },
}
