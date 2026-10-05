# opencode plugins

Two project-scoped plugins. Both are **agent tooling**: they shape how an agent behaves in this
repository and affect nothing about the shipped engine, its tests, or its artifacts.

| plugin | mechanism | what it does |
|---|---|---|
| `claims-gate.ts` | `ctx.permission.hook("evaluate")` | **Denies** a Markdown edit carrying a citation that is provably wrong, and a PowerShell command matching `Set-Content`/`Out-File`/`Add-Content` with `-Encoding utf8` |
| `rules-inject.ts` | `ctx.session.hook("context")` | **Injects** three rules on every model call, capped at three |

The distinction is the point. `claims-gate` is the only blocking hook opencode offers; `rules-inject`
cannot prevent anything, it can only make sure the rules are present when the request is assembled.
One addresses a rule that is broken, the other a rule that is forgotten.

## Install

Nothing to install — the plugins import nothing. Restart the opencode service, or nothing loads:
plugins are read at service start.

## Why `package.json` exists here

For `"type": "module"`, and for nothing else. It declares no dependencies.

The history is worth keeping, because the obvious fix was tried and did not work. `@opencode/plugin`
resolves only inside the global config directory, where `thinking-guard.ts` lives, so these plugins
originally imported it, and opencode **skipped both files silently** — it does not crash, so nothing
else reported it:

```
WARN failed to load plugin target=...\.opencode\plugins\claims-gate.ts
  cause="Cause([Die(ResolveMessage: Cannot find package '@opencode/plugin' imported from ...)]"
```

Installing the package into `.opencode/plugins/node_modules` made `node` resolve it and did **not**
make the service resolve it — byte-identical packages, across a restart. So the plugins no longer
import it at all: `Plugin.define({ id, setup })` is a typing helper over a plain object, and the
default export is that object. The dependency, the committed lockfile and the 139 MiB install went
with the import. `test_plugin_loads.mjs` asserts the import's **absence**, because that absence is
the property that makes these files loadable here.

## Tests

Three harnesses, and the split between them is deliberate:

| harness | question it answers | needs the install |
|---|---|---|
| `test_claims_gate.mjs` | is the deny/allow decision logic right? | no — stubs the package, so it runs anywhere |
| `test_rules_inject.mjs` | does the injected text arrive, on every call, without accumulating? | no — same stub |
| `test_plugin_loads.mjs` | **would opencode actually load these?** | no — it asserts the package is *not* imported |

The stub is correct for testing decision logic in isolation and **wrong** for asking whether the
service can load the file, because it makes an unresolvable import resolvable. That is how both
plugins passed their tests while being dead in the service. `test_plugin_loads.mjs` exists to close
that gap and runs in the pre-commit hook.

```sh
node --experimental-strip-types tools/opencode/test_plugin_loads.mjs
NODE_OPTIONS="--import=data:text/javascript,import{register}from'node:module';register('file:///C:/AI/ninfer-v3-windows/tools/opencode/stub-opencode-plugin.mjs');" \
  node --experimental-strip-types tools/opencode/test_claims_gate.mjs
```

## Verifying after a restart

The harness proves the file loads in Node. It cannot prove the running service loaded it, because a
plugin cannot affect the session that wrote it. The service log is the only external signal:

```sh
grep "failed to load plugin" ~/.local/share/opencode/log/opencode.log
```

No output means both loaded. `claims-gate.ts` also appends every decision to
`%TEMP%/opencode/claims-gate.log`, where a `SETUP` line means it ran.
