# ADR-0008: The shared catalog's handle is never populated, so a shared prefix can never serve

**Status:** accepted

## Context

`SharedCatalogEntry::handle` is a `std::optional<SharedPrefixHandle>`, and every reuse path that could
serve a shared prefix requires it:

```cpp
if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
```

The field is written in exactly five places, and none of them fills it: `reset()` at four sites (the
catalog sweeps and `clear_shared_entry`), and one `emplace` -- at `resource_manager.h:1040`, which is
the **private** catalog's publication, not the shared one. So a shared entry is published with a state,
a summary and an advanced revision, and an **empty** handle, and is then invisible to every lookup that
could use it.

It cannot be filled where it is needed either. `SharedPrefixHandle` is a program-layer class
(`program.h:379`): movable-only, non-copyable, holding `const void* owner_`, `index_` and `generation_`
-- a pointer into the program's own storage. Only `detail::RuntimeContractAccess`, its friend, can
construct one. The runtime therefore has to be *given* a handle, and no result struct carries one:
`MaterializationSharedSourceResult` has a `SharedPrefixSummary` and nothing else, and
`MaterializationSharedVictimResult` is the same.

Evidence that this is the whole of the shared-prefix defect:

- The reproduction's request log shows the capture is **offered and planned** -- `offered 1`, every
  refusal branch 0 -- while `shared_stable_prefix` selections stay 0 and `shared_owners_degraded`
  climbs. A published-but-unusable entry produces exactly that: the action path does not check
  `handle`, the reuse path does.
- The launcher table already recorded, from an earlier session, that **"raising the shared bound alone
  changed nothing"**. It could not have: `--max-shared-prefixes` sizes a catalog whose entries are
  never usable.
- `shared_active_references` is 0 and `shared_owners_evicted` is 0 with host KV at 653 MiB of 8 GiB, so
  no pool and no reference was involved.

## Decision

Hand the handle over. The program exposes the handle it creates for a shared prefix on the publication
result, and the runtime stores it in the entry at publication, next to the summary it already stores.
`detail::RuntimeContractAccess` is the existing seam for a program-owned handle crossing to the
runtime, so no new mechanism is introduced.

Until that lands, the shared path is inert and should not be reasoned about as if it worked: four
candidate causes were measured out in an earlier pass (pool capacity, search budget, retention weight,
structural credit) on the assumption that the entries were usable.

## Consequences

- ADR-0007's shared-prefix paragraph, which attributes the symptom to the valuation having no term for
  a future reuse, is **superseded** for the shared case: the entries are unusable, so their valuation
  never mattered. The private turn-closure finding in that ADR is untouched.
- Once the handle is stored, the shared path becomes measurable for the first time -- the reproduction
  should show `shared_stable_prefix` selections above zero -- and only then can the retention questions
  ADR-0007 raises be answered for it.
- A lane that pins `--max-shared-prefixes` currently pays for a catalog it cannot use.
