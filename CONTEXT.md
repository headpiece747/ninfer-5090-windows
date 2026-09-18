# Context

Domain vocabulary for this repo. **Concepts and names only.** No ceilings, ports, speeds or
artifact sizes: those live in one profile table under `tools/release/` and drift every time
they are restated. If a fact here needs a number, it belongs there or in the launcher that
ships it.

Two names carry an artifact, not a preference: the **QUASAR artifact** is our own
quantisation-aware-trained image, and the **NVFP4-full artifact** is the fuller one published
by cometkim. "ninfer" on its own is ambiguous and should not be used.

## Launch surface

**profile** — one shippable combination of artifact, spec route, vision and ceiling. A profile is
data, not a file: the launcher, the verifier, the measurement harnesses and the docs are all
consumers of it.

**launcher** — the `.bat` a user double-clicks to start one profile. It resolves the engine and
artifact beside itself, checks both exist, and starts one engine.

**model id** — the string the engine enforces on every request. The engine rejects a request
naming anything else, so a launcher's model id and the entry a client uses must match exactly.
The display name a client shows is free text and can change without consequence.

**ceiling** — the highest context the engine accepts for a profile. It is refused above that
point with byte accounting, never silently reduced, so a ceiling is a measured value rather
than a configured one.

## Speculation

**spec route** — how draft tokens are proposed: **MTP** or **DFlash2**. MTP is lighter and needs
only the artifact's MTP head; DFlash2 needs a companion module and proposes deeper.

**draft depth** — how many tokens a route proposes per round. Chosen per profile by measurement.

**acceptance** — the fraction of proposed tokens the target commits. It does not predict
throughput: tokens committed per round matters more, so depth is chosen on measured decode rate.

**proposal head** — the optimised head the `--lm-head-draft` flag selects. Its value differs per
profile, and it can cost headroom as well as buy speed.

## Memory

**artifact** — the `.ninfer` file. The **v3 container** has a JSON index, declared bindings and
named components; the **v2 container** is rejected outright by a v3 engine, so upgrading is
mandatory rather than optional.

**device weights** — what the artifact costs in VRAM once materialised. This, not the file size,
is what bounds a ceiling.

**KV pool** — the context state, sized automatically from the VRAM left after weights. Its dtype
trades precision for room.

**prefix cache** — retention of prepared context so a later request reuses it instead of
re-prefilling. Retention is bounded by three budgets: **shared prefixes**, **private
continuations**, and **anchors**. These, not the host pool's size, are what limit it.

## Practices

**measured, not assumed** — every number a profile ships is backed by a record. Where a claim
cannot be measured, say so rather than inferring it.

**generated, not transcribed** — a fact restated by hand is a drift site. Two documented drifts
in this repo's docs came from transcription, not from wrong measurement.
