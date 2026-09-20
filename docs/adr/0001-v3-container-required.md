# ADR-0001: A v3 container is required, and v2 artifacts are upgraded offline

**Status:** accepted

## Context

The v3 reader rejects a v2 container with a message naming the upgrade tool. The QUASAR artifact
is published by cometkim as v2; the NVFP4-full artifact is published as v3. So one of the two
artifacts we ship needs a conversion step before it can run at all.

## Decision

`download_model.py` stages the v2 artifact under its own name (`...qat.v2.ninfer`) and prints the
upgrade command with resolved absolute paths. The upgrader ships inside the release archive,
together with the `chat_templates/` data it reads. The upgrader produces the `.v3.ninfer` path
the launchers expect.

## Consequences

- The upgrade needs Python and `huggingface_hub` on the user's machine.
- The downloader must never target the `.v3.` path directly. It did once: a re-run after the
  upgrade would see a size mismatch, re-download, and overwrite the upgraded artifact with v2
  content, silently undoing it. That was a real bug, found in review.
- The archive must ship `chat_templates/` alongside the tool or the upgrade fails with
  `FileNotFoundError`. Also a real bug, from the same review.

## Why this needs recording

The obvious "simplification" is to point the downloader at the launcher's filename. That is the
exact change that caused the clobber.

## Amendment (2026-09-20): the QUASAR repository was republished as v3

`cometkim/Qwen3.8-27B-nvfp4qat-NInfer` now serves a **v3** container (`NINFER\0\x03`, size
18,638,510,576, sha256 `8b86901a…`), so both artifacts download and run directly and the Context
above describes neither repository any more. What that changes:

- `download_model.py` targets the `.v3.ninfer` path directly again, and the clobber in
  Consequences cannot recur while the source is v3: a re-download overwrites a v3 file with the
  same v3 file. The `...qat.v2.ninfer` staging name went with the v2 source.
- The upgrader still ships, for a copy fetched before the republish.
- **Re-derive a pin from the repository's blob metadata**
  (`https://huggingface.co/api/models/<repo>?blobs=true`), never from a local copy. A local copy is
  not the published file in more than name: this one is a different container whose upgrader emits
  the fused draft binding (ADR-0003) rather than the repository's upstream-shaped one, and the two
  measure 314.3 against 341.7 tok/s. A size check cannot see that; the pin can, once it is current.
