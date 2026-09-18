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
