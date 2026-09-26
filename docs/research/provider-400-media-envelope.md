docs(research): the provider 400 is the media envelope, and the config key that hid it

Reproduced, after a long narrowing. The desktop app reports a rejected request as
`Provider request failed with HTTP 400` with a stack trace and no body, and the engine's reply -- which
names the cause -- is thrown away by the app. The engine's own log has it, so a lane started detached
from a console puts the only usable evidence nowhere.

What it is:

    12 images at 3840x2160   HTTP 400  media_budget_exceeded
                             "vision raw patches exceed processor budget"
    12 images at 1280x720    HTTP 200

Images reach the engine at their unresized 4K size because the image limits sat under `attachment`,
which is V1's key and which V2 does not read; V2's key is `media`. So the intended 1280x720 was never
applied and the 2000x2000 defaults governed instead. Twelve 4K images exceed the vision budget; the
same twelve at the intended size pass.

Nothing else was the cause, and each was excluded by measurement rather than by argument. Passed: a
275,000-token prompt (clamped, not rejected), a 48 MB body, a 7680x4320 single image, progressive,
CMYK, 16-bit, palette, interlaced and animated images, 64 images at 64x64, every reasoning-effort
value, tools, streaming, store, metadata, and the full header set the app sends -- including the
`gzip, deflate, br, zstd` accept-encoding that no probe had used. The engine rejects exactly four
things, each with its own words: `image_detail_not_supported` for a `detail` other than `auto`,
`invalid_media` for bytes it cannot decode, `modality_not_supported` for a part from another protocol,
and this envelope.

opencode's own request was captured and refuted the first hypothesis: its image part carries only
`url`, never `detail`, and 89876 characters of session text plus an image returns 200.

Also recorded: `v3_profile_matrix.py` stops every `ninfer-serve.exe` on the machine before a sweep, so
a measurement takes down a chat session's lane -- which is how a working lane was mistaken for a broken
provider in the middle of this.
